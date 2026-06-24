# Copyright (c) OpenMMLab. All rights reserved.
"""Unified FlashOCC merged OM inference + GT/Pred/input visualization.

Uses ``flashocc_car_grid_merged.om`` (img -> occ logits). Saves per-sample
BEV grids, input camera images, optional GT comparison, and profile charts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from mmcv import Config, DictAction

import mmdet
from mmdet3d.datasets import build_dataloader, build_dataset

if mmdet.__version__ > '2.23.0':
    from mmdet.utils import setup_multi_processes
else:
    from mmdet3d.utils import setup_multi_processes

try:
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.export_onnx_split_npu import (  # noqa: E402
    _get_sample_batch,
    _import_plugin,
)
from tools.run_split_infer_npu import (  # noqa: E402
    AclSession,
    _StageTimer,
    _print_profile_block,
    _save_profile_report,
    _sync_npu,
)
from tools.run_unified_infer_npu import (  # noqa: E402
    _img_from_data,
    _load_manifest,
    _resolve_e2e_om,
)
from tools.visualize_car_grid_test_detailed import (  # noqa: E402
    build_detailed_panel,
    put_title,
    save_assets,
    write_report,
)

E2E_PROFILE_STAGES = [
    'img_to_host',
    'e2e_om',
    'occ_decode',
    'om_total',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Merged OM inference with visualization and profiling')
    parser.add_argument(
        'config',
        default='projects/configs/flashocc/flashocc-r50-car-grid-trt.py',
        nargs='?',
        help='deploy config')
    parser.add_argument(
        'manifest',
        default='work_dirs/onnx_unified_car_grid/'
                'flashocc_car_grid_unified_deploy_manifest.json',
        nargs='?',
        help='unified deploy manifest json')
    parser.add_argument(
        '--om-path',
        default='work_dirs/onnx_unified_car_grid/flashocc_car_grid_merged.om',
        help='merged .om path')
    parser.add_argument(
        '--out-dir',
        default='work_dirs/unified_viz_results',
        help='directory for panels, grids, profile outputs')
    parser.add_argument('--sample-idx', type=int, default=0)
    parser.add_argument(
        '--samples', type=int, default=0,
        help='number of frames (0 = all in dataset)')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--profile-iters', type=int, default=3)
    parser.add_argument('--profile-warmup', type=int, default=2)
    parser.add_argument(
        '--profile-out',
        default=None,
        help='JSON path for profile report (default: <out-dir>/profile.json)')
    parser.add_argument(
        '--no-gt',
        action='store_true',
        help='skip GT panels when occ labels are unavailable')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    return parser.parse_args()


def decode_occ_logits(logits: np.ndarray, occ_shape: tuple) -> np.ndarray:
    """Logits -> semantic grid (Gx, Gy, Z)."""
    arr = np.asarray(logits, dtype=np.float32).reshape(occ_shape)
    if arr.ndim == 5:
        pred = arr.argmax(axis=-1)[0]
    elif arr.ndim == 4:
        pred = arr.argmax(axis=-1)
    else:
        raise ValueError(f'unexpected occ logits ndim after reshape: {arr.ndim}')
    return pred.astype(np.int64)


def _has_gt(info: dict, no_gt: bool) -> bool:
    if no_gt:
        return False
    occ = info.get('occ_path')
    if not occ:
        return False
    labels = ROOT / occ / 'labels.npz'
    return labels.is_file()


def _build_input_only_panel(info: dict, pred: np.ndarray) -> np.ndarray:
    """Fallback panel when GT is missing: cameras + pred BEV only."""
    from tools.visualize_car_grid_test_detailed import (
        bev_sem, hstack_row, load_rgb, put_title, resolve_cam_path)

    row_h = 260
    rows = []
    cam_row = []
    for cam in ['CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']:
        rgb = load_rgb(resolve_cam_path(info, cam, 'data_path'))
        cam_row.append(put_title(rgb, cam))
    rows.append(hstack_row(cam_row, row_h))

    pred_row = [
        put_title(bev_sem(pred[:, :, 0]), 'Pred z0'),
        put_title(bev_sem(pred[:, :, 1]), 'Pred z1'),
    ]
    rows.append(hstack_row(pred_row, row_h))

    panel_w = max(r.shape[1] for r in rows)
    canvas = np.zeros((sum(r.shape[0] for r in rows), panel_w, 3), np.uint8)
    y = 0
    for r in rows:
        canvas[y:y + r.shape[0], :r.shape[1]] = r
        y += r.shape[0]
    stem = info.get('sample_stem', info.get('token', 'sample')[:8])
    return put_title(canvas, f'{stem} | pred only (no GT)')


def render_profile_chart(
        timings: dict,
        ordered_keys: list[str],
        title: str,
        out_path: Path,
        width: int = 720,
        height: int = 360) -> None:
    keys = [k for k in ordered_keys if k in timings and k != ordered_keys[-1]]
    if not keys:
        return
    vals = [timings[k] * 1000.0 for k in keys]
    total_ms = sum(vals)
    canvas = np.ones((height, width, 3), np.uint8) * 255
    cv2.putText(canvas, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (20, 20, 20), 2, cv2.LINE_AA)
    bar_x0, bar_y0 = 160, 50
    bar_h, gap = 28, 8
    colors = [
        (70, 130, 220), (60, 180, 75), (220, 120, 60),
        (180, 80, 180), (80, 180, 180), (200, 200, 60),
    ]
    max_ms = max(vals) if vals else 1.0
    plot_w = width - bar_x0 - 24
    for i, (key, ms) in enumerate(zip(keys, vals)):
        y = bar_y0 + i * (bar_h + gap)
        w = int(ms / max_ms * plot_w) if max_ms > 0 else 0
        color = colors[i % len(colors)]
        cv2.rectangle(canvas, (bar_x0, y), (bar_x0 + w, y + bar_h), color, -1)
        pct = 100.0 * ms / total_ms if total_ms > 0 else 0.0
        label = f'{key}  {ms:.2f}ms ({pct:.1f}%)'
        cv2.putText(canvas, label, (12, y + bar_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(canvas, f'total {total_ms:.2f} ms', (12, height - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 1, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def run_e2e_timed(acl_sess, img_np: np.ndarray, occ_shape: tuple,
                  timer: _StageTimer | None = None):
    if timer is None:
        occ_raw = acl_sess.infer('e2e', img_np)[0]
        pred = decode_occ_logits(occ_raw, occ_shape)
        return pred, occ_raw

    with timer.measure('om_total'):
        with timer.measure('img_to_host'):
            img_np = np.ascontiguousarray(img_np, dtype=np.float32)
        with timer.measure('e2e_om'):
            occ_raw = acl_sess.infer('e2e', img_np)[0]
        with timer.measure('occ_decode'):
            pred = decode_occ_logits(occ_raw, occ_shape)
    return pred, occ_raw



def main():
    args = parse_args()
    os.chdir(ROOT)

    manifest = _load_manifest(args.manifest)
    om_path = _resolve_e2e_om(manifest, args.manifest, args.om_path)
    if not os.path.isfile(om_path):
        raise FileNotFoundError(f'Missing merged OM: {om_path}')

    occ_shape = tuple(manifest.get('tensor_shapes', {}).get(
        'occ_out_0', (1, 200, 200, 2, 3)))

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg = compat_cfg(cfg)
    setup_multi_processes(cfg)
    _import_plugin(cfg, args.config)

    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False)

    n_total = len(dataset)
    n_run = n_total if args.samples <= 0 else min(args.samples, n_total - args.sample_idx)
    if n_run <= 0:
        raise ValueError(f'no samples to run (dataset={n_total}, sample_idx={args.sample_idx})')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / 'samples'
    vis_dir.mkdir(parents=True, exist_ok=True)

    print(f'merged OM: {om_path}')
    print(f'output dir: {out_dir}')
    print(f'samples: {n_run} (from idx {args.sample_idx})')

    acl_sess = AclSession(device_id=args.gpu_id)
    acl_sess.load('e2e', om_path)
    all_meta = []
    thumbs = []

    try:
        for si in range(n_run):
            idx = args.sample_idx + si
            data = _get_sample_batch(data_loader, idx)
            info = dataset.data_infos[idx]
            img_np = _img_from_data(data)
            pred, occ_raw = run_e2e_timed(acl_sess, img_np, occ_shape)

            stem = info.get('sample_stem', info.get('token', f's{idx}')[:8])
            tag = f'{si:02d}_{stem}'
            sample_dir = vis_dir / tag
            sample_dir.mkdir(parents=True, exist_ok=True)

            np.savez_compressed(
                sample_dir / 'pred.npz',
                pred=pred,
                logits=occ_raw.reshape(occ_shape))

            meta = {
                'index': idx,
                'stem': stem,
                'token': info.get('token', ''),
                'scene': info.get('scene_name', ''),
                'has_gt': _has_gt(info, args.no_gt),
            }

            if meta['has_gt']:
                panel = build_detailed_panel(info, pred, meta)
                save_assets(info, pred, sample_dir)
            else:
                panel = _build_input_only_panel(info, pred)
                from tools.visualize_car_grid_test_detailed import (
                    bev_sem, load_rgb, resolve_cam_path)
                for cam in ['CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']:
                    cv2.imwrite(
                        str(sample_dir / f'{cam}.png'),
                        load_rgb(resolve_cam_path(info, cam, 'data_path'), 1024))
                cv2.imwrite(str(sample_dir / 'pred_z0.png'), bev_sem(pred[:, :, 0]))
                cv2.imwrite(str(sample_dir / 'pred_z1.png'), bev_sem(pred[:, :, 1]))

            cv2.imwrite(str(sample_dir / 'detailed_panel.png'), panel)
            if meta['has_gt']:
                write_report(meta, sample_dir / 'report.txt')

            thumb = cv2.resize(
                panel, (920, max(1, int(920 * panel.shape[0] / panel.shape[1]))))
            title = stem
            if meta.get('mIoU') is not None:
                title += f' mIoU={meta["mIoU"]:.3f}'
            thumbs.append(put_title(thumb, title))
            meta['panel'] = os.path.relpath(
                str(sample_dir / 'detailed_panel.png'), str(ROOT))
            all_meta.append(meta)

            print(f'--- sample {idx} ({stem}) ---')
            print(f'  pred shape={pred.shape} '
                  f'classes={np.unique(pred).tolist()} '
                  f'has_gt={meta["has_gt"]}')

        if args.profile and n_run > 0:
            data = _get_sample_batch(data_loader, args.sample_idx)
            img_np = _img_from_data(data)
            print(f'\n[profile] warmup={args.profile_warmup}, '
                  f'iters={args.profile_iters}')
            for _ in range(args.profile_warmup):
                run_e2e_timed(acl_sess, img_np, occ_shape)
            _sync_npu()

            timer = _StageTimer()
            for _ in range(args.profile_iters):
                run_e2e_timed(acl_sess, img_np, occ_shape, timer=timer)
            avg = timer.average(args.profile_iters)
            _print_profile_block(
                f'E2E merged OM (avg of {args.profile_iters} iters)',
                avg, E2E_PROFILE_STAGES)

            profile_out = args.profile_out or str(out_dir / 'profile.json')
            report = {
                'om_path': om_path,
                'profile_warmup': args.profile_warmup,
                'profile_iters': args.profile_iters,
                'e2e_merged_seconds': avg,
            }
            _save_profile_report(profile_out, report)
            render_profile_chart(
                avg, E2E_PROFILE_STAGES,
                f'E2E merged OM profile ({args.profile_iters} iters)',
                out_dir / 'profile_chart.png')

    finally:
        acl_sess.close()

    if thumbs:
        cols = 2
        rows_n = (len(thumbs) + cols - 1) // cols
        th, tw = thumbs[0].shape[:2]
        grid = np.zeros((rows_n * th, cols * tw, 3), np.uint8)
        for i, t in enumerate(thumbs):
            r, c = divmod(i, cols)
            grid[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = t
        cv2.imwrite(str(out_dir / 'summary_grid.png'), grid)

    manifest_path = out_dir / 'manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(all_meta, f, indent=2, ensure_ascii=False)

    print(f'\nSaved {len(all_meta)} sample visualizations -> {vis_dir}')
    print(f'  summary: {out_dir / "summary_grid.png"}')
    if args.profile:
        print(f'  profile: {out_dir / "profile.json"}')
        print(f'  profile chart: {out_dir / "profile_chart.png"}')


if __name__ == '__main__':
    main()
