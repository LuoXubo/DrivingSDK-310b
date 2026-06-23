#!/usr/bin/env python3
"""Detailed GT vs Pred visualization for car_perception_grid test results."""
from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GX, GY = 200, 200
EGO_GX, EGO_GY = 100, 100
DEPTH_INVALID = 60000.0

SEM_COLORS = {0: (60, 200, 60), 1: (40, 40, 220), 255: (160, 160, 160)}
CLASS_NAMES = ['passable', 'car', 'unknown']


def put_title(img: np.ndarray, text: str, bar_h: int = 32) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], bar_h), (0, 0, 0), -1)
    cv2.putText(out, text, (8, bar_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def load_rgb(path: Path, max_side: int = 480) -> np.ndarray:
    p = path if path.is_absolute() else ROOT / path
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(p)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)))
    return img


def colorize_depth(path: Path, max_side: int = 480) -> np.ndarray:
    p = path if path.is_absolute() else ROOT / path
    depth = np.load(p).astype(np.float32)
    valid = (depth > 0.05) & (depth < DEPTH_INVALID)
    vis = np.zeros(depth.shape, np.uint8)
    if valid.any():
        lo, hi = np.percentile(depth[valid], [2, 98])
        norm = np.clip((depth - lo) / max(hi - lo, 1e-3), 0, 1)
        vis[valid] = (norm[valid] * 255).astype(np.uint8)
    color = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    h, w = color.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        color = cv2.resize(color, (int(w * s), int(h * s)))
    return color


def bev_sem(sem2d: np.ndarray, mask2d: np.ndarray | None = None,
            size: int = 400) -> np.ndarray:
    out = np.full((GX, GY, 3), 40, np.uint8)
    for val, color in SEM_COLORS.items():
        out[sem2d == val] = color
    if mask2d is not None:
        m = mask2d.astype(bool)
        blend = out.copy()
        blend[m] = (blend[m] * 0.45 + np.array((255, 180, 0)) * 0.55).astype(np.uint8)
        out = blend
    out = cv2.flip(out, 0)
    out = cv2.resize(out, (size, size), interpolation=cv2.INTER_NEAREST)
    ex = int(EGO_GX / GX * size)
    ey = int((GY - 1 - EGO_GY) / GY * size)
    cv2.drawMarker(out, (ex, ey), (255, 255, 0), cv2.MARKER_CROSS, 14, 2)
    return out


def bev_error(gt2d: np.ndarray, pred2d: np.ndarray, mask2d: np.ndarray,
              size: int = 400) -> np.ndarray:
    """Yellow=correct, cyan=FN (missed car), magenta=FP (false car), gray=ignore/outside mask."""
    valid = mask2d.astype(bool) & (gt2d != 255)
    out = np.full((GX, GY, 3), 40, np.uint8)
    out[~valid] = (70, 70, 70)
    correct = valid & (gt2d == pred2d)
    out[correct] = (40, 220, 40)
    fn = valid & (gt2d == 1) & (pred2d != 1)
    fp = valid & (gt2d != 1) & (pred2d == 1)
    out[fn] = (255, 255, 0)
    out[fp] = (255, 0, 255)
    out = cv2.flip(out, 0)
    out = cv2.resize(out, (size, size), interpolation=cv2.INTER_NEAREST)
    return out


def bev_mask(mask2d: np.ndarray, size: int = 400, label: str = '') -> np.ndarray:
    m = (mask2d > 0).astype(np.uint8) * 255
    m = cv2.flip(m, 0)
    color = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
    out = cv2.resize(color, (size, size), interpolation=cv2.INTER_NEAREST)
    if m.sum() == 0:
        cv2.putText(out, label or 'empty layer', (20, size // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2, cv2.LINE_AA)
    return out


def load_seg_color(path: Path | None, max_side: int = 480) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    cls = raw[:, :, 2] if raw.ndim == 3 else raw
    color = np.zeros((*cls.shape, 3), np.uint8)
    color[cls == 0] = (60, 200, 60)
    color[cls == 1] = (40, 40, 220)
    color[cls == 2] = (180, 180, 40)
    h, w = color.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        color = cv2.resize(color, (int(w * s), int(h * s)))
    return color


def overlay_seg_on_rgb(rgb: np.ndarray, seg_path: Path | None, alpha: float = 0.45):
    if seg_path is None or not seg_path.exists():
        return rgb
    raw = cv2.imread(str(seg_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return rgb
    cls = raw[:, :, 2] if raw.ndim == 3 else raw
    if cls.shape[:2] != rgb.shape[:2]:
        cls = cv2.resize(cls, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    color = np.zeros_like(rgb)
    color[cls == 0] = (60, 200, 60)
    color[cls == 1] = (40, 40, 220)
    color[cls == 2] = (180, 180, 40)
    mask = cls != 2
    out = rgb.copy()
    out[mask] = (rgb[mask] * (1 - alpha) + color[mask] * alpha).astype(np.uint8)
    return out


def seg_path_for(info: dict, stem: str, cam_id: int) -> Path | None:
    pose = info.get('src_pose_dir')
    if not pose:
        return None
    p = Path(pose) / f'{stem}_seg_cam{cam_id}_raw.png'
    return p if p.exists() else None


def bev_car_only(sem2d: np.ndarray, mask2d: np.ndarray, size: int = 400) -> np.ndarray:
    out = np.zeros((GX, GY, 3), np.uint8)
    car = (sem2d == 1) & mask2d.astype(bool)
    out[car] = (40, 40, 220)
    out = cv2.flip(out, 0)
    return cv2.resize(out, (size, size), interpolation=cv2.INTER_NEAREST)


def metrics_panel(gt: np.ndarray, pred: np.ndarray, mc: np.ndarray,
                  layer: int, size: int = 400) -> np.ndarray:
    s_gt, s_pred = gt[:, :, layer], pred[:, :, layer]
    m = mc[:, :, layer].astype(bool)
    valid = m & (s_gt != 255)
    panel = np.zeros((size, size, 3), np.uint8)
    if not valid.any():
        cv2.putText(panel, 'no valid voxels', (20, size // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        return panel

    g, p = s_gt[valid], s_pred[valid]
    tp_car = int(((g == 1) & (p == 1)).sum())
    fn_car = int(((g == 1) & (p != 1)).sum())
    fp_car = int(((g != 1) & (p == 1)).sum())
    tp_pass = int(((g == 0) & (p == 0)).sum())
    iou_car = tp_car / max(tp_car + fn_car + fp_car, 1)
    iou_pass = tp_pass / max(int((g == 0).sum()) + int((p == 0).sum()) - tp_pass, 1)

    lines = [
        f'z{layer} (cam mask only)',
        f'valid: {int(valid.sum())}',
        f'GT car: {int((g == 1).sum())}  Pred car: {int((p == 1).sum())}',
        f'car TP/FN/FP: {tp_car}/{fn_car}/{fp_car}',
        f'car IoU: {iou_car:.3f}',
        f'passable IoU: {iou_pass:.3f}',
        '',
        'error colors:',
        'green=correct',
        'yellow=missed car (FN)',
        'magenta=false car (FP)',
    ]
    for i, line in enumerate(lines):
        cv2.putText(panel, line, (10, 26 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (230, 230, 230), 1, cv2.LINE_AA)
    return panel


def hstack_row(images: list[np.ndarray], h: int) -> np.ndarray:
    parts = []
    for im in images:
        nw = max(1, int(im.shape[1] * h / im.shape[0]))
        parts.append(cv2.resize(im, (nw, h)))
    return np.hstack(parts)


def resolve_cam_path(info: dict, cam: str, key: str) -> Path:
    p = info['cams'][cam][key]
    return Path(p) if os.path.isabs(p) else ROOT / p


def sample_metrics(gt: np.ndarray, pred: np.ndarray, mc: np.ndarray) -> dict:
    valid = mc.astype(bool) & (gt != 255)
    g, p = gt[valid], pred[valid]
    hist = np.zeros((3, 3), dtype=np.int64)
    for c in range(3):
        for d in range(3):
            hist[c, d] = int(((g == c) & (p == d)).sum())
    ious = []
    for c in range(3):
        tp = hist[c, c]
        denom = hist[c, :].sum() + hist[:, c].sum() - tp
        ious.append(float(tp / denom) if denom > 0 else float('nan'))
    return {
        'mIoU': float(np.nanmean(ious)),
        'class_iou': {CLASS_NAMES[i]: ious[i] for i in range(3)},
        'car_gt': int((g == 1).sum()),
        'car_pred': int((p == 1).sum()),
        'car_tp': int(hist[1, 1]),
        'car_fn': int(hist[1, 0]) + int(hist[1, 2]),
        'car_fp': int(hist[0, 1]) + int(hist[2, 1]),
        'valid_voxels': int(valid.sum()),
    }


def build_detailed_panel(info: dict, pred: np.ndarray, meta: dict) -> np.ndarray:
    gt_pack = np.load(ROOT / info['occ_path'] / 'labels.npz')
    gt = gt_pack['semantics']
    mc = gt_pack['mask_camera']
    ml = gt_pack['mask_lidar']
    stem = info.get('sample_stem', info['token'][:8])
    row_h = 260
    rows = []

    cam_row, depth_row, seg_row, overlay_row = [], [], [], []
    for cam_id, cam in [(1, 'CAM_FRONT_LEFT'), (2, 'CAM_FRONT_RIGHT')]:
        rgb = load_rgb(resolve_cam_path(info, cam, 'data_path'))
        cam_row.append(put_title(rgb, cam))
        depth_row.append(put_title(colorize_depth(resolve_cam_path(info, cam, 'depth_path')),
                                   f'{cam} depth'))
        seg_p = seg_path_for(info, stem, cam_id)
        seg_img = load_seg_color(seg_p)
        if seg_img is not None:
            seg_row.append(put_title(seg_img, f'{cam} seg'))
            overlay_row.append(put_title(overlay_seg_on_rgb(rgb, seg_p), f'{cam} rgb+seg'))
        else:
            blank = np.zeros_like(rgb)
            seg_row.append(put_title(blank, f'{cam} seg N/A'))
            overlay_row.append(put_title(rgb, f'{cam} rgb only'))

    rows.append(hstack_row(cam_row, row_h))
    rows.append(hstack_row(depth_row, row_h))
    rows.append(hstack_row(seg_row, row_h))
    rows.append(hstack_row(overlay_row, row_h))

    mc_any = (mc[:, :, 0] > 0) | (mc[:, :, 1] > 0)
    mask_row = [
        put_title(bev_mask(mc[:, :, 0], label='z0 empty (see z1)'), 'cam mask z0'),
        put_title(bev_mask(mc[:, :, 1]), 'cam mask z1'),
        put_title(bev_mask(mc_any.astype(np.uint8)), 'cam mask z0|z1'),
        put_title(bev_mask(ml[:, :, 0]), 'lidar mask z0'),
        put_title(bev_mask(ml[:, :, 1]), 'lidar mask z1'),
    ]
    rows.append(hstack_row(mask_row, row_h))

    bev_row0 = [
        put_title(bev_sem(gt[:, :, 0], mc[:, :, 0]), 'GT z0'),
        put_title(bev_sem(pred[:, :, 0], mc[:, :, 0]), 'Pred z0'),
        put_title(bev_error(gt[:, :, 0], pred[:, :, 0], mc[:, :, 0]), 'Error z0'),
        put_title(bev_car_only(gt[:, :, 0], mc[:, :, 0]), 'GT car z0'),
        put_title(bev_car_only(pred[:, :, 0], mc[:, :, 0]), 'Pred car z0'),
        put_title(metrics_panel(gt, pred, mc, 0), 'z0 metrics'),
    ]
    bev_row1 = [
        put_title(bev_sem(gt[:, :, 1], mc[:, :, 1]), 'GT z1'),
        put_title(bev_sem(pred[:, :, 1], mc[:, :, 1]), 'Pred z1'),
        put_title(bev_error(gt[:, :, 1], pred[:, :, 1], mc[:, :, 1]), 'Error z1'),
        put_title(bev_car_only(gt[:, :, 1], mc[:, :, 1]), 'GT car z1'),
        put_title(bev_car_only(pred[:, :, 1], mc[:, :, 1]), 'Pred car z1'),
        put_title(metrics_panel(gt, pred, mc, 1), 'z1 metrics'),
    ]
    rows.append(hstack_row(bev_row0, row_h))
    rows.append(hstack_row(bev_row1, row_h))

    panel_w = max(r.shape[1] for r in rows)
    title_h = 44
    canvas = np.zeros((sum(r.shape[0] for r in rows) + title_h, panel_w, 3), np.uint8)
    y = 0
    for r in rows:
        canvas[y:y + r.shape[0], :r.shape[1]] = r
        y += r.shape[0]

    m = sample_metrics(gt, pred, mc)
    title = (f'{stem} | mIoU={m["mIoU"]:.3f} car={m["class_iou"]["car"]:.3f} '
             f'pass={m["class_iou"]["passable"]:.3f} | '
             f'car TP/FN/FP={m["car_tp"]}/{m["car_fn"]}/{m["car_fp"]}')
    cv2.rectangle(canvas, (0, y), (panel_w, canvas.shape[0]), (25, 25, 25), -1)
    cv2.putText(canvas, title, (10, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 2, cv2.LINE_AA)
    meta.update(m)
    meta['stem'] = stem
    return canvas


def write_report(meta: dict, path: Path):
    lines = [
        f"stem: {meta.get('stem')}",
        f"tag: {meta.get('tag')}",
        f"split: {meta.get('split')} index={meta.get('index')}",
        f"mIoU: {meta.get('mIoU', 0):.4f}",
        f"class_iou: {meta.get('class_iou')}",
        f"valid_voxels: {meta.get('valid_voxels')}",
        f"car_gt: {meta.get('car_gt')} car_pred: {meta.get('car_pred')}",
        f"car TP/FN/FP: {meta.get('car_tp')}/{meta.get('car_fn')}/{meta.get('car_fp')}",
        '',
        'Legend:',
        '  GT/Pred: green=passable, red=car, gray=ignore, orange=cam mask',
        '  Error: green=correct, yellow=missed car (FN), magenta=false car (FP)',
        '  cam mask z0 often empty; supervision mainly on z1',
    ]
    path.write_text('\n'.join(lines), encoding='utf-8')


def load_pred_for_sample(eval_dir: Path, idx: int, tag: str, stem: str) -> np.ndarray | None:
    vis = eval_dir / 'vis_panels'
    candidates = [
        vis / f'{idx:02d}_{tag}' / 'pred.npz',
        vis / f'{idx:02d}_{tag}_{stem}' / 'pred.npz',
    ]
    if vis.exists():
        for d in vis.iterdir():
            if stem in d.name and (d / 'pred.npz').exists():
                candidates.append(d / 'pred.npz')
    for pred_path in candidates:
        if pred_path.exists():
            z = np.load(pred_path)
            return z['pred'] if 'pred' in z else z[z.files[0]]
    return None


def save_assets(info: dict, pred: np.ndarray, out_dir: Path):
    gt_pack = np.load(ROOT / info['occ_path'] / 'labels.npz')
    gt, mc, ml = gt_pack['semantics'], gt_pack['mask_camera'], gt_pack['mask_lidar']
    mc_any = ((mc[:, :, 0] > 0) | (mc[:, :, 1] > 0)).astype(np.uint8)
    for zl in [0, 1]:
        cv2.imwrite(str(out_dir / f'gt_z{zl}.png'), bev_sem(gt[:, :, zl], mc[:, :, zl]))
        cv2.imwrite(str(out_dir / f'pred_z{zl}.png'), bev_sem(pred[:, :, zl], mc[:, :, zl]))
        cv2.imwrite(str(out_dir / f'error_z{zl}.png'),
                    bev_error(gt[:, :, zl], pred[:, :, zl], mc[:, :, zl]))
        cv2.imwrite(str(out_dir / f'cam_mask_z{zl}.png'),
                    bev_mask(mc[:, :, zl], label='z0 empty (see z1)' if zl == 0 else ''))
        cv2.imwrite(str(out_dir / f'lidar_mask_z{zl}.png'), bev_mask(ml[:, :, zl]))
    cv2.imwrite(str(out_dir / 'cam_mask_combined.png'), bev_mask(mc_any))
    for cam in ['CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']:
        cv2.imwrite(str(out_dir / f'{cam}.png'),
                    load_rgb(resolve_cam_path(info, cam, 'data_path'), max_side=1024))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-dir', default='work_dirs/car_grid_test_results')
    parser.add_argument('--out-dir', default='work_dirs/car_grid_test_results/vis_detailed')
    parser.add_argument('--test-pkl',
                        default='data/car_perception_grid/nuscenes/'
                                'bevdetv2-nuscenes_infos_test_eval.pkl')
    parser.add_argument('--checkpoint', default='',
                        help='if set and preds missing, run inference first')
    parser.add_argument('--config',
                        default='projects/configs/flashocc/flashocc-r50-car-grid.py')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--max-samples', type=int, default=0)
    args = parser.parse_args()

    os.chdir(ROOT)
    eval_dir = ROOT / args.eval_dir
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.load(open(eval_dir / 'test_manifest.json'))
    if args.max_samples > 0:
        manifest = manifest[:args.max_samples]

    test_pkl = ROOT / args.test_pkl
    infos = pickle.load(open(test_pkl, 'rb'))['infos']

    preds: list[np.ndarray | None] = []
    need_infer = []
    for i, m in enumerate(manifest):
        p = load_pred_for_sample(eval_dir, i, m['tag'], m['stem'])
        preds.append(p)
        if p is None:
            need_infer.append(i)

    if need_infer and args.checkpoint:
        print(f'Running inference for {len(need_infer)} samples without saved preds ...')
        from eval_car_grid_occ import run_inference, _import_plugin
        from mmcv import Config
        cfg = Config.fromfile(args.config)
        cfg.merge_from_dict({
            'data.test.ann_file': str(test_pkl.relative_to(ROOT)),
            'data.test.data_root': 'data/car_perception_grid/nuscenes/',
        })
        _import_plugin(cfg, args.config)
        from mx_driving.patcher import PatcherBuilder, Patch
        from mx_driving.patcher import batch_matmul, resnet_add_relu
        pb = (PatcherBuilder()
              .add_module_patch('torch', Patch(batch_matmul))
              .add_module_patch('mmdet', Patch(resnet_add_relu)))
        with pb.build():
            _, infer_preds = run_inference(cfg, args.checkpoint, args.gpu_id)
        for j, p in enumerate(infer_preds):
            if preds[j] is None:
                preds[j] = p

    if any(p is None for p in preds):
        missing = sum(p is None for p in preds)
        raise FileNotFoundError(
            f'{missing} samples missing pred.npz under {eval_dir}/vis_panels. '
            f'Re-run: bash test/eval_car_grid.sh <checkpoint> or pass --checkpoint')

    thumbs = []
    all_meta = []
    for i, (m, pred, info) in enumerate(zip(manifest, preds, infos)):
        meta = dict(m)
        panel = build_detailed_panel(info, pred, meta)
        sample_dir = out_dir / f'{i:02d}_{m["tag"]}'
        sample_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(sample_dir / 'detailed_panel.png'), panel)
        write_report(meta, sample_dir / 'report.txt')
        np.savez_compressed(sample_dir / 'pred.npz', pred=pred)
        save_assets(info, pred, sample_dir)

        thumb = cv2.resize(panel, (920, int(920 * panel.shape[0] / panel.shape[1])))
        car_iou = meta.get('class_iou', {}).get('car', 0)
        thumbs.append(put_title(thumb, f'[{i}] {m["stem"]} carIoU={car_iou:.3f}'))
        meta['panel'] = str((sample_dir / 'detailed_panel.png').relative_to(ROOT))
        all_meta.append(meta)

    if thumbs:
        cols = 2
        rows_n = (len(thumbs) + cols - 1) // cols
        th, tw = thumbs[0].shape[:2]
        grid = np.zeros((rows_n * th, cols * tw, 3), np.uint8)
        for i, t in enumerate(thumbs):
            r, c = divmod(i, cols)
            grid[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = t
        cv2.imwrite(str(out_dir / 'summary_grid.png'), grid)

    legend = np.zeros((260, 700, 3), np.uint8)
    items = [
        ('GT passable', (60, 200, 60)),
        ('GT/Pred car', (40, 40, 220)),
        ('cam mask', (255, 180, 0)),
        ('correct', (40, 220, 40)),
        ('missed car FN', (255, 255, 0)),
        ('false car FP', (255, 0, 255)),
    ]
    for i, (name, color) in enumerate(items):
        y = 24 + i * 38
        cv2.rectangle(legend, (12, y - 14), (36, y + 6), color, -1)
        cv2.putText(legend, name, (48, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / 'legend.png'), legend)

    with open(out_dir / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(all_meta, f, indent=2, ensure_ascii=False)

    print(f'Saved {len(all_meta)} detailed test panels -> {out_dir}')
    print(f'  summary: {out_dir / "summary_grid.png"}')


if __name__ == '__main__':
    main()
