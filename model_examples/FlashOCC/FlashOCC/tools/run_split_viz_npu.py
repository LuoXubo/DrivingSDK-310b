# Copyright (c) OpenMMLab. All rights reserved.
"""Split deploy Torch NPU inference + GT/Pred visualization (test10).

Uses PyTorch on NPU: eval-aligned part1 -> bev_pool_v3 -> eval-aligned part3.
Output layout matches ``run_unified_viz_npu.py`` but saved under a separate dir.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch_npu  # noqa: F401
from torch_npu.contrib import transfer_to_npu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.export_onnx_split_npu import (  # noqa: E402
    Part3ExportWrapper,
    _get_bev_pool_metas_v3,
    _get_sample_batch,
    build_part1_wrapper,
    part1_layout_from_manifest,
)
from tools.run_split_infer_npu import (  # noqa: E402
    _StageTimer,
    _print_profile_block,
    _save_profile_report,
    _sync_npu,
)
from tools.run_unified_viz_npu import (  # noqa: E402
    _build_input_only_panel,
    _has_gt,
    decode_occ_logits,
    render_profile_chart,
)
from tools.split_deploy_infer import (  # noqa: E402
    DEFAULT_OCC_SHAPE,
    build_deploy_model,
    run_split_pt_stages,
)
from tools.visualize_car_grid_test_detailed import (  # noqa: E402
    build_detailed_panel,
    put_title,
    save_assets,
    write_report,
)

PT_PROFILE_STAGES = [
    'part1',
    'bev_pool',
    'part3',
    'occ_decode',
    'pt_total',
]

def parse_args():
    p = argparse.ArgumentParser(
        description='Split deploy Torch NPU inference with visualization')
    p.add_argument(
        '--deploy-config',
        default='projects/configs/flashocc/flashocc-r50-car-grid-trt.py')
    p.add_argument(
        '--checkpoint',
        default='work_dirs/car_grid_v4/epoch_12_ema.pth')
    p.add_argument(
        '--split-manifest',
        default='work_dirs/onnx_split_car_grid_eval/'
                'flashocc_car_grid_deploy_manifest.json')
    p.add_argument(
        '--out-dir',
        default='work_dirs/test10_torch_npu_viz')
    p.add_argument('--sample-idx', type=int, default=0)
    p.add_argument(
        '--samples', type=int, default=0,
        help='0 = all samples in dataset')
    p.add_argument('--gpu-id', type=int, default=0)
    p.add_argument('--profile', action='store_true')
    p.add_argument('--profile-iters', type=int, default=3)
    p.add_argument('--profile-warmup', type=int, default=2)
    p.add_argument('--no-gt', action='store_true')
    return p.parse_args()


@torch.no_grad()
def run_split_pt_timed(model, part1, part3, img, ranks_bev, ranks_depth,
                       ranks_feat, device, data, occ_shape,
                       timer: _StageTimer | None = None,
                       bev_mode='bev_pool_v3'):
    if timer is None:
        stages = run_split_pt_stages(
            model, part1, part3, img, ranks_bev, ranks_depth, ranks_feat,
            device, occ_shape, data=data, bev_mode=bev_mode)
        return stages['pred'], stages['occ_np']

    with timer.measure('pt_total'):
        with timer.measure('part1'):
            tran, depth = part1(img)
        with timer.measure('bev_pool'):
            if bev_mode == 'vt_core' and data is not None:
                from tools.split_deploy_infer import run_bev_view_transform_core
                bev = run_bev_view_transform_core(
                    model, data, img, tran, depth, device)
            else:
                from tools.export_onnx_split_npu import _part1_to_bev_pool_inputs
                from tools.split_deploy_infer import run_bev_pool_v3
                depth_bev, feat_bev = _part1_to_bev_pool_inputs(
                    tran, depth, model.img_view_transformer)
                bev = run_bev_pool_v3(
                    model, depth_bev, feat_bev,
                    ranks_bev, ranks_depth, ranks_feat)
        with timer.measure('part3'):
            outs = part3(bev)
            occ_logits = outs[0] if isinstance(outs, (list, tuple)) else outs
        with timer.measure('occ_decode'):
            occ_np = occ_logits.detach().float().cpu().numpy()
            pred = decode_occ_logits(occ_np, occ_shape)
    return pred, occ_np


def main():
    args = parse_args()
    os.chdir(ROOT)

    with open(args.split_manifest, encoding='utf-8') as f:
        manifest = json.load(f)
    occ_shape = tuple(manifest.get('tensor_shapes', {}).get(
        'occ_out_0', DEFAULT_OCC_SHAPE))
    part1_layout = part1_layout_from_manifest(manifest)

    model, _, dataset, data_loader, device = build_deploy_model(
        args.deploy_config, args.checkpoint, args.gpu_id, eval_config=None)

    part1 = build_part1_wrapper(model, part1_layout).eval()
    part3 = Part3ExportWrapper(model).eval()

    n_total = len(dataset)
    n_run = n_total if args.samples <= 0 else min(
        args.samples, n_total - args.sample_idx)
    if n_run <= 0:
        raise ValueError(f'no samples (dataset={n_total}, idx={args.sample_idx})')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / 'samples'
    vis_dir.mkdir(parents=True, exist_ok=True)

    print(f'backend: split_deploy_torch_npu (part1+bev_pool_v3+part3 on NPU)')
    print(f'checkpoint: {args.checkpoint}')
    print(f'output dir: {out_dir}')
    print(f'samples: {n_run} (from idx {args.sample_idx})')

    all_meta = []
    thumbs = []

    for si in range(n_run):
        idx = args.sample_idx + si
        data = _get_sample_batch(data_loader, idx)
        info = dataset.data_infos[idx]
        img, ranks_bev, ranks_depth, ranks_feat = _get_bev_pool_metas_v3(
            model, data, device, layout=part1_layout)

        pred, occ_np = run_split_pt_timed(
            model, part1, part3, img, ranks_bev, ranks_depth, ranks_feat,
            device, data, occ_shape)

        stem = info.get('sample_stem', info.get('token', f's{idx}')[:8])
        tag = f'{si:02d}_{stem}'
        sample_dir = vis_dir / tag
        sample_dir.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            sample_dir / 'pred.npz',
            pred=pred,
            logits=occ_np.reshape(occ_shape))

        meta = {
            'index': idx,
            'stem': stem,
            'token': info.get('token', ''),
            'scene': info.get('scene_name', ''),
            'has_gt': _has_gt(info, args.no_gt),
            'backend': 'split_deploy_torch_npu',
        }

        if meta['has_gt']:
            panel = build_detailed_panel(info, pred, meta)
            save_assets(info, pred, sample_dir)
        else:
            panel = _build_input_only_panel(info, pred)

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
        print(f'  pred shape={pred.shape} classes={np.unique(pred).tolist()} '
              f'has_gt={meta["has_gt"]}')

    if args.profile and n_run > 0:
        data = _get_sample_batch(data_loader, args.sample_idx)
        img, ranks_bev, ranks_depth, ranks_feat = _get_bev_pool_metas_v3(
            model, data, device, layout=part1_layout)
        print(f'\n[profile] warmup={args.profile_warmup}, '
              f'iters={args.profile_iters}')
        for _ in range(args.profile_warmup):
            run_split_pt_timed(
                model, part1, part3, img, ranks_bev, ranks_depth, ranks_feat,
                device, data, occ_shape)
        _sync_npu()

        timer = _StageTimer()
        for _ in range(args.profile_iters):
            run_split_pt_timed(
                model, part1, part3, img, ranks_bev, ranks_depth, ranks_feat,
                device, data, occ_shape, timer=timer)
        avg = timer.average(args.profile_iters)
        _print_profile_block(
            f'Split deploy Torch NPU (avg of {args.profile_iters} iters)',
            avg, PT_PROFILE_STAGES)

        profile_out = str(out_dir / 'profile.json')
        report = {
            'backend': 'split_deploy_torch_npu',
            'checkpoint': args.checkpoint,
            'profile_warmup': args.profile_warmup,
            'profile_iters': args.profile_iters,
            'split_pt_seconds': avg,
        }
        _save_profile_report(profile_out, report)
        render_profile_chart(
            avg, PT_PROFILE_STAGES,
            f'Split deploy Torch NPU ({args.profile_iters} iters)',
            out_dir / 'profile_chart.png')

    if thumbs:
        cols = 2
        rows_n = (len(thumbs) + cols - 1) // cols
        th, tw = thumbs[0].shape[:2]
        grid = np.zeros((rows_n * th, cols * tw, 3), np.uint8)
        for i, t in enumerate(thumbs):
            r, c = divmod(i, cols)
            grid[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = t
        cv2.imwrite(str(out_dir / 'summary_grid.png'), grid)

    meta_doc = {
        'backend': 'split_deploy_torch_npu',
        'checkpoint': args.checkpoint,
        'split_manifest': args.split_manifest,
        'samples': all_meta,
    }
    with open(out_dir / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(meta_doc, f, indent=2, ensure_ascii=False)

    print(f'\nSaved {len(all_meta)} Torch NPU visualizations -> {vis_dir}')
    print(f'  summary: {out_dir / "summary_grid.png"}')
    if args.profile:
        print(f'  profile: {out_dir / "profile.json"}')


if __name__ == '__main__':
    from mx_driving.patcher import PatcherBuilder, Patch
    from mx_driving.patcher import batch_matmul, resnet_add_relu, resnet_fp16

    pb = (PatcherBuilder()
          .add_module_patch('torch', Patch(batch_matmul))
          .add_module_patch('mmdet', Patch(resnet_add_relu))
          .add_module_patch('mmdet', Patch(resnet_fp16)))
    with pb.build():
        main()
