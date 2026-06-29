#!/usr/bin/env python3
"""Stage-by-stage diagnosis and path alignment on test10."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch_npu  # noqa: F401
from mmcv import DictAction
from mmcv.device.npu import NPUDataParallel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.export_onnx_split_npu import (  # noqa: E402
    Part3ExportWrapper,
    _get_bev_pool_metas_v3,
    _get_sample_batch,
    build_part1_wrapper,
    part1_layout_from_manifest,
)
from tools.run_split_infer_npu import AclSession, _resolve_om_paths  # noqa: E402
from tools.split_deploy_infer import (  # noqa: E402
    DEFAULT_OCC_SHAPE,
    build_deploy_model,
    run_eval_full,
    run_om_stages_vt_core,
    run_split_pt_stages,
    semantic_overlap,
    tensor_diff,
)
from tools.run_unified_viz_npu import decode_occ_logits  # noqa: E402
from tools.visualize_car_grid_test_detailed import sample_metrics  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description='Path alignment + stage diagnosis on test10')
    p.add_argument('--deploy-config',
                   default='projects/configs/flashocc/flashocc-r50-car-grid-trt.py')
    p.add_argument('--eval-config',
                   default='projects/configs/flashocc/flashocc-r50-car-grid.py')
    p.add_argument('--checkpoint',
                   default='work_dirs/car_grid_v4/epoch_12_ema.pth')
    p.add_argument('--manifest',
                   default='work_dirs/onnx_split_car_grid/'
                            'flashocc_car_grid_deploy_manifest.json')
    p.add_argument('--sample-idx', type=int, default=0)
    p.add_argument('--all-samples', action='store_true')
    p.add_argument('--gpu-id', type=int, default=0)
    p.add_argument('--out', default='work_dirs/path_alignment_test10.json')
    p.add_argument('--cfg-options', nargs='+', action=DictAction)
    return p.parse_args()


def _print_diff(name, entry):
    if entry.get('status') == 'SHAPE':
        print(f"  [SHAPE] {name}")
        return
    if 'sem_match' in entry:
        print(f"  [{name}] sem_match={entry['sem_match']:.4f} "
              f"car_overlap={entry['car_overlap']}")
        return
    print(f"  [{entry.get('status', '?')}] {name}: "
          f"max_abs={entry['max_abs']:.6f} rel_max={entry.get('rel_max', 0):.6f}")


def main():
    args = parse_args()
    os.chdir(ROOT)

    manifest, part1_om, part3_om, _ = _resolve_om_paths(args.manifest, None)
    occ_shape = tuple(manifest.get('tensor_shapes', {}).get(
        'occ_out_0', DEFAULT_OCC_SHAPE))

    model, eval_model, dataset, loader, device = build_deploy_model(
        args.deploy_config, args.checkpoint, args.gpu_id,
        eval_config=args.eval_config)
    eval_wrapped = NPUDataParallel(eval_model.cuda(), device_ids=[args.gpu_id])
    part1_layout = part1_layout_from_manifest(manifest)
    part1 = build_part1_wrapper(model, part1_layout).eval()
    part3 = Part3ExportWrapper(model).eval()

    acl = AclSession(device_id=args.gpu_id)
    acl.load('part1', part1_om)
    acl.load('part3', part3_om)

    indices = range(len(dataset)) if args.all_samples else [args.sample_idx]
    report = {'samples': [], 'aggregate': {}}
    path_gaps, stage_agg = [], []

    print('=' * 72)
    print('Path alignment: full_eval vs split_deploy_pt; stages: split_pt vs OM')
    print('=' * 72)

    try:
        for idx in indices:
            data = _get_sample_batch(loader, idx)
            info = dataset.data_infos[idx]
            stem = info.get('sample_stem', info['token'][:8])
            print(f'\n--- sample {idx}: {stem} ---')

            img, ranks_bev, ranks_depth, ranks_feat = _get_bev_pool_metas_v3(
                model, data, device, layout=part1_layout)
            pt = run_split_pt_stages(
                model, part1, part3, img, ranks_bev, ranks_depth, ranks_feat,
                device, occ_shape, data=data, bev_mode='vt_core')
            pred_eval = run_eval_full(eval_wrapped, data, device, args.gpu_id)
            path_gap = semantic_overlap(pred_eval, pt['pred'])

            gt_npz = np.load(os.path.join(info['occ_path'], 'labels.npz'))
            meta_eval = sample_metrics(
                gt_npz['semantics'], pred_eval, gt_npz['mask_camera'])
            meta_pt = sample_metrics(
                gt_npz['semantics'], pt['pred'], gt_npz['mask_camera'])

            print('[path] full_eval vs split_pt (vt_core)')
            _print_diff('semantics', path_gap)
            print(f'  mIoU eval={meta_eval["mIoU"]:.4f} split_pt={meta_pt["mIoU"]:.4f} '
                  f'delta={meta_eval["mIoU"] - meta_pt["mIoU"]:+.4f}')

            om = run_om_stages_vt_core(
                acl, 'part1', 'part3', model, data, img, device, occ_shape)
            pred_om = om['pred']
            path_gap_om = semantic_overlap(pred_eval, pred_om)
            stages = {
                'part1_tran_feat': tensor_diff(pt['tran_feat'], om['tran_feat']),
                'part1_depth': tensor_diff(pt['depth'], om['depth']),
                'part2_bev_feat': tensor_diff(pt['bev_feat'], om['bev_feat']),
                'part3_occ_logits': tensor_diff(pt['occ_np'], om['occ_np']),
                'semantics': semantic_overlap(pt['pred'], pred_om),
            }
            meta_om = sample_metrics(
                gt_npz['semantics'], pred_om, gt_npz['mask_camera'])
            print('[stages] split_pt vs OM')
            for k, v in stages.items():
                _print_diff(k, v)
            print(f'  mIoU split_pt={meta_pt["mIoU"]:.4f} OM={meta_om["mIoU"]:.4f} '
                  f'delta={meta_pt["mIoU"] - meta_om["mIoU"]:+.4f}')
            print('[path] full_eval vs OM (vt_core)')
            _print_diff('semantics', path_gap_om)

            sample_rec = {
                'index': idx,
                'stem': stem,
                'path_gap_eval_vs_split_pt': path_gap,
                'path_gap_eval_vs_om': path_gap_om,
                'mIoU': {
                    'full_eval': meta_eval['mIoU'],
                    'split_pt': meta_pt['mIoU'],
                    'OM': meta_om['mIoU'],
                },
                'stage_diff': stages,
            }
            report['samples'].append(sample_rec)
            path_gaps.append(path_gap)
            stage_agg.append(stages)
    finally:
        acl.close()

    if path_gaps:
        report['aggregate']['path_gap'] = {
            'mean_sem_match': float(np.mean([g['sem_match'] for g in path_gaps])),
            'mean_car_overlap': float(np.mean([g['car_overlap'] for g in path_gaps])),
            'mean_mIoU_eval': float(np.mean(
                [s['mIoU']['full_eval'] for s in report['samples']])),
            'mean_mIoU_split_pt': float(np.mean(
                [s['mIoU']['split_pt'] for s in report['samples']])),
            'mean_mIoU_OM': float(np.mean(
                [s['mIoU']['OM'] for s in report['samples']])),
        }
    if stage_agg:
        for key in ('part1_tran_feat', 'part1_depth', 'part2_bev_feat', 'part3_occ_logits'):
            report['aggregate'].setdefault('stage_om_vs_split_pt', {})[key] = {
                'mean_max_abs': float(np.mean([s[key]['max_abs'] for s in stage_agg])),
            }
        report['aggregate']['stage_om_vs_split_pt']['semantics'] = {
            'mean_sem_match': float(np.mean(
                [s['semantics']['sem_match'] for s in stage_agg])),
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'\nWrote {out}')
    if report.get('aggregate'):
        ag = report['aggregate']
        if 'path_gap' in ag:
            pg = ag['path_gap']
            print(f"Path gap (eval vs split_pt): sem_match={pg['mean_sem_match']:.4f} "
                  f"mIoU eval={pg['mean_mIoU_eval']:.4f} split_pt={pg['mean_mIoU_split_pt']:.4f}")
        if 'stage_om_vs_split_pt' in ag:
            st = ag['stage_om_vs_split_pt']
            print(f"OM vs split_pt: part3 occ mean_max_abs="
                  f"{st['part3_occ_logits']['mean_max_abs']:.6f} "
                  f"sem_match={st['semantics']['mean_sem_match']:.4f}")


if __name__ == '__main__':
    from mx_driving.patcher import PatcherBuilder, Patch
    from mx_driving.patcher import batch_matmul, resnet_add_relu, resnet_fp16

    pb = (PatcherBuilder()
          .add_module_patch('torch', Patch(batch_matmul))
          .add_module_patch('mmdet', Patch(resnet_add_relu))
          .add_module_patch('mmdet', Patch(resnet_fp16)))
    with pb.build():
        main()
