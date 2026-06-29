#!/usr/bin/env python3
"""Compare split deploy PyTorch vs OM on car_grid test10; optional eval path gap."""
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
from mmcv import Config, DictAction
from mmcv.device.npu import NPUDataParallel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.eval_car_grid_occ import compute_car_metrics, _import_plugin  # noqa: E402
from tools.export_onnx_split_npu import (  # noqa: E402
    Part3ExportWrapper,
    _get_bev_pool_metas_v3,
    _get_sample_batch,
    _import_plugin as _import_deploy_plugin,
    _part1_to_bev_pool_inputs,
    build_part1_wrapper,
    part1_layout_from_manifest,
)
from tools.run_split_infer_npu import (  # noqa: E402
    AclSession,
    _resolve_om_paths,
    run_om_pipeline,
)
from tools.run_unified_infer_npu import _img_from_data, _load_manifest, _resolve_e2e_om  # noqa: E402
from tools.run_unified_viz_npu import decode_occ_logits, run_e2e_timed  # noqa: E402
from tools.split_deploy_infer import (  # noqa: E402
    DEFAULT_OCC_SHAPE,
    build_deploy_model,
    run_eval_full,
    run_om_stages_vt_core,
    run_split_pt_bev_from_part1,
    run_split_pt_stages,
    semantic_overlap,
    tensor_diff,
)
from tools.visualize_car_grid_test_detailed import (  # noqa: E402
    bev_car_only,
    bev_error,
    bev_sem,
    hstack_row,
    put_title,
    sample_metrics,
)

TEST10_PKL = 'data/car_perception_grid/nuscenes/bevdetv2-nuscenes_infos_test10.pkl'


def aggregate_summary(per_sample_meta: list[dict]) -> dict:
    return {
        'mean_mIoU': float(np.mean([m['mIoU'] for m in per_sample_meta])),
        'mean_car_iou': float(np.mean([m['class_iou']['car'] for m in per_sample_meta])),
        'mean_passable_iou': float(
            np.mean([m['class_iou']['passable'] for m in per_sample_meta])),
    }


def build_compare_bev_panel(info, pred_ref, pred_om, meta_ref, meta_om,
                          ref_label='split_pt', pred_eval=None, meta_eval=None):
    gt_pack = np.load(ROOT / info['occ_path'] / 'labels.npz')
    gt = gt_pack['semantics']
    mc = gt_pack['mask_camera']
    stem = info.get('sample_stem', info['token'][:8])
    row_h = 320
    rows = []
    for layer in (0, 1):
        row = [
            put_title(bev_sem(gt[:, :, layer], mc[:, :, layer]), f'GT z{layer}'),
        ]
        if pred_eval is not None:
            row.append(put_title(bev_sem(pred_eval[:, :, layer], mc[:, :, layer]),
                               f'eval z{layer}'))
            row.append(put_title(
                bev_error(gt[:, :, layer], pred_eval[:, :, layer], mc[:, :, layer]),
                f'eval err z{layer}'))
        row.extend([
            put_title(bev_sem(pred_ref[:, :, layer], mc[:, :, layer]),
                      f'{ref_label} z{layer}'),
            put_title(bev_error(gt[:, :, layer], pred_ref[:, :, layer], mc[:, :, layer]),
                      f'{ref_label} err z{layer}'),
            put_title(bev_sem(pred_om[:, :, layer], mc[:, :, layer]), f'OM z{layer}'),
            put_title(bev_error(gt[:, :, layer], pred_om[:, :, layer], mc[:, :, layer]),
                      f'OM err z{layer}'),
        ])
        rows.append(hstack_row(row, row_h))
    # car-only row for clearer visualization
    car_row = [
        put_title(bev_car_only(gt[:, :, 1], mc[:, :, 1]), 'GT car z1'),
    ]
    if pred_eval is not None:
        car_row.append(put_title(bev_car_only(pred_eval[:, :, 1], mc[:, :, 1]), 'eval car z1'))
    car_row.extend([
        put_title(bev_car_only(pred_ref[:, :, 1], mc[:, :, 1]), f'{ref_label} car z1'),
        put_title(bev_car_only(pred_om[:, :, 1], mc[:, :, 1]), 'OM car z1'),
    ])
    rows.append(hstack_row(car_row, row_h))
    panel_w = max(r.shape[1] for r in rows)
    title_h = 52
    canvas = np.zeros((sum(r.shape[0] for r in rows) + title_h, panel_w, 3), np.uint8)
    y = 0
    for r in rows:
        canvas[y:y + r.shape[0], :r.shape[1]] = r
        y += r.shape[0]
    winner = 'tie'
    if meta_ref['mIoU'] > meta_om['mIoU'] + 1e-6:
        winner = ref_label
    elif meta_om['mIoU'] > meta_ref['mIoU'] + 1e-6:
        winner = 'OM'
    if meta_eval is not None:
        title = (
            f'{stem} | eval mIoU={meta_eval["mIoU"]:.3f} car={meta_eval["class_iou"]["car"]:.3f} | '
            f'OM mIoU={meta_om["mIoU"]:.3f} car={meta_om["class_iou"]["car"]:.3f} | '
            f'{ref_label} mIoU={meta_ref["mIoU"]:.3f}')
    else:
        title = (
            f'{stem} | {ref_label} mIoU={meta_ref["mIoU"]:.3f} '
            f'car={meta_ref["class_iou"]["car"]:.3f} | '
            f'OM mIoU={meta_om["mIoU"]:.3f} car={meta_om["class_iou"]["car"]:.3f} | '
            f'better={winner}')
    cv2.rectangle(canvas, (0, y), (panel_w, canvas.shape[0]), (25, 25, 25), -1)
    cv2.putText(canvas, title, (10, y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def _stage_diffs(pt_stages, om_stages):
    """Per-stage numeric diff: split PT ref vs OM."""
    diffs = {
        'part1_tran_feat': tensor_diff(pt_stages['tran_feat'], om_stages['tran_feat']),
        'part1_depth': tensor_diff(pt_stages['depth'], om_stages['depth']),
        'part2_bev_feat': tensor_diff(pt_stages['bev_feat'], om_stages['bev_feat']),
        'part3_occ_logits': tensor_diff(pt_stages['occ_np'], om_stages['occ_np']),
        'semantics': semantic_overlap(pt_stages['pred'], om_stages['pred']),
    }
    return diffs


@torch.no_grad()
def run_split_pt_and_om(cfg_path, checkpoint, manifest_path, gpu_id,
                        part1_om=None, include_eval_ref=False, eval_config=None):
    """Run split PT + split OM (+ optional full eval) on test10, per-sample stages."""
    manifest, default_part1_om, part3_om, _ = _resolve_om_paths(manifest_path, None)
    part1_om = part1_om or default_part1_om
    occ_shape = tuple(manifest.get('tensor_shapes', {}).get(
        'occ_out_0', DEFAULT_OCC_SHAPE))

    eval_cfg = eval_config if include_eval_ref else None
    model, eval_model, dataset, loader, device = build_deploy_model(
        cfg_path, checkpoint, gpu_id, eval_config=eval_cfg)

    part1_layout = part1_layout_from_manifest(manifest)
    part1 = build_part1_wrapper(model, part1_layout).eval()
    part3 = Part3ExportWrapper(model).eval()
    eval_wrapped = None
    if eval_model is not None:
        eval_wrapped = NPUDataParallel(eval_model.cuda(), device_ids=[gpu_id])

    acl_sess = AclSession(device_id=gpu_id)
    acl_sess.load('part1', part1_om)
    acl_sess.load('part3', part3_om)

    preds_pt, preds_om, preds_eval = [], [], []
    stage_reports, path_gaps = [], []

    try:
        for idx in range(len(dataset)):
            data = _get_sample_batch(loader, idx)
            img, ranks_bev, ranks_depth, ranks_feat = _get_bev_pool_metas_v3(
                model, data, device, layout=part1_layout)

            pt = run_split_pt_stages(
                model, part1, part3, img, ranks_bev, ranks_depth, ranks_feat,
                device, occ_shape, data=data, bev_mode='vt_core')
            preds_pt.append(pt['pred'])

            om = run_om_stages_vt_core(
                acl_sess, 'part1', 'part3', model, data, img, device, occ_shape)
            preds_om.append(om['pred'])
            stage_reports.append(_stage_diffs(pt, om))

            if eval_wrapped is not None:
                pred_eval = run_eval_full(eval_wrapped, data, device, gpu_id)
                preds_eval.append(pred_eval)
                path_gaps.append(semantic_overlap(pred_eval, om['pred']))
    finally:
        acl_sess.close()

    return {
        'dataset': dataset,
        'preds_pt': preds_pt,
        'preds_om': preds_om,
        'preds_eval': preds_eval,
        'stage_reports': stage_reports,
        'path_gaps': path_gaps,
        'part1_om': part1_om,
        'part3_om': part3_om,
    }


def run_merged_om_inference(cfg_path, manifest_path, om_path, gpu_id, preds_pt_ref):
    """Merged OM inference; semantic diff vs split PT reference."""
    from mmdet3d.datasets import build_dataset

    manifest = _load_manifest(manifest_path)
    om_path = _resolve_e2e_om(manifest, manifest_path, om_path)
    part1_layout = part1_layout_from_manifest(manifest)
    occ_shape = tuple(manifest.get('tensor_shapes', {}).get(
        'occ_out_0', DEFAULT_OCC_SHAPE))

    cfg = Config.fromfile(cfg_path)
    cfg.merge_from_dict({
        'data.test.ann_file': TEST10_PKL,
        'data.test.data_root': 'data/car_perception_grid/nuscenes/',
    })
    _import_deploy_plugin(cfg, cfg_path)
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader_from_cfg(cfg)

    acl_sess = AclSession(device_id=gpu_id)
    acl_sess.load('e2e', om_path)
    preds_om, stage_reports = [], []
    try:
        for idx in range(len(dataset)):
            data = _get_sample_batch(loader, idx)
            img_np = _img_from_data(data, layout=part1_layout).astype(np.float32)
            occ_raw = acl_sess.infer('e2e', img_np)[0]
            pred = decode_occ_logits(occ_raw, occ_shape)
            preds_om.append(pred)
            stage_reports.append({
                'semantics': semantic_overlap(preds_pt_ref[idx], pred),
            })
    finally:
        acl_sess.close()
    return dataset, preds_om, stage_reports


def build_dataloader_from_cfg(cfg):
    from mmdet3d.datasets import build_dataloader, build_dataset
    dataset = build_dataset(cfg.data.test)
    return build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare split deploy PyTorch vs OM on car_grid test10')
    parser.add_argument('--config',
                        default='projects/configs/flashocc/flashocc-r50-car-grid.py')
    parser.add_argument('--deploy-config',
                        default='projects/configs/flashocc/flashocc-r50-car-grid-trt.py')
    parser.add_argument('--checkpoint',
                        default='work_dirs/car_grid_v4/epoch_12_ema.pth')
    parser.add_argument('--om-mode', choices=('split', 'merged'), default='split')
    parser.add_argument('--split-manifest',
                        default='work_dirs/onnx_split_car_grid/'
                                'flashocc_car_grid_deploy_manifest.json')
    parser.add_argument('--part1-om', default=None)
    parser.add_argument('--om-label', default=None)
    parser.add_argument('--manifest',
                        default='work_dirs/onnx_unified_car_grid/'
                                'flashocc_car_grid_unified_deploy_manifest.json')
    parser.add_argument('--om-path',
                        default='work_dirs/onnx_unified_car_grid/flashocc_car_grid_merged.om')
    parser.add_argument('--out-dir',
                        default='work_dirs/compare_torch_npu_vs_om_test10')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--skip-torch', action='store_true',
                        help='reuse pred_torch.npz from --torch-ref-dir')
    parser.add_argument('--torch-ref-dir', default=None)
    parser.add_argument('--skip-om', action='store_true')
    parser.add_argument('--include-eval-ref', action='store_true',
                        help='also run full BEVDetOCC eval and report path gap vs split PT')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    return parser.parse_args()


def _aggregate_stage_reports(reports: list[dict]) -> dict:
    if not reports or 'part1_tran_feat' not in reports[0]:
        return {}
    keys = ['part1_tran_feat', 'part1_depth', 'part2_bev_feat', 'part3_occ_logits']
    out = {}
    for k in keys:
        out[k] = {
            'mean_max_abs': float(np.mean([r[k]['max_abs'] for r in reports])),
            'max_max_abs': float(np.max([r[k]['max_abs'] for r in reports])),
            'all_ok': all(r[k].get('allclose', False) for r in reports),
        }
    sem = [r['semantics'] for r in reports]
    out['semantics'] = {
        'mean_sem_match': float(np.mean([s['sem_match'] for s in sem])),
        'mean_car_overlap': float(np.mean([s['car_overlap'] for s in sem])),
    }
    return out


def main():
    args = parse_args()
    os.chdir(ROOT)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / 'samples'
    samples_dir.mkdir(parents=True, exist_ok=True)

    stage_reports = []
    path_gaps = []
    preds_eval = []

    if not args.skip_torch:
        if args.om_mode == 'split' and not args.skip_om:
            print('[1/4] split deploy PyTorch + OM (stage-wise) on test10 ...')
            bundle = run_split_pt_and_om(
                args.deploy_config, args.checkpoint, args.split_manifest,
                args.gpu_id, part1_om=args.part1_om,
                include_eval_ref=args.include_eval_ref,
                eval_config=args.config)
            dataset = bundle['dataset']
            preds_pt = bundle['preds_pt']
            preds_om = bundle['preds_om']
            stage_reports = bundle['stage_reports']
            path_gaps = bundle['path_gaps']
            preds_eval = bundle['preds_eval']
        else:
            print('[1/4] split deploy PyTorch on test10 ...')
            model, eval_model, dataset, loader, device = build_deploy_model(
                args.deploy_config, args.checkpoint, args.gpu_id,
                eval_config=args.config if args.include_eval_ref else None)
            from tools.split_deploy_infer import run_split_pt_inference
            preds_pt = run_split_pt_inference(model, dataset, loader, device)
            preds_om = []
            stage_reports = []
            path_gaps = []
            preds_eval = []
            if args.include_eval_ref and eval_model is not None:
                eval_wrapped = NPUDataParallel(eval_model.cuda(), device_ids=[args.gpu_id])
                for idx in range(len(dataset)):
                    data = _get_sample_batch(loader, idx)
                    pred_eval = run_eval_full(eval_wrapped, data, device, args.gpu_id)
                    preds_eval.append(pred_eval)
                    path_gaps.append(semantic_overlap(pred_eval, preds_pt[idx]))
            if args.om_mode == 'merged' and not args.skip_om:
                print('[2/4] merged OM inference ...')
                dataset, preds_om, stage_reports = run_merged_om_inference(
                    args.deploy_config, args.manifest, args.om_path,
                    args.gpu_id, preds_pt)
            elif args.skip_om:
                preds_om = list(preds_pt)
    else:
        cfg = Config.fromfile(args.config)
        cfg.merge_from_dict({
            'data.test.ann_file': TEST10_PKL,
            'data.test.data_root': 'data/car_perception_grid/nuscenes/',
        })
        _import_plugin(cfg, args.config)
        from mmdet3d.datasets import build_dataset
        dataset = build_dataset(cfg.data.test)
        preds_pt = []
        ref_dir = Path(args.torch_ref_dir) if args.torch_ref_dir else samples_dir
        ref_samples = ref_dir / 'samples' if (ref_dir / 'samples').is_dir() else ref_dir
        for d in sorted(ref_samples.glob('*/pred_torch.npz')):
            preds_pt.append(np.load(d)['pred'])
        if not preds_pt:
            raise RuntimeError(f'--skip-torch but no pred_torch.npz under {ref_samples}')

        if not args.skip_om:
            if args.om_mode == 'split':
                print('[2/4] split OM only (--skip-torch) ...')
                _, preds_om = _run_split_om_only(
                    args.deploy_config, args.checkpoint, args.split_manifest,
                    args.gpu_id, part1_om=args.part1_om)
            else:
                print('[2/4] merged OM ...')
                _, preds_om, _ = run_merged_om_inference(
                    args.deploy_config, args.manifest, args.om_path,
                    args.gpu_id, preds_pt)
        else:
            preds_om = list(preds_pt)

    if len(preds_pt) != len(dataset) or len(preds_om) != len(dataset):
        raise RuntimeError(
            f'sample count mismatch: dataset={len(dataset)} '
            f'split_pt={len(preds_pt)} om={len(preds_om)}')

    gt_packs = []
    per_sample = []
    thumbs = []
    ref_label = 'split_deploy_pt'
    print('[3/4] Building comparison panels ...')
    for i, info in enumerate(dataset.data_infos):
        gt_npz = np.load(os.path.join(info['occ_path'], 'labels.npz'))
        gt_pack = {
            'semantics': gt_npz['semantics'],
            'mask_camera': gt_npz['mask_camera'],
            'mask_lidar': gt_npz['mask_lidar'],
        }
        gt_packs.append(gt_pack)
        pred_pt = preds_pt[i]
        pred_om = preds_om[i]
        stem = info.get('sample_stem', info['token'][:8])
        tag = f'{i:02d}_{stem}'
        sample_dir = samples_dir / tag
        sample_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(sample_dir / 'pred_torch.npz', pred=pred_pt)
        np.savez_compressed(sample_dir / 'pred_split_pt.npz', pred=pred_pt)
        np.savez_compressed(sample_dir / 'pred_om.npz', pred=pred_om)
        if preds_eval:
            np.savez_compressed(sample_dir / 'pred_eval.npz', pred=preds_eval[i])

        meta_pt = sample_metrics(gt_npz['semantics'], pred_pt, gt_npz['mask_camera'])
        meta_om = sample_metrics(gt_npz['semantics'], pred_om, gt_npz['mask_camera'])
        meta_pt['backend'] = ref_label
        meta_om['backend'] = args.om_label or args.om_mode

        entry = {
            'index': i,
            'stem': stem,
            'split_deploy_pt': meta_pt,
            'OM': meta_om,
            'better_mIoU': (
                ref_label if meta_pt['mIoU'] > meta_om['mIoU'] + 1e-6
                else 'OM' if meta_om['mIoU'] > meta_pt['mIoU'] + 1e-6
                else 'tie'),
            'compare_panel': str(sample_dir / 'compare_bev_panel.png'),
        }
        if stage_reports and i < len(stage_reports):
            entry['stage_diff'] = stage_reports[i]
        if path_gaps and i < len(path_gaps):
            entry['path_gap_eval_vs_om'] = path_gaps[i]
            if preds_eval:
                meta_eval = sample_metrics(
                    gt_npz['semantics'], preds_eval[i], gt_npz['mask_camera'])
                meta_eval['backend'] = 'full_eval'
                entry['full_eval'] = meta_eval

        meta_eval = entry.get('full_eval')
        pred_eval_i = preds_eval[i] if preds_eval else None
        compare_panel = build_compare_bev_panel(
            info, pred_pt, pred_om, meta_pt, meta_om, ref_label=ref_label,
            pred_eval=pred_eval_i, meta_eval=meta_eval)
        cv2.imwrite(str(sample_dir / 'compare_bev_panel.png'), compare_panel)
        per_sample.append(entry)
        thumb = put_title(
            cv2.resize(compare_panel,
                       (1100, int(1100 * compare_panel.shape[0] / compare_panel.shape[1]))),
            f'{stem} | {ref_label}={meta_pt["mIoU"]:.3f} OM={meta_om["mIoU"]:.3f}')
        thumbs.append(thumb)

    summary_pt = aggregate_summary([s['split_deploy_pt'] for s in per_sample])
    summary_om = aggregate_summary([s['OM'] for s in per_sample])
    pt_wins = sum(1 for s in per_sample if s['better_mIoU'] == ref_label)
    om_wins = sum(1 for s in per_sample if s['better_mIoU'] == 'OM')
    ties = sum(1 for s in per_sample if s['better_mIoU'] == 'tie')
    overall_better = (
        ref_label if summary_pt['mean_mIoU'] > summary_om['mean_mIoU'] + 1e-6
        else 'OM' if summary_om['mean_mIoU'] > summary_pt['mean_mIoU'] + 1e-6
        else 'tie')

    report = {
        'test_pkl': TEST10_PKL,
        'checkpoint': args.checkpoint,
        'torch_ref': ref_label,
        'om_mode': args.om_mode,
        'om_label': args.om_label or args.om_mode,
        'part1_om': args.part1_om,
        'om_path': args.om_path if args.om_mode == 'merged' else args.split_manifest,
        'num_samples': len(per_sample),
        'overall_better_mIoU': overall_better,
        'per_sample_wins': {ref_label: pt_wins, 'OM': om_wins, 'tie': ties},
        'split_deploy_pt': {
            'aggregate': summary_pt,
            'metrics': compute_car_metrics(preds_pt, gt_packs),
        },
        'OM': {
            'aggregate': summary_om,
            'metrics': compute_car_metrics(preds_om, gt_packs),
        },
        'stage_aggregate': _aggregate_stage_reports(stage_reports),
        'per_sample': per_sample,
    }
    if preds_eval:
        summary_eval = aggregate_summary(
            [s['full_eval'] for s in per_sample if 'full_eval' in s])
        report['full_eval'] = {
            'aggregate': summary_eval,
            'metrics': compute_car_metrics(preds_eval, gt_packs),
            'path_gap_vs_om': {
                'mean_sem_match': float(np.mean([g['sem_match'] for g in path_gaps])),
                'mean_car_overlap': float(np.mean([g['car_overlap'] for g in path_gaps])),
            },
        }
    # backward-compat keys for chain script
    report['torch_npu'] = report['split_deploy_pt']

    print('[4/4] Saving summary ...')
    with open(out_dir / 'comparison_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    om_desc = args.om_path if args.om_mode == 'merged' else (args.part1_om or 'split default')
    summary_lines = [
        'FlashOCC car_grid test10: split_deploy_pt vs OM',
        f'torch_ref: {ref_label}',
        f'om_mode: {args.om_mode}',
        f'om_label: {args.om_label or args.om_mode}',
        f'samples: {len(per_sample)}',
        f'checkpoint: {args.checkpoint}',
        f'om: {om_desc}',
        '',
        f'Overall better (mean mIoU): {overall_better}',
        f'  split_deploy_pt mean mIoU: {summary_pt["mean_mIoU"]:.4f} '
        f'(car IoU {summary_pt["mean_car_iou"]:.4f}, '
        f'passable IoU {summary_pt["mean_passable_iou"]:.4f})',
        f'  OM mean mIoU:              {summary_om["mean_mIoU"]:.4f} '
        f'(car IoU {summary_om["mean_car_iou"]:.4f}, '
        f'passable IoU {summary_om["mean_passable_iou"]:.4f})',
        f'  delta (split_pt - OM): {summary_pt["mean_mIoU"] - summary_om["mean_mIoU"]:.4f}',
    ]
    if report.get('stage_aggregate'):
        sa = report['stage_aggregate']
        summary_lines += [
            '',
            'Stage aggregate (split_pt vs OM):',
            f'  part1 tran_feat mean_max_abs: {sa["part1_tran_feat"]["mean_max_abs"]:.6f}',
            f'  part1 depth     mean_max_abs: {sa["part1_depth"]["mean_max_abs"]:.6f}',
            f'  part2 bev_feat  mean_max_abs: {sa["part2_bev_feat"]["mean_max_abs"]:.6f}',
            f'  part3 occ       mean_max_abs: {sa["part3_occ_logits"]["mean_max_abs"]:.6f}',
            f'  semantics match: {sa["semantics"]["mean_sem_match"]:.4f}',
        ]
    if report.get('full_eval'):
        fe = report['full_eval']
        summary_lines += [
            '',
            f'Path gap (full_eval vs OM):',
            f'  full_eval mean mIoU: {fe["aggregate"]["mean_mIoU"]:.4f}',
            f'  OM mean mIoU: {summary_om["mean_mIoU"]:.4f}',
            f'  sem_match eval vs OM: {fe["path_gap_vs_om"]["mean_sem_match"]:.4f}',
            f'  car overlap eval vs OM: {fe["path_gap_vs_om"]["mean_car_overlap"]:.0f}',
        ]
    summary_lines.append(f'\nPer-sample wins: {ref_label}={pt_wins}, OM={om_wins}, tie={ties}')
    (out_dir / 'comparison_summary.txt').write_text('\n'.join(summary_lines), encoding='utf-8')

    if thumbs:
        cols = 2
        rows_n = (len(thumbs) + cols - 1) // cols
        th, tw = thumbs[0].shape[:2]
        grid = np.zeros((rows_n * th, cols * tw, 3), np.uint8)
        for i, t in enumerate(thumbs):
            r, c = divmod(i, cols)
            grid[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = t
        cv2.imwrite(str(out_dir / 'summary_grid.png'), grid)

    print('\n' + '\n'.join(summary_lines))
    print(f'\nResults saved to: {out_dir}')


def _run_split_om_only(cfg_path, checkpoint, manifest_path, gpu_id, part1_om=None):
    """OM-only path when reusing saved split PT preds."""
    from mmcv import Config
    from mmcv.cnn import fuse_conv_bn
    from mmcv.runner import load_checkpoint
    from mmdet3d.datasets import build_dataloader, build_dataset
    from mmdet3d.models import build_model

    manifest, default_part1_om, part3_om, _ = _resolve_om_paths(manifest_path, None)
    part1_om = part1_om or default_part1_om
    part1_layout = part1_layout_from_manifest(manifest)
    occ_shape = tuple(manifest.get('tensor_shapes', {}).get(
        'occ_out_0', DEFAULT_OCC_SHAPE))

    cfg = Config.fromfile(cfg_path)
    cfg.merge_from_dict({
        'data.test.ann_file': TEST10_PKL,
        'data.test.data_root': 'data/car_perception_grid/nuscenes/',
    })
    _import_deploy_plugin(cfg, cfg_path)
    if not cfg.model.type.endswith('TRT'):
        cfg.model.type += 'TRT'
    cfg.data.test.test_mode = True
    cfg.model.train_cfg = None
    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader_from_cfg(cfg)
    device = torch.device(f'npu:{gpu_id}')
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, checkpoint, map_location='cpu')
    model = fuse_conv_bn(model).to(device).eval()

    acl_sess = AclSession(device_id=gpu_id)
    acl_sess.load('part1', part1_om)
    acl_sess.load('part3', part3_om)
    preds = []
    try:
        for idx in range(len(dataset)):
            data = _get_sample_batch(loader, idx)
            img, ranks_bev, ranks_depth, ranks_feat = _get_bev_pool_metas_v3(
                model, data, device, layout=part1_layout)
            occ_raw, _ = run_om_pipeline(
                acl_sess, 'part1', 'part3', model, img,
                ranks_bev, ranks_depth, ranks_feat, device, data=data, bev_mode='vt_core')
            preds.append(decode_occ_logits(occ_raw, occ_shape))
    finally:
        acl_sess.close()
    return dataset, preds


if __name__ == '__main__':
    from mx_driving.patcher import PatcherBuilder, Patch
    from mx_driving.patcher import batch_matmul, resnet_add_relu, resnet_fp16

    pb = (PatcherBuilder()
          .add_module_patch('torch', Patch(batch_matmul))
          .add_module_patch('mmdet', Patch(resnet_add_relu))
          .add_module_patch('mmdet', Patch(resnet_fp16)))
    with pb.build():
        main()
