#!/usr/bin/env python3
"""Reverse-order layer comparison: Torch NPU (split_pt + full_eval) vs OM (split + merged).

Reports from output (semantics) back to input (img), with cross-feed isolation and
part1/part3 submodule breakdown.
"""
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

from tools.compare_split_vs_test import (  # noqa: E402
    testpy_stage1,
    testpy_stage2,
    testpy_stage3,
)
from tools.export_onnx_split_npu import (  # noqa: E402
    Part3ExportWrapper,
    _get_bev_pool_metas_v3,
    _get_sample_batch,
    _import_plugin,
    _part1_to_bev_pool_inputs,
    _to_device,
    build_part1_wrapper,
    part1_layout_from_manifest,
    run_bev_pool_v3,
)
from tools.run_split_infer_npu import AclSession, _resolve_om_paths  # noqa: E402
from tools.run_unified_infer_npu import _img_from_data  # noqa: E402
from tools.run_unified_viz_npu import decode_occ_logits  # noqa: E402
from tools.split_deploy_infer import (  # noqa: E402
    DEFAULT_OCC_SHAPE,
    build_deploy_model,
    run_bev_view_transform_core,
    run_eval_full,
    run_split_pt_stages,
    semantic_overlap,
    tensor_diff,
)
from tools.visualize_car_grid_test_detailed import sample_metrics  # noqa: E402

TEST10_PKL = 'data/car_perception_grid/nuscenes/bevdetv2-nuscenes_infos_test10.pkl'


def parse_args():
    p = argparse.ArgumentParser(description='Reverse layer compare: NPU vs OM')
    p.add_argument('--deploy-config',
                   default='projects/configs/flashocc/flashocc-r50-car-grid-trt.py')
    p.add_argument('--eval-config',
                   default='projects/configs/flashocc/flashocc-r50-car-grid.py')
    p.add_argument('--checkpoint',
                   default='work_dirs/car_grid_v4/epoch_12_ema.pth')
    p.add_argument('--split-manifest',
                   default='work_dirs/onnx_split_car_grid_eval/'
                            'flashocc_car_grid_deploy_manifest.json')
    p.add_argument('--merged-om',
                   default='work_dirs/onnx_unified_car_grid/'
                            'flashocc_car_grid_merged_segment_sum.om')
    p.add_argument('--sample-idx', type=int, default=0)
    p.add_argument('--gpu-id', type=int, default=0)
    p.add_argument('--out', default='work_dirs/layer_compare_sample0/report.json')
    p.add_argument('--save-tensors', action='store_true')
    p.add_argument('--cfg-options', nargs='+', action=DictAction)
    return p.parse_args()


@torch.no_grad()
def _run_eval_stages(eval_model, img_inputs, occ_shape):
    """Full eval BEVDetOCC decomposed into stage tensors."""
    t_tran, t_depth, t_feat, t_prepared = testpy_stage1(eval_model, img_inputs)
    t_bev = testpy_stage2(eval_model, t_prepared, t_feat, t_depth, t_tran)
    t_occ = testpy_stage3(eval_model, t_bev)
    t_tran_flat = t_tran.permute(0, 2, 3, 1).contiguous().flatten(0, 2)
    t_depth_flat = t_depth.contiguous().flatten()
    pred = decode_occ_logits(
        t_occ.detach().float().cpu().numpy(), occ_shape)
    return {
        'encoder_feat': t_feat,
        'tran_feat': t_tran_flat,
        'depth': t_depth_flat,
        'bev_feat': t_bev,
        'occ_np': t_occ.detach().float().cpu().numpy(),
        'pred': pred,
    }


@torch.no_grad()
def _run_part1_submodules(model, img):
    """Split deploy part1 internal: image_encoder feat + depth_net outputs."""
    if img.dim() == 4 and img.shape[1] == 3:
        img = img.unsqueeze(0)
    feat, _ = model.image_encoder(img)
    b, n, c, h, w = feat.shape
    x = feat.view(b * n, c, h, w)
    vt = model.img_view_transformer
    dn_out = vt.depth_net(x)
    depth = dn_out[:, :vt.D].softmax(dim=1)
    tran_feat = dn_out[:, vt.D:(vt.D + vt.out_channels)]
    tran_flat = tran_feat.permute(0, 2, 3, 1).contiguous().flatten(0, 2)
    depth_flat = depth.reshape(-1)
    return {
        'encoder_feat': feat,
        'depth_net_depth': depth,
        'depth_net_tran': tran_feat,
        'tran_feat': tran_flat,
        'depth': depth_flat,
    }


@torch.no_grad()
def _run_part3_submodules_deploy(model, bev_feat):
    """Part3 internal on BEVDetOCCTRT: backbone -> neck -> occ_head."""
    from projects.mmdet3d_plugin.models.detectors.bevdet_occ import (
        _trt_reshape_part3_input)

    vt = model.img_view_transformer
    x = _trt_reshape_part3_input(vt, bev_feat.contiguous().reshape(-1))
    x = x.permute(0, 3, 1, 2).contiguous()
    bev_bb = model.img_bev_encoder_backbone(x)
    bev_neck = model.img_bev_encoder_neck(bev_bb)
    occ_out = model.occ_head(bev_neck)
    if isinstance(occ_out, (list, tuple)):
        occ_out = occ_out[0]
    return {
        'bev_encoder_backbone': bev_bb,
        'bev_encoder_neck': bev_neck,
        'occ_logits': occ_out,
    }


@torch.no_grad()
def _run_eval_part3_on_bev(eval_model, bev_feat):
    """Eval bev_encoder + occ_head on shared bev_feat."""
    x = eval_model.bev_encoder(bev_feat)
    outs = eval_model.occ_head(x)
    occ = outs[0] if isinstance(outs, (list, tuple)) else outs
    return occ.detach().float().cpu().numpy()


@torch.no_grad()
def _run_part3_pt(part3, bev_feat):
    outs = part3(bev_feat)
    occ = outs[0] if isinstance(outs, (list, tuple)) else outs
    return occ.detach().float().cpu().numpy()


@torch.no_grad()
def _run_part3_om(acl, bev_feat):
    bev_np = bev_feat.detach().cpu().numpy().astype(np.float32)
    return acl.infer('part3', bev_np)[0]


def _maybe_save(save_dir, name, arr):
    if save_dir is None:
        return
    save_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().float().cpu().numpy()
    np.save(save_dir / f'{name}.npy', np.asarray(arr))


def _diff_entry(ref, test, label):
    if 'sem_match' in (ref if isinstance(ref, dict) else {}):
        return ref
    d = tensor_diff(ref, test)
    d['label'] = label
    return d


def _is_diverged(entry):
    if entry is None:
        return False
    if 'sem_match' in entry:
        return entry['sem_match'] < 0.95
    status = entry.get('status', 'OK')
    rel = entry.get('rel_max', 0.0)
    return status == 'FAIL' or rel > 0.01


def _find_first_divergence(layers):
    """Scan output->input; return first layer with significant divergence."""
    order = [
        'L5_semantics',
        'L4_part3_occ_logits',
        'L3_part2_bev_feat',
        'L2_part1_depth',
        'L1_part1_tran_feat',
    ]
    for key in order:
        block = layers.get(key, {})
        for pair_name, entry in block.items():
            if pair_name.startswith('cross_feed'):
                continue
            if _is_diverged(entry):
                return {
                    'layer': key,
                    'pair': pair_name,
                    'entry': entry,
                }
    return None


def _format_diff_line(entry):
    if entry is None:
        return '  (n/a)'
    if 'sem_match' in entry:
        return (f"  sem_match={entry['sem_match']:.4f} "
                f"car_overlap={entry['car_overlap']} "
                f"status={'FAIL' if entry['sem_match'] < 0.95 else 'OK'}")
    st = entry.get('status', '?')
    return (f"  max_abs={entry.get('max_abs', 0):.6f} "
            f"rel_max={entry.get('rel_max', 0):.6f} status={st}")


def _write_txt_report(report, txt_path):
    lines = []
    lines.append(f"Sample {report['sample_idx']}: {report['stem']}")
    lines.append(f"part1_om: {report['part1_om']}")
    lines.append(f"part3_om: {report['part3_om']}")
    lines.append(f"merged_om: {report['merged_om']}")
    lines.append('')
    fd = report.get('first_divergence')
    if fd:
        lines.append(f"FIRST_DIVERGENCE: {fd['layer']} / {fd['pair']}")
        lines.append(_format_diff_line(fd['entry']).strip())
    else:
        lines.append('FIRST_DIVERGENCE: none (all pairs within tolerance)')
    lines.append('')
    lines.append('=== Reverse layer report (output -> input) ===')
    for key in sorted(report['layers'].keys(), reverse=True):
        lines.append(f'\n[{key}]')
        block = report['layers'][key]
        for pair_name, entry in block.items():
            lines.append(f'  {pair_name}:')
            if isinstance(entry, str):
                lines.append(f'  {entry}')
            else:
                lines.append(_format_diff_line(entry))
    if report.get('submodules'):
        lines.append('\n=== Part1 submodules (split_pt vs split_OM) ===')
        for k, v in report['submodules'].get('part1', {}).items():
            lines.append(f'  {k}: {_format_diff_line(v).strip()}')
        lines.append('\n=== Part3 submodules (split_pt vs full_eval, same PT bev) ===')
        for k, v in report['submodules'].get('part3_pt_vs_eval', {}).items():
            lines.append(f'  {k}: {_format_diff_line(v).strip()}')
    if report.get('cross_feed'):
        lines.append('\n=== Cross-feed isolation ===')
        for k, v in report['cross_feed'].items():
            lines.append(f'  {k}: {_format_diff_line(v).strip()}')
    if report.get('miou'):
        lines.append('\n=== mIoU ===')
        for k, v in report['miou'].items():
            lines.append(f'  {k}: {v:.4f}')
    txt_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    args = parse_args()
    os.chdir(ROOT)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tensor_dir = out_path.parent / 'tensors' if args.save_tensors else None

    manifest, part1_om, part3_om, _ = _resolve_om_paths(args.split_manifest, None)
    occ_shape = tuple(manifest.get('tensor_shapes', {}).get(
        'occ_out_0', DEFAULT_OCC_SHAPE))
    part1_layout = part1_layout_from_manifest(manifest)

    print('[1/2] Loading deploy + eval models ...')
    model, eval_model, dataset, loader, device = build_deploy_model(
        args.deploy_config, args.checkpoint, args.gpu_id,
        eval_config=args.eval_config)

    data = _get_sample_batch(loader, args.sample_idx)
    info = dataset.data_infos[args.sample_idx]
    stem = info.get('sample_stem', info['token'][:8])

    print('[2/2] full_eval decomposition + split PT/OM + merged OM ...')
    img_inputs = _to_device(data['img_inputs'][0], device)
    eval_stages = _run_eval_stages(eval_model, img_inputs, occ_shape)
    eval_wrapped = NPUDataParallel(eval_model.cuda(), device_ids=[args.gpu_id])
    pred_eval = run_eval_full(eval_wrapped, data, device, args.gpu_id)
    img, ranks_bev, ranks_depth, ranks_feat = _get_bev_pool_metas_v3(
        model, data, device, layout=part1_layout)
    part1 = build_part1_wrapper(model, part1_layout).eval()
    part3 = Part3ExportWrapper(model).eval()

    pt_vt = run_split_pt_stages(
        model, part1, part3, img, ranks_bev, ranks_depth, ranks_feat,
        device, occ_shape, data=data, bev_mode='vt_core')
    pt_pool = run_split_pt_stages(
        model, part1, part3, img, ranks_bev, ranks_depth, ranks_feat,
        device, occ_shape, data=data, bev_mode='bev_pool_v3')

    acl = AclSession(device_id=args.gpu_id)
    acl.load('part1', part1_om)
    acl.load('part3', part3_om)
    img_np = img.detach().cpu().numpy().astype(np.float32)
    om_tran, om_depth = acl.infer('part1', img_np)
    om_bev = run_bev_view_transform_core(
        model, data, img, om_tran, om_depth, device)
    om_occ = acl.infer('part3', om_bev.detach().cpu().numpy().astype(np.float32))[0]
    om_pred = decode_occ_logits(om_occ, occ_shape)
    om_stages = {
        'tran_feat': om_tran,
        'depth': om_depth,
        'bev_feat': om_bev,
        'occ_np': om_occ,
        'pred': om_pred,
    }

    # Cross-feed
    pt_bev = pt_vt['bev_feat']
    cross_pt_occ = _run_part3_pt(part3, pt_bev)
    cross_om_on_pt_bev = _run_part3_om(acl, pt_bev)
    cross_pt_on_om_bev = _run_part3_pt(part3, om_bev)
    cross_eval_on_pt_bev = _run_eval_part3_on_bev(eval_model, pt_bev)
    cross_feed = {
        'OM_part3_on_PT_bev_vs_PT_part3': tensor_diff(
            cross_pt_occ, cross_om_on_pt_bev),
        'PT_part3_on_OM_bev_vs_OM_part3': tensor_diff(
            cross_pt_on_om_bev, om_occ),
        'eval_part3_on_PT_bev_vs_deploy_part3': tensor_diff(
            cross_eval_on_pt_bev, cross_pt_occ),
    }

    acl_merged = AclSession(device_id=args.gpu_id)
    acl_merged.load('merged', args.merged_om)
    merged_img = _img_from_data(data, layout=part1_layout).astype(np.float32)
    merged_occ = acl_merged.infer('merged', merged_img)[0]
    merged_pred = decode_occ_logits(merged_occ, occ_shape)
    acl_merged.close()
    acl.close()

    # Submodules
    pt_p1_sub = _run_part1_submodules(model, img)
    om_p1_sub = {
        'tran_feat': om_tran,
        'depth': om_depth,
    }
    part1_sub = {
        'tran_feat_flat': tensor_diff(pt_p1_sub['tran_feat'], om_tran),
        'depth_flat': tensor_diff(pt_p1_sub['depth'], om_depth),
        'encoder_feat': tensor_diff(
            pt_p1_sub['encoder_feat'],
            eval_stages['encoder_feat']),
    }
    part3_pt_vs_eval = {
        'occ_logits': tensor_diff(cross_eval_on_pt_bev, cross_pt_occ),
    }

    gt_npz = np.load(os.path.join(info['occ_path'], 'labels.npz'))
    miou = {
        'full_eval': sample_metrics(
            gt_npz['semantics'], pred_eval, gt_npz['mask_camera'])['mIoU'],
        'split_pt': sample_metrics(
            gt_npz['semantics'], pt_vt['pred'], gt_npz['mask_camera'])['mIoU'],
        'split_OM': sample_metrics(
            gt_npz['semantics'], om_pred, gt_npz['mask_camera'])['mIoU'],
        'merged_OM': sample_metrics(
            gt_npz['semantics'], merged_pred, gt_npz['mask_camera'])['mIoU'],
    }

    layers = {
        'L5_semantics': {
            'split_pt_vs_split_OM': semantic_overlap(pt_vt['pred'], om_pred),
            'full_eval_vs_split_OM': semantic_overlap(pred_eval, om_pred),
            'full_eval_vs_split_pt': semantic_overlap(pred_eval, pt_vt['pred']),
            'split_pt_vs_merged_OM': semantic_overlap(pt_vt['pred'], merged_pred),
            'full_eval_vs_merged_OM': semantic_overlap(pred_eval, merged_pred),
        },
        'L4_part3_occ_logits': {
            'split_pt_vs_split_OM': tensor_diff(pt_vt['occ_np'], om_occ),
            'full_eval_vs_split_OM': tensor_diff(eval_stages['occ_np'], om_occ),
            'full_eval_vs_split_pt': tensor_diff(eval_stages['occ_np'], pt_vt['occ_np']),
            'eval_part3_on_PT_bev_vs_deploy_part3': tensor_diff(
                cross_eval_on_pt_bev, cross_pt_occ),
            'split_pt_vs_merged_OM': tensor_diff(pt_vt['occ_np'], merged_occ),
        },
        'L3_part2_bev_feat': {
            'split_pt_vt_core_vs_split_OM': tensor_diff(pt_vt['bev_feat'], om_bev),
            'full_eval_vs_split_pt_vt_core': tensor_diff(
                eval_stages['bev_feat'], pt_vt['bev_feat']),
            'full_eval_vs_split_OM': tensor_diff(eval_stages['bev_feat'], om_bev),
            'split_pt_vt_core_vs_bev_pool_v3': tensor_diff(
                pt_vt['bev_feat'], pt_pool['bev_feat']),
            'split_pt_bev_pool_v3_vs_split_OM': tensor_diff(
                pt_pool['bev_feat'], om_bev),
        },
        'L2_part1_depth': {
            'split_pt_vs_split_OM': tensor_diff(pt_vt['depth'], om_depth),
            'full_eval_vs_split_OM': tensor_diff(eval_stages['depth'], om_depth),
            'full_eval_vs_split_pt': tensor_diff(eval_stages['depth'], pt_vt['depth']),
        },
        'L1_part1_tran_feat': {
            'split_pt_vs_split_OM': tensor_diff(pt_vt['tran_feat'], om_tran),
            'full_eval_vs_split_OM': tensor_diff(eval_stages['tran_feat'], om_tran),
            'full_eval_vs_split_pt': tensor_diff(
                eval_stages['tran_feat'], pt_vt['tran_feat']),
        },
        'L0_img': {
            'note': 'shared dataloader batch (identical input)',
        },
    }

    report = {
        'sample_idx': args.sample_idx,
        'stem': stem,
        'part1_om': part1_om,
        'part3_om': part3_om,
        'merged_om': args.merged_om,
        'layers': layers,
        'cross_feed': cross_feed,
        'submodules': {
            'part1': part1_sub,
            'part3_pt_vs_eval': part3_pt_vs_eval,
        },
        'miou': miou,
    }
    report['first_divergence'] = _find_first_divergence(layers)
    report['interpretation'] = _build_interpretation(report)

    if tensor_dir:
        for prefix, stages in (
            ('eval', eval_stages),
            ('split_pt', pt_vt),
            ('split_om', om_stages),
        ):
            for k, v in stages.items():
                if k == 'pred':
                    continue
                _maybe_save(tensor_dir, f'{prefix}_{k}', v)
        _maybe_save(tensor_dir, 'merged_occ', merged_occ)

    out_path.write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
    txt_path = out_path.with_suffix('.txt')
    _write_txt_report(report, txt_path)
    print(f'Wrote {out_path}')
    print(f'Wrote {txt_path}')
    if report['first_divergence']:
        fd = report['first_divergence']
        print(f"FIRST_DIVERGENCE: {fd['layer']} / {fd['pair']}")
    print(f"Interpretation: {report['interpretation']}")


def _build_interpretation(report):
    """Heuristic diagnosis from cross-feed and layer diffs."""
    cf = report.get('cross_feed', {})
    om_p3_ok = cf.get('OM_part3_on_PT_bev_vs_PT_part3', {}).get('allclose', False)
    layers = report['layers']
    p1_om = layers.get('L1_part1_tran_feat', {}).get('split_pt_vs_split_OM', {})
    eval_vs_pt = layers.get('L5_semantics', {}).get('full_eval_vs_split_pt', {})
    eval_vs_om = layers.get('L5_semantics', {}).get('full_eval_vs_split_OM', {})

    parts = []
    if p1_om.get('status') in ('WARN', 'FAIL'):
        parts.append('split OM part1 shows numeric drift vs split_pt')
    if om_p3_ok:
        parts.append('part3 OM matches PT when given identical PT bev (OM conversion OK)')
    elif cf.get('OM_part3_on_PT_bev_vs_PT_part3', {}).get('status') == 'FAIL':
        parts.append('part3 OM conversion error (cross-feed FAIL on PT bev)')

    if eval_vs_pt.get('sem_match', 1.0) < 0.95:
        parts.append('deploy path gap: full_eval vs split_pt semantics diverge')
    if eval_vs_om.get('sem_match', 1.0) < 0.95:
        if eval_vs_pt.get('sem_match', 1.0) < 0.95:
            parts.append('full_eval vs OM gap largely from deploy path, not OM alone')
        else:
            parts.append('OM-specific semantic gap vs full_eval')

    merged = layers.get('L5_semantics', {}).get('split_pt_vs_merged_OM', {})
    if merged.get('sem_match', 1.0) < 0.99:
        parts.append('merged OM differs from split_pt at semantics layer')
    return '; '.join(parts) if parts else 'all comparisons within expected tolerance'


if __name__ == '__main__':
    from mx_driving.patcher import PatcherBuilder, Patch
    from mx_driving.patcher import batch_matmul, resnet_add_relu, resnet_fp16

    pb = (PatcherBuilder()
          .add_module_patch('torch', Patch(batch_matmul))
          .add_module_patch('mmdet', Patch(resnet_add_relu))
          .add_module_patch('mmdet', Patch(resnet_fp16)))
    with pb.build():
        main()
