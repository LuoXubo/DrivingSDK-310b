#!/usr/bin/env python3
"""Shared split deploy inference: PyTorch chain and stage-wise tensors."""
from __future__ import annotations

import numpy as np
import torch
from mmcv import Config
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model

from tools.export_onnx_split_npu import (
    Part1ExportWrapper,
    Part3ExportWrapper,
    _get_bev_pool_metas_v3,
    _get_sample_batch,
    _import_plugin,
    _part1_to_bev_pool_inputs,
    _run_split_forward,
    _to_device,
    build_part1_wrapper,
    part1_layout_from_manifest,
    run_bev_pool_v3,
)
from tools.run_unified_viz_npu import decode_occ_logits

TEST10_PKL = 'data/car_perception_grid/nuscenes/bevdetv2-nuscenes_infos_test10.pkl'
DEFAULT_OCC_SHAPE = (1, 200, 200, 2, 3)


def _compat_cfg(cfg):
    try:
        from mmdet.utils import compat_cfg
    except ImportError:
        from mmdet3d.utils import compat_cfg
    return compat_cfg(cfg)


def build_deploy_model(cfg_path, checkpoint, gpu_id, eval_config=None):
    """Build BEVDetOCCTRT on NPU; optionally build BEVDetOCC eval model."""
    cfg = Config.fromfile(cfg_path)
    cfg.merge_from_dict({
        'data.test.ann_file': TEST10_PKL,
        'data.test.data_root': 'data/car_perception_grid/nuscenes/',
        'data.val.ann_file': TEST10_PKL,
        'data.val.data_root': 'data/car_perception_grid/nuscenes/',
    })
    cfg = _compat_cfg(cfg)
    _import_plugin(cfg, cfg_path)
    if not cfg.model.type.endswith('TRT'):
        cfg.model.type += 'TRT'
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    if hasattr(cfg.model, 'img_view_transformer'):
        cfg.model.img_view_transformer.accelerate = True
    cfg.model.train_cfg = None

    device = torch.device(f'npu:{gpu_id}')
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, checkpoint, map_location='cpu')
    model = fuse_conv_bn(model).to(device).eval()

    eval_model = None
    if eval_config is not None:
        ecfg = Config.fromfile(eval_config)
        ecfg.merge_from_dict({
            'data.test.ann_file': TEST10_PKL,
            'data.test.data_root': 'data/car_perception_grid/nuscenes/',
        })
        ecfg = _compat_cfg(ecfg)
        _import_plugin(ecfg, eval_config)
        ecfg.model.pretrained = None
        ecfg.model.train_cfg = None
        if isinstance(ecfg.data.test, dict):
            ecfg.data.test.test_mode = True
        eval_model = build_model(ecfg.model, test_cfg=ecfg.get('test_cfg'))
        load_checkpoint(eval_model, checkpoint, map_location='cpu')
        eval_model = fuse_conv_bn(eval_model).to(device).eval()

    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False)
    return model, eval_model, dataset, loader, device


def tensor_diff(ref, test, rtol=1e-2, atol=1e-3):
    """Numeric diff between tensors/arrays."""
    if isinstance(ref, torch.Tensor):
        r = ref.detach().float().cpu().numpy().reshape(-1)
    else:
        r = np.asarray(ref, dtype=np.float32).reshape(-1)
    if isinstance(test, torch.Tensor):
        t = test.detach().float().cpu().numpy().reshape(-1)
    else:
        t = np.asarray(test, dtype=np.float32).reshape(-1)
    if r.shape != t.shape:
        return {
            'status': 'SHAPE',
            'ref_shape': list(np.asarray(ref).shape),
            'test_shape': list(np.asarray(test).shape),
        }
    d = np.abs(r - t)
    scale = float(np.abs(r).max()) + 1e-6
    ok = bool(np.allclose(r, t, rtol=rtol, atol=atol))
    status = 'OK' if ok else ('WARN' if d.max() < 0.05 else 'FAIL')
    return {
        'status': status,
        'max_abs': float(d.max()),
        'mean_abs': float(d.mean()),
        'rel_max': float(d.max() / scale),
        'allclose': ok,
    }


def semantic_overlap(pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """Compare two semantic grids."""
    a = pred_a.astype(np.int64)
    b = pred_b.astype(np.int64)
    match = float((a == b).mean())
    car_a = int((a == 1).sum())
    car_b = int((b == 1).sum())
    car_tp = int(((a == 1) & (b == 1)).sum())
    return {
        'sem_match': match,
        'car_a': car_a,
        'car_b': car_b,
        'car_overlap': car_tp,
    }


@torch.no_grad()
def _flat_part1_to_vt_tensors(tran_flat, depth_flat, view_transformer, num_cam=None):
    """Unflatten part1 outputs to (B*N, C, fH, fW) and (B*N, D, fH, fW)."""
    vt = view_transformer
    fH = int(vt.frustum.shape[1])
    fW = int(vt.frustum.shape[2])
    feat_c = int(vt.out_channels)
    d = int(vt.D)
    if isinstance(tran_flat, torch.Tensor):
        tran_flat = tran_flat.detach()
        depth_flat = depth_flat.detach()
    else:
        tran_flat = torch.from_numpy(np.asarray(tran_flat, dtype=np.float32))
        depth_flat = torch.from_numpy(np.asarray(depth_flat, dtype=np.float32))
    n = num_cam or int(tran_flat.numel() // (fH * fW * feat_c))
    tran_feat = tran_flat.reshape(n, fH, fW, feat_c).permute(0, 3, 1, 2).contiguous()
    depth = depth_flat.reshape(n, d, fH, fW).contiguous()
    return tran_feat, depth


def _num_cam_from_img(img):
    if isinstance(img, torch.Tensor):
        if img.dim() == 5:
            return int(img.shape[1])
        return int(img.shape[0])
    img = np.asarray(img)
    if img.ndim == 5:
        return int(img.shape[1])
    return int(img.shape[0])


@torch.no_grad()
def run_bev_view_transform_core(model, data, img, tran_flat, depth_flat, device):
    """Eval-equivalent BEV: view_transform_core with per-sample geometry."""
    vt = model.img_view_transformer
    inputs = _to_device(data['img_inputs'][0], device)
    prepared = model.prepare_inputs(inputs)
    feat, _ = model.image_encoder(prepared[0])
    tran_feat, depth = _flat_part1_to_vt_tensors(
        tran_flat, depth_flat, vt, num_cam=_num_cam_from_img(img))
    tran_feat = tran_feat.to(device)
    depth = depth.to(device)
    vt_in = [feat] + list(prepared[1:7])
    if vt.accelerate:
        vt.initial_flag = True
        vt.pre_compute(vt_in)
    bev_feat, _ = vt.view_transform_core(vt_in, depth, tran_feat)
    return bev_feat.contiguous()


@torch.no_grad()
def run_om_stages_vt_core(acl_sess, part1_key, part3_key, model, data, img,
                          device, occ_shape=DEFAULT_OCC_SHAPE):
    """OM part1 + eval-equivalent view_transform + OM part3."""
    img_np = img.detach().cpu().numpy().astype(np.float32)
    tran_flat, depth_flat = acl_sess.infer(part1_key, img_np)
    bev_feat = run_bev_view_transform_core(
        model, data, img, tran_flat, depth_flat, device)
    bev_np = bev_feat.detach().cpu().numpy().astype(np.float32)
    occ_raw = acl_sess.infer(part3_key, bev_np)[0]
    pred = decode_occ_logits(occ_raw, occ_shape)
    return {
        'tran_feat': tran_flat,
        'depth': depth_flat,
        'bev_feat': bev_feat,
        'occ_np': occ_raw,
        'pred': pred,
    }


@torch.no_grad()
def run_split_pt_stages(model, part1, part3, img, ranks_bev, ranks_depth, ranks_feat,
                        device, occ_shape=DEFAULT_OCC_SHAPE, data=None,
                        bev_mode='vt_core'):
    """Split deploy PyTorch: part1 -> bev -> part3."""
    tran, depth = part1(img)
    if bev_mode == 'vt_core' and data is not None:
        bev = run_bev_view_transform_core(model, data, img, tran, depth, device)
        outs = part3(bev)
        occ_logits = outs[0] if isinstance(outs, (list, tuple)) else outs
    else:
        depth_bev, feat_bev = _part1_to_bev_pool_inputs(
            tran, depth, model.img_view_transformer)
        bev = run_bev_pool_v3(
            model, depth_bev, feat_bev, ranks_bev, ranks_depth, ranks_feat)
        outs = part3(bev)
        occ_logits = outs[0] if isinstance(outs, (list, tuple)) else outs
    occ_np = occ_logits.detach().float().cpu().numpy()
    pred = decode_occ_logits(occ_np, occ_shape)
    return {
        'tran_feat': tran,
        'depth': depth,
        'bev_feat': bev,
        'occ_logits': occ_logits,
        'occ_np': occ_np,
        'pred': pred,
    }


@torch.no_grad()
def run_split_pt_bev_from_part1(model, tran_feat, depth, ranks_bev, ranks_depth, ranks_feat):
    depth_bev, feat_bev = _part1_to_bev_pool_inputs(
        tran_feat, depth, model.img_view_transformer)
    return run_bev_pool_v3(
        model, depth_bev, feat_bev, ranks_bev, ranks_depth, ranks_feat)


@torch.no_grad()
def run_eval_full(eval_model, data, device, gpu_id=0):
    """BEVDetOCC full eval path (same calling convention as eval_car_grid_occ)."""
    from mmcv.device.npu import NPUDataParallel

    if not isinstance(eval_model, NPUDataParallel):
        eval_model = NPUDataParallel(eval_model.cuda(), device_ids=[gpu_id])

    def _to_device(obj):
        if torch.is_tensor(obj):
            return obj.to(device)
        if isinstance(obj, (list, tuple)):
            return type(obj)(_to_device(x) for x in obj)
        return obj

    batch = {k: _to_device(v) for k, v in data.items()}
    result = eval_model(return_loss=False, rescale=True, **batch)
    out = result[0]
    if isinstance(out, dict) and 'pred_occ' in out:
        pred = out['pred_occ']
    else:
        pred = out
    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()
    if pred.ndim == 4 and pred.shape[0] == 1:
        pred = pred[0]
    return pred.astype(np.int64)


@torch.no_grad()
def run_split_pt_inference(model, dataset, loader, device, occ_shape=DEFAULT_OCC_SHAPE,
                           part1_layout='eval'):
    """All test10 samples through split deploy PyTorch."""
    part1 = build_part1_wrapper(model, part1_layout).eval()
    part3 = Part3ExportWrapper(model).eval()
    preds = []
    for idx in range(len(dataset)):
        data = _get_sample_batch(loader, idx)
        img, ranks_bev, ranks_depth, ranks_feat = _get_bev_pool_metas_v3(
            model, data, device, layout=part1_layout)
        stages = run_split_pt_stages(
            model, part1, part3, img, ranks_bev, ranks_depth, ranks_feat,
            device, occ_shape, data=data, bev_mode='vt_core')
        preds.append(stages['pred'])
    return preds
