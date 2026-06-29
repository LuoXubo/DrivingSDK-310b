# Copyright (c) OpenMMLab. All rights reserved.
"""Compare test.py inference vs split deploy (part1 / bev_pool_v3 / part3).

Reference (ground truth for deployment alignment):
  test.py path = BEVDetOCC + flashocc-r50.py
    stage1: image_encoder + depth_net  -> tran_feat, depth
    stage2: view_transform (bev_pool_v3 inside LSS) -> bev_feat
    stage3: bev_encoder + occ_head -> occ logits

Split deploy path = BEVDetOCCTRT + flashocc-r50-trt.py (same checkpoint)
    stage1: Part1ExportWrapper / part1.onnx / part1.om
    stage2: bev_pool_v3 (mx_driving), ranks from test model
    stage3: Part3ExportWrapper / part3.onnx / part3.om

Optional: ONNX (onnxruntime CPU), OM (ACL) per stage.
"""
import argparse
import importlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch_npu
from torch_npu.contrib import transfer_to_npu
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import load_checkpoint

import mmdet
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model

if mmdet.__version__ > '2.23.0':
    from mmdet.utils import setup_multi_processes
else:
    from mmdet3d.utils import setup_multi_processes

try:
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg

sys.path.insert(0, os.getcwd())

from tools.export_onnx_split_npu import (  # noqa: E402
    Part1EvalAlignedExportWrapper,
    Part1ExportWrapper,
    Part3ExportWrapper,
    _get_sample_batch,
    _import_plugin,
    _part1_to_bev_pool_inputs,
    run_bev_pool_v3,
)
from tools.run_split_infer_npu import (  # noqa: E402
    AclSession,
    _load_img_and_ranks,
    _resolve_om_paths,
    _to_device,
)


def parse_args():
    p = argparse.ArgumentParser(
        description='Compare test.py vs split deploy, stage by stage')
    p.add_argument(
        '--test-config',
        default='projects/configs/flashocc/flashocc-r50.py',
        help='config used by tools/test.py')
    p.add_argument(
        '--deploy-config',
        default='projects/configs/flashocc/flashocc-r50-trt.py',
        help='deploy / ONNX split config')
    p.add_argument('checkpoint')
    p.add_argument(
        '--manifest',
        default='work_dirs/onnx_split/flashocc_r50_deploy_manifest.json',
        help='manifest with onnx/om paths')
    p.add_argument('--sample-idx', type=int, default=0)
    p.add_argument('--gpu-id', type=int, default=0)
    p.add_argument('--fuse-conv-bn', action='store_true')
    p.add_argument(
        '--no-acceleration',
        action='store_true',
        help='disable VT accelerate on both models')
    p.add_argument('--check-onnx', action='store_true', help='compare part1/3 ONNX')
    p.add_argument('--check-om', action='store_true', help='compare part1/3 OM')
    p.add_argument(
        '--sequential-load',
        action='store_true',
        default=True,
        help='run test.py ref first, free NPU, then split (avoids OOM)')
    p.add_argument(
        '--no-sequential-load',
        action='store_false',
        dest='sequential_load',
        help='keep both models on NPU (may OOM)')
    p.add_argument('--cfg-options', nargs='+', action=DictAction)
    return p.parse_args()


def _diff(name, ref, test, rtol=1e-2, atol=1e-3):
    """Print max/mean abs diff; ref is ground truth (test.py)."""
    r = ref.detach().float().reshape(-1).cpu().numpy()
    t = test.detach().float().reshape(-1).cpu().numpy()
    if r.shape != t.shape:
        print(f'[{name}] SHAPE ref={tuple(ref.shape)} test={tuple(test.shape)}')
        return
    d = np.abs(r - t)
    scale = float(np.abs(r).max()) + 1e-6
    ok = np.allclose(r, t, rtol=rtol, atol=atol)
    print(
        f'[{name}] max_abs={d.max():.6f} mean_abs={d.mean():.6f} '
        f'rel_max={d.max()/scale:.6f} allclose={ok}')


def _build_model(cfg_path, args, as_trt=False):
    cfg = Config.fromfile(cfg_path)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg = compat_cfg(cfg)
    setup_multi_processes(cfg)
    _import_plugin(cfg, cfg_path)

    if as_trt and not cfg.model.type.endswith('TRT'):
        cfg.model.type = cfg.model.type + 'TRT'

    cfg.model.pretrained = None
    if hasattr(cfg.model, 'img_view_transformer'):
        if args.no_acceleration:
            cfg.model.img_view_transformer.accelerate = False
        else:
            cfg.model.img_view_transformer.accelerate = True

    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    device = torch.device(f'npu:{args.gpu_id}')
    return model.to(device).eval(), device


# ---------------------------------------------------------------------------
# test.py reference (BEVDetOCC) — decomposed
# ---------------------------------------------------------------------------

@torch.no_grad()
def testpy_stage1(model, img_inputs):
    """Same math as LSSViewTransformer.forward up to pool."""
    prepared = model.prepare_inputs(img_inputs)
    feat, _ = model.image_encoder(prepared[0])  # (B,N,C,fH,fW)
    b, n, c, h, w = feat.shape
    x = feat.view(b * n, c, h, w)
    vt = model.img_view_transformer
    out = vt.depth_net(x)
    depth = out[:, :vt.D].softmax(dim=1)
    tran_feat = out[:, vt.D:vt.D + vt.out_channels]
    return tran_feat, depth, feat, prepared


@torch.no_grad()
def testpy_stage2(model, prepared, feat, depth, tran_feat):
    vt = model.img_view_transformer
    vt_in = [feat] + list(prepared[1:7])
    if vt.accelerate:
        vt.pre_compute(vt_in)
    bev_feat, _ = vt.view_transform_core(vt_in, depth, tran_feat)
    return bev_feat


@torch.no_grad()
def testpy_stage3(model, bev_feat):
    x = model.bev_encoder(bev_feat)
    outs = model.occ_head(x)
    if isinstance(outs, (list, tuple)):
        return outs[0]
    return outs


@torch.no_grad()
def testpy_full(model, img_inputs):
    """Full chain as test.py: extract_feat -> occ_head (no get_occ_gpu)."""
    img_feats, _, _ = model.extract_feat(
        points=None, img_inputs=img_inputs, img_metas=None)
    x = img_feats[0]
    if getattr(model, 'upsample', False):
        x = F.interpolate(
            x, scale_factor=2, mode='bilinear', align_corners=True)
    outs = model.occ_head(x)
    if isinstance(outs, (list, tuple)):
        return outs[0]
    return outs


# ---------------------------------------------------------------------------
# Split deploy (BEVDetOCCTRT)
# ---------------------------------------------------------------------------

@torch.no_grad()
def split_stage1(model_trt, img):
    """Legacy part1: (N,3,H,W) via forward_part1."""
    part1 = Part1ExportWrapper(model_trt).eval()
    return part1(img)


@torch.no_grad()
def split_stage1_eval(model_trt, img_bn):
    """Eval-aligned part1: (B,N,3,H,W) via image_encoder + depth_net."""
    part1 = Part1EvalAlignedExportWrapper(model_trt).eval()
    return part1(img_bn)


@torch.no_grad()
def split_stage2_pool(model_trt, depth_bev, feat_bev, ranks_bev, ranks_depth,
                      ranks_feat):
    return run_bev_pool_v3(
        model_trt, depth_bev, feat_bev, ranks_bev, ranks_depth, ranks_feat)


@torch.no_grad()
def split_stage3(model_trt, bev_feat):
    part3 = Part3ExportWrapper(model_trt).eval()
    outs = part3(bev_feat)
    if isinstance(outs, (list, tuple)):
        return outs[0]
    return outs


def _onnx_run(path, feed):
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    names = [i.name for i in sess.get_inputs()]
    return sess.run(None, {names[0]: feed})


def _load_data(args, device):
    test_cfg = Config.fromfile(args.test_config)
    if args.cfg_options is not None:
        test_cfg.merge_from_dict(args.cfg_options)
    test_cfg = compat_cfg(test_cfg)
    if isinstance(test_cfg.data.test, dict):
        test_cfg.data.test.test_mode = True
    dataset = build_dataset(test_cfg.data.test)
    data_loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=0,
        dist=False, shuffle=False)
    data = _get_sample_batch(data_loader, args.sample_idx)
    return _to_device(data['img_inputs'][0], device)


def _run_testpy_reference(model_test, img_inputs):
    """Run test.py decomposition; return CPU tensors."""
    t_tran, t_depth, t_feat, t_prepared = testpy_stage1(model_test, img_inputs)
    t_bev = testpy_stage2(model_test, t_prepared, t_feat, t_depth, t_tran)
    t_occ = testpy_stage3(model_test, t_bev)
    t_full = testpy_full(model_test, img_inputs)
    t_tran_flat = t_tran.permute(0, 2, 3, 1).contiguous().flatten(0, 2).cpu()
    t_depth_flat = t_depth.contiguous().flatten().cpu()
    img6 = t_prepared[0].squeeze(0).float().contiguous().cpu()
    if img6.shape[0] > 6:
        img6 = img6[:6]
    img_bn = t_prepared[0].float().contiguous().cpu()
    if img_bn.shape[1] > 6:
        img_bn = img_bn[:, :6]
    return {
        'tran_flat': t_tran_flat,
        'depth_flat': t_depth_flat,
        'bev': t_bev.cpu(),
        'occ': t_occ.cpu(),
        'full_occ': t_full.cpu(),
        'img6': img6,
        'img_bn': img_bn,
        'prepared': [x.cpu() for x in t_prepared],
    }


def main():
    args = parse_args()
    manifest, part1_om, part3_om, meta_npz = _resolve_om_paths(args.manifest, None)
    prefix = os.path.basename(args.manifest).replace('_deploy_manifest.json', '')
    work_dir = os.path.dirname(os.path.abspath(args.manifest))
    part1_onnx = os.path.join(work_dir, f'{prefix}_part1.onnx')
    part3_onnx = os.path.join(work_dir, f'{prefix}_part3.onnx')

    print('=' * 60)
    print('Reference: test.py  (BEVDetOCC + flashocc-r50.py)')
    print('Split:     deploy   (BEVDetOCCTRT + flashocc-r50-trt.py)')
    print('=' * 60)

    device = torch.device(f'npu:{args.gpu_id}')

    # Phase A: test.py reference only
    print('\n[phase A] Loading BEVDetOCC (test.py reference)...')
    model_test, _ = _build_model(args.test_config, args, as_trt=False)
    img_inputs = _load_data(args, device)
    ref = _run_testpy_reference(model_test, img_inputs)
    if args.sequential_load:
        del model_test
        torch.npu.empty_cache()

    # Phase B: split deploy
    print('[phase B] Loading BEVDetOCCTRT (split deploy)...')
    model_split, _ = _build_model(args.deploy_config, args, as_trt=True)
    img_inputs = _load_data(args, device)
    ranks_pack = model_split.get_bev_pool_input(img_inputs)
    if isinstance(ranks_pack, torch.Tensor):
        ranks_pack = (ranks_pack,)
    ranks_bev, ranks_depth, ranks_feat = [
        t.int().contiguous() for t in ranks_pack[:3]]

    img6 = ref['img6'].to(device)
    img_bn = ref['img_bn'].to(device)
    t_tran_flat = ref['tran_flat']
    t_depth_flat = ref['depth_flat']
    t_bev = ref['bev'].to(device)
    t_occ = ref['occ']
    t_full = ref['full_occ']

    # ----- Stage 1 -----
    print('\n--- Stage 1: image_encoder + depth_net (tran_feat, depth) ---')
    s_tran_flat, s_depth_flat = split_stage1_eval(model_split, img_bn)
    _diff('testpy vs split-PyTorch(eval part1) tran_feat', t_tran_flat, s_tran_flat.cpu())
    _diff('testpy vs split-PyTorch(eval part1) depth', t_depth_flat, s_depth_flat.cpu())

    s_tran_legacy, s_depth_legacy = split_stage1(model_split, img6)
    _diff('testpy vs split-PyTorch(legacy part1) tran_feat', t_tran_flat, s_tran_legacy.cpu())
    _diff('testpy vs split-PyTorch(legacy part1) depth', t_depth_flat, s_depth_legacy.cpu())

    if args.check_onnx and os.path.isfile(part1_onnx):
        o_tran, o_depth = _onnx_run(part1_onnx, img_bn.cpu().numpy())
        _diff('testpy vs part1-ONNX tran_feat', t_tran_flat, torch.from_numpy(o_tran))
        _diff('testpy vs part1-ONNX depth', t_depth_flat, torch.from_numpy(o_depth))

    if args.check_om and os.path.isfile(part1_om):
        acl = AclSession(device_id=args.gpu_id)
        acl.load('part1', part1_om)
        o_tran, o_depth = acl.infer('part1', img_bn.cpu().numpy())
        acl.close()
        _diff('testpy vs part1-OM tran_feat', t_tran_flat, torch.from_numpy(o_tran))
        _diff('testpy vs part1-OM depth', t_depth_flat, torch.from_numpy(o_depth))

    # ----- Stage 2 -----
    print('\n--- Stage 2: bev_pool_v3 -> bev_feat [1,64,200,200] ---')
    depth_bev, feat_bev = _part1_to_bev_pool_inputs(
        s_tran_flat, s_depth_flat, model_split.img_view_transformer)
    s_bev = split_stage2_pool(
        model_split, depth_bev, feat_bev, ranks_bev, ranks_depth, ranks_feat)
    _diff('testpy vs split-PyTorch bev_feat', t_bev, s_bev)

    # ----- Stage 3 -----
    print('\n--- Stage 3: bev_encoder + occ_head -> occ logits ---')
    s_occ = split_stage3(model_split, s_bev)
    _diff('testpy vs split-PyTorch occ', t_occ, s_occ.cpu())

    if args.check_onnx and os.path.isfile(part3_onnx):
        bev_np = s_bev.detach().cpu().numpy().astype(np.float32)
        o_occ = _onnx_run(part3_onnx, bev_np)[0]
        _diff('testpy vs part3-ONNX occ', t_occ, torch.from_numpy(o_occ).view_as(t_occ))

    if args.check_om and os.path.isfile(part3_om):
        acl = AclSession(device_id=args.gpu_id)
        acl.load('part3', part3_om)
        o_occ = acl.infer('part3', s_bev.detach().cpu().numpy())[0]
        acl.close()
        _diff('testpy vs part3-OM occ', t_occ, torch.from_numpy(o_occ).view_as(t_occ))

    # ----- End-to-end -----
    print('\n--- End-to-end: test.py extract_feat+occ_head vs split 1+2+3 ---')
    _diff('testpy-full vs split-chain occ', t_full, s_occ.cpu())

    print('\nDone. Reference is always test.py (BEVDetOCC); split/ONNX/OM vs each stage.')


if __name__ == '__main__':
    from mx_driving.patcher import PatcherBuilder, Patch
    from mx_driving.patcher import batch_matmul, resnet_add_relu

    pb = (PatcherBuilder()
          .add_module_patch('torch', Patch(batch_matmul))
          .add_module_patch('mmdet', Patch(resnet_add_relu)))
    with pb.build():
        main()
