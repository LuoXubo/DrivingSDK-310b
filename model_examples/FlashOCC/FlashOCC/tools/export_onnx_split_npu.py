# Copyright (c) OpenMMLab. All rights reserved.
"""Split FlashOCC export for Ascend NPU deploy.

Pipeline:
  img --[part1.onnx / OM]--> depth, tran_feat
        --[bev_pool_v3 on NPU, NOT in ONNX]--> bev_feat
        --[part3.onnx / OM]--> occ logits

Middle BEV pool uses mx_driving.bev_pool_v3 (same as training), not TRTBEVPoolv2.
"""
import argparse
import importlib
import json
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.onnx
import torch_npu
from torch_npu.contrib import transfer_to_npu
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import load_checkpoint

import mmdet
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.datasets import replace_ImageToTensor
from mx_driving import bev_pool_v3

if mmdet.__version__ > '2.23.0':
    from mmdet.utils import setup_multi_processes
else:
    from mmdet3d.utils import setup_multi_processes

try:
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg

sys.path.insert(0, os.getcwd())


class Part1ExportWrapper(nn.Module):
    """img -> (tran_feat, depth) flattened, same as BEVDetOCCTRT.forward_part1."""

    def __init__(self, detector):
        super().__init__()
        self.detector = detector

    def forward(self, img):
        return self.detector.forward_part1(img)


class Part3ExportWrapper(nn.Module):
    """bev_feat (NCHW) -> occ head output."""

    def __init__(self, detector):
        super().__init__()
        self.detector = detector

    def forward(self, bev_feat):
        outs = self.detector.forward_part3(bev_feat.contiguous().reshape(-1))
        if isinstance(outs, (list, tuple)):
            return tuple(outs)
        return (outs,)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export FlashOCC part1/part3 ONNX; bev_pool_v3 stays on NPU')
    parser.add_argument('config', help='config, e.g. flashocc-r50-trt.py')
    parser.add_argument('checkpoint', help='checkpoint path')
    parser.add_argument('work_dir', help='output directory')
    parser.add_argument('--prefix', default='flashocc', help='file name prefix')
    parser.add_argument('--opset-version', type=int, default=11)
    parser.add_argument('--fuse-conv-bn', action='store_true')
    parser.add_argument('--sample-idx', type=int, default=0)
    parser.add_argument('--no-acceleration', action='store_true')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument(
        '--skip-verify', action='store_true', help='skip split pipeline verify')
    parser.add_argument(
        '--parallel-export',
        action='store_true',
        help='reserved; ONNX export runs sequentially (shared weights)')
    parser.add_argument(
        '--export-devices',
        type=str,
        default='0',
        help='logical NPU id for ONNX export')
    parser.add_argument(
        '--parallel-atc',
        action='store_true',
        default=True,
        help='emit atc script that runs part1/part3 conversion in parallel')
    parser.add_argument(
        '--no-parallel-atc',
        action='store_false',
        dest='parallel_atc',
        help='emit sequential atc commands in shell script')
    parser.add_argument(
        '--export-on-cpu',
        action='store_true',
        default=True,
        help='trace ONNX on CPU (recommended; NPU trace often diverges)')
    parser.add_argument(
        '--export-on-npu',
        action='store_false',
        dest='export_on_cpu',
        help='trace ONNX on NPU (legacy)')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    return parser.parse_args()


def _import_plugin(cfg, config_path):
    if not (hasattr(cfg, 'plugin') and cfg.plugin):
        return
    if hasattr(cfg, 'plugin_dir'):
        _module_dir = os.path.dirname(cfg.plugin_dir).split('/')
    else:
        _module_dir = os.path.dirname(config_path).split('/')
    _module_path = _module_dir[0]
    for m in _module_dir[1:]:
        _module_path = _module_path + '.' + m
    print(_module_path)
    importlib.import_module(_module_path)


def _to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(x, device) for x in obj)
    return obj


def _get_sample_batch(data_loader, sample_idx):
    for i, data in enumerate(data_loader):
        if i == sample_idx:
            return data
    raise IndexError(f'sample_idx={sample_idx} out of range')


def _get_bev_pool_metas_v3(model, data, device):
    """Return img and 3 rank tensors for bev_pool_v3 (NPU training path)."""
    inputs = _to_device(data['img_inputs'][0], device)
    img = inputs[0].squeeze(0).float().contiguous()
    if img.shape[0] > 6:
        img = img[:6]
    metas = model.get_bev_pool_input(inputs)
    if isinstance(metas, torch.Tensor):
        metas = (metas,)
    if len(metas) != 3:
        raise ValueError(
            f'Expected 3 bev-pool tensors from NPU prepare, got {len(metas)}')
    ranks_bev, ranks_depth, ranks_feat = metas
    if ranks_bev is None:
        raise RuntimeError('Empty ranks_bev; try another --sample-idx')
    return (
        img,
        ranks_bev.int().contiguous(),
        ranks_depth.int().contiguous(),
        ranks_feat.int().contiguous(),
    )


def _part1_spatial_shapes(detector, num_cam=6):
    """(N, D, fH, fW, C) from LSS frustum; D=88 for depth [1,45,0.5], not 16."""
    vt = detector.img_view_transformer
    return (
        num_cam,
        int(vt.D),
        int(vt.frustum.shape[1]),
        int(vt.frustum.shape[2]),
        int(vt.out_channels),
    )


def _part1_to_bev_pool_inputs(tran_feat_flat, depth_flat, view_transformer, num_cam=6):
    """Reshape part1 outputs to layouts used by bev_pool_v3."""
    vt = view_transformer
    d = int(vt.D)
    fH = int(vt.frustum.shape[1])
    fW = int(vt.frustum.shape[2])
    feat_c = int(vt.out_channels)
    num_cam = int(tran_feat_flat.numel() // (fH * fW * feat_c))

    tran_feat = tran_feat_flat.reshape(num_cam, fH, fW, feat_c)
    depth = depth_flat.reshape(num_cam, d, fH, fW)
    depth = depth.unsqueeze(0)  # (1, N, D, fH, fW)
    feat = tran_feat.unsqueeze(0)  # (1, N, fH, fW, C)
    return depth.contiguous(), feat.contiguous()


def run_bev_pool_v3(detector, depth, feat, ranks_bev, ranks_depth, ranks_feat):
    """Run mx_driving bev_pool_v3 (training path, no interval tensors)."""
    vt = detector.img_view_transformer
    grid_size = vt.grid_size
    bev_feat_shape = (
        depth.shape[0],
        int(grid_size[2]),
        int(grid_size[1]),
        int(grid_size[0]),
        feat.shape[-1],
    )
    bev_feat = bev_pool_v3(
        depth, feat, ranks_depth, ranks_feat, ranks_bev, bev_feat_shape)
    if vt.collapse_z:
        bev_feat = torch.cat(bev_feat.unbind(dim=2), 1)
    return bev_feat.contiguous()


def _check_onnx(path):
    try:
        import onnx
        from onnx import shape_inference
    except ImportError as exc:
        warnings.warn(f'onnx not installed, skip checker: {exc}')
        return
    model = onnx.load(path)
    try:
        onnx.checker.check_model(model)
        print(f'ONNX check passed: {path}')
    except Exception as exc:
        print(f'ONNX check failed: {path}: {exc}')
        return
    try:
        onnx.save(shape_inference.infer_shapes(model), path)
    except Exception as exc:
        warnings.warn(f'shape inference skipped: {exc}')


def _disable_checkpointing(module):
    """ONNX trace fails when ResNet uses torch.utils.checkpoint."""
    for m in module.modules():
        if hasattr(m, 'with_cp'):
            m.with_cp = False


def _export_part1(part1, img, out_path, opset):
    part1.eval()
    _disable_checkpointing(part1)
    torch.onnx.export(
        part1,
        (img,),
        out_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=['img'],
        output_names=['tran_feat', 'depth'],
    )
    print(f'Exported part1 ONNX: {out_path}')
    _check_onnx(out_path)


def _export_part3(part3, bev_feat, out_path, opset, output_names):
    part3.eval()
    _disable_checkpointing(part3)
    torch.onnx.export(
        part3,
        (bev_feat,),
        out_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=['bev_feat'],
        output_names=output_names,
    )
    print(f'Exported part3 ONNX: {out_path}')
    _check_onnx(out_path)


def _run_split_forward(detector, part1, part3, img, ranks_bev, ranks_depth,
                       ranks_feat):
    """Single forward: part1 -> bev_pool_v3 -> part3."""
    with torch.no_grad():
        tran_feat, depth = part1(img)
        depth_bev, feat_bev = _part1_to_bev_pool_inputs(
            tran_feat, depth, detector.img_view_transformer)
        bev_feat = run_bev_pool_v3(
            detector, depth_bev, feat_bev, ranks_bev, ranks_depth, ranks_feat)
        outs = part3(bev_feat)
        if isinstance(outs, (list, tuple)):
            outs_list = list(outs)
        else:
            outs_list = [outs]
    return tran_feat, depth, depth_bev, feat_bev, bev_feat, outs_list


def _parse_device_ids(device_str):
    ids = [int(x.strip()) for x in device_str.split(',') if x.strip() != '']
    if len(ids) < 2:
        raise ValueError('--export-devices needs at least two ids, e.g. 0,1')
    return ids[0], ids[1]


def _export_part1_on_device(part1, img, out_path, opset, device_id):
    dev = torch.device(f'npu:{device_id}')
    torch.npu.set_device(dev)
    _export_part1(part1.to(dev), img.to(dev), out_path, opset)


def _export_part3_on_device(part3, bev_feat, out_path, opset, output_names,
                            device_id):
    dev = torch.device(f'npu:{device_id}')
    torch.npu.set_device(dev)
    _export_part3(part3.to(dev), bev_feat.to(dev), out_path, opset, output_names)


def _export_onnx_parallel(part1, part3, img, bev_feat, part1_onnx, part3_onnx,
                          opset, output_names, dev0, dev1):
    """Export part1/part3 sequentially on one NPU.

    part1/part3 wrappers share one detector; moving it to two NPUs in parallel
    races parameters. ONNX trace is CPU-heavy anyway.
    """
    warnings.warn(
        'Parallel ONNX export disabled: shared detector cannot be on two NPUs. '
        'Using sequential export on npu:{}.'.format(dev0))
    _export_onnx_sequential(
        part1, part3, img, bev_feat, part1_onnx, part3_onnx,
        opset, output_names, dev0)


def _export_onnx_sequential(part1, part3, img, bev_feat, part1_onnx, part3_onnx,
                            opset, output_names, device_id, export_on_cpu=True):
    if export_on_cpu:
        cpu = torch.device('cpu')
        print('ONNX export on CPU (recommended for ATC/OM accuracy)')
        _export_part1(part1.to(cpu), img.to(cpu), part1_onnx, opset)
        _export_part3(
            part3.to(cpu), bev_feat.to(cpu), part3_onnx, opset, output_names)
        return
    dev = torch.device(f'npu:{device_id}')
    torch.npu.set_device(dev)
    _export_part1_on_device(part1, img, part1_onnx, opset, device_id)
    _export_part3_on_device(
        part3, bev_feat, part3_onnx, opset, output_names, device_id)


def _verify_onnx_part1(part1, img, onnx_path, device):
    """Optional CPU onnxruntime check vs PyTorch part1."""
    try:
        import onnxruntime as ort
    except ImportError:
        warnings.warn('onnxruntime not installed; skip ONNX numeric verify')
        return
    cpu = torch.device('cpu')
    with torch.no_grad():
        tf_ref, d_ref = part1.to(cpu)(img.to(cpu))
    sess = ort.InferenceSession(
        onnx_path, providers=['CPUExecutionProvider'])
    tf_onnx, d_onnx = sess.run(None, {'img': img.detach().cpu().numpy()})
    tf_diff = np.max(np.abs(tf_ref.numpy() - tf_onnx))
    d_diff = np.max(np.abs(d_ref.numpy() - d_onnx))
    print(f'[verify-onnx] part1 tran_feat max_abs={tf_diff:.6f} depth max_abs={d_diff:.6f}')
    if tf_diff > 1e-2 or d_diff > 1e-2:
        warnings.warn(
            'part1 ONNX differs from PyTorch; re-export on CPU or check opset')


def _write_manifest(work_dir, prefix, manifest):
    path = os.path.join(work_dir, f'{prefix}_deploy_manifest.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    print(f'Wrote manifest: {path}')


def _write_atc_script(work_dir, prefix, manifest, parallel_atc=True):
    script_path = os.path.join(work_dir, f'atc_convert_{prefix}.sh')
    p1 = manifest['onnx']['part1']
    p3 = manifest['onnx']['part3']
    bev = manifest['tensor_shapes']['bev_feat_nchw']
    img = manifest['tensor_shapes']['img']
    soc = manifest.get('soc_version', 'Ascend910B3')
    p1_name = os.path.basename(p1)
    p3_name = os.path.basename(p3)
    atc1 = f"""atc --model="${{WORK_DIR}}/{p1_name}" \\
    --framework=5 \\
    --output="${{WORK_DIR}}/{prefix}_part1" \\
    --input_format=NCHW \\
    --input_shape="img:{','.join(map(str, img))}" \\
    --soc_version="${{SOC_VERSION}}" \\
    --precision_mode=allow_fp32_to_fp16"""

    atc3 = f"""atc --model="${{WORK_DIR}}/{p3_name}" \\
    --framework=5 \\
    --output="${{WORK_DIR}}/{prefix}_part3" \\
    --input_format=NCHW \\
    --input_shape="bev_feat:{','.join(map(str, bev))}" \\
    --soc_version="${{SOC_VERSION}}" \\
    --precision_mode=allow_fp32_to_fp16"""

    if parallel_atc:
        atc_block = f"""# Part1 / Part3 OM builds run in parallel
{atc1} &
PID1=$!
{atc3} &
PID2=$!
wait "$PID1"
wait "$PID2"
"""
    else:
        atc_block = f"""# Part1: backbone + depth head
{atc1}

# Part3: BEV encoder + occ head
{atc3}
"""

    content = f"""#!/usr/bin/env bash
# Auto-generated ATC template. Edit soc_version / paths before use.
set -euo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh

SOC_VERSION="{soc}"
WORK_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"

{atc_block}
echo "OM files:"
echo "  ${{WORK_DIR}}/{prefix}_part1.om"
echo "  ${{WORK_DIR}}/{prefix}_part3.om"
echo "Middle bev_pool_v3: run on NPU via mx_driving (see {prefix}_bev_pool_meta_v3.npz)"
"""
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(content)
    os.chmod(script_path, 0o755)
    print(f'Wrote ATC script: {script_path}')


def build_model_and_data(args, device=None):
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg = compat_cfg(cfg)
    setup_multi_processes(cfg)
    _import_plugin(cfg, args.config)

    if not cfg.model.type.endswith('TRT'):
        cfg.model.type = cfg.model.type + 'TRT'

    cfg.model.pretrained = None
    cfg.gpu_ids = [args.gpu_id]
    if device is None:
        device = torch.device(
            'cpu' if getattr(args, 'export_on_cpu', False)
            else f'npu:{args.gpu_id}')

    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True

    test_loader_cfg = {
        **cfg.data.get('test_dataloader', {}),
        'samples_per_gpu': 1,
        'workers_per_gpu': 0,
        'dist': False,
        'shuffle': False,
    }

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    if (not args.no_acceleration
            and hasattr(cfg.model, 'img_view_transformer')):
        cfg.model.img_view_transformer.accelerate = True

    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    model = model.to(device).eval()

    data = _get_sample_batch(data_loader, args.sample_idx)
    return model, data, device


def _get_img_cpu(data):
    inputs = data['img_inputs'][0]
    img = inputs[0].squeeze(0).float().contiguous()
    if img.shape[0] > 6:
        img = img[:6]
    return img


def main():
    args = parse_args()
    os.makedirs(args.work_dir, exist_ok=True)

    prefix = args.prefix
    part1_onnx = os.path.join(args.work_dir, f'{prefix}_part1.onnx')
    part3_onnx = os.path.join(args.work_dir, f'{prefix}_part3.onnx')
    meta_npz = os.path.join(args.work_dir, f'{prefix}_bev_pool_meta_v3.npz')

    if args.export_on_cpu:
        print('CPU ONNX export mode: model/img stay on CPU; part3 uses dummy bev_feat shape')
        model, data, device = build_model_and_data(args, device=torch.device('cpu'))
        img = _get_img_cpu(data)
        bev_shape = (1, 64, 200, 200)
        if hasattr(model, 'img_view_transformer'):
            vt = model.img_view_transformer
            gs = vt.grid_size
            if getattr(vt, 'collapse_z', False):
                bev_shape = (1, int(gs[2]) * int(gs[0]), int(gs[1]), int(gs[1]))
        bev_feat = torch.randn(bev_shape, dtype=torch.float32)
        part1 = Part1ExportWrapper(model).eval()
        part3 = Part3ExportWrapper(model).eval()
        output_names = ['occ_out_0']
        export_dev = int(args.export_devices.split(',')[0].strip())
        _export_onnx_sequential(
            part1, part3, img, bev_feat, part1_onnx, part3_onnx,
            args.opset_version, output_names, export_dev, export_on_cpu=True)
        _verify_onnx_part1(part1, img, part1_onnx, device)
        if os.path.isfile(meta_npz):
            print(f'Keep existing bev_pool meta: {meta_npz}')
        else:
            warnings.warn(
                f'{meta_npz} missing; run once without --export-on-cpu on NPU to generate ranks_*')
        manifest_path = os.path.join(args.work_dir, f'{prefix}_deploy_manifest.json')
        if os.path.isfile(manifest_path):
            with open(manifest_path, encoding='utf-8') as f:
                manifest = json.load(f)
            manifest['onnx']['part1'] = part1_onnx
            manifest['onnx']['part3'] = part3_onnx
            manifest['notes'] = (
                manifest.get('notes', '') + ' part1/part3 ONNX re-exported on CPU.')
        else:
            num_cam, d, fh, fw, c = _part1_spatial_shapes(model)
            manifest = {
                'pipeline': [
                    'part1_onnx: img -> tran_feat, depth',
                    'bev_pool_v3: NPU native (mx_driving), not in ONNX',
                    'part3_onnx: bev_feat -> occ',
                ],
                'onnx': {
                    'part1': part1_onnx,
                    'part3': part3_onnx,
                    'bev_pool_meta': meta_npz,
                },
                'tensor_shapes': {
                    'img': list(img.shape),
                    'part1_layout': {
                        'num_cam': num_cam, 'D': d, 'fH': fh, 'fW': fw, 'C': c,
                    },
                    'bev_feat_nchw': list(bev_feat.shape),
                    'occ_out_0': [1, 200, 200, 16, 18],
                },
                'soc_version': 'Ascend910B3',
            }
        _write_manifest(args.work_dir, prefix, manifest)
        _write_atc_script(args.work_dir, prefix, manifest, args.parallel_atc)
        print('Done. Re-run atc_convert script, then tools/run_split_infer_npu.py')
        return

    model, data, device = build_model_and_data(args)
    img, ranks_bev, ranks_depth, ranks_feat = _get_bev_pool_metas_v3(
        model, data, device)

    part1 = Part1ExportWrapper(model).to(device).eval()
    part3 = Part3ExportWrapper(model).to(device).eval()

    tran_feat, depth, depth_bev, feat_bev, bev_feat, outs_list = _run_split_forward(
        model, part1, part3, img, ranks_bev, ranks_depth, ranks_feat)
    if not args.skip_verify:
        print(f'[verify] split pipeline ok, occ shape={tuple(outs_list[0].shape)}')

    output_names = [f'occ_out_{i}' for i in range(len(outs_list))]

    export_dev = int(args.export_devices.split(',')[0].strip())
    _export_onnx_sequential(
        part1, part3, img, bev_feat, part1_onnx, part3_onnx,
        args.opset_version, output_names, export_dev,
        export_on_cpu=False)

    np.savez(
        meta_npz,
        ranks_bev=ranks_bev.detach().cpu().numpy(),
        ranks_depth=ranks_depth.detach().cpu().numpy(),
        ranks_feat=ranks_feat.detach().cpu().numpy(),
    )
    print(f'Saved bev_pool_v3 meta: {meta_npz}')

    img_shape = list(img.shape)
    bev_shape = list(bev_feat.shape)
    num_cam, d, fh, fw, c = _part1_spatial_shapes(model)
    manifest = {
        'pipeline': [
            'part1_onnx: img -> tran_feat, depth',
            'bev_pool_v3: NPU native (mx_driving), not in ONNX',
            'part3_onnx: bev_feat -> occ',
        ],
        'onnx': {
            'part1': part1_onnx,
            'part3': part3_onnx,
            'bev_pool_meta': meta_npz,
        },
        'tensor_shapes': {
            'img': img_shape,
            'part1_layout': {
                'num_cam': num_cam,
                'D': d,
                'fH': fh,
                'fW': fw,
                'C': c,
            },
            'tran_feat_flat': list(tran_feat.shape),
            'depth_flat': list(depth.shape),
            'depth_bev': list(depth_bev.shape),
            'feat_bev': list(feat_bev.shape),
            'bev_feat_nchw': bev_shape,
            'occ_out_0': list(outs_list[0].shape),
        },
        'soc_version': 'Ascend910B3',
        'notes': (
            'Fixed camera rig: ranks_* can be reused. '
            'For OM INT8, quantize part1/part3 separately with AMCT.'
        ),
    }
    manifest['parallel'] = {
        'onnx_export': 'sequential',
        'export_device': export_dev,
        'atc': args.parallel_atc,
    }
    _write_manifest(args.work_dir, prefix, manifest)
    _write_atc_script(args.work_dir, prefix, manifest, parallel_atc=args.parallel_atc)
    print('Done. Next: run atc_convert_{}.sh after checking shapes.'.format(prefix))


if __name__ == '__main__':
    from mx_driving.patcher import PatcherBuilder, Patch
    from mx_driving.patcher import batch_matmul
    from mx_driving.patcher import resnet_add_relu

    pb = (PatcherBuilder()
          .add_module_patch('torch', Patch(batch_matmul))
          .add_module_patch('mmdet', Patch(resnet_add_relu)))
    with pb.build():
        main()
