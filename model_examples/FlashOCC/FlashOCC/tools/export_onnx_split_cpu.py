# Copyright (c) OpenMMLab. All rights reserved.
"""Export FlashOCC part1/part3 ONNX on CPU only (no torch_npu).

Use this when NPU-traced ONNX diverges from PyTorch. After export, run ATC and
tools/run_split_infer_npu.py on NPU.
"""
import argparse
import os
import sys
import warnings

import numpy as np
import torch
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
    Part1ExportWrapper,
    Part3ExportWrapper,
    _check_onnx,
    _disable_checkpointing,
    _export_part1,
    _export_part3,
    _get_sample_batch,
    _import_plugin,
    _part1_spatial_shapes,
    _write_atc_script,
    _write_manifest,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Export FlashOCC ONNX on CPU')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('work_dir')
    parser.add_argument('--prefix', default='flashocc')
    parser.add_argument('--opset-version', type=int, default=11)
    parser.add_argument('--fuse-conv-bn', action='store_true')
    parser.add_argument('--sample-idx', type=int, default=0)
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    return parser.parse_args()


def _get_img_cpu(data):
    inputs = data['img_inputs'][0]
    img = inputs[0].squeeze(0).float().contiguous()
    if img.shape[0] > 6:
        img = img[:6]
    return img


def _verify_onnx_part1(part1, img, onnx_path):
    try:
        import onnxruntime as ort
    except ImportError:
        warnings.warn('onnxruntime not installed; skip numeric verify')
        return
    with torch.no_grad():
        tf_ref, d_ref = part1(img)
    tf_onnx, d_onnx = ort.InferenceSession(
        onnx_path, providers=['CPUExecutionProvider']).run(
            None, {'img': img.numpy()})
    print(
        f'[verify-onnx] part1 tran_feat max_abs={np.max(np.abs(tf_ref.numpy()-tf_onnx)):.6f} '
        f'depth max_abs={np.max(np.abs(d_ref.numpy()-d_onnx)):.6f}')


def main():
    args = parse_args()
    os.makedirs(args.work_dir, exist_ok=True)

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg = compat_cfg(cfg)
    setup_multi_processes(cfg)
    _import_plugin(cfg, args.config)

    if not cfg.model.type.endswith('TRT'):
        cfg.model.type = cfg.model.type + 'TRT'
    cfg.model.pretrained = None

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
    )

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    model = model.cpu().eval()

    data = _get_sample_batch(data_loader, args.sample_idx)
    img = _get_img_cpu(data)
    # part3 expects flattened BEV then reshapes internally; NCHW (1, 64, 200, 200)
    bev_feat = torch.randn(1, 64, 200, 200, dtype=torch.float32)

    part1 = Part1ExportWrapper(model).eval()
    part3 = Part3ExportWrapper(model).eval()
    _disable_checkpointing(part1)
    _disable_checkpointing(part3)

    prefix = args.prefix
    part1_onnx = os.path.join(args.work_dir, f'{prefix}_part1.onnx')
    part3_onnx = os.path.join(args.work_dir, f'{prefix}_part3.onnx')
    meta_npz = os.path.join(args.work_dir, f'{prefix}_bev_pool_meta_v3.npz')

    _export_part1(part1, img, part1_onnx, args.opset_version)
    _export_part3(part3, bev_feat, part3_onnx, args.opset_version, ['occ_out_0'])
    _verify_onnx_part1(part1, img, part1_onnx)

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
            'bev_feat_nchw': [1, 64, 200, 200],
            'occ_out_0': [1, 200, 200, 16, 18],
        },
        'soc_version': 'Ascend910B3',
        'notes': 'ONNX exported on CPU via export_onnx_split_cpu.py',
    }
    _write_manifest(args.work_dir, prefix, manifest)
    _write_atc_script(args.work_dir, prefix, manifest, parallel_atc=True)
    print('Done. Run: bash work_dirs/onnx_split/atc_convert_{}.sh'.format(prefix))


if __name__ == '__main__':
    main()
