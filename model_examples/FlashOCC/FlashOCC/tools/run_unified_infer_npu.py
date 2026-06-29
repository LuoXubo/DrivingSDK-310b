# Copyright (c) OpenMMLab. All rights reserved.
"""Single OM end-to-end FlashOCC inference (img -> occ)."""
import argparse
import json
import os
import sys
import time

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

sys.path.insert(0, os.getcwd())

from tools.export_onnx_split_npu import (  # noqa: E402
    _get_sample_batch,
    _import_plugin,
    _to_device,
)
from tools.run_split_infer_npu import (  # noqa: E402
    AclSession,
    _sync_npu,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Unified FlashOCC e2e OM inference')
    parser.add_argument('config', help='deploy config')
    parser.add_argument('manifest', help='*_unified_deploy_manifest.json')
    parser.add_argument('--om-path', default=None, help='override e2e .om path')
    parser.add_argument('--sample-idx', type=int, default=0)
    parser.add_argument(
        '--samples', type=int, default=1,
        help='number of dataloader frames to run (from sample-idx)')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--profile-iters', type=int, default=3)
    parser.add_argument('--profile-warmup', type=int, default=2)
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    return parser.parse_args()


def _load_manifest(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _resolve_e2e_om(manifest, manifest_path, override):
    if override:
        return override
    base = os.path.dirname(os.path.abspath(manifest_path))
    onnx_map = manifest.get('onnx', {})
    for key in ('merged', 'e2e', 'merged_from_split'):
        p = onnx_map.get(key, '')
        if p:
            cand = p if os.path.isabs(p) else os.path.join(base, os.path.basename(p))
            om = cand.replace('.onnx', '.om')
            if os.path.isfile(om):
                return om
    stem = os.path.basename(manifest_path).replace(
        '_unified_deploy_manifest.json', '').replace('_unified', '')
    for suffix in ('_merged', '_e2e'):
        cand = os.path.join(base, f'{stem}{suffix}.om')
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError('Cannot resolve e2e OM path; pass --om-path')


def _img_from_data(data, layout='eval'):
    from tools.export_onnx_split_npu import _part1_img_from_tensor
    inputs = _to_device(data['img_inputs'][0], torch.device('cpu'))
    return _part1_img_from_tensor(inputs[0], layout=layout).numpy()


def main():
    args = parse_args()
    manifest = _load_manifest(args.manifest)
    om_path = _resolve_e2e_om(manifest, args.manifest, args.om_path)
    if not os.path.isfile(om_path):
        raise FileNotFoundError(f'Missing e2e OM: {om_path}')

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

    occ_shape = tuple(manifest.get('tensor_shapes', {}).get('occ_out_0', (-1,)))
    from tools.export_onnx_split_npu import part1_layout_from_manifest
    part1_layout = part1_layout_from_manifest(manifest)
    print(f'e2e OM: {om_path} (part1 img layout={part1_layout})')

    acl_sess = AclSession(device_id=args.gpu_id)
    acl_sess.load('e2e', om_path)
    try:
        n = max(args.samples, 1)
        first_img = None
        for si in range(n):
            idx = args.sample_idx + si
            data = _get_sample_batch(data_loader, idx)
            img_np = _img_from_data(data, layout=part1_layout)
            if first_img is None:
                first_img = img_np
            occ = acl_sess.infer('e2e', img_np)[0]
            print(f'--- sample {idx} ---')
            print(f'img shape={img_np.shape} '
                  f'occ finite={np.isfinite(occ).all()} '
                  f'min/max={occ.min():.4f}/{occ.max():.4f} '
                  f'expected_occ_shape={occ_shape}')

        if args.profile and first_img is not None:
            for _ in range(args.profile_warmup):
                acl_sess.infer('e2e', first_img)
            _sync_npu()
            t0 = time.perf_counter()
            for _ in range(args.profile_iters):
                acl_sess.infer('e2e', first_img)
            _sync_npu()
            elapsed = time.perf_counter() - t0
            ms = elapsed / args.profile_iters * 1000
            print(f'[profile] e2e OM: {ms:.2f} ms/iter '
                  f'({args.profile_iters} iters, warmup={args.profile_warmup})')
    finally:
        acl_sess.close()


if __name__ == '__main__':
    main()
