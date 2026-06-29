# Copyright (c) OpenMMLab. All rights reserved.
"""Export FlashOCC part1 + part2 (BEVPoolV3) + part3 ONNX and unified e2e ONNX.

Pipeline options:
  Split:  part1.onnx + part2.onnx + part3.onnx
  Unified: flashocc_*_e2e.onnx  (img -> occ, ranks baked as constants)

After export, run the generated ``atc_convert_*_e2e.sh`` to build a single .om.
Requires ``cust_onnx_parsers.so`` with BEVPoolV3 parser (onnx_plugin/).
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn

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

from projects.mmdet3d_plugin.ops.bev_pool_v2.onnx_bev_pool_v3 import (  # noqa: E402
    bev_pool_v3_cpu_reference,
    bev_pool_v3_segment_sum_export,
    onnx_bev_pool_v3,
)
from tools.export_onnx_split_npu import (  # noqa: E402
    Part1EvalAlignedExportWrapper,
    Part3EvalAlignedExportWrapper,
    Part3ExportWrapper,
    _atc_precision_flags,
    _check_onnx,
    _disable_checkpointing,
    _export_part1,
    _export_part3,
    _get_bev_pool_metas_v3,
    _get_img_cpu,
    _import_plugin,
    _part1_spatial_shapes,
    _part1_to_bev_pool_inputs,
    _to_device,
    _verify_onnx_part1,
    build_part1_wrapper,
    build_part3_wrapper,
    _write_manifest,
    build_model_and_data,
)


class Part2ExportWrapper(nn.Module):
    """part1 outputs (tran_feat, depth flat) -> bev_feat NCHW."""

    def __init__(self, detector, ranks_bev, ranks_depth, ranks_feat,
                 use_std_ops=False, use_segment_sum=True):
        super().__init__()
        self.detector = detector
        self.use_std_ops = use_std_ops
        self.use_segment_sum = use_segment_sum and not use_std_ops
        vt = detector.img_view_transformer
        self.collapse_z = bool(getattr(vt, 'collapse_z', False))
        self.register_buffer('ranks_bev', ranks_bev.int().contiguous())
        self.register_buffer('ranks_depth', ranks_depth.int().contiguous())
        self.register_buffer('ranks_feat', ranks_feat.int().contiguous())
        self.bev_b = 1
        self.bev_d = int(vt.grid_size[2])
        self.bev_h = int(vt.grid_size[1])
        self.bev_w = int(vt.grid_size[0])
        self.feat_c = int(vt.out_channels)

    def forward(self, tran_feat, depth):
        depth_bev, feat_bev = _part1_to_bev_pool_inputs(
            tran_feat, depth, self.detector.img_view_transformer)
        bev_shape = (
            self.bev_b, self.bev_d, self.bev_h, self.bev_w, self.feat_c)
        if self.use_segment_sum:
            bev_feat = bev_pool_v3_segment_sum_export(
                depth_bev, feat_bev,
                self.ranks_depth, self.ranks_feat, self.ranks_bev,
                *bev_shape)
        elif self.use_std_ops:
            bev_feat = bev_pool_v3_cpu_reference(
                depth_bev, feat_bev,
                self.ranks_depth, self.ranks_feat, self.ranks_bev,
                *bev_shape)
        else:
            bev_feat = onnx_bev_pool_v3(
                depth_bev, feat_bev,
                self.ranks_depth, self.ranks_feat, self.ranks_bev,
                bev_shape)
        if self.collapse_z:
            bev_feat = torch.cat(bev_feat.unbind(dim=2), 1)
        return bev_feat.contiguous()


class EndToEndExportWrapper(nn.Module):
    """img -> occ (part1 + BEV pool + eval-aligned part3)."""

    def __init__(self, detector, ranks_bev, ranks_depth, ranks_feat,
                 output_occ_only=True, use_std_ops=False,
                 use_segment_sum=True):
        super().__init__()
        self.detector = detector
        self.output_occ_only = output_occ_only
        self.part1 = build_part1_wrapper(detector, layout='eval')
        self.part2 = Part2ExportWrapper(
            detector, ranks_bev, ranks_depth, ranks_feat,
            use_std_ops=use_std_ops, use_segment_sum=use_segment_sum)
        self.part3 = Part3EvalAlignedExportWrapper(detector)

    def forward(self, img):
        tran_feat, depth = self.part1(img)
        bev_feat = self.part2(tran_feat, depth)
        occ = self.part3(bev_feat)
        if self.output_occ_only:
            return occ
        return occ


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export FlashOCC split + unified e2e ONNX with BEVPoolV3')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('work_dir')
    parser.add_argument('--prefix', default='flashocc')
    parser.add_argument('--opset-version', type=int, default=11)
    parser.add_argument('--fuse-conv-bn', action='store_true')
    parser.add_argument('--sample-idx', type=int, default=0)
    parser.add_argument('--no-acceleration', action='store_true')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--soc-version', default='Ascend310B1')
    parser.add_argument(
        '--skip-verify', action='store_true',
        help='skip numeric verify against PyTorch split path')
    parser.add_argument(
        '--skip-split', action='store_true',
        help='only export unified e2e onnx (skip part1/2/3 split files)')
    parser.add_argument(
        '--skip-e2e', action='store_true',
        help='only export split part1/part2/part3 onnx')
    parser.add_argument(
        '--part2-std-ops', action='store_true',
        help='export part2 with index_add ScatterElements (broken on OM)')
    parser.add_argument(
        '--part2-segment-sum', action='store_true', default=True,
        help='export part2 with segment-sum cumsum path (default, OM-safe)')
    parser.add_argument(
        '--no-part2-segment-sum', action='store_false', dest='part2_segment_sum')
    parser.add_argument(
        '--part1-layout', default='eval', choices=('eval', 'legacy'))
    parser.add_argument(
        '--atc-part1-precision', default='force_fp32',
        choices=('force_fp32', 'allow_fp32_to_fp16'))
    parser.add_argument(
        '--atc-merged-precision', default='force_fp32',
        choices=('force_fp32', 'allow_fp32_to_fp16'))
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    return parser.parse_args()


def _export_onnx_model(model, args, out_path, opset, input_names, output_names,
                       skip_torch_checker=False):
    """Export ONNX; optionally skip PyTorch proto checker for custom ops."""
    model.eval()
    _disable_checkpointing(model)
    if not skip_torch_checker:
        torch.onnx.export(
            model, args, out_path,
            export_params=True, opset_version=opset,
            do_constant_folding=True,
            input_names=input_names, output_names=output_names)
    else:
        import torch._C as _torch_c
        orig = _torch_c._check_onnx_proto
        _torch_c._check_onnx_proto = lambda proto: None
        try:
            torch.onnx.export(
                model, args, out_path,
                export_params=True, opset_version=opset,
                do_constant_folding=True,
                input_names=input_names, output_names=output_names)
        finally:
            _torch_c._check_onnx_proto = orig
    print(f'Exported ONNX: {out_path}')
    _check_onnx_lenient(out_path)


def _check_onnx_lenient(path):
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
        warnings.warn(f'ONNX checker skipped (custom op?): {path}: {exc}')
    try:
        onnx.save(shape_inference.infer_shapes(model), path)
    except Exception as exc:
        warnings.warn(f'shape inference skipped: {exc}')


def _export_part2(part2, tran_feat, depth, out_path, opset, traceable=True):
    _export_onnx_model(
        part2, (tran_feat, depth), out_path, opset,
        input_names=['tran_feat', 'depth'],
        output_names=['bev_feat'],
        skip_torch_checker=traceable)


def _export_e2e(e2e, img, out_path, opset, output_names, traceable=True):
    _export_onnx_model(
        e2e, (img,), out_path, opset,
        input_names=['img'],
        output_names=output_names,
        skip_torch_checker=traceable)


def _verify_part2_bev(part2, tran_feat, depth, detector, rtol=2e-3):
    from tools.export_onnx_split_npu import _part1_to_bev_pool_inputs

    vt = detector.img_view_transformer
    bev_shape = (1, int(vt.grid_size[2]), int(vt.grid_size[1]),
                 int(vt.grid_size[0]), int(vt.out_channels))
    with torch.no_grad():
        bev_pt = part2(tran_feat, depth)
        depth_bev, feat_bev = _part1_to_bev_pool_inputs(tran_feat, depth, vt)
        bev_ref = bev_pool_v3_cpu_reference(
            depth_bev, feat_bev, part2.ranks_depth, part2.ranks_feat,
            part2.ranks_bev, *bev_shape)
        if bool(getattr(vt, 'collapse_z', False)):
            bev_ref = torch.cat(bev_ref.unbind(dim=2), 1)
    diff = (bev_pt - bev_ref).abs().max().item()
    print(f'[verify] part2 vs cpu_reference max_abs={diff:.6f}')
    if diff > rtol:
        raise RuntimeError(f'part2 export path differs from cpu_reference: {diff:.6f}')


def _write_atc_merged_script(work_dir, prefix, manifest, om_suffix='segment_sum'):
    script_path = os.path.join(
        work_dir, f'atc_convert_{prefix}_merged_{om_suffix}.sh')
    merged_onnx = manifest['onnx']['merged']
    merged_name = os.path.basename(merged_onnx)
    img = manifest['tensor_shapes']['img']
    soc = manifest.get('soc_version', 'Ascend310B1')
    img_shape = ','.join(map(str, img))
    prec = manifest.get('atc_merged_precision', 'force_fp32')
    flags = _atc_precision_flags(
        prec, 'Softmax,CumSum,ScatterElements,ReduceSum,Gather')
    out_stem = f'{prefix}_merged_{om_suffix}'
    content = f"""#!/usr/bin/env bash
# Auto-generated: merged FlashOCC ONNX (part1+segment-sum part2+part3) -> OM.
set -euo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh

SOC_VERSION="{soc}"
WORK_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
MERGED_ONNX="${{WORK_DIR}}/{merged_name}"
OUT="${{WORK_DIR}}/{out_stem}"

atc --model="${{MERGED_ONNX}}" \\
    --framework=5 \\
    --output="${{OUT}}" \\
    --input_format=NCHW \\
    --input_shape="img:{img_shape}" \\
    --soc_version="${{SOC_VERSION}}" \\
    {flags}

echo "Merged OM: ${{OUT}}.om"
"""
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(content)
    os.chmod(script_path, 0o755)
    print(f'Wrote merged ATC script: {script_path}')


def _write_atc_e2e_script(work_dir, prefix, manifest):
    script_path = os.path.join(work_dir, f'atc_convert_{prefix}_e2e.sh')
    e2e_onnx = manifest['onnx']['e2e']
    e2e_name = os.path.basename(e2e_onnx)
    img = manifest['tensor_shapes']['img']
    soc = manifest.get('soc_version', 'Ascend310B1')
    img_shape = ','.join(map(str, img))
    content = f"""#!/usr/bin/env bash
# Auto-generated: convert unified FlashOCC e2e ONNX -> single OM.
# Requires cust_onnx_parsers.so (BEVPoolV3) under ASCEND_CUSTOM_OPP_PATH.
set -euo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh

SOC_VERSION="{soc}"
WORK_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
E2E_ONNX="${{WORK_DIR}}/{e2e_name}"
OUT="${{WORK_DIR}}/{prefix}_e2e"

if [[ -z "${{ASCEND_CUSTOM_OPP_PATH:-}}" ]]; then
  echo "Set ASCEND_CUSTOM_OPP_PATH to include cust_onnx_parsers.so" >&2
  exit 1
fi

atc --model="${{E2E_ONNX}}" \\
    --framework=5 \\
    --output="${{OUT}}" \\
    --input_format=NCHW \\
    --input_shape="img:{img_shape}" \\
    --soc_version="${{SOC_VERSION}}" \\
    --precision_mode=allow_fp32_to_fp16

echo "Unified OM: ${{OUT}}.om"
echo "Run: python3 tools/run_unified_infer_npu.py ..."
"""
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(content)
    os.chmod(script_path, 0o755)
    print(f'Wrote e2e ATC script: {script_path}')


def _write_atc_split_script(work_dir, prefix, manifest):
    """ATC for part1/part2/part3 separately (optional fallback)."""
    script_path = os.path.join(work_dir, f'atc_convert_{prefix}_split3.sh')
    p1 = os.path.basename(manifest['onnx']['part1'])
    p2 = os.path.basename(manifest['onnx']['part2'])
    p3 = os.path.basename(manifest['onnx']['part3'])
    img = manifest['tensor_shapes']['img']
    layout = manifest['tensor_shapes']['part1_layout']
    bev = manifest['tensor_shapes']['bev_feat_nchw']
    soc = manifest.get('soc_version', 'Ascend310B1')
    nc, d, fh, fw, c = (layout['num_cam'], layout['D'], layout['fH'],
                        layout['fW'], layout['C'])
    tran_n = nc * fh * fw * c
    depth_n = nc * d * fh * fw
    content = f"""#!/usr/bin/env bash
set -euo pipefail
source /usr/local/Ascend/ascend-toolkit/set_env.sh
WORK_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
SOC_VERSION="{soc}"

atc --model="${{WORK_DIR}}/{p1}" --framework=5 \\
    --output="${{WORK_DIR}}/{prefix}_part1" --input_format=NCHW \\
    --input_shape="img:{','.join(map(str, img))}" \\
    --soc_version="${{SOC_VERSION}}" --precision_mode=allow_fp32_to_fp16 &

atc --model="${{WORK_DIR}}/{p2}" --framework=5 \\
    --output="${{WORK_DIR}}/{prefix}_part2" --input_format=ND \\
    --input_shape="tran_feat:{tran_n};depth:{depth_n}" \\
    --soc_version="${{SOC_VERSION}}" --precision_mode=allow_fp32_to_fp16 &

atc --model="${{WORK_DIR}}/{p3}" --framework=5 \\
    --output="${{WORK_DIR}}/{prefix}_part3" --input_format=NCHW \\
    --input_shape="bev_feat:{','.join(map(str, bev))}" \\
    --soc_version="${{SOC_VERSION}}" --precision_mode=allow_fp32_to_fp16 &
wait
echo "Split OM: part1/part2/part3 under ${{WORK_DIR}}"
"""
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(content)
    os.chmod(script_path, 0o755)
    print(f'Wrote split3 ATC script: {script_path}')


def _get_ranks_from_model(model, data, device):
    inputs = _to_device(data['img_inputs'][0], device)
    metas = model.get_bev_pool_input(inputs)
    if isinstance(metas, torch.Tensor):
        metas = (metas,)
    if len(metas) != 3:
        raise ValueError(f'Expected 3 rank tensors, got {len(metas)}')
    ranks_bev, ranks_depth, ranks_feat = metas
    return (
        ranks_bev.int().contiguous(),
        ranks_depth.int().contiguous(),
        ranks_feat.int().contiguous(),
    )


def main():
    args = parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    prefix = args.prefix
    use_segment_sum = args.part2_segment_sum and not args.part2_std_ops

    args.export_on_cpu = True
    model, data, _ = build_model_and_data(args, device=torch.device('cpu'))
    img = _get_img_cpu(data, layout=args.part1_layout)
    ranks_bev, ranks_depth, ranks_feat = _get_ranks_from_model(
        model, data, torch.device('cpu'))

    part1 = build_part1_wrapper(model, args.part1_layout).eval()
    part2 = Part2ExportWrapper(
        model, ranks_bev, ranks_depth, ranks_feat,
        use_std_ops=args.part2_std_ops,
        use_segment_sum=use_segment_sum).eval()
    part3 = build_part3_wrapper(model).eval()
    e2e = EndToEndExportWrapper(
        model, ranks_bev, ranks_depth, ranks_feat,
        use_std_ops=args.part2_std_ops,
        use_segment_sum=use_segment_sum).eval()

    with torch.no_grad():
        tran_feat, depth = part1(img)
        bev_ref = part2(tran_feat, depth)
        occ_ref = e2e(img)
    print(f'[prepare] img={tuple(img.shape)} bev={tuple(bev_ref.shape)} '
          f'occ={tuple(occ_ref.shape)} ranks={ranks_bev.shape[0]} '
          f'part2_segment_sum={use_segment_sum}')

    if not args.skip_verify:
        _verify_part2_bev(part2, tran_feat, depth, model)

    part1_onnx = os.path.join(args.work_dir, f'{prefix}_part1.onnx')
    part2_onnx = os.path.join(args.work_dir, f'{prefix}_part2.onnx')
    part3_onnx = os.path.join(args.work_dir, f'{prefix}_part3.onnx')
    e2e_onnx = os.path.join(args.work_dir, f'{prefix}_e2e.onnx')
    merged_onnx = os.path.join(args.work_dir, f'{prefix}_merged.onnx')
    meta_npz = os.path.join(args.work_dir, f'{prefix}_bev_pool_meta_v3.npz')

    output_names = ['occ_out_0']
    opset = args.opset_version
    traceable_part2 = use_segment_sum or args.part2_std_ops

    if not args.skip_split:
        _export_part1(part1, img, part1_onnx, opset)
        _verify_onnx_part1(part1, img, part1_onnx, torch.device('cpu'))
        _export_part2(part2, tran_feat, depth, part2_onnx, opset, traceable_part2)
        _export_part3(part3, bev_ref, part3_onnx, opset, output_names)

    if not args.skip_e2e:
        _export_e2e(e2e, img, e2e_onnx, opset, output_names, traceable=True)

    if not args.skip_split:
        from tools.merge_onnx_flashocc import merge_flashocc_onnx
        merge_flashocc_onnx(part1_onnx, part2_onnx, part3_onnx, merged_onnx)

    np.savez(
        meta_npz,
        ranks_bev=ranks_bev.numpy(),
        ranks_depth=ranks_depth.numpy(),
        ranks_feat=ranks_feat.numpy(),
    )

    num_cam, d, fh, fw, c = _part1_spatial_shapes(model)
    part2_mode = (
        'segment_sum' if use_segment_sum else
        ('std_ops' if args.part2_std_ops else 'bevpoolv3_custom'))
    manifest = {
        'pipeline': [
            'part1_onnx: img -> tran_feat, depth (eval-aligned, FP32 depth head)',
            f'part2_onnx: tran_feat, depth -> bev_feat ({part2_mode})',
            'part3_onnx: bev_feat -> occ (eval bev_encoder+occ_head)',
            'merged_onnx: part1+part2+part3 composed graph',
        ],
        'onnx': {
            'part1': part1_onnx,
            'part2': part2_onnx,
            'part3': part3_onnx,
            'merged': merged_onnx,
            'e2e': e2e_onnx,
            'bev_pool_meta': meta_npz,
        },
        'tensor_shapes': {
            'img': list(img.shape),
            'part1_layout': {
                'num_cam': num_cam, 'D': d, 'fH': fh, 'fW': fw, 'C': c,
            },
            'tran_feat_flat': list(tran_feat.shape),
            'depth_flat': list(depth.shape),
            'bev_feat_nchw': list(bev_ref.shape),
            'occ_out_0': list(occ_ref.shape),
        },
        'soc_version': args.soc_version,
        'part1_img_layout': args.part1_layout,
        'part2_mode': part2_mode,
        'atc_merged_precision': args.atc_merged_precision,
        'atc_part1_precision': args.atc_part1_precision,
        'notes': (
            f'part2={part2_mode}; part3=eval_aligned; '
            f'merged OM: atc_convert_{prefix}_merged_segment_sum.sh'
        ),
    }
    _write_manifest(args.work_dir, f'{prefix}_unified', manifest)
    if not args.skip_e2e:
        _write_atc_e2e_script(args.work_dir, prefix, manifest)
    if not args.skip_split:
        _write_atc_split_script(args.work_dir, prefix, manifest)
        _write_atc_merged_script(args.work_dir, prefix, manifest)

    print('Done.')
    if not args.skip_split:
        print(f'  merged ONNX: {merged_onnx}')
        print(f'  ATC merged: bash {args.work_dir}/atc_convert_{prefix}_merged_segment_sum.sh')
    if not args.skip_e2e:
        print(f'  e2e ONNX: {e2e_onnx}')


if __name__ == '__main__':
    main()
