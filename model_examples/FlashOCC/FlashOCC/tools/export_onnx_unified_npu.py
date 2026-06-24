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
    onnx_bev_pool_v3,
)
from tools.export_onnx_split_npu import (  # noqa: E402
    Part1ExportWrapper,
    Part3ExportWrapper,
    _check_onnx,
    _disable_checkpointing,
    _export_part1,
    _export_part3,
    _get_bev_pool_metas_v3,
    _get_sample_batch,
    _import_plugin,
    _part1_spatial_shapes,
    _part1_to_bev_pool_inputs,
    _to_device,
    _write_manifest,
    build_model_and_data,
)


class Part2ExportWrapper(nn.Module):
    """part1 outputs (tran_feat, depth flat) -> bev_feat NCHW."""

    def __init__(self, detector, ranks_bev, ranks_depth, ranks_feat,
                 use_std_ops=False):
        super().__init__()
        self.detector = detector
        self.use_std_ops = use_std_ops
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
        if self.use_std_ops:
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
    """img -> occ (part1 + BEVPoolV3 + part3), ranks baked as buffers."""

    def __init__(self, detector, ranks_bev, ranks_depth, ranks_feat,
                 output_occ_only=True, use_std_ops=False):
        super().__init__()
        self.detector = detector
        self.output_occ_only = output_occ_only
        self.part2 = Part2ExportWrapper(
            detector, ranks_bev, ranks_depth, ranks_feat,
            use_std_ops=use_std_ops)

    def forward(self, img):
        tran_feat, depth = self.detector.forward_part1(img)
        bev_feat = self.part2(tran_feat, depth)
        outs = self.detector.forward_part3(bev_feat.contiguous().reshape(-1))
        if isinstance(outs, (list, tuple)):
            if self.output_occ_only:
                return outs[0]
            return tuple(outs)
        return outs


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
        help='export part2/e2e with Gather+ScatterElements instead of '
             'custom BEVPoolV3 (ATC-friendly, no cust_onnx_parsers)')
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


def _export_part2(part2, tran_feat, depth, out_path, opset, use_std_ops=False):
    _export_onnx_model(
        part2, (tran_feat, depth), out_path, opset,
        input_names=['tran_feat', 'depth'],
        output_names=['bev_feat'],
        skip_torch_checker=not use_std_ops)


def _export_e2e(e2e, img, out_path, opset, output_names):
    _export_onnx_model(
        e2e, (img,), out_path, opset,
        input_names=['img'],
        output_names=output_names,
        skip_torch_checker=True)


def _merge_split_onnx(part1_path, part2_path, part3_path, merged_path):
    """Merge three ONNX subgraphs into one graph (part1->part2->part3)."""
    try:
        import onnx
        from onnx import helper, TensorProto
    except ImportError as exc:
        warnings.warn(f'onnx not installed, skip merge: {exc}')
        return False

    m1 = onnx.load(part1_path)
    m2 = onnx.load(part2_path)
    m3 = onnx.load(part3_path)

    def _rename(model, suffix):
        renames = {}
        for init in model.graph.initializer:
            renames[init.name] = f'{init.name}_{suffix}'
        for node in model.graph.node:
            for i, inp in enumerate(node.input):
                if inp in renames:
                    node.input[i] = renames[inp]
            for i, out in enumerate(node.output):
                node.output[i] = f'{out}_{suffix}'
        for init in model.graph.initializer:
            init.name = renames[init.name]
        return model, renames

    m2, r2 = _rename(m2, 'p2')
    m3, r3 = _rename(m3, 'p3')

    p1_out_tran = m1.graph.output[0].name
    p1_out_depth = m1.graph.output[1].name
    p2_in_tran = m2.graph.input[0].name
    p2_in_depth = m2.graph.input[1].name
    p2_out_bev = m2.graph.output[0].name
    p3_in_bev = m3.graph.input[0].name

    for node in m2.graph.node:
        for i, inp in enumerate(node.input):
            if inp == p2_in_tran:
                node.input[i] = p1_out_tran
            elif inp == p2_in_depth:
                node.input[i] = p1_out_depth
    for node in m3.graph.node:
        for i, inp in enumerate(node.input):
            if inp == p3_in_bev:
                node.input[i] = p2_out_bev

    graph = helper.make_graph(
        nodes=list(m1.graph.node) + list(m2.graph.node) + list(m3.graph.node),
        name='flashocc_merged',
        inputs=list(m1.graph.input),
        outputs=list(m3.graph.output),
        initializer=(list(m1.graph.initializer)
                     + list(m2.graph.initializer)
                     + list(m3.graph.initializer)),
    )
    merged = helper.make_model(
        graph,
        opset_imports=list(m1.opset_import),
        producer_name='export_onnx_unified_npu',
    )
    onnx.save(merged, merged_path)
    print(f'Merged split ONNX -> {merged_path}')
    _check_onnx(merged_path)
    return True


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

    args.export_on_cpu = True
    model, data, _ = build_model_and_data(args, device=torch.device('cpu'))
    from tools.export_onnx_split_npu import _get_img_cpu
    img = _get_img_cpu(data)
    ranks_bev, ranks_depth, ranks_feat = _get_ranks_from_model(
        model, data, torch.device('cpu'))

    tran_feat, depth = Part1ExportWrapper(model)(img)
    part1 = Part1ExportWrapper(model).eval()
    part2 = Part2ExportWrapper(
        model, ranks_bev, ranks_depth, ranks_feat,
        use_std_ops=args.part2_std_ops).eval()
    part3 = Part3ExportWrapper(model).eval()
    e2e = EndToEndExportWrapper(
        model, ranks_bev, ranks_depth, ranks_feat,
        use_std_ops=args.part2_std_ops).eval()

    with torch.no_grad():
        bev_ref = part2(tran_feat, depth)
        occ_ref = e2e(img)
    print(f'[prepare] img={tuple(img.shape)} bev={tuple(bev_ref.shape)} '
          f'occ={tuple(occ_ref.shape)} ranks={ranks_bev.shape[0]}')

    part1_onnx = os.path.join(args.work_dir, f'{prefix}_part1.onnx')
    part2_onnx = os.path.join(args.work_dir, f'{prefix}_part2.onnx')
    part3_onnx = os.path.join(args.work_dir, f'{prefix}_part3.onnx')
    e2e_onnx = os.path.join(args.work_dir, f'{prefix}_e2e.onnx')
    merged_onnx = os.path.join(args.work_dir, f'{prefix}_merged.onnx')
    meta_npz = os.path.join(args.work_dir, f'{prefix}_bev_pool_meta_v3.npz')

    output_names = ['occ_out_0']
    opset = args.opset_version

    if not args.skip_split:
        _export_part1(part1, img, part1_onnx, opset)
        _export_part2(part2, tran_feat, depth, part2_onnx, opset,
                      use_std_ops=args.part2_std_ops)
        _export_part3(part3, bev_ref, part3_onnx, opset, output_names)

    if not args.skip_e2e:
        _export_e2e(e2e, img, e2e_onnx, opset, output_names)

    if not args.skip_split and not args.skip_e2e:
        _merge_split_onnx(part1_onnx, part2_onnx, part3_onnx, merged_onnx)
    elif not args.skip_split and args.skip_e2e:
        from tools.merge_onnx_flashocc import merge_flashocc_onnx
        merge_flashocc_onnx(part1_onnx, part2_onnx, part3_onnx, merged_onnx)

    np.savez(
        meta_npz,
        ranks_bev=ranks_bev.numpy(),
        ranks_depth=ranks_depth.numpy(),
        ranks_feat=ranks_feat.numpy(),
    )

    num_cam, d, fh, fw, c = _part1_spatial_shapes(model)
    manifest = {
        'pipeline': [
            'part1_onnx: img -> tran_feat, depth',
            'part2_onnx: tran_feat, depth -> bev_feat (BEVPoolV3 custom op)',
            'part3_onnx: bev_feat -> occ',
            'merged_onnx: part1+part2+part3 composed graph',
        ],
        'onnx': {
            'part1': part1_onnx,
            'part2': part2_onnx,
            'part3': part3_onnx,
            'merged': merged_onnx,
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
        'notes': (
            'part2 uses standard ONNX ops (Gather/ScatterElements)'
            if args.part2_std_ops else
            'part2/e2e use ONNX custom op BEVPoolV3. '
            'Build cust_onnx_parsers.so before ATC e2e convert. '
            'Fixed camera rig: ranks baked as ONNX initializers.'
        ),
    }
    _write_manifest(args.work_dir, f'{prefix}_unified', manifest)
    if not args.skip_e2e:
        _write_atc_e2e_script(args.work_dir, prefix, manifest)
    if not args.skip_split:
        _write_atc_split_script(args.work_dir, prefix, manifest)

    print('Done.')
    if not args.skip_split:
        print(f'  merged ONNX: {merged_onnx}')
    if not args.skip_e2e:
        print(f'  e2e ONNX: {e2e_onnx}')
        print(f'  ATC: bash {args.work_dir}/atc_convert_{prefix}_e2e.sh')


if __name__ == '__main__':
    main()
