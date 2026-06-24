# Copyright (c) OpenMMLab. All rights reserved.
"""Merge FlashOCC part1 + part2 + part3 ONNX into a single graph."""
import argparse
import os
import sys

import onnx
from onnx import checker, shape_inference
from onnx.compose import add_prefix, merge_models


def parse_args():
    parser = argparse.ArgumentParser(description='Merge FlashOCC split ONNX files')
    parser.add_argument('part1', help='part1.onnx (img -> tran_feat, depth)')
    parser.add_argument('part2', help='part2.onnx (tran_feat, depth -> bev_feat)')
    parser.add_argument('part3', help='part3.onnx (bev_feat -> occ)')
    parser.add_argument('output', help='merged output .onnx path')
    return parser.parse_args()


def _io_names(model, kind):
    coll = model.graph.input if kind == 'in' else model.graph.output
    return [x.name for x in coll]


def merge_flashocc_onnx(part1_path, part2_path, part3_path, output_path):
    m1 = onnx.load(part1_path)
    m2 = onnx.load(part2_path)
    m3 = onnx.load(part3_path)

    p1_out = _io_names(m1, 'out')
    p2_in = _io_names(m2, 'in')
    p2_out = _io_names(m2, 'out')
    p3_in = _io_names(m3, 'in')

    if len(p1_out) != 2 or len(p2_in) != 2 or len(p2_out) != 1 or len(p3_in) != 1:
        raise ValueError(
            f'unexpected IO: p1_out={p1_out} p2_in={p2_in} p2_out={p2_out} p3_in={p3_in}')

    m2p = add_prefix(m2, prefix='p2/')
    m3p = add_prefix(m3, prefix='p3/')
    io12 = [
        (p1_out[0], f'p2/{p2_in[0]}'),
        (p1_out[1], f'p2/{p2_in[1]}'),
    ]
    io23 = [(f'p2/{p2_out[0]}', f'p3/{p3_in[0]}')]

    # merge_models runs onnx checker; BEVPoolV3 is a custom op.
    _orig_check = checker.check_model
    checker.check_model = lambda model, *a, **k: None
    try:
        m12 = merge_models(m1, m2p, io_map=io12)
        merged = merge_models(m12, m3p, io_map=io23)
    finally:
        checker.check_model = _orig_check

    try:
        merged = shape_inference.infer_shapes(merged)
    except Exception as exc:
        print(f'warn: shape inference skipped: {exc}')

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    onnx.save(merged, output_path)
    try:
        checker.check_model(merged)
        print(f'ONNX check passed: {output_path}')
    except Exception as exc:
        print(f'ONNX check skipped (custom op): {exc}')
    print(f'Merged ONNX saved: {output_path}')
    print(f'  inputs : {_io_names(merged, "in")}')
    print(f'  outputs: {_io_names(merged, "out")}')
    return output_path


def main():
    args = parse_args()
    merge_flashocc_onnx(args.part1, args.part2, args.part3, args.output)


if __name__ == '__main__':
    main()
