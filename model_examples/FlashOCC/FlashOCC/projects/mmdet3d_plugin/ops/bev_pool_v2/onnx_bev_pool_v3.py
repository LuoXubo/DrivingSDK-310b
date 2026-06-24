# Copyright (c) OpenMMLab. All rights reserved.
"""ONNX-exportable BEVPoolV3 wrapper for Ascend deploy.

Exports as custom ONNX op ``BEVPoolV3`` (mapped to CANN BEVPoolV3 via
onnx_plugin/onnx_bev_pool_v3.cpp).  Forward uses a CPU reference so
``torch.onnx.export`` can trace on CPU without NPU.
"""
import torch


def bev_pool_v3_cpu_reference(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                              b, d, h, w, c):
    """Vectorized CPU reference matching mx_driving.bev_pool_v3 (with depth)."""
    depth_flat = depth.reshape(-1)
    feat_flat = feat.reshape(-1, c)
    rd = ranks_depth.long()
    rf = ranks_feat.long()
    rb = ranks_bev.long()
    weighted = (feat_flat.index_select(0, rf)
                * depth_flat.index_select(0, rd).unsqueeze(-1))
    out_flat = torch.zeros(b * d * h * w, c, dtype=feat.dtype, device=feat.device)
    out_flat.index_add_(0, rb, weighted)
    out = out_flat.view(b, d, h, w, c)
    return out.permute(0, 4, 1, 2, 3).contiguous()


class OnnxBEVPoolV3(torch.autograd.Function):
    """Custom ONNX op for BEVPoolV3; attrs match CANN op definition."""

    @staticmethod
    def symbolic(g, depth, feat, ranks_depth, ranks_feat, ranks_bev, b, d, h,
                 w, c):
        return g.op(
            'BEVPoolV3',
            depth,
            feat,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            b_i=int(b),
            d_i=int(d),
            h_i=int(h),
            w_i=int(w),
            c_i=int(c),
            with_depth_i=1,
        )

    @staticmethod
    def forward(ctx, depth, feat, ranks_depth, ranks_feat, ranks_bev, b, d, h,
                w, c):
        return bev_pool_v3_cpu_reference(
            depth, feat, ranks_depth, ranks_feat, ranks_bev, b, d, h, w, c)


def onnx_bev_pool_v3(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                     bev_feat_shape):
    """Same signature as mx_driving.bev_pool_v3 for drop-in export."""
    b, d, h, w, c = bev_feat_shape
    return OnnxBEVPoolV3.apply(
        depth.contiguous().float(),
        feat.contiguous().float(),
        ranks_depth.contiguous().int(),
        ranks_feat.contiguous().int(),
        ranks_bev.contiguous().int(),
        int(b), int(d), int(h), int(w), int(c),
    )


def _register_bev_pool_v3_onnx_symbolic():
    """Register BEVPoolV3 for torch.onnx export checker (opset 11)."""
    try:
        from torch.onnx import register_custom_op_symbolic
    except ImportError:
        return

    def _symbolic(g, depth, feat, ranks_depth, ranks_feat, ranks_bev, b, d, h,
                  w, c):
        return g.op(
            'BEVPoolV3',
            depth,
            feat,
            ranks_depth,
            ranks_feat,
            ranks_bev,
            b_i=int(b),
            d_i=int(d),
            h_i=int(h),
            w_i=int(w),
            c_i=int(c),
            with_depth_i=1,
        )

    register_custom_op_symbolic('::BEVPoolV3', _symbolic, 11)


_register_bev_pool_v3_onnx_symbolic()
