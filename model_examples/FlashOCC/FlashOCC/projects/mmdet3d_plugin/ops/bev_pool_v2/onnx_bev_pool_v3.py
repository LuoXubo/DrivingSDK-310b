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


def bev_pool_v3_segment_sum_export(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                                   b, d, h, w, c):
    """ONNX-traceable BEV pool matching index_add_ via sort + segment cumsum."""
    depth_flat = depth.reshape(-1)
    feat_flat = feat.reshape(-1, c)
    rd = ranks_depth.long()
    rf = ranks_feat.long()
    rb = ranks_bev.long()
    weighted = (feat_flat.index_select(0, rf)
                * depth_flat.index_select(0, rd).unsqueeze(-1))

    n = int(rb.numel())
    bev_slots = b * d * h * w
    out_flat = weighted.new_zeros(bev_slots, c)
    if n == 0:
        out = out_flat.view(b, d, h, w, c)
        return out.permute(0, 4, 1, 2, 3).contiguous()

    sort_idx = torch.argsort(rb)
    rb_sorted = rb[sort_idx]
    w_sorted = weighted[sort_idx]

    cumsum = torch.cumsum(w_sorted, dim=0)
    seg_end = torch.cat([
        rb_sorted[1:] != rb_sorted[:-1],
        w_sorted.new_ones(1, dtype=torch.bool),
    ])
    seg_start = torch.cat([
        w_sorted.new_ones(1, dtype=torch.bool),
        rb_sorted[1:] != rb_sorted[:-1],
    ])
    end_idx = seg_end.nonzero(as_tuple=False).squeeze(-1)
    start_idx = seg_start.nonzero(as_tuple=False).squeeze(-1)

    end_vals = cumsum[end_idx]
    start_vals = torch.zeros_like(end_vals)
    if start_idx.numel() > 1:
        start_vals[1:] = cumsum[start_idx[1:] - 1]

    seg_sums = end_vals - start_vals
    unique_rb = rb_sorted[end_idx]
    out_flat = out_flat.scatter_add(
        0, unique_rb.unsqueeze(1).expand(-1, c), seg_sums)
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
