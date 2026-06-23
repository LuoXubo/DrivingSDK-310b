# Copyright (c) OpenMMLab. All rights reserved.
"""End-to-end split FlashOCC inference on Ascend NPU.

Pipeline:
  img --[part1.om]--> tran_feat, depth
        --[bev_pool_v3 via mx_driving]--> bev_feat
        --[part3.om]--> occ logits

Compares OM+bev_pool_v3 vs PyTorch split path; optional latency benchmark and
per-module profiling (--profile).
"""
import argparse
import importlib
import json
import os
import sys
import time

import numpy as np
import torch
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
    Part1ExportWrapper,
    Part3ExportWrapper,
    _get_bev_pool_metas_v3,
    _get_sample_batch,
    _import_plugin,
    _part1_to_bev_pool_inputs,
    _run_split_forward,
    _to_device,
    run_bev_pool_v3,
)

def _import_acl():
    import acl as _acl
    return _acl


class OmModel:
    """Single .om model via ACL (host -> device -> host)."""

    def __init__(self, om_path, acl_mod):
        self._acl = acl_mod
        self.om_path = om_path
        self.model_id, ret = acl_mod.mdl.load_from_file(om_path)
        if ret != 0:
            raise RuntimeError(f'acl.mdl.load_from_file failed: {om_path} ret={ret}')
        acl = acl_mod
        ACL_MEM_MALLOC_HUGE_FIRST = 0
        ACL_MEMCPY_HOST_TO_DEVICE = 1
        ACL_MEMCPY_DEVICE_TO_HOST = 2
        self.desc = acl.mdl.create_desc()
        acl.mdl.get_desc(self.desc, self.model_id)
        self.input_dataset = acl.mdl.create_dataset()
        self.output_dataset = acl.mdl.create_dataset()
        self.input_data = []
        self.output_data = []
        for i in range(acl.mdl.get_num_inputs(self.desc)):
            sz = acl.mdl.get_input_size_by_index(self.desc, i)
            buf, _ = acl.rt.malloc(sz, ACL_MEM_MALLOC_HUGE_FIRST)
            db = acl.create_data_buffer(buf, sz)
            acl.mdl.add_dataset_buffer(self.input_dataset, db)
            self.input_data.append({'buffer': buf, 'size': sz})
        for i in range(acl.mdl.get_num_outputs(self.desc)):
            sz = acl.mdl.get_output_size_by_index(self.desc, i)
            buf, _ = acl.rt.malloc(sz, ACL_MEM_MALLOC_HUGE_FIRST)
            db = acl.create_data_buffer(buf, sz)
            acl.mdl.add_dataset_buffer(self.output_dataset, db)
            self.output_data.append({'buffer': buf, 'size': sz})
        self._ACL_MEMCPY_HOST_TO_DEVICE = ACL_MEMCPY_HOST_TO_DEVICE
        self._ACL_MEMCPY_DEVICE_TO_HOST = ACL_MEMCPY_DEVICE_TO_HOST

    def infer(self, inp):
        acl = self._acl
        inp = np.ascontiguousarray(inp, dtype=np.float32)
        expect = self.input_data[0]['size']
        if inp.nbytes != expect:
            raise ValueError(
                f'{self.om_path}: input nbytes {inp.nbytes} != expected {expect}')
        ptr = acl.util.bytes_to_ptr(inp.tobytes())
        ret = acl.rt.memcpy(
            self.input_data[0]['buffer'], expect, ptr, expect,
            self._ACL_MEMCPY_HOST_TO_DEVICE)
        if ret != 0:
            raise RuntimeError(f'H2D memcpy failed for {self.om_path}, ret={ret}')
        ret = acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset)
        if ret != 0:
            msg = acl.get_recent_err_msg()
            raise RuntimeError(
                f'acl.mdl.execute failed for {self.om_path}, ret={ret}, msg={msg}')
        outs = []
        for od in self.output_data:
            host_buf, _ = acl.rt.malloc_host(od['size'])
            ret = acl.rt.memcpy(
                host_buf, od['size'], od['buffer'], od['size'],
                self._ACL_MEMCPY_DEVICE_TO_HOST)
            if ret != 0:
                raise RuntimeError(f'D2H memcpy failed for {self.om_path}, ret={ret}')
            raw = acl.util.ptr_to_bytes(host_buf, od['size'])
            outs.append(np.frombuffer(raw, dtype=np.float32).copy())
            acl.rt.free_host(host_buf)
        return outs

    def destroy(self):
        acl = self._acl
        for item in self.input_data:
            acl.rt.free(item['buffer'])
        for item in self.output_data:
            acl.rt.free(item['buffer'])
        n_in = acl.mdl.get_dataset_num_buffers(self.input_dataset)
        for i in range(n_in):
            db = acl.mdl.get_dataset_buffer(self.input_dataset, i)
            if db:
                acl.destroy_data_buffer(db)
        n_out = acl.mdl.get_dataset_num_buffers(self.output_dataset)
        for i in range(n_out):
            db = acl.mdl.get_dataset_buffer(self.output_dataset, i)
            if db:
                acl.destroy_data_buffer(db)
        acl.mdl.destroy_dataset(self.input_dataset)
        acl.mdl.destroy_dataset(self.output_dataset)
        acl.mdl.unload(self.model_id)
        acl.mdl.destroy_desc(self.desc)


class AclSession:
    """ACL session for OM models; reuses torch_npu runtime context (no create_context)."""

    def __init__(self, device_id=0):
        self.acl = _import_acl()
        ret = self.acl.init()
        # 100002: ACL already initialized (e.g. by torch_npu)
        if ret not in (0, 100002):
            raise RuntimeError(f'acl.init failed, ret={ret}')
        self._finalize_acl = (ret == 0)
        self.device_id = device_id
        self.models = {}

    def load(self, name, om_path):
        self.models[name] = OmModel(om_path, self.acl)

    def infer(self, name, inp):
        return self.models[name].infer(inp)

    def close(self):
        for model in self.models.values():
            model.destroy()
        if self._finalize_acl:
            self.acl.finalize()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Split FlashOCC OM inference: part1.om + bev_pool_v3 + part3.om')
    parser.add_argument('config', help='deploy config, e.g. flashocc-r50-trt.py')
    parser.add_argument('checkpoint', help='checkpoint for ranks / pytorch ref')
    parser.add_argument(
        'manifest',
        help='deploy manifest json from export_onnx_split_npu.py')
    parser.add_argument(
        '--work-dir',
        default=None,
        help='override manifest directory for .om paths')
    parser.add_argument('--sample-idx', type=int, default=0)
    parser.add_argument(
        '--samples',
        type=int,
        default=1,
        help='number of dataloader samples to compare')
    parser.add_argument(
        '--benchmark-iters',
        type=int,
        default=0,
        help='if >0, benchmark OM pipeline for this many iterations')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--fuse-conv-bn', action='store_true')
    parser.add_argument('--gpu-id', type=int, default=0, help='logical NPU id')
    parser.add_argument(
        '--no-acceleration',
        action='store_true',
        help='disable img_view_transformer accelerate')
    parser.add_argument(
        '--use-cached-ranks',
        action='store_true',
        default=True,
        help='load ranks_* from bev_pool_meta npz (fixed rig, default on)')
    parser.add_argument(
        '--no-cached-ranks',
        action='store_false',
        dest='use_cached_ranks',
        help='recompute ranks via model.get_bev_pool_input')
    parser.add_argument(
        '--om-only',
        action='store_true',
        help='run OM pipeline only (skip PyTorch reference forward)')
    parser.add_argument(
        '--profile',
        action='store_true',
        help='print per-module latency after warmup (OM split path)')
    parser.add_argument(
        '--profile-detail',
        action='store_true',
        help='also profile PyTorch sub-modules (backbone/neck/occ head, etc.)')
    parser.add_argument(
        '--profile-warmup',
        type=int,
        default=2,
        help='warmup iterations before profile timing (default: 2)')
    parser.add_argument(
        '--profile-iters',
        type=int,
        default=3,
        help='timed iterations for --profile (default: 3)')
    parser.add_argument(
        '--profile-out',
        default=None,
        help='optional JSON path to save profile report')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    return parser.parse_args()


class _StageTimer:
    """Accumulate wall-clock seconds per named stage (NPU sync before/after)."""

    def __init__(self):
        self.times = {}

    def add(self, name, dt):
        self.times[name] = self.times.get(name, 0.0) + dt

    def merge(self, other):
        for name, dt in other.times.items():
            self.add(name, dt)

    def average(self, n_iters):
        n = max(int(n_iters), 1)
        return {k: v / n for k, v in self.times.items()}

    class _Ctx:
        def __init__(self, timer, name):
            self._timer = timer
            self._name = name
            self._t0 = 0.0

        def __enter__(self):
            _sync_npu()
            self._t0 = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc, tb):
            _sync_npu()
            self._timer.add(self._name, time.perf_counter() - self._t0)
            return False

    def measure(self, name):
        return self._Ctx(self, name)


OM_PROFILE_STAGES = [
    'img_to_host',
    'part1_om',
    'part1_to_bevpool',
    'bev_pool_v3',
    'bev_to_host',
    'part3_om',
    'om_total',
]

PYTORCH_PROFILE_STAGES = [
    'img_backbone',
    'img_neck',
    'depth_net',
    'part1_to_bevpool',
    'bev_pool_v3',
    'bev_encoder_backbone',
    'bev_encoder_neck',
    'occ_head',
    'pytorch_total',
]


def _print_profile_block(title, timings, ordered_keys):
    total_key = ordered_keys[-1]
    total_s = timings.get(total_key, sum(timings.values()))
    print(f'\n========== {title} ==========')
    print(f'{"stage":<28} {"seconds":>10} {"percent":>8}')
    print('-' * 48)
    for key in ordered_keys:
        if key == total_key:
            continue
        if key not in timings:
            continue
        sec = timings[key]
        pct = 100.0 * sec / total_s if total_s > 0 else 0.0
        print(f'{key:<28} {sec:>10.3f} {pct:>7.1f}%')
    print('-' * 48)
    print(f'{"profiled total":<28} {total_s:>10.3f}')


def _save_profile_report(path, report):
    abspath = os.path.abspath(path)
    parent = os.path.dirname(abspath)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(abspath, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f'profile report saved: {abspath}')


def _load_img_and_ranks(model, data, device, meta_npz, use_cached_ranks):
    inputs = _to_device(data['img_inputs'][0], device)
    img = inputs[0].squeeze(0).float().contiguous()
    if img.shape[0] > 6:
        img = img[:6]
    if use_cached_ranks and os.path.isfile(meta_npz):
        meta = np.load(meta_npz)
        ranks_bev = torch.from_numpy(meta['ranks_bev']).int().to(device)
        ranks_depth = torch.from_numpy(meta['ranks_depth']).int().to(device)
        ranks_feat = torch.from_numpy(meta['ranks_feat']).int().to(device)
        return img, ranks_bev.contiguous(), ranks_depth.contiguous(), ranks_feat.contiguous()
    return _get_bev_pool_metas_v3(model, data, device)


def _resolve_om_paths(manifest_path, work_dir_override):
    with open(manifest_path, encoding='utf-8') as f:
        manifest = json.load(f)
    base = work_dir_override or os.path.dirname(os.path.abspath(manifest_path))
    prefix = os.path.basename(manifest_path).replace('_deploy_manifest.json', '')
    part1_om = os.path.join(base, f'{prefix}_part1.om')
    part3_om = os.path.join(base, f'{prefix}_part3.om')
    meta_npz = os.path.join(base, f'{prefix}_bev_pool_meta_v3.npz')
    if not os.path.isfile(part1_om):
        part1_om = os.path.join(base, os.path.basename(
            manifest.get('onnx', {}).get('part1', '')).replace('.onnx', '.om'))
    if not os.path.isfile(part3_om):
        part3_om = os.path.join(base, os.path.basename(
            manifest.get('onnx', {}).get('part3', '')).replace('.onnx', '.om'))
    return manifest, part1_om, part3_om, meta_npz


def _sync_npu():
    if torch.npu.is_available():
        torch.npu.synchronize()


def _build_model_and_loader(args):
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
    device = torch.device(f'npu:{args.gpu_id}')

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
    return model, data_loader, device


def _compare_occ(ref, test, name='occ'):
    ref_np = ref.detach().float().cpu().numpy()
    test_np = np.asarray(test, dtype=np.float32)
    if ref_np.shape != test_np.shape:
        print(f'[{name}] shape mismatch ref={ref_np.shape} test={test_np.shape}')
        return
    diff = np.abs(ref_np - test_np)
    print(
        f'[{name}] max_abs={diff.max():.6f} mean_abs={diff.mean():.6f} '
        f'rel_max={diff.max() / (np.abs(ref_np).max() + 1e-6):.6f}')


def run_om_pipeline(acl_sess, part1, part3, model, img, ranks_bev, ranks_depth,
                      ranks_feat, device, timer=None):
    """part1.om -> bev_pool_v3 -> part3.om."""
    if timer is None:
        img_np = img.detach().cpu().numpy()
        tran_flat, depth_flat = acl_sess.infer('part1', img_np)
        tran_feat = torch.from_numpy(tran_flat).to(device)
        depth = torch.from_numpy(depth_flat).to(device)
        depth_bev, feat_bev = _part1_to_bev_pool_inputs(
            tran_feat, depth, model.img_view_transformer)
        bev_feat = run_bev_pool_v3(
            model, depth_bev, feat_bev, ranks_bev, ranks_depth, ranks_feat)
        bev_np = bev_feat.detach().cpu().numpy()
        occ_list = acl_sess.infer('part3', bev_np)
        return occ_list[0], bev_feat

    with timer.measure('om_total'):
        with timer.measure('img_to_host'):
            img_np = img.detach().cpu().numpy()
        with timer.measure('part1_om'):
            tran_flat, depth_flat = acl_sess.infer('part1', img_np)
        tran_feat = torch.from_numpy(tran_flat).to(device)
        depth = torch.from_numpy(depth_flat).to(device)
        with timer.measure('part1_to_bevpool'):
            depth_bev, feat_bev = _part1_to_bev_pool_inputs(
                tran_feat, depth, model.img_view_transformer)
        with timer.measure('bev_pool_v3'):
            bev_feat = run_bev_pool_v3(
                model, depth_bev, feat_bev, ranks_bev, ranks_depth, ranks_feat)
        with timer.measure('bev_to_host'):
            bev_np = bev_feat.detach().cpu().numpy()
        with timer.measure('part3_om'):
            occ_list = acl_sess.infer('part3', bev_np)
    return occ_list[0], bev_feat


def profile_pytorch_stages(model, img, ranks_bev, ranks_depth, ranks_feat,
                           timer):
    """Fine-grained PyTorch eager timing on the same tensors as OM deploy."""
    from projects.mmdet3d_plugin.models.detectors.bevdet_occ import (
        _trt_reshape_part3_input)

    detector = model
    vt = detector.img_view_transformer
    with torch.no_grad():
        with timer.measure('pytorch_total'):
            with timer.measure('img_backbone'):
                x = detector.img_backbone(img)
            with timer.measure('img_neck'):
                x = detector.img_neck(x)
            with timer.measure('depth_net'):
                x = vt.depth_net(x[0])
                depth = x[:, :vt.D].softmax(dim=1)
                tran_feat = x[:, vt.D:(vt.D + vt.out_channels)]
                tran_feat = tran_feat.permute(0, 2, 3, 1).flatten(0, 2)
                depth = depth.reshape(-1)
            with timer.measure('part1_to_bevpool'):
                depth_bev, feat_bev = _part1_to_bev_pool_inputs(
                    tran_feat, depth, vt)
            with timer.measure('bev_pool_v3'):
                bev_feat = run_bev_pool_v3(
                    detector, depth_bev, feat_bev, ranks_bev, ranks_depth,
                    ranks_feat)
            with timer.measure('bev_encoder_backbone'):
                x_bev = _trt_reshape_part3_input(
                    vt, bev_feat.contiguous().reshape(-1))
                x_bev = x_bev.permute(0, 3, 1, 2).contiguous()
                bev_feature = detector.img_bev_encoder_backbone(x_bev)
            with timer.measure('bev_encoder_neck'):
                occ_bev_feature = detector.img_bev_encoder_neck(bev_feature)
            with timer.measure('occ_head'):
                detector.occ_head(occ_bev_feature)


def main():
    args = parse_args()
    manifest, part1_om, part3_om, meta_npz = _resolve_om_paths(
        args.manifest, args.work_dir)
    for p in (part1_om, part3_om):
        if not os.path.isfile(p):
            raise FileNotFoundError(f'Missing OM: {p}')

    print(f'part1 OM: {part1_om}')
    print(f'part3 OM: {part3_om}')
    if os.path.isfile(meta_npz):
        print(f'bev_pool meta (optional): {meta_npz}')

    model, data_loader, device = _build_model_and_loader(args)
    part1 = Part1ExportWrapper(model).eval()
    part3 = Part3ExportWrapper(model).eval()

    # Pre-fetch tensors before ACL context (torch_npu + extra acl context can clash).
    sample_batches = []
    n = max(args.samples, 1)
    for si in range(n):
        idx = si if args.samples > 1 else args.sample_idx
        data = _get_sample_batch(data_loader, idx)
        img, ranks_bev, ranks_depth, ranks_feat = _load_img_and_ranks(
            model, data, device, meta_npz, args.use_cached_ranks)
        sample_batches.append((img, ranks_bev, ranks_depth, ranks_feat))

    acl_sess = AclSession(device_id=args.gpu_id)
    acl_sess.load('part1', part1_om)
    acl_sess.load('part3', part3_om)

    try:
        for si, (img, ranks_bev, ranks_depth, ranks_feat) in enumerate(sample_batches):

            occ_om, bev_feat = run_om_pipeline(
                acl_sess, part1, part3, model, img, ranks_bev, ranks_depth,
                ranks_feat, device)
            print(f'--- sample {si} ---')
            occ_shape = tuple(manifest.get('tensor_shapes', {}).get(
                'occ_out_0', (1, 200, 200, 16, 18)))
            print(f'om occ shape={occ_shape} finite={np.isfinite(occ_om).all()}')
            if not args.om_only:
                _, _, _, _, _, outs_list = _run_split_forward(
                    model, part1, part3, img, ranks_bev, ranks_depth, ranks_feat)
                print(f'pytorch occ shape={tuple(outs_list[0].shape)}')
                _compare_occ(outs_list[0], occ_om.reshape(occ_shape))
            else:
                print(f'om occ min/max={occ_om.min():.4f}/{occ_om.max():.4f}')

        if args.benchmark_iters > 0:
            img, ranks_bev, ranks_depth, ranks_feat = sample_batches[0]
            img_np = img.detach().cpu().numpy()

            for _ in range(args.warmup):
                run_om_pipeline(
                    acl_sess, part1, part3, model, img, ranks_bev, ranks_depth,
                    ranks_feat, device)
            _sync_npu()

            t0 = time.perf_counter()
            for _ in range(args.benchmark_iters):
                run_om_pipeline(
                    acl_sess, part1, part3, model, img, ranks_bev, ranks_depth,
                    ranks_feat, device)
            _sync_npu()
            elapsed = time.perf_counter() - t0
            ms = elapsed / args.benchmark_iters * 1000
            print(
                f'[benchmark] OM split pipeline: {ms:.2f} ms/iter '
                f'({args.benchmark_iters} iters, warmup={args.warmup})')

        if args.profile:
            img, ranks_bev, ranks_depth, ranks_feat = sample_batches[0]
            print(
                f'\n[profile] warmup={args.profile_warmup}, '
                f'iters={args.profile_iters}')
            for _ in range(args.profile_warmup):
                run_om_pipeline(
                    acl_sess, part1, part3, model, img, ranks_bev, ranks_depth,
                    ranks_feat, device)
            _sync_npu()

            om_timer = _StageTimer()
            for _ in range(args.profile_iters):
                run_om_pipeline(
                    acl_sess, part1, part3, model, img, ranks_bev, ranks_depth,
                    ranks_feat, device, timer=om_timer)
            om_avg = om_timer.average(args.profile_iters)
            _print_profile_block(
                f'OM deploy path (avg of {args.profile_iters} iters)',
                om_avg, OM_PROFILE_STAGES)

            report = {
                'profile_warmup': args.profile_warmup,
                'profile_iters': args.profile_iters,
                'om_deploy_seconds': om_avg,
            }

            if args.profile_detail:
                for _ in range(args.profile_warmup):
                    profile_pytorch_stages(
                        model, img, ranks_bev, ranks_depth, ranks_feat,
                        _StageTimer())
                _sync_npu()
                pt_timer = _StageTimer()
                for _ in range(args.profile_iters):
                    profile_pytorch_stages(
                        model, img, ranks_bev, ranks_depth, ranks_feat,
                        pt_timer)
                pt_avg = pt_timer.average(args.profile_iters)
                _print_profile_block(
                    f'PyTorch sub-modules (avg of {args.profile_iters} iters)',
                    pt_avg, PYTORCH_PROFILE_STAGES)
                report['pytorch_eager_seconds'] = pt_avg

            out_path = args.profile_out
            if out_path is None:
                out_path = os.path.join(
                    os.path.dirname(part1_om), 'profile_report.json')
            _save_profile_report(out_path, report)
    finally:
        acl_sess.close()


if __name__ == '__main__':
    from mx_driving.patcher import PatcherBuilder, Patch
    from mx_driving.patcher import batch_matmul, resnet_add_relu, resnet_fp16

    pb = (PatcherBuilder()
          .add_module_patch('torch', Patch(batch_matmul))
          .add_module_patch('mmdet', Patch(resnet_add_relu))
          .add_module_patch('mmdet', Patch(resnet_fp16)))
    with pb.build():
        main()
