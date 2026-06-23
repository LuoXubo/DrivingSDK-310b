#!/usr/bin/env python3
"""Evaluate car_perception_grid FlashOCC checkpoint and visualize pred vs GT."""
from __future__ import annotations

import argparse
import importlib
import json
import os
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
import torch_npu  # noqa: F401
from mmcv import Config
from mmcv.device.npu import NPUDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
GX, GY = 200, 200
EGO_GX, EGO_GY = 100, 100
CAR_CLASS_NAMES = ['passable', 'car', 'unknown']

SEM_COLORS = {
    0: (60, 200, 60),
    1: (40, 40, 220),
    2: (180, 180, 40),
    255: (160, 160, 160),
}


def _import_plugin(cfg: Config, config_path: str) -> None:
    if not cfg.get('plugin', False):
        return
    if hasattr(cfg, 'plugin_dir'):
        parts = os.path.dirname(cfg.plugin_dir).split('/')
    else:
        parts = os.path.dirname(config_path).split('/')
    module_path = parts[0]
    for part in parts[1:]:
        module_path = module_path + '.' + part
    importlib.import_module(module_path)


def pick_test_samples(train_infos, val_infos, max_samples: int = 12):
    """Pick a diverse test subset: all val-car frames + spread passable samples."""
    picks: list[tuple[str, int, str]] = []

    for i, info in enumerate(val_infos):
        sem = np.load(os.path.join(info['occ_path'], 'labels.npz'))['semantics']
        if (sem == 1).any():
            picks.append(('val', i, f'val_car_{info["sample_stem"]}'))

    anchors = [
        ('val', 0, 'val_start'),
        ('val', len(val_infos) // 2, 'val_mid'),
        ('train', 0, 'train_start'),
        ('train', len(train_infos) // 2, 'train_mid'),
        ('train', len(train_infos) - 1, 'train_end'),
    ]
    for item in anchors:
        picks.append(item)

    seen = set()
    out = []
    for item in picks:
        key = (item[0], item[1])
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out[:max_samples]


def build_test_pkl(train_pkl: Path, val_pkl: Path, out_pkl: Path,
                   max_samples: int) -> list[dict]:
    train_data = pickle.load(open(train_pkl, 'rb'))
    val_data = pickle.load(open(val_pkl, 'rb'))
    picks = pick_test_samples(train_data['infos'], val_data['infos'], max_samples)

    infos = []
    manifest = []
    for split, idx, tag in picks:
        info = (val_data if split == 'val' else train_data)['infos'][idx]
        infos.append(info)
        manifest.append({
            'split': split, 'index': idx, 'tag': tag,
            'stem': info.get('sample_stem', info['token'][:8]),
            'token': info['token'],
            'scene': info.get('scene_name', 'unknown'),
        })

    meta = dict(train_data.get('metadata', {}))
    meta['purpose'] = 'car_grid_test_eval'
    meta['num_samples'] = len(infos)
    with open(out_pkl, 'wb') as f:
        pickle.dump({'infos': infos, 'metadata': meta}, f)
    return manifest


def occ_layer_to_bgr(sem: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    crop = sem[:GX, :GY]
    out = np.zeros((GX, GY, 3), dtype=np.uint8)
    for val, color in SEM_COLORS.items():
        out[crop == val] = color
    if mask is not None:
        m = mask[:GX, :GY].astype(bool)
        overlay = out.copy()
        overlay[m] = (overlay[m] * 0.5 + np.array((255, 180, 0)) * 0.5).astype(np.uint8)
        out = overlay
    out = cv2.flip(out, 0)
    return cv2.resize(out, (400, 400), interpolation=cv2.INTER_NEAREST)


def diff_layer_to_bgr(gt: np.ndarray, pred: np.ndarray,
                      mask: np.ndarray | None = None) -> np.ndarray:
    g = gt[:GX, :GY]
    p = pred[:GX, :GY]
    out = np.zeros((GX, GY, 3), dtype=np.uint8)
    for val, color in SEM_COLORS.items():
        out[g == val] = color
    wrong = (g != p) & (g != 255)
    if mask is not None:
        wrong &= mask[:GX, :GY].astype(bool)
    out[wrong] = (0, 255, 255)
    if mask is not None:
        m = mask[:GX, :GY].astype(bool)
        overlay = out.copy()
        overlay[m] = (overlay[m] * 0.5 + np.array((255, 180, 0)) * 0.5).astype(np.uint8)
        out = overlay
    out = cv2.flip(out, 0)
    return cv2.resize(out, (400, 400), interpolation=cv2.INTER_NEAREST)


def put_title(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def hstack_resize(images: list[np.ndarray], target_h: int) -> np.ndarray:
    resized = []
    for img in images:
        h, w = img.shape[:2]
        new_w = max(1, int(w * target_h / h))
        resized.append(cv2.resize(img, (new_w, target_h)))
    return np.hstack(resized)


def compute_car_metrics(preds: list[np.ndarray], gts: list[dict]) -> dict:
    """3-class mIoU on camera-masked voxels, ignoring gt=255."""
    num_classes = 3
    hist = np.zeros((num_classes, num_classes), dtype=np.int64)
    per_sample = []

    for pred, gt_pack in zip(preds, gts):
        gt = gt_pack['semantics']
        mask = gt_pack['mask_camera'].astype(bool)
        valid = mask & (gt != 255)
        if not valid.any():
            per_sample.append({'miou': 0.0, 'valid_voxels': 0})
            continue
        p = pred[valid].astype(np.int64)
        g = gt[valid].astype(np.int64)
        for c in range(num_classes):
            for d in range(num_classes):
                hist[c, d] += int(((g == c) & (p == d)).sum())
        sample_hist = np.zeros((num_classes, num_classes), dtype=np.int64)
        for c in range(num_classes):
            for d in range(num_classes):
                sample_hist[c, d] = int(((g == c) & (p == d)).sum())
        ious = []
        for c in range(num_classes):
            tp = sample_hist[c, c]
            denom = sample_hist[c, :].sum() + sample_hist[:, c].sum() - tp
            ious.append(float(tp / denom) if denom > 0 else float('nan'))
        per_sample.append({
            'miou': float(np.nanmean(ious)),
            'class_iou': {CAR_CLASS_NAMES[i]: ious[i] for i in range(num_classes)},
            'valid_voxels': int(valid.sum()),
            'car_gt_voxels': int((g == 1).sum()),
            'car_pred_voxels': int((p == 1).sum()),
        })

    class_ious = []
    for c in range(num_classes):
        tp = hist[c, c]
        denom = hist[c, :].sum() + hist[:, c].sum() - tp
        class_ious.append(float(tp / denom) if denom > 0 else float('nan'))
    return {
        'num_samples': len(preds),
        'mIoU': float(np.nanmean(class_ious)),
        'class_mIoU': {CAR_CLASS_NAMES[i]: class_ious[i] for i in range(num_classes)},
        'confusion_hist': hist.tolist(),
        'per_sample': per_sample,
    }


def build_compare_panel(info: dict, pred: np.ndarray, root: Path) -> np.ndarray:
    occ = np.load(os.path.join(info['occ_path'], 'labels.npz'))
    gt = occ['semantics']
    mc = occ['mask_camera']
    stem = info.get('sample_stem', info['token'][:8])

    row = [
        put_title(occ_layer_to_bgr(gt[:, :, 0], mc[:, :, 0]), 'GT z0'),
        put_title(occ_layer_to_bgr(pred[:, :, 0], mc[:, :, 0]), 'Pred z0'),
        put_title(diff_layer_to_bgr(gt[:, :, 0], pred[:, :, 0], mc[:, :, 0]), 'Diff z0'),
        put_title(occ_layer_to_bgr(gt[:, :, 1], mc[:, :, 1]), 'GT z1'),
        put_title(occ_layer_to_bgr(pred[:, :, 1], mc[:, :, 1]), 'Pred z1'),
        put_title(diff_layer_to_bgr(gt[:, :, 1], pred[:, :, 1], mc[:, :, 1]), 'Diff z1'),
    ]
    panel = hstack_resize(row, 280)
    title_h = 36
    canvas = np.zeros((panel.shape[0] + title_h, panel.shape[1], 3), np.uint8)
    canvas[:panel.shape[0]] = panel
    cv2.putText(canvas, f'{stem} | GT vs Pred (yellow=wrong)', (10, panel.shape[0] + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def run_inference(cfg: Config, checkpoint: str, gpu_id: int = 0):
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.data.test.test_mode = True
    cfg.gpu_ids = [gpu_id]

    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False)

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, checkpoint, map_location='cpu')
    model = NPUDataParallel(model.cuda(), device_ids=[gpu_id])
    model.eval()

    preds = []
    with torch.no_grad():
        for data in loader:
            result = model(return_loss=False, rescale=True, **data)
            if isinstance(result[0], dict) and 'pred_occ' in result[0]:
                arr = result[0]['pred_occ']
                if torch.is_tensor(arr):
                    arr = arr.cpu().numpy()
            else:
                arr = result[0]
                if torch.is_tensor(arr):
                    arr = arr.cpu().numpy()
            preds.append(arr)
    return dataset, preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        default='projects/configs/flashocc/flashocc-r50-car-grid.py')
    parser.add_argument('--checkpoint',
                        default='work_dirs/car_grid_2p_npu67/epoch_24_ema.pth')
    parser.add_argument('--train-pkl',
                        default='data/car_perception_grid/nuscenes/'
                                'bevdetv2-nuscenes_infos_train.pkl')
    parser.add_argument('--val-pkl',
                        default='data/car_perception_grid/nuscenes/'
                                'bevdetv2-nuscenes_infos_val.pkl')
    parser.add_argument('--test-pkl',
                        default='data/car_perception_grid/nuscenes/'
                                'bevdetv2-nuscenes_infos_test_eval.pkl')
    parser.add_argument('--out-dir',
                        default='work_dirs/car_grid_test_results')
    parser.add_argument('--max-samples', type=int, default=12)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--detailed-vis', action='store_true', default=True,
                        help='generate detailed GT vs Pred panels')
    args = parser.parse_args()

    os.chdir(ROOT)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = out_dir / 'vis_panels'
    vis_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_test_pkl(
        ROOT / args.train_pkl, ROOT / args.val_pkl,
        ROOT / args.test_pkl, args.max_samples)
    with open(out_dir / 'test_manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict({
        'data.test.ann_file': args.test_pkl,
        'data.test.data_root': 'data/car_perception_grid/nuscenes/',
    })
    _import_plugin(cfg, args.config)

    print(f'[1/3] Inference on {len(manifest)} test samples ...')
    dataset, preds = run_inference(cfg, args.checkpoint, args.gpu_id)

    gt_packs = []
    thumbs = []
    sample_metrics = []
    for i, (info, pred) in enumerate(zip(dataset.data_infos, preds)):
        gt_npz = np.load(os.path.join(info['occ_path'], 'labels.npz'))
        gt_pack = {
            'semantics': gt_npz['semantics'],
            'mask_camera': gt_npz['mask_camera'],
            'mask_lidar': gt_npz['mask_lidar'],
        }
        gt_packs.append(gt_pack)

        tag = manifest[i]['tag']
        sample_dir = vis_dir / f'{i:02d}_{tag}'
        sample_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(sample_dir / 'pred.npz', pred=pred, gt=gt_npz)
        panel = build_compare_panel(info, pred, ROOT)
        cv2.imwrite(str(sample_dir / 'compare_panel.png'), panel)
        thumbs.append(put_title(
            cv2.resize(panel, (900, int(900 * panel.shape[0] / panel.shape[1]))),
            manifest[i]['stem']))

        sample_metric = compute_car_metrics([pred], [gt_pack])['per_sample'][0]
        sample_metric.update(manifest[i])
        sample_metrics.append(sample_metric)

    print('[2/3] Computing metrics ...')
    metrics = compute_car_metrics(preds, gt_packs)
    metrics['per_sample_detail'] = sample_metrics
    with open(out_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    if thumbs:
        cols = 2
        rows = (len(thumbs) + cols - 1) // cols
        th, tw = thumbs[0].shape[:2]
        grid = np.zeros((rows * th, cols * tw, 3), np.uint8)
        for i, t in enumerate(thumbs):
            r, c = divmod(i, cols)
            grid[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = t
        cv2.imwrite(str(out_dir / 'summary_grid.png'), grid)

    print('[3/3] Done.')
    print(f'  test pkl : {args.test_pkl} ({len(manifest)} samples)')
    print(f'  metrics  : {out_dir / "metrics.json"}')
    print(f'  panels   : {vis_dir}')
    print(f'  summary  : {out_dir / "summary_grid.png"}')
    print(f'  mIoU (camera mask, 3-class) = {metrics["mIoU"]:.4f}')
    for k, v in metrics['class_mIoU'].items():
        val = 'nan' if v != v else f'{v:.4f}'
        print(f'    {k}: {val}')

    if args.detailed_vis:
        print('[4/4] Building detailed test visualizations ...')
        import subprocess
        import sys
        vis_script = ROOT / 'tools' / 'visualize_car_grid_test_detailed.py'
        cmd = [
            sys.executable, str(vis_script),
            '--eval-dir', str(out_dir),
            '--out-dir', str(out_dir / 'vis_detailed'),
        ]
        subprocess.run(cmd, cwd=ROOT, check=True)
        print(f'  detailed : {out_dir / "vis_detailed"}')


if __name__ == '__main__':
    from mx_driving.patcher import PatcherBuilder, Patch
    from mx_driving.patcher import batch_matmul, resnet_add_relu, resnet_fp16

    pb = (PatcherBuilder()
          .add_module_patch('torch', Patch(batch_matmul))
          .add_module_patch('mmdet', Patch(resnet_add_relu))
          .add_module_patch('mmdet', Patch(resnet_fp16)))
    with pb.build():
        main()
