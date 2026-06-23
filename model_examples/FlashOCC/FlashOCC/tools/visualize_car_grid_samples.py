#!/usr/bin/env python3
"""Export visual panels for car_perception_grid FlashOCC training samples."""
from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DX, DY = 67, 67
DEPTH_INVALID = 60000.0

SEM_COLORS = {
    0: (60, 200, 60),      # passable - green (BGR)
    1: (40, 40, 220),      # car - red
    255: (160, 160, 160),  # ignore - gray
}


def load_rgb(path: Path, max_side: int = 512) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def colorize_depth(path: Path, max_side: int = 512) -> np.ndarray:
    depth = np.load(path).astype(np.float32)
    valid = (depth > 0.05) & (depth < DEPTH_INVALID)
    vis = np.zeros(depth.shape, dtype=np.uint8)
    if valid.any():
        d = depth[valid]
        lo, hi = np.percentile(d, [2, 98])
        norm = np.clip((depth - lo) / max(hi - lo, 1e-3), 0, 1)
        vis[valid] = (norm[valid] * 255).astype(np.uint8)
    color = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    h, w = color.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        color = cv2.resize(color, (int(w * scale), int(h * scale)))
    return color


def load_seg(path: Path | None, max_side: int = 512) -> np.ndarray | None:
    if path is None or not path.exists():
        return None
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    if raw.ndim == 3:
        cls = raw[:, :, 2]
    else:
        cls = raw
    color = np.zeros((*cls.shape, 3), dtype=np.uint8)
    color[cls == 0] = (60, 200, 60)
    color[cls == 1] = (40, 40, 220)
    color[cls == 2] = (180, 180, 40)
    h, w = color.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        color = cv2.resize(color, (int(w * scale), int(h * scale)))
    return color


def occ_layer_to_bgr(sem: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    crop = sem[:DX, :DY]
    out = np.zeros((DX, DY, 3), dtype=np.uint8)
    for val, color in SEM_COLORS.items():
        out[crop == val] = color
    if mask is not None:
        m = mask[:DX, :DY].astype(bool)
        overlay = out.copy()
        overlay[m] = (overlay[m] * 0.5 + np.array((255, 180, 0)) * 0.5).astype(np.uint8)
        out = overlay
    out = cv2.flip(out, 0)  # near ego at bottom for intuitive forward-up view
    out = cv2.resize(out, (400, 400), interpolation=cv2.INTER_NEAREST)
    return out


def mask_to_bgr(mask: np.ndarray) -> np.ndarray:
    crop = (mask[:DX, :DY] > 0).astype(np.uint8) * 255
    crop = cv2.flip(crop, 0)
    color = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    return cv2.resize(color, (400, 400), interpolation=cv2.INTER_NEAREST)


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


def build_panel(info: dict, root: Path) -> tuple[np.ndarray, dict]:
    stem = info.get('sample_stem', info['token'][:8])
    scene = info.get('scene_name', 'unknown')
    occ = np.load(os.path.join(info['occ_path'], 'labels.npz'))
    sem = occ['semantics']
    mc = occ['mask_camera']
    ml = occ['mask_lidar']

    cams = ['CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']
    cam_imgs, depth_imgs, seg_imgs = [], [], []
    for cam in cams:
        cp = root / info['cams'][cam]['data_path']
        dp = root / info['cams'][cam]['depth_path']
        cam_imgs.append(put_title(load_rgb(cp), cam))
        depth_imgs.append(put_title(colorize_depth(dp), f'{cam} depth'))

        seg_path = None
        pose_dir = info.get('src_pose_dir')
        if pose_dir:
            cam_id = 1 if cam == 'CAM_FRONT_LEFT' else 2
            seg_path = Path(pose_dir) / f'{stem}_seg_cam{cam_id}_raw.png'
        seg = load_seg(seg_path)
        if seg is not None:
            seg_imgs.append(put_title(seg, f'{cam} seg'))
        else:
            seg_imgs.append(put_title(np.zeros((256, 256, 3), np.uint8),
                                      f'{cam} seg (missing)'))

    occ0 = put_title(occ_layer_to_bgr(sem[:, :, 0], mc[:, :, 0]),
                     'OCC z0 sem+cam')
    occ1 = put_title(occ_layer_to_bgr(sem[:, :, 1], mc[:, :, 1]),
                     'OCC z1 sem+cam')
    ml0 = put_title(mask_to_bgr(ml[:, :, 0]), 'lidar mask z0')
    ml1 = put_title(mask_to_bgr(ml[:, :, 1]), 'lidar mask z1')

    row_h = 280
    row1 = hstack_resize(cam_imgs + depth_imgs, row_h)
    row2 = hstack_resize(seg_imgs + [occ0, occ1, ml0, ml1], row_h)
    panel_w = max(row1.shape[1], row2.shape[1])
    canvas = np.zeros((row1.shape[0] + row2.shape[0] + 40, panel_w, 3), np.uint8)
    canvas[:row1.shape[0], :row1.shape[1]] = row1
    canvas[row1.shape[0]:row1.shape[0] + row2.shape[0], :row2.shape[1]] = row2
    title = f'{scene} | {stem} | sem unique={np.unique(sem).tolist()}'
    cv2.rectangle(canvas, (0, row1.shape[0] + row2.shape[0]),
                  (panel_w, canvas.shape[0]), (30, 30, 30), -1)
    cv2.putText(canvas, title, (10, row1.shape[0] + row2.shape[0] + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    meta = {
        'scene': scene,
        'stem': stem,
        'token': info['token'],
        'occ_path': info['occ_path'],
        'sem_unique': np.unique(sem).astype(int).tolist(),
        'mask_camera_ratio': float(mc.mean()),
        'mask_lidar_ratio': float(ml.mean()),
        'car_voxels': int((sem == 1).sum()),
        'passable_voxels': int((sem == 0).sum()),
        'ignore_voxels': int((sem == 255).sum()),
    }
    return canvas, meta


def pick_default_indices(train_infos, val_infos) -> list[tuple[str, int, str]]:
    """Return (split, index, tag) for a diverse 8-sample set."""
    picks: list[tuple[str, int, str]] = []
    n_train = len(train_infos)
    for idx, tag in [(0, 'train_start'), (n_train // 4, 'train_q1'),
                     (n_train // 2, 'train_mid'), (n_train - 1, 'train_end')]:
        picks.append(('train', idx, tag))

    for i, info in enumerate(val_infos):
        z = np.load(os.path.join(info['occ_path'], 'labels.npz'))
        if (z['semantics'] == 1).any():
            picks.append(('val', i, f'val_car_{info["sample_stem"]}'))

    for i, info in enumerate(val_infos):
        if info['scene_name'] == 'scene-near_20260610_224039':
            picks.append(('val', i, 'val_near_scene'))
            break

    # dedupe while preserving order
    seen = set()
    out = []
    for item in picks:
        key = (item[0], item[1])
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out[:10]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl-train',
                        default='data/car_perception_grid/nuscenes/'
                                'bevdetv2-nuscenes_infos_train.pkl')
    parser.add_argument('--pkl-val',
                        default='data/car_perception_grid/nuscenes/'
                                'bevdetv2-nuscenes_infos_val.pkl')
    parser.add_argument('--out-dir',
                        default='data/car_perception_grid/vis_train_samples')
    parser.add_argument('--indices', default='',
                        help='Comma list of train indices, e.g. 0,100,500')
    parser.add_argument('--max-samples', type=int, default=8)
    args = parser.parse_args()

    root = ROOT
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_infos = pickle.load(open(root / args.pkl_train, 'rb'))['infos']
    val_infos = pickle.load(open(root / args.pkl_val, 'rb'))['infos']

    if args.indices:
        picks = [('train', int(x), f'train_{int(x)}')
                 for x in args.indices.split(',') if x.strip()]
    else:
        picks = pick_default_indices(train_infos, val_infos)
    picks = picks[:args.max_samples]

    manifest = []
    thumbs = []
    for split, idx, tag in picks:
        info = train_infos[idx] if split == 'train' else val_infos[idx]
        panel, meta = build_panel(info, root)
        meta.update({'split': split, 'index': idx, 'tag': tag})
        sample_dir = out_dir / f'{split}_{idx:04d}_{tag}'
        sample_dir.mkdir(parents=True, exist_ok=True)
        panel_path = sample_dir / 'panel.png'
        cv2.imwrite(str(panel_path), panel)
        thumb = cv2.resize(panel, (800, int(800 * panel.shape[0] / panel.shape[1])))
        thumbs.append(put_title(thumb, f'{split}[{idx}] {meta["stem"]}'))
        manifest.append({**meta, 'panel': str(panel_path.relative_to(root))})

    if thumbs:
        cols = 2
        rows = (len(thumbs) + cols - 1) // cols
        thumb_h, thumb_w = thumbs[0].shape[:2]
        grid = np.zeros((rows * thumb_h, cols * thumb_w, 3), np.uint8)
        for i, t in enumerate(thumbs):
            r, c = divmod(i, cols)
            grid[r * thumb_h:(r + 1) * thumb_h,
                 c * thumb_w:(c + 1) * thumb_w] = t
        cv2.imwrite(str(out_dir / 'summary_grid.png'), grid)

    with open(out_dir / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    legend = np.zeros((120, 500, 3), np.uint8)
    items = [('0 passable', SEM_COLORS[0]), ('1 car', SEM_COLORS[1]),
             ('255 ignore', SEM_COLORS[255]), ('cam mask tint', (255, 180, 0))]
    for i, (name, color) in enumerate(items):
        y = 20 + i * 28
        cv2.rectangle(legend, (10, y - 14), (34, y + 6), color, -1)
        cv2.putText(legend, name, (44, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / 'legend.png'), legend)

    print(f'Saved {len(manifest)} samples to {out_dir}')
    print(f'  summary: {out_dir / "summary_grid.png"}')
    print(f'  manifest: {out_dir / "manifest.json"}')
    for m in manifest:
        print(f'  - {m["tag"]}: car_voxels={m["car_voxels"]}, '
              f'passable={m["passable_voxels"]}, ignore={m["ignore_voxels"]}, '
              f'cam_mask={m["mask_camera_ratio"]:.3f}, '
              f'lidar_mask={m["mask_lidar_ratio"]:.3f}')


if __name__ == '__main__':
    main()
