#!/usr/bin/env python3
"""Detailed training-sample visualization for car_perception_grid annotation QA."""
from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GX, GY = 200, 200
EGO_GX, EGO_GY = 100, 100
DEPTH_INVALID = 60000.0

SEM_COLORS = {
    0: (60, 200, 60),
    1: (40, 40, 220),
    255: (160, 160, 160),
}
SEM_NAMES = {0: 'passable', 1: 'car', 255: 'ignore'}


def put_title(img: np.ndarray, text: str, bar_h: int = 32) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], bar_h), (0, 0, 0), -1)
    cv2.putText(out, text, (8, bar_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def load_rgb(path: Path, max_side: int = 480) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)))
    return img


def colorize_depth(path: Path, max_side: int = 480) -> np.ndarray:
    depth = np.load(path).astype(np.float32)
    valid = (depth > 0.05) & (depth < DEPTH_INVALID)
    vis = np.zeros(depth.shape, np.uint8)
    if valid.any():
        lo, hi = np.percentile(depth[valid], [2, 98])
        norm = np.clip((depth - lo) / max(hi - lo, 1e-3), 0, 1)
        vis[valid] = (norm[valid] * 255).astype(np.uint8)
    color = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    h, w = color.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        color = cv2.resize(color, (int(w * s), int(h * s)))
    return color


def load_seg_color(path: Path | None, max_side: int = 480) -> tuple[np.ndarray | None, dict]:
    if path is None or not path.exists():
        return None, {}
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None, {}
    cls = raw[:, :, 2] if raw.ndim == 3 else raw
    color = np.zeros((*cls.shape, 3), np.uint8)
    color[cls == 0] = (60, 200, 60)
    color[cls == 1] = (40, 40, 220)
    color[cls == 2] = (180, 180, 40)
    stats = {
        'passable_px': int((cls == 0).sum()),
        'car_px': int((cls == 1).sum()),
        'unknown_px': int((cls == 2).sum()),
    }
    h, w = color.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        color = cv2.resize(color, (int(w * s), int(h * s)))
    return color, stats


def overlay_seg_on_rgb(rgb: np.ndarray, seg_path: Path | None, alpha: float = 0.45):
    if seg_path is None or not seg_path.exists():
        return rgb
    raw = cv2.imread(str(seg_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return rgb
    cls = raw[:, :, 2] if raw.ndim == 3 else raw
    if cls.shape[:2] != rgb.shape[:2]:
        cls = cv2.resize(cls, (rgb.shape[1], rgb.shape[0]),
                         interpolation=cv2.INTER_NEAREST)
    color = np.zeros_like(rgb)
    color[cls == 0] = (60, 200, 60)
    color[cls == 1] = (40, 40, 220)
    color[cls == 2] = (180, 180, 40)
    mask = cls != 2
    out = rgb.copy()
    out[mask] = (rgb[mask] * (1 - alpha) + color[mask] * alpha).astype(np.uint8)
    return out


def bev_sem(sem2d: np.ndarray, mask2d: np.ndarray | None = None,
            show_ego: bool = True, size: int = 520) -> np.ndarray:
    out = np.full((GX, GY, 3), 40, np.uint8)
    for val, color in SEM_COLORS.items():
        out[sem2d == val] = color
    if mask2d is not None:
        m = mask2d.astype(bool)
        blend = out.copy()
        blend[m] = (blend[m] * 0.45 + np.array((255, 180, 0)) * 0.55).astype(np.uint8)
        out = blend
    out = cv2.flip(out, 0)
    out = cv2.resize(out, (size, size), interpolation=cv2.INTER_NEAREST)
    if show_ego:
        ex = int(EGO_GX / GX * size)
        ey = int((GY - 1 - EGO_GY) / GY * size)
        cv2.drawMarker(out, (ex, ey), (255, 255, 0), cv2.MARKER_CROSS, 18, 2)
        cv2.putText(out, 'ego', (ex + 8, ey - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 0), 1, cv2.LINE_AA)
    return out


def bev_mask(mask2d: np.ndarray, size: int = 520) -> np.ndarray:
    m = (mask2d > 0).astype(np.uint8) * 255
    m = cv2.flip(m, 0)
    color = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
    return cv2.resize(color, (size, size), interpolation=cv2.INTER_NEAREST)


def bev_counts_panel(sem: np.ndarray, mc: np.ndarray, ml: np.ndarray, layer: int,
                     size: int = 520) -> np.ndarray:
    s = sem[:, :, layer]
    valid = mc[:, :, layer].astype(bool)
    panel = np.zeros((size, size, 3), np.uint8)
    lines = [
        f'layer z{layer}',
        f'unique: {np.unique(s).tolist()}',
        f'passable: {(s == 0).sum()}',
        f'car: {(s == 1).sum()}',
        f'ignore: {(s == 255).sum()}',
        f'cam_mask: {valid.sum()} ({valid.mean()*100:.2f}%)',
        f'lidar_mask: {ml[:, :, layer].sum()}',
        f'car_in_cam: {((s == 1) & valid).sum()}',
        f'pass_in_cam: {((s == 0) & valid).sum()}',
    ]
    for i, line in enumerate(lines):
        cv2.putText(panel, line, (12, 28 + i * 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (230, 230, 230), 1, cv2.LINE_AA)
    return panel


def hstack_row(images: list[np.ndarray], h: int) -> np.ndarray:
    parts = []
    for im in images:
        nh = h
        nw = max(1, int(im.shape[1] * h / im.shape[0]))
        parts.append(cv2.resize(im, (nw, nh)))
    return np.hstack(parts)


def pick_detailed_samples(train_infos: list, n: int = 10) -> list[tuple[int, str]]:
    """Pick diverse train indices for QA."""
    scored = []
    for i, info in enumerate(train_infos):
        z = np.load(os.path.join(info['occ_path'], 'labels.npz'))
        sem, mc, ml = z['semantics'], z['mask_camera'], z['mask_lidar']
        car = int((sem == 1).sum())
        car_cam = int(((sem == 1) & mc.astype(bool)).sum())
        scored.append((i, car, car_cam, float(mc.mean()), float(ml.mean()),
                       info.get('scene_name', ''), info.get('sample_stem', '')))

    picks: list[tuple[int, str]] = []

    # car-rich in camera mask
    car_cam_rank = sorted(scored, key=lambda x: x[2], reverse=True)
    for row in car_cam_rank:
        if row[2] > 0 and (row[0], 'car_in_cam') not in [(p[0], p[1]) for p in picks]:
            picks.append((row[0], 'car_in_cam'))
        if sum(1 for p in picks if p[1] == 'car_in_cam') >= 4:
            break

    # high car voxel count
    car_rank = sorted(scored, key=lambda x: x[1], reverse=True)
    for row in car_rank[:2]:
        tag = f'car_vox_{row[1]}'
        if (row[0], tag) not in [(p[0], p[1]) for p in picks]:
            picks.append((row[0], tag))

    # passable only (no car)
    for row in scored:
        if row[1] == 0 and row[3] > 0:
            picks.append((row[0], 'passable_only'))
            break

    # high lidar mask
    lidar_rank = sorted(scored, key=lambda x: x[4], reverse=True)
    for row in lidar_rank[:2]:
        if row[4] > 0.01:
            picks.append((row[0], 'high_lidar'))

    # near scene
    for row in scored:
        if 'near_' in row[5]:
            picks.append((row[0], 'near_scene'))
            break

    # spread anchors
    anchors = [0, len(train_infos) // 4, len(train_infos) // 2,
               len(train_infos) * 3 // 4, len(train_infos) - 1]
    for j, idx in enumerate(anchors):
        picks.append((idx, f'anchor_{j}'))

    seen = set()
    out = []
    for idx, tag in picks:
        if idx not in seen:
            seen.add(idx)
            out.append((idx, tag))
        if len(out) >= n:
            break
    return out


def build_detailed_panel(info: dict, root: Path) -> tuple[np.ndarray, dict, dict]:
    stem = info.get('sample_stem', info['token'][:8])
    scene = info.get('scene_name', 'unknown')
    occ = np.load(os.path.join(root, info['occ_path'], 'labels.npz'))
    sem, mc, ml = occ['semantics'], occ['mask_camera'], occ['mask_lidar']

    files = {}
    row_h = 300
    rows = []

    cam_row = []
    depth_row = []
    seg_row = []
    overlay_row = []
    seg_stats = {}

    for cam in ['CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']:
        cam_id = 1 if cam == 'CAM_FRONT_LEFT' else 2
        cp = root / info['cams'][cam]['data_path']
        dp = root / info['cams'][cam]['depth_path']
        rgb = load_rgb(cp)
        cam_row.append(put_title(rgb, cam))
        depth_row.append(put_title(colorize_depth(dp), f'{cam} depth'))

        seg_path = None
        if info.get('src_pose_dir'):
            seg_path = Path(info['src_pose_dir']) / f'{stem}_seg_cam{cam_id}_raw.png'
        seg_img, st = load_seg_color(seg_path)
        seg_stats[cam] = st
        if seg_img is not None:
            seg_row.append(put_title(seg_img, f'{cam} seg'))
            overlay_row.append(put_title(overlay_seg_on_rgb(rgb, seg_path),
                                         f'{cam} rgb+seg'))
            files[f'seg_{cam}'] = str(seg_path) if seg_path else ''
        else:
            blank = np.zeros((rgb.shape[0], rgb.shape[1], 3), np.uint8)
            seg_row.append(put_title(blank, f'{cam} seg missing'))
            overlay_row.append(put_title(rgb, f'{cam} rgb only'))

    rows.append(hstack_row(cam_row, row_h))
    rows.append(hstack_row(depth_row, row_h))
    rows.append(hstack_row(seg_row, row_h))
    rows.append(hstack_row(overlay_row, row_h))

    bev_z0 = [
        put_title(bev_sem(sem[:, :, 0]), 'BEV z0 sem'),
        put_title(bev_sem(sem[:, :, 0], mc[:, :, 0]), 'BEV z0 sem+cam'),
        put_title(bev_mask(mc[:, :, 0]), 'BEV z0 cam mask'),
        put_title(bev_mask(ml[:, :, 0]), 'BEV z0 lidar mask'),
        put_title(bev_counts_panel(sem, mc, ml, 0), 'z0 stats'),
    ]
    bev_z1 = [
        put_title(bev_sem(sem[:, :, 1]), 'BEV z1 sem'),
        put_title(bev_sem(sem[:, :, 1], mc[:, :, 1]), 'BEV z1 sem+cam'),
        put_title(bev_mask(mc[:, :, 1]), 'BEV z1 cam mask'),
        put_title(bev_mask(ml[:, :, 1]), 'BEV z1 lidar mask'),
        put_title(bev_counts_panel(sem, mc, ml, 1), 'z1 stats'),
    ]
    rows.append(hstack_row(bev_z0, row_h))
    rows.append(hstack_row(bev_z1, row_h))

    # mask bbox on full grid
    active = np.argwhere(mc.any(axis=2))
    bbox = None
    if len(active):
        bbox = [int(active[:, 0].min()), int(active[:, 1].min()),
                int(active[:, 0].max()), int(active[:, 1].max())]

    panel_w = max(r.shape[1] for r in rows)
    total_h = sum(r.shape[0] for r in rows) + 50
    canvas = np.zeros((total_h, panel_w, 3), np.uint8)
    y = 0
    for r in rows:
        canvas[y:y + r.shape[0], :r.shape[1]] = r
        y += r.shape[0]

    title = (f'{scene} | {stem} | car_vox={(sem==1).sum()} '
             f'pass_vox={(sem==0).sum()} ignore={(sem==255).sum()} '
             f'cam_bbox={bbox}')
    cv2.rectangle(canvas, (0, y), (panel_w, total_h), (25, 25, 25), -1)
    cv2.putText(canvas, title, (10, y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 2, cv2.LINE_AA)

    meta = {
        'scene': scene,
        'stem': stem,
        'token': info['token'],
        'index': info.get('_index'),
        'tag': info.get('_tag'),
        'sem_unique': np.unique(sem).astype(int).tolist(),
        'car_voxels': int((sem == 1).sum()),
        'passable_voxels': int((sem == 0).sum()),
        'ignore_voxels': int((sem == 255).sum()),
        'car_in_cam': int(((sem == 1) & mc.astype(bool)).sum()),
        'mask_camera_ratio': float(mc.mean()),
        'mask_lidar_ratio': float(ml.mean()),
        'cam_mask_bbox': bbox,
        'seg_stats': seg_stats,
        'occ_path': info['occ_path'],
    }
    return canvas, meta, files


def save_individual_assets(info: dict, root: Path, out_dir: Path):
    stem = info.get('sample_stem', 'sample')
    occ = np.load(os.path.join(root, info['occ_path'], 'labels.npz'))
    for name, arr in occ.items():
        pass
    for zl in [0, 1]:
        cv2.imwrite(str(out_dir / f'bev_z{zl}_sem.png'),
                    bev_sem(occ['semantics'][:, :, zl]))
        cv2.imwrite(str(out_dir / f'bev_z{zl}_sem_cam.png'),
                    bev_sem(occ['semantics'][:, :, zl], occ['mask_camera'][:, :, zl]))
        cv2.imwrite(str(out_dir / f'bev_z{zl}_cam_mask.png'),
                    bev_mask(occ['mask_camera'][:, :, zl]))
        cv2.imwrite(str(out_dir / f'bev_z{zl}_lidar_mask.png'),
                    bev_mask(occ['mask_lidar'][:, :, zl]))
    for cam in ['CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']:
        cp = root / info['cams'][cam]['data_path']
        cv2.imwrite(str(out_dir / f'{cam}.png'), load_rgb(cp, max_side=1024))


def write_report(meta: dict, path: Path):
    lines = [
        f"scene: {meta['scene']}",
        f"stem: {meta['stem']}",
        f"token: {meta['token']}",
        f"tag: {meta.get('tag')}",
        f"sem_unique: {meta['sem_unique']}",
        f"car_voxels: {meta['car_voxels']}",
        f"passable_voxels: {meta['passable_voxels']}",
        f"ignore_voxels: {meta['ignore_voxels']}",
        f"car_in_cam_mask: {meta['car_in_cam']}",
        f"mask_camera_ratio: {meta['mask_camera_ratio']:.4f}",
        f"mask_lidar_ratio: {meta['mask_lidar_ratio']:.4f}",
        f"cam_mask_bbox [gx0,gy0,gx1,gy1]: {meta['cam_mask_bbox']}",
        "seg_stats:",
    ]
    for cam, st in meta.get('seg_stats', {}).items():
        lines.append(f"  {cam}: {st}")
    lines += [
        "",
        "Legend:",
        "  green=passable(0), red=car(1), gray=ignore(255), yellow=ego",
        "  orange tint = camera-visible voxels in BEV",
        "  BEV coords: FlashOCC grid 200x200, ego at (100,100), x forward up",
    ]
    path.write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl-train',
                        default='data/car_perception_grid/nuscenes/'
                                'bevdetv2-nuscenes_infos_train.pkl')
    parser.add_argument('--out-dir',
                        default='data/car_perception_grid/vis_train_detailed')
    parser.add_argument('--indices', default='',
                        help='Comma-separated train indices')
    parser.add_argument('--max-samples', type=int, default=10)
    parser.add_argument('--save-assets', action='store_true', default=True)
    args = parser.parse_args()

    root = ROOT
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_infos = pickle.load(open(root / args.pkl_train, 'rb'))['infos']

    if args.indices.strip():
        picks = [(int(x), f'train_{int(x)}') for x in args.indices.split(',') if x.strip()]
    else:
        picks = pick_detailed_samples(train_infos, args.max_samples)

    manifest = []
    thumbs = []
    for idx, tag in picks:
        info = dict(train_infos[idx])
        info['_index'] = idx
        info['_tag'] = tag
        panel, meta, _ = build_detailed_panel(info, root)
        sample_dir = out_dir / f'{idx:04d}_{tag}_{meta["stem"]}'
        sample_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(sample_dir / 'detailed_panel.png'), panel)
        write_report(meta, sample_dir / 'report.txt')
        if args.save_assets:
            save_individual_assets(info, root, sample_dir)
        thumb = cv2.resize(panel, (900, int(900 * panel.shape[0] / panel.shape[1])))
        thumbs.append(put_title(thumb, f'[{idx}] {meta["stem"]} car={meta["car_voxels"]}'))
        meta['panel'] = str((sample_dir / 'detailed_panel.png').relative_to(root))
        meta['report'] = str((sample_dir / 'report.txt').relative_to(root))
        manifest.append(meta)

    if thumbs:
        cols = 2
        rows_n = (len(thumbs) + cols - 1) // cols
        th, tw = thumbs[0].shape[:2]
        grid = np.zeros((rows_n * th, cols * tw, 3), np.uint8)
        for i, t in enumerate(thumbs):
            r, c = divmod(i, cols)
            grid[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = t
        cv2.imwrite(str(out_dir / 'summary_grid.png'), grid)

    legend = np.zeros((200, 640, 3), np.uint8)
    items = [
        ('0 passable', SEM_COLORS[0]),
        ('1 car obstacle', SEM_COLORS[1]),
        ('255 ignore', SEM_COLORS[255]),
        ('cam mask tint', (255, 180, 0)),
        ('seg unknown', (180, 180, 40)),
        ('ego marker', (255, 255, 0)),
    ]
    for i, (name, color) in enumerate(items):
        y = 24 + i * 30
        cv2.rectangle(legend, (12, y - 14), (36, y + 6), color, -1)
        cv2.putText(legend, name, (48, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / 'legend.png'), legend)

    with open(out_dir / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f'Saved {len(manifest)} detailed samples -> {out_dir}')
    print(f'  summary: {out_dir / "summary_grid.png"}')
    for m in manifest:
        print(f'  [{m.get("tag")}] {m["stem"]}: car={m["car_voxels"]}, '
              f'car_in_cam={m["car_in_cam"]}, cam_mask={m["mask_camera_ratio"]:.3f}')


if __name__ == '__main__':
    main()
