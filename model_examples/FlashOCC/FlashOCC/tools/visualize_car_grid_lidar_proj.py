#!/usr/bin/env python3
"""Project training-set lidar point clouds onto camera images for QA."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from car_grid_calibration import (  # noqa: E402
    DEPTH_INVALID,
    load_lidar_from_info,
    lidar_capture_ok,
    project_body_to_image,
    read_lidar_meta,
    sensor2ego_matrix,
)


def depth_colormap(depths: np.ndarray, max_depth: float = 30.0) -> np.ndarray:
    t = np.clip(depths / max(max_depth, 1e-3), 0, 1)
    idx = (t * 255).astype(np.uint8)
    return cv2.applyColorMap(idx.reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)


def draw_points_on_image(
        img: np.ndarray,
        uv: np.ndarray,
        depths: np.ndarray,
        radius: int = 2,
        max_depth: float = 30.0,
) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    if uv.shape[0] == 0:
        return out
    colors = depth_colormap(depths, max_depth)
    for i in np.argsort(-depths):
        u, v = int(round(uv[i, 0])), int(round(uv[i, 1]))
        if 0 <= u < w and 0 <= v < h:
            cv2.circle(out, (u, v), radius, tuple(int(c) for c in colors[i]), -1,
                       lineType=cv2.LINE_AA)
    return out


def load_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def put_title(img: np.ndarray, text: str, bar_h: int = 36) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], bar_h), (0, 0, 0), -1)
    cv2.putText(out, text, (8, bar_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def depth_error_overlay(
        img: np.ndarray,
        uv: np.ndarray,
        depths: np.ndarray,
        depth_map: np.ndarray,
        sensor2ego: np.ndarray,
        radius: int = 2,
) -> tuple[np.ndarray, dict]:
    """Compare projected lidar with depth-unprojected body geometry."""
    out = img.copy()
    h, w = out.shape[:2]
    Kinv = np.linalg.inv(np.array([
        [610.177856, 0, 512],
        [0, 610.177856, 512],
        [0, 0, 1],
    ], dtype=np.float64))
    errs = []
    geom = []
    matched = 0
    for i in range(uv.shape[0]):
        u, v = int(round(uv[i, 0])), int(round(uv[i, 1]))
        if not (0 <= u < w and 0 <= v < h):
            continue
        gt = float(depth_map[v, u])
        if gt <= 0.05 or gt >= DEPTH_INVALID:
            color = (120, 120, 120)
        else:
            err = abs(depths[i] - gt)
            errs.append(err)
            pc = Kinv @ np.array([u * gt, v * gt, gt])
            body_d = (sensor2ego @ np.array([*pc, 1.0]))[:3]
            # caller stores body points; here we only have projected depth.
            if err < 0.5:
                color = (60, 220, 60)
                matched += 1
            elif err < 1.5:
                color = (0, 200, 255)
            else:
                color = (40, 40, 220)
        cv2.circle(out, (u, v), radius, color, -1, lineType=cv2.LINE_AA)
    stats = {
        'n_projected_in_fov': int(uv.shape[0]),
        'n_depth_compared': len(errs),
        'n_depth_match_lt0.5m': matched,
        'depth_err_mean': float(np.mean(errs)) if errs else None,
        'depth_err_p50': float(np.median(errs)) if errs else None,
        'geom_err_p50': float(np.median(geom)) if geom else None,
    }
    return out, stats


def pick_samples(train_infos: list, n: int, valid_only: bool) -> list[tuple[int, str]]:
    scored = []
    for i, info in enumerate(train_infos):
        pose = Path(info.get('src_pose_dir', ''))
        stem = info.get('sample_stem', '')
        meta = read_lidar_meta(pose, stem)
        captured = lidar_capture_ok(meta)
        scored.append((i, captured, info.get('scene_name', ''), stem))

    picks: list[tuple[int, str]] = []
    for row in scored:
        if row[1]:
            picks.append((row[0], 'lidar_ok'))
        if len(picks) >= max(3, n):
            break

    if not valid_only:
        anchors = [0, len(train_infos) // 4, len(train_infos) // 2,
                   len(train_infos) * 3 // 4, len(train_infos) - 1]
        for j, idx in enumerate(anchors):
            picks.append((idx, f'anchor_{j}'))

    seen = set()
    out = []
    for idx, tag in picks:
        if idx in seen:
            continue
        if valid_only:
            meta = read_lidar_meta(
                Path(train_infos[idx].get('src_pose_dir', '')),
                train_infos[idx].get('sample_stem', ''))
            if not lidar_capture_ok(meta):
                continue
        seen.add(idx)
        out.append((idx, tag))
        if len(out) >= n:
            break
    return out


def build_panel(info: dict, root: Path, subsample: int = 3) -> tuple[np.ndarray, dict]:
    stem = info.get('sample_stem', info['token'][:8])
    scene = info.get('scene_name', 'unknown')
    pts, lidar_src, lidar_meta = load_lidar_from_info(info)

    cam_panels = []
    meta = {
        'scene': scene,
        'stem': stem,
        'token': info['token'],
        'tag': info.get('_tag'),
        'index': info.get('_index'),
        'lidar_src': lidar_src,
        'lidar_points': int(pts.shape[0]),
        'lidar_frame': lidar_meta.get('frame'),
        'lidar_captured': lidar_meta.get('captured'),
        'lidar_warning': lidar_meta.get('warning'),
        'cams': {},
    }

    for cam in ['CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']:
        cam_data = info['cams'][cam]
        img_path = root / cam_data['data_path']
        depth_path = root / cam_data['depth_path']
        rgb = load_rgb(img_path)
        K = np.array(cam_data['cam_intrinsic'], dtype=np.float64)
        sensor2ego = sensor2ego_matrix(
            cam_data['sensor2ego_rotation'],
            cam_data['sensor2ego_translation'],
        )

        uv, depths, _ = project_body_to_image(
            pts, K, sensor2ego, subsample=subsample)
        proj = draw_points_on_image(rgb, uv, depths, radius=2)

        depth_map = np.load(depth_path).astype(np.float32)
        err_img, err_stats = depth_error_overlay(
            rgb, uv, depths, depth_map, sensor2ego, radius=2)

        meta['cams'][cam] = {
            'n_in_fov': int(uv.shape[0]),
            'depth_compare': err_stats,
        }

        row = np.hstack([
            put_title(rgb, f'{cam} RGB'),
            put_title(proj, f'{cam} lidar proj ({uv.shape[0]} pts)'),
            put_title(err_img, f'{cam} depth check'),
        ])
        cam_panels.append(row)

    body = np.vstack(cam_panels)
    title_h = 56
    canvas = np.zeros((body.shape[0] + title_h, body.shape[1], 3), np.uint8)
    canvas[title_h:, :] = body

    status = 'OK' if lidar_meta.get('captured') else 'INVALID/STALE'
    title = (f'[{info.get("_index")}] {scene} | {stem} | lidar={status} '
             f'| frame={lidar_meta.get("frame")} | pts={pts.shape[0]}')
    cv2.putText(canvas, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (255, 255, 255), 2, cv2.LINE_AA)
    if lidar_meta.get('warning'):
        cv2.putText(canvas, lidar_meta['warning'], (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 180, 255), 1, cv2.LINE_AA)
    legend = ('body-frame lidar -> camera | green depth err<0.5m | '
              'yellow<1.5m | red>=1.5m | gray=no depth')
    cv2.putText(canvas, legend, (10, title_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (200, 200, 200), 1, cv2.LINE_AA)
    return canvas, meta


def make_summary_grid(panels: list[np.ndarray], thumb_h: int = 280) -> np.ndarray:
    thumbs = []
    for p in panels:
        h, w = p.shape[:2]
        nh = thumb_h
        nw = max(1, int(w * nh / h))
        thumbs.append(cv2.resize(p, (nw, nh)))
    cols = min(3, len(thumbs))
    rows = []
    for i in range(0, len(thumbs), cols):
        chunk = thumbs[i:i + cols]
        max_w = max(t.shape[1] for t in chunk)
        padded = []
        for t in chunk:
            if t.shape[1] < max_w:
                pad = np.zeros((t.shape[0], max_w - t.shape[1], 3), np.uint8)
                t = np.hstack([t, pad])
            padded.append(t)
        while len(padded) < cols:
            padded.append(np.zeros_like(padded[0]))
        rows.append(np.hstack(padded))
    max_w = max(r.shape[1] for r in rows)
    return np.vstack([
        np.hstack([r, np.zeros((r.shape[0], max_w - r.shape[1], 3), np.uint8)])
        if r.shape[1] < max_w else r
        for r in rows
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='train', choices=['train', 'val'])
    parser.add_argument('--max-samples', type=int, default=8)
    parser.add_argument('--indices', default='', help='comma-separated indices')
    parser.add_argument('--subsample', type=int, default=3)
    parser.add_argument('--valid-lidar-only', action='store_true',
                        help='only visualize samples with lidar_captured=true')
    parser.add_argument('--out', default='data/car_perception_grid/vis_train_lidar_proj')
    args = parser.parse_args()

    pkl_path = ROOT / f'data/car_perception_grid/nuscenes/bevdetv2-nuscenes_infos_{args.split}.pkl'
    infos = pickle.load(open(pkl_path, 'rb'))['infos']
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.indices.strip():
        picks = [(int(x), f'idx_{x}') for x in args.indices.split(',') if x.strip()]
    else:
        picks = pick_samples(infos, args.max_samples, args.valid_lidar_only)

    manifest = []
    panels = []
    for idx, tag in picks:
        info = dict(infos[idx])
        info['_index'] = idx
        info['_tag'] = tag
        panel, meta = build_panel(info, ROOT, subsample=args.subsample)
        name = f'{idx:04d}_{args.split}_{tag}_{info.get("sample_stem", "sample")}'
        out_path = out_dir / f'{name}.png'
        cv2.imwrite(str(out_path), panel)
        panels.append(panel)
        meta['output'] = str(out_path.relative_to(ROOT))
        manifest.append(meta)
        fl = meta['cams']['CAM_FRONT_LEFT']['depth_compare']
        print(f'Wrote {out_path} | captured={meta["lidar_captured"]} '
              f'pts={meta["lidar_points"]} depth_p50={fl.get("depth_err_p50")}')

    if panels:
        cv2.imwrite(str(out_dir / 'summary_grid.png'), make_summary_grid(panels))
    with open(out_dir / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f'Wrote {len(panels)} panels to {out_dir}')


if __name__ == '__main__':
    main()
