#!/usr/bin/env python3
"""Convert /data/car_perception_grid to FlashOCC training layout.

Maps GT into FlashOCC ego BEV grid: x/y in [-40, 40] m, step 0.4 m -> 200x200.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from car_grid_calibration import load_lidar_body  # noqa: E402

# Raw OCC grid (67x67 @ 0.15m) used in *_occ.npz and sensor rasterization
X_NEAR, X_FAR = 0.1, 9.85
Y_MIN, Y_MAX = -5.05, 5.0
Z_MIN, Z_MAX = -2.0, 1.0
RES_M = 0.15
DX_RAW, DY_RAW = 67, 67
DZ_OCC = 16
DZ_OUT = 2
Z_LAYER_SPLIT = 8
Z_SPLIT = -0.5

# FlashOCC / BEVDet grid (must match flashocc-r50-perf.py grid_config)
GRID_X = (-40.0, 40.0, 0.4)
GRID_Y = (-40.0, 40.0, 0.4)
GX, GY, GZ = 200, 200, DZ_OUT

DEPTH_INVALID = 60000.0
SEM_PASSABLE, SEM_CAR, SEM_IGNORE = 0, 1, 255
# v3 default: denser projection (step=1 ~6% mask vs step=2 ~4%).
DEFAULT_RASTER_STEP = 1
RASTER_STEP = DEFAULT_RASTER_STEP

CAM_INTRINSIC = np.array([
    [610.1778564453125, 0.0, 512.0],
    [0.0, 610.1778564453125, 512.0],
    [0.0, 0.0, 1.0],
], dtype=np.float64)

CAM_EXTRINSICS = {
    'CAM_FRONT_LEFT': {
        'sensor2ego_rotation': [0.5792279653389899, 0.40557978767223324,
                                0.4055797876722333, 0.5792279653389898],
        'sensor2ego_translation': [0.0, -0.155, 0.0],
    },
    'CAM_FRONT_RIGHT': {
        'sensor2ego_rotation': [0.5792279653389899, 0.40557978767223324,
                                0.4055797876722333, 0.5792279653389898],
        'sensor2ego_translation': [0.0, 0.155, 0.0],
    },
}

CAM_MAP = {1: 'CAM_FRONT_LEFT', 2: 'CAM_FRONT_RIGHT'}


def quat_to_rot(w, x, y, z):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def sensor2ego_matrix(rot_q, trans):
    w, x, y, z = rot_q
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = quat_to_rot(w, x, y, z)
    mat[:3, 3] = np.asarray(trans, dtype=np.float64)
    return mat


def read_ply_xyz(path: Path) -> np.ndarray:
    with open(path, 'rb') as f:
        header = b''
        while True:
            line = f.readline()
            header += line
            if b'end_header' in line:
                break
        n_verts = 0
        for line in header.decode('utf-8', 'ignore').splitlines():
            if line.startswith('element vertex'):
                n_verts = int(line.split()[-1])
        if n_verts == 0:
            return np.zeros((0, 3), dtype=np.float32)
        raw = f.read(n_verts * 12)
    pts = np.frombuffer(raw, dtype='<f4').reshape(-1, 3)
    return pts.astype(np.float32)


def write_lidar_bin(path: Path, pts: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = pts.shape[0]
    buf = np.zeros((n, 5), dtype=np.float32)
    buf[:, :3] = pts
    buf.tofile(path)


def token_from_stem(stem: str) -> str:
    return hashlib.md5(stem.encode()).hexdigest()


def ego_to_grid(x: float, y: float, z: float):
    """Map ego xyz (m) to FlashOCC voxel (gx, gy, layer)."""
    gx = int(np.floor((x - GRID_X[0]) / GRID_X[2]))
    gy = int(np.floor((y - GRID_Y[0]) / GRID_Y[2]))
    if not (0 <= gx < GX and 0 <= gy < GY):
        return None
    layer = 0 if z < Z_SPLIT else 1
    return gx, gy, layer


def raw_index_to_ego(ix: int, iy: int) -> tuple[float, float]:
    x = X_FAR - ix * RES_M
    y = Y_MIN + iy * RES_M
    return x, y


def empty_grid():
    sem = np.full((GX, GY, GZ), SEM_IGNORE, dtype=np.uint8)
    mc = np.zeros((GX, GY, GZ), dtype=np.uint8)
    ml = np.zeros((GX, GY, GZ), dtype=np.uint8)
    return sem, mc, ml


def write_voxel(sem, mc, ml, x, y, z, label, from_camera=False, from_lidar=False):
    idx = ego_to_grid(x, y, z)
    if idx is None:
        return
    gx, gy, layer = idx
    if from_camera:
        mc[gx, gy, layer] = 1
        if label == SEM_CAR:
            sem[gx, gy, layer] = SEM_CAR
        elif label == SEM_PASSABLE and sem[gx, gy, layer] != SEM_CAR:
            sem[gx, gy, layer] = SEM_PASSABLE
    if from_lidar:
        ml[gx, gy, layer] = 1


def collapse_occ_16_to_2(sem16, mc16, ml16):
    sem2 = np.full((DX_RAW, DY_RAW, DZ_OUT), SEM_IGNORE, dtype=np.uint8)
    mc2 = np.zeros((DX_RAW, DY_RAW, DZ_OUT), dtype=np.uint8)
    ml2 = np.zeros((DX_RAW, DY_RAW, DZ_OUT), dtype=np.uint8)
    splits = [(0, Z_LAYER_SPLIT), (Z_LAYER_SPLIT, DZ_OCC)]
    for layer, (z0, z1) in enumerate(splits):
        s = sem16[:, :, z0:z1]
        m = mc16[:, :, z0:z1]
        ml = ml16[:, :, z0:z1]
        vis = m.astype(bool)
        car = (s == 1) & vis
        free = (s == 0) & vis
        ml_vis = ml.astype(bool).any(axis=2)
        mc2[vis.any(axis=2)] = 1
        sem2[car.any(axis=2)] = SEM_CAR
        sem2[free.any(axis=2) & ~car.any(axis=2)] = SEM_PASSABLE
        ml2[ml_vis] = 1
    return sem2, mc2, ml2


def remap_semantics(arr):
    out = arr.astype(np.uint8, copy=True)
    out[out == 2] = SEM_IGNORE
    return out


def map_raw_grid_to_flashocc(sem2, mc2, ml2):
    """Place 67x67 raw grid cells into FlashOCC 200x200 ego grid."""
    sem, mc, ml = empty_grid()
    for ix in range(DX_RAW):
        for iy in range(DY_RAW):
            x, y = raw_index_to_ego(ix, iy)
            for layer in range(DZ_OUT):
                gx = int(np.floor((x - GRID_X[0]) / GRID_X[2]))
                gy = int(np.floor((y - GRID_Y[0]) / GRID_Y[2]))
                if 0 <= gx < GX and 0 <= gy < GY:
                    sem[gx, gy, layer] = sem2[ix, iy, layer]
                    mc[gx, gy, layer] = mc2[ix, iy, layer]
                    ml[gx, gy, layer] = ml2[ix, iy, layer]
    return sem, mc, ml


def build_gt_from_occ_npz(occ_path: Path):
    z = np.load(occ_path)
    sem = remap_semantics(z['semantics'])
    mc = z['mask_camera'].astype(np.uint8)
    ml = z['mask_lidar'].astype(np.uint8)
    sem2, mc2, ml2 = collapse_occ_16_to_2(sem, mc, ml)
    return map_raw_grid_to_flashocc(sem2, mc2, ml2)


def decode_seg_class(seg_path: Path) -> np.ndarray:
    import cv2
    img = cv2.imread(str(seg_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(seg_path)
    if img.ndim == 3:
        # Class id is stored in R channel of BGR (verified on dataset).
        return img[:, :, 2].astype(np.uint8)
    return img.astype(np.uint8)


def _propagate_mask_columns(mask: np.ndarray) -> np.ndarray:
    """If any z layer is visible at (gx, gy), mark all z layers there."""
    cols = mask.any(axis=2)
    for z in range(GZ):
        mask[:, :, z] |= cols.astype(mask.dtype)
    return mask


def _rasterize_lidar(pts: np.ndarray, ml: np.ndarray):
    """Project lidar xyz to BEV columns; mark all z layers per occupied column."""
    if pts.shape[0] == 0:
        return
    gx = np.floor((pts[:, 0] - GRID_X[0]) / GRID_X[2]).astype(np.int32)
    gy = np.floor((pts[:, 1] - GRID_Y[0]) / GRID_Y[2]).astype(np.int32)
    ok = (gx >= 0) & (gx < GX) & (gy >= 0) & (gy < GY)
    gx, gy = gx[ok], gy[ok]
    if gx.size == 0:
        return
    cols = np.unique(np.ravel_multi_index((gx, gy), (GX, GY)))
    gx_u, gy_u = np.unravel_index(cols, (GX, GY))
    ml[gx_u, gy_u, 0] = 1
    ml[gx_u, gy_u, 1] = 1


def _finalize_sensor_masks(mc: np.ndarray, ml: np.ndarray):
    """Unify masks across z layers; extend lidar mask to camera-visible columns."""
    _propagate_mask_columns(mc)
    _propagate_mask_columns(ml)
    # PLY lidar often covers shorter range than camera depth; keep ml >= mc.
    ml[:] = np.maximum(ml, mc)


def _rasterize_cam(depth, seg, T, sem, mc):
    K_inv = np.linalg.inv(CAM_INTRINSIC)
    h, w = depth.shape
    step = RASTER_STEP
    vs = np.arange(0, h, step)
    us = np.arange(0, w, step)
    uu, vv = np.meshgrid(us, vs)
    d = depth[vv, uu].astype(np.float64)
    cls = seg[vv, uu].astype(np.int32)
    valid = (d > 0.05) & (d < DEPTH_INVALID) & (cls != 2)
    if not np.any(valid):
        return
    d = d[valid]
    cls = cls[valid]
    u = uu[valid].astype(np.float64)
    v = vv[valid].astype(np.float64)
    pts = (K_inv @ np.stack([u * d, v * d, d], axis=0)).T
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    ego = (T @ np.hstack([pts, ones]).T).T[:, :3]

    gx = np.floor((ego[:, 0] - GRID_X[0]) / GRID_X[2]).astype(np.int32)
    gy = np.floor((ego[:, 1] - GRID_Y[0]) / GRID_Y[2]).astype(np.int32)
    layer = (ego[:, 2] >= Z_SPLIT).astype(np.int32)
    ok = (gx >= 0) & (gx < GX) & (gy >= 0) & (gy < GY)
    gx, gy, layer, cls = gx[ok], gy[ok], layer[ok], cls[ok]
    if gx.size == 0:
        return
    mc[gx, gy, layer] = 1
    car = cls == 1
    free = cls == 0
    if car.any():
        sem[gx[car], gy[car], layer[car]] = SEM_CAR
    if free.any():
        flat = np.ravel_multi_index((gx[free], gy[free], layer[free]), (GX, GY, GZ))
        cur = sem.ravel()[flat]
        sem.ravel()[flat[cur != SEM_CAR]] = SEM_PASSABLE


def build_gt_from_sensors(stem: str, pose_dir: Path):
    sem, mc, ml = empty_grid()

    for cam_id, cam_name in CAM_MAP.items():
        depth_path = pose_dir / f'{stem}_cam{cam_id}_depth.npy'
        seg_path = pose_dir / f'{stem}_seg_cam{cam_id}_raw.png'
        if not depth_path.exists() or not seg_path.exists():
            continue
        depth = np.load(depth_path)
        seg = decode_seg_class(seg_path)
        T = sensor2ego_matrix(
            CAM_EXTRINSICS[cam_name]['sensor2ego_rotation'],
            CAM_EXTRINSICS[cam_name]['sensor2ego_translation'],
        )
        _rasterize_cam(depth, seg, T, sem, mc)

    pts_body, _, lidar_meta = load_lidar_body(pose_dir, stem)
    if pts_body.shape[0] > 0 and lidar_meta.get('captured', True):
        _rasterize_lidar(pts_body, ml)

    _finalize_sensor_masks(mc, ml)
    return sem, mc, ml


def build_gt_v2_legacy(stem: str, pose_dir: Path, occ_src: Path | None):
    """Previous pipeline: occ.npz-only when present, else sensors @ RASTER_STEP=2."""
    if occ_src is not None and occ_src.exists():
        return build_gt_from_occ_npz(occ_src)
    old_step = RASTER_STEP
    globals()['RASTER_STEP'] = 2
    try:
        return build_gt_from_sensors(stem, pose_dir)
    finally:
        globals()['RASTER_STEP'] = old_step


def build_gt(stem: str, pose_dir: Path, occ_src: Path | None):
    """v3: sensor rasterization for masks; optional occ.npz semantics overlay."""
    sem, mc, ml = build_gt_from_sensors(stem, pose_dir)
    if occ_src is None or not occ_src.exists():
        return sem, mc, ml

    sem_occ, mc_occ, ml_occ = build_gt_from_occ_npz(occ_src)
    occ_vis = mc_occ.astype(bool)
    car = occ_vis & (sem_occ == SEM_CAR)
    free = occ_vis & (sem_occ == SEM_PASSABLE)
    sem[car] = SEM_CAR
    sem[free & (sem != SEM_CAR)] = SEM_PASSABLE
    mc = np.maximum(mc, mc_occ)
    ml = np.maximum(ml, ml_occ)
    _finalize_sensor_masks(mc, ml)
    return sem, mc, ml


def mask_stats(sem, mc, ml):
    mc_ratio = float(mc.mean())
    ml_ratio = float(ml.mean())
    car = int((sem == SEM_CAR).sum())
    car_in_cam = int(((sem == SEM_CAR) & mc.astype(bool)).sum())
    passable = int((sem == SEM_PASSABLE).sum())
    masked = int(mc.sum())
    return {
        'mask_camera_ratio': mc_ratio,
        'mask_lidar_ratio': ml_ratio,
        'car_voxels': car,
        'car_in_cam': car_in_cam,
        'passable_voxels': passable,
        'masked_voxels': masked,
        'car_per_masked': car_in_cam / max(masked, 1),
    }


def compare_mask_coverage(infos, label: str):
    ratios, lidar_ratios, cars, cars_in_cam = [], [], [], []
    for info in infos:
        pose_dir = Path(info['src_pose_dir'])
        stem = info['sample_stem']
        occ_src = pose_dir / f'{stem}_occ.npz'
        sem, mc, ml = build_gt(stem, pose_dir, occ_src)
        st = mask_stats(sem, mc, ml)
        ratios.append(st['mask_camera_ratio'])
        lidar_ratios.append(st['mask_lidar_ratio'])
        cars.append(st['car_voxels'])
        cars_in_cam.append(st['car_in_cam'])
    ratios = np.asarray(ratios)
    lidar_ratios = np.asarray(lidar_ratios)
    print(f'  [{label}] n={len(infos)}')
    print(f'    mask_camera: mean={ratios.mean():.4f} '
          f'p50={np.median(ratios):.4f} min={ratios.min():.4f} max={ratios.max():.4f}')
    print(f'    mask_lidar:  mean={lidar_ratios.mean():.4f}')
    print(f'    car_voxels:  total={sum(cars)} samples_with_car={sum(c>0 for c in cars)}')
    print(f'    car_in_cam:  total={sum(cars_in_cam)}')
    return ratios.mean()


def compare_mask_legacy(infos, label: str):
    ratios, lidar_ratios, cars, cars_in_cam = [], [], [], []
    for info in infos:
        pose_dir = Path(info['src_pose_dir'])
        stem = info['sample_stem']
        occ_src = pose_dir / f'{stem}_occ.npz'
        sem, mc, ml = build_gt_v2_legacy(stem, pose_dir, occ_src)
        st = mask_stats(sem, mc, ml)
        ratios.append(st['mask_camera_ratio'])
        lidar_ratios.append(st['mask_lidar_ratio'])
        cars.append(st['car_voxels'])
        cars_in_cam.append(st['car_in_cam'])
    ratios = np.asarray(ratios)
    lidar_ratios = np.asarray(lidar_ratios)
    print(f'  [{label}] n={len(infos)}')
    print(f'    mask_camera: mean={ratios.mean():.4f} '
          f'p50={np.median(ratios):.4f} min={ratios.min():.4f} max={ratios.max():.4f}')
    print(f'    mask_lidar:  mean={lidar_ratios.mean():.4f}')
    print(f'    car_voxels:  total={sum(cars)} samples_with_car={sum(c>0 for c in cars)}')
    print(f'    car_in_cam:  total={sum(cars_in_cam)}')
    return ratios.mean()


def compare_on_disk(infos, label: str):
    ratios, lidar_ratios, cars, cars_in_cam = [], [], [], []
    root = Path(__file__).resolve().parents[1]
    for info in infos:
        z = np.load(root / info['occ_path'] / 'labels.npz')
        sem, mc, ml = z['semantics'], z['mask_camera'], z['mask_lidar']
        st = mask_stats(sem, mc, ml)
        ratios.append(st['mask_camera_ratio'])
        lidar_ratios.append(st['mask_lidar_ratio'])
        cars.append(st['car_voxels'])
        cars_in_cam.append(st['car_in_cam'])
    ratios = np.asarray(ratios)
    lidar_ratios = np.asarray(lidar_ratios)
    print(f'  [{label}] n={len(infos)}')
    print(f'    mask_camera: mean={ratios.mean():.4f} '
          f'p50={np.median(ratios):.4f} min={ratios.min():.4f} max={ratios.max():.4f}')
    print(f'    mask_lidar:  mean={lidar_ratios.mean():.4f}')
    print(f'    car_voxels:  total={sum(cars)} samples_with_car={sum(c>0 for c in cars)}')
    print(f'    car_in_cam:  total={sum(cars_in_cam)}')
    return ratios.mean()


def discover_samples(src_root: Path):
    samples = []
    for session_dir in sorted(src_root.iterdir()):
        if not session_dir.is_dir():
            continue
        poses = session_dir / 'poses'
        if not poses.is_dir():
            continue
        scene = session_dir.name
        for pose_dir in sorted(poses.iterdir()):
            if not pose_dir.is_dir():
                continue
            for cam1 in sorted(pose_dir.glob('*_cam1.png')):
                if '_seg_cam' in cam1.name:
                    continue
                stem = cam1.name[:-len('_cam1.png')]
                cam2 = pose_dir / f'{stem}_cam2.png'
                if not cam2.exists():
                    continue
                samples.append({
                    'scene': scene,
                    'pose': pose_dir.name,
                    'stem': stem,
                    'pose_dir': pose_dir,
                })
    return samples


def make_info(sample, out_root: Path, flashocc_root: Path):
    stem = sample['stem']
    pose_dir = sample['pose_dir']
    token = token_from_stem(stem)
    scene_name = f"scene-{sample['scene']}"
    ts = int(hashlib.md5(stem.encode()).hexdigest()[:12], 16)

    def rel(p):
        return os.path.relpath(p, flashocc_root)

    cams = {}
    for cam_id, cam_name in CAM_MAP.items():
        img_src = pose_dir / f'{stem}_cam{cam_id}.png'
        depth_src = pose_dir / f'{stem}_cam{cam_id}_depth.npy'
        ext = CAM_EXTRINSICS[cam_name]
        cams[cam_name] = {
            'data_path': str(img_src.resolve()),
            'depth_path': str(depth_src.resolve()),
            'sample_data_token': token[:16],
            'sensor2ego_rotation': ext['sensor2ego_rotation'],
            'sensor2ego_translation': ext['sensor2ego_translation'],
            'ego2global_rotation': [1.0, 0.0, 0.0, 0.0],
            'ego2global_translation': [0.0, 0.0, 0.0],
            'timestamp': ts,
            'cam_intrinsic': CAM_INTRINSIC.tolist(),
        }

    lidar_dst = out_root / 'samples' / 'LIDAR_TOP' / f'{stem}.bin'
    lidar_dst.parent.mkdir(parents=True, exist_ok=True)
    if not lidar_dst.exists():
        pts, _, _ = load_lidar_body(pose_dir, stem)
        if pts.shape[0] == 0:
            pts = np.zeros((1, 3), dtype=np.float32)
        write_lidar_bin(lidar_dst, pts)

    occ_src = pose_dir / f'{stem}_occ.npz'
    gt_dir = out_root / 'gts' / scene_name / token
    gt_dir.mkdir(parents=True, exist_ok=True)
    labels_path = gt_dir / 'labels.npz'
    sem, mc, ml = build_gt(stem, pose_dir, occ_src)
    np.savez_compressed(labels_path, semantics=sem, mask_camera=mc, mask_lidar=ml)

    info = {
        'lidar_path': rel(lidar_dst),
        'token': token,
        'sweeps': [],
        'cams': cams,
        'lidar2ego_translation': [0.0, 0.0, 0.0],
        'lidar2ego_rotation': [1.0, 0.0, 0.0, 0.0],
        'ego2global_translation': [0.0, 0.0, 0.0],
        'ego2global_rotation': [1.0, 0.0, 0.0, 0.0],
        'timestamp': ts,
        'ann_infos': ([], []),
        'scene_token': hashlib.md5(sample['scene'].encode()).hexdigest(),
        'scene_name': scene_name,
        'occ_path': rel(gt_dir),
        'sample_stem': stem,
        'src_pose_dir': str(pose_dir),
    }
    return info, sem


def split_train_val(infos_with_sem, val_ratio: float, seed: int):
    """Stratified split: keep car frames in both train and val."""
    rng = np.random.RandomState(seed)
    car_idx, plain_idx = [], []
    for i, (info, sem) in enumerate(infos_with_sem):
        if (sem == SEM_CAR).any():
            car_idx.append(i)
        else:
            plain_idx.append(i)
    rng.shuffle(car_idx)
    rng.shuffle(plain_idx)

    n_val_car = max(1, int(len(car_idx) * val_ratio))
    n_val_plain = max(1, int(len(plain_idx) * val_ratio))
    val_set = set(car_idx[:n_val_car] + plain_idx[:n_val_plain])

    train, val = [], []
    for i, (info, _) in enumerate(infos_with_sem):
        (val if i in val_set else train).append(info)
    return train, val


def run_mask_compare(flashocc_root: Path):
    print('Mask coverage comparison (v2 legacy vs v3 new vs on-disk GT)')
    print(f'  v2 legacy: occ.npz-only when present, else sensors @ RASTER_STEP=2')
    print(f'  v3 new:    sensors @ RASTER_STEP={RASTER_STEP}, occ.npz semantics overlay')
    for split in ('train', 'val'):
        pkl_path = flashocc_root / 'data/car_perception_grid/nuscenes' / \
            f'bevdetv2-nuscenes_infos_{split}.pkl'
        if not pkl_path.exists():
            print(f'  skip {split}: {pkl_path} not found')
            continue
        infos = pickle.load(open(pkl_path, 'rb'))['infos']
        print(f'\n== {split} ({len(infos)} samples) ==')
        old_mean = compare_mask_legacy(infos, 'v2 legacy (recomputed)')
        new_mean = compare_mask_coverage(infos, 'v3 new (recomputed)')
        disk_mean = compare_on_disk(infos, 'on-disk labels.npz')
        delta = new_mean - old_mean
        print(f'  delta v3-v2: {delta:+.4f} ({delta/old_mean*100:+.1f}%)')
        print(f'  delta v3-disk: {new_mean-disk_mean:+.4f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='/data/car_perception_grid')
    parser.add_argument('--out', default=None)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-samples', type=int, default=0)
    parser.add_argument('--raster-step', type=int, default=DEFAULT_RASTER_STEP,
                        help='pixel step for depth+seg projection (default: 1)')
    parser.add_argument('--compare-mask', action='store_true',
                        help='compare v2/v3 mask coverage on existing pkl, no rebuild')
    parser.add_argument('--rebuild-labels-only', action='store_true',
                        help='rebuild labels.npz from pkl src_pose_dir without rescanning src')
    parser.add_argument('--force', action='store_true',
                        help='rebuild all labels.npz')
    args = parser.parse_args()

    global RASTER_STEP
    RASTER_STEP = args.raster_step
    flashocc_root = Path(__file__).resolve().parents[1]

    if args.compare_mask:
        run_mask_compare(flashocc_root)
        return

    if args.rebuild_labels_only:
        out_root = Path(args.out) if args.out else \
            flashocc_root / 'data/car_perception_grid/nuscenes'
        for split in ('train', 'val'):
            pkl_path = out_root / f'bevdetv2-nuscenes_infos_{split}.pkl'
            infos = pickle.load(open(pkl_path, 'rb'))['infos']
            print(f'Rebuilding labels for {split}: {len(infos)} samples')
            for i, info in enumerate(infos):
                if i % 200 == 0:
                    print(f'  [{i}/{len(infos)}]')
                pose_dir = Path(info['src_pose_dir'])
                stem = info['sample_stem']
                occ_src = pose_dir / f'{stem}_occ.npz'
                sem, mc, ml = build_gt(stem, pose_dir, occ_src)
                labels_path = flashocc_root / info['occ_path'] / 'labels.npz'
                np.savez_compressed(labels_path, semantics=sem,
                                    mask_camera=mc, mask_lidar=ml)
            mc_mean = np.mean([
                np.load(flashocc_root / x['occ_path'] / 'labels.npz')['mask_camera'].mean()
                for x in infos
            ])
            car_n = sum(
                (np.load(flashocc_root / x['occ_path'] / 'labels.npz')['semantics'] == SEM_CAR).any()
                for x in infos
            )
            print(f'  {split}: car={car_n}, avg_cam_mask={mc_mean:.4f}')
        return
    out_root = Path(args.out) if args.out else flashocc_root / 'data/car_perception_grid/nuscenes'
    src_root = Path(args.src)
    out_root.mkdir(parents=True, exist_ok=True)

    samples = discover_samples(src_root)
    if args.max_samples > 0:
        samples = samples[:args.max_samples]
    print(f'Found {len(samples)} samples under {src_root}')

    infos_with_sem = []
    for i, s in enumerate(samples):
        if i % 100 == 0:
            print(f'  [{i}/{len(samples)}] {s["stem"]}')
        info, sem = make_info(s, out_root, flashocc_root)
        infos_with_sem.append((info, sem))

    train_infos, val_infos = split_train_val(
        infos_with_sem, args.val_ratio, args.seed)

    def count_car(part):
        n = 0
        for info in part:
            sem = np.load(os.path.join(info['occ_path'], 'labels.npz'))['semantics']
            if (sem == SEM_CAR).any():
                n += 1
        return n

    for split, part in [('train', train_infos), ('val', val_infos)]:
        pkl = {
            'infos': part,
            'metadata': {
                'version': 'car_perception_grid_v3',
                'src': str(src_root),
                'dz': DZ_OUT,
                'num_classes': 3,
                'grid': {'x': GRID_X, 'y': GRID_Y, 'shape': [GX, GY, GZ]},
            },
        }
        pkl_path = out_root / f'bevdetv2-nuscenes_infos_{split}.pkl'
        with open(pkl_path, 'wb') as f:
            pickle.dump(pkl, f)
        car_n = count_car(part)
        mc_mean = np.mean([
            np.load(os.path.join(x['occ_path'], 'labels.npz'))['mask_camera'].mean()
            for x in part[:50]
        ])
        print(f'Wrote {pkl_path}: {len(part)} samples, car={car_n}, '
              f'avg_cam_mask~{mc_mean:.3f} (first 50)')

    # Rewrite camera/depth paths to nuscenes/samples for Docker portability.
    import subprocess
    import sys
    link_script = Path(__file__).resolve().parent / 'link_car_perception_media.py'
    r = subprocess.run([sys.executable, str(link_script)], cwd=flashocc_root)
    if r.returncode == 0:
        print('Linked media -> nuscenes/samples and updated pkl paths.')
    else:
        print('Warning: link_car_perception_media.py exited with', r.returncode)


if __name__ == '__main__':
    main()
