"""Shared lidar/camera geometry helpers for car_perception_grid."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEPTH_INVALID = 60000.0

# Stable UL01 mount: lidar sensor frame -> body/ego frame.
LIDAR_SENSOR2BODY = np.array([
    [1.0, 0.0, 0.0, 0.5],
    [0.0, 0.9396926, 0.34202015, 0.0],
    [0.0, -0.34202015, 0.9396926, -0.7],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)

# UE body frame uses Y-right; FlashOCC/moon_stereo ego uses Y-left.
BODY_UE_TO_EGO = np.diag([1.0, -1.0, 1.0, 1.0]).astype(np.float64)


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
    return np.frombuffer(raw, dtype='<f4').reshape(-1, 3).astype(np.float32)


def read_lidar_meta(pose_dir: Path, stem: str) -> dict | None:
    meta_path = pose_dir / f'{stem}_ul01_meta.json'
    if not meta_path.exists():
        return None
    with open(meta_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def lidar_capture_ok(meta: dict | None) -> bool:
    if not meta:
        return False
    return bool(meta.get('lidar_captured')) and int(meta.get('lidar_points_sensor', 0)) > 0


def body_to_ego(pts_body: np.ndarray) -> np.ndarray:
    hom = np.hstack([pts_body.astype(np.float64), np.ones((pts_body.shape[0], 1))])
    return (BODY_UE_TO_EGO @ hom.T).T[:, :3].astype(np.float32)


def load_lidar_body(pose_dir: Path, stem: str | None = None) -> tuple[np.ndarray, str, dict]:
    """Load lidar points in body/ego frame.

    Prefer `{pose}_lidar.ply` when body cloud was saved. Otherwise transform
    `{pose}_lidar_sensor.ply` with the fixed sensor->body extrinsic.
    """
    pose_dir = Path(pose_dir)
    meta = read_lidar_meta(pose_dir, stem) if stem else None
    captured = lidar_capture_ok(meta)

    body_ply = pose_dir / f'{pose_dir.name}_lidar.ply'
    sensor_ply = pose_dir / f'{pose_dir.name}_lidar_sensor.ply'

    if body_ply.exists() and (captured or meta is None or meta.get('lidar_body_saved')):
        pts = body_to_ego(read_ply_xyz(body_ply))
        return pts, str(body_ply), {
            'frame': 'body->ego',
            'captured': captured,
            'meta': meta,
        }

    if sensor_ply.exists() and captured:
        sensor = read_ply_xyz(sensor_ply)
        hom = np.hstack([sensor.astype(np.float64), np.ones((sensor.shape[0], 1))])
        pts = body_to_ego((LIDAR_SENSOR2BODY @ hom.T).T[:, :3])
        return pts, str(sensor_ply), {
            'frame': 'sensor->body->ego',
            'captured': captured,
            'meta': meta,
        }

    if sensor_ply.exists():
        pts = body_to_ego(read_ply_xyz(sensor_ply))
        return pts, str(sensor_ply), {
            'frame': 'sensor_raw',
            'captured': False,
            'meta': meta,
            'warning': 'lidar_captured=false; ply may not match camera view',
        }

    return np.zeros((0, 3), dtype=np.float32), 'missing', {
        'frame': 'missing',
        'captured': False,
        'meta': meta,
    }


def load_lidar_from_info(info: dict) -> tuple[np.ndarray, str, dict]:
    pose_dir = Path(info.get('src_pose_dir', ''))
    stem = info.get('sample_stem')
    return load_lidar_body(pose_dir, stem)


def project_body_to_image(
        pts_body: np.ndarray,
        K: np.ndarray,
        sensor2ego: np.ndarray,
        max_depth: float = 80.0,
        subsample: int = 1,
):
    """Project body-frame points to the camera image."""
    if pts_body.shape[0] == 0:
        return np.zeros((0, 2)), np.zeros(0), np.zeros(0, dtype=bool)

    pts = pts_body[::max(1, subsample)].astype(np.float64)
    ego2cam = np.linalg.inv(sensor2ego)
    hom = np.hstack([pts, np.ones((pts.shape[0], 1))])
    cam = (ego2cam @ hom.T).T[:, :3]

    valid = (cam[:, 2] > 0.1) & (cam[:, 2] < max_depth)
    cam_v = cam[valid]
    if cam_v.shape[0] == 0:
        return np.zeros((0, 2)), np.zeros(0), valid

    uvh = (K @ cam_v.T).T
    uv = uvh[:, :2] / uvh[:, 2:3]
    return uv, cam_v[:, 2], valid
