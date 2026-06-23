#!/usr/bin/env python3
"""Hardlink images/depth into nuscenes/samples/ and rewrite pkl paths for Docker."""
from __future__ import annotations

import os
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NUSC = ROOT / 'data/car_perception_grid/nuscenes'
PREFIX = '/data/car_perception_grid/'


def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        import shutil
        shutil.copy2(src, dst)


def fix_pkl(path: Path, flashocc_root: Path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    for info in data['infos']:
        for cam_name, cam in info['cams'].items():
            for key, sub in [('data_path', ''), ('depth_path', 'depth')]:
                p = Path(cam[key])
                if str(p).startswith('data/car_perception_grid'):
                    continue
                if not str(p).startswith(PREFIX):
                    continue
                # fix mistaken seg_cam stems from older pkl
                if key == 'depth_path' and '_seg_cam' in p.name:
                    p = Path(str(p).replace('_seg_cam', '_cam'))
                if not p.exists():
                    continue
                rel = p.relative_to(PREFIX.rstrip('/'))
                stem = rel.name
                if key == 'data_path':
                    dst = NUSC / 'samples' / cam_name / stem
                else:
                    dst = NUSC / 'samples' / sub / cam_name / stem
                link_or_copy(p, dst)
                cam[key] = os.path.relpath(dst, flashocc_root)
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def main():
    flashocc_root = ROOT
    for name in ['bevdetv2-nuscenes_infos_train.pkl',
                  'bevdetv2-nuscenes_infos_val.pkl']:
        p = NUSC / name
        print(f'Processing {p} ...')
        fix_pkl(p, flashocc_root)
    print('Done. Media hardlinked under', NUSC / 'samples')


if __name__ == '__main__':
    main()
