#!/usr/bin/env python3
"""Drop invalid samples and ensure depth/image paths exist."""
import os
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NUSC = ROOT / 'data/car_perception_grid/nuscenes'


def valid(info):
    stem = info.get('sample_stem', '')
    if stem.endswith('_seg') or '_seg_cam' in stem:
        return False
    for cam in info['cams'].values():
        for key in ('data_path', 'depth_path'):
            p = cam[key]
            if not os.path.isabs(p):
                p = str(ROOT / p)
            if not os.path.exists(p):
                return False
    gt = info['occ_path']
    if not os.path.isabs(gt):
        gt = str(ROOT / gt)
    return os.path.exists(os.path.join(gt, 'labels.npz'))


def main():
    for name in ['bevdetv2-nuscenes_infos_train.pkl',
                  'bevdetv2-nuscenes_infos_val.pkl']:
        path = NUSC / name
        with open(path, 'rb') as f:
            data = pickle.load(f)
        before = len(data['infos'])
        data['infos'] = [i for i in data['infos'] if valid(i)]
        after = len(data['infos'])
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f'{name}: {before} -> {after} samples')


if __name__ == '__main__':
    main()
