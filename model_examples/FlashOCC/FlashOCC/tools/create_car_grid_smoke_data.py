#!/usr/bin/env python3
"""Create 1-frame smoke val set for 310B flow test (random fake images, no full dataset)."""
import os
import pickle
import time

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(ROOT, 'data/car_perception_grid/nuscenes')
SRC_PKL = os.path.join(DATA_ROOT, 'bevdetv2-nuscenes_infos_val.pkl')
OUT_PKL = os.path.join(DATA_ROOT, 'bevdetv2-nuscenes_infos_val_smoke.pkl')

SMOKE_LEFT = 'data/car_perception_grid/nuscenes/samples/CAM_FRONT_LEFT/smoke_cam1.png'
SMOKE_RIGHT = 'data/car_perception_grid/nuscenes/samples/CAM_FRONT_RIGHT/smoke_cam2.png'


def _write_gray_png(path, size=(1024, 1024), seed=0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rng = np.random.RandomState(seed)
    arr = rng.randint(80, 180, size=(size[1], size[0]), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def main():
    base_seed = int(os.environ.get('SMOKE_RANDOM_SEED', time.time())) & 0xFFFFFFFF
    left_abs = os.path.join(ROOT, SMOKE_LEFT)
    right_abs = os.path.join(ROOT, SMOKE_RIGHT)
    _write_gray_png(left_abs, seed=base_seed)
    _write_gray_png(right_abs, seed=base_seed + 1)
    print(f'Random smoke seed: {base_seed}')

    with open(SRC_PKL, 'rb') as f:
        data = pickle.load(f)
    info = data['infos'][0].copy()
    info['cams']['CAM_FRONT_LEFT'] = dict(info['cams']['CAM_FRONT_LEFT'])
    info['cams']['CAM_FRONT_RIGHT'] = dict(info['cams']['CAM_FRONT_RIGHT'])
    info['cams']['CAM_FRONT_LEFT']['data_path'] = SMOKE_LEFT
    info['cams']['CAM_FRONT_RIGHT']['data_path'] = SMOKE_RIGHT
    smoke = {'infos': [info], 'metadata': data.get('metadata', {})}
    with open(OUT_PKL, 'wb') as f:
        pickle.dump(smoke, f)

    print('Wrote smoke val pkl:', OUT_PKL)
    print('Images:', left_abs, right_abs)


if __name__ == '__main__':
    main()
