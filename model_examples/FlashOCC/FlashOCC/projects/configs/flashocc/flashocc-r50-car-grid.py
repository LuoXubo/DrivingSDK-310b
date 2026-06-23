# FlashOCC for car_perception_grid: 2 cams, 3 classes, Dz=2.
# Prepare data: python tools/create_car_perception_flashocc.py
_base_ = ['flashocc-r50-perf.py']

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
]

data_root = 'data/car_perception_grid/nuscenes/'

grid_config = dict(
    x=[-40, 40, 0.4],
    y=[-40, 40, 0.4],
    z=[-1, 5.4, 6.4],
    depth=[0.5, 12.0, 0.5],
)

data_config = dict(
    cams=['CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT'],
    Ncams=2,
    input_size=(256, 704),
    src_size=(1024, 1024),
    resize=(0.0, 0.0),
    rot=(0.0, 0.0),
    flip=False,
    crop_h=(0.0, 0.0),
    resize_test=0.00,
)

bda_aug_conf = dict(
    rot_lim=(0., 0.),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.0,
    flip_dy_ratio=0.0,
)

file_client_args = dict(backend='disk')

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        sequential=False,
        load_depth=True,
        depth_invalid=60000.0),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=True),
    dict(type='LoadOccGTFromFile'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=['img_inputs', 'gt_depth', 'voxel_semantics',
              'mask_lidar', 'mask_camera']),
]

test_pipeline = [
    dict(
        type='PrepareImageInputs',
        data_config=data_config,
        sequential=False,
        load_depth=True,
        depth_invalid=60000.0),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img_inputs']),
        ]),
]

share_data_config = dict(
    type='NuScenesDatasetOccpancy',
    data_root=data_root,
    classes=class_names,
    modality=dict(
        use_lidar=False,
        use_camera=True,
        use_radar=False,
        use_map=False,
        use_external=False),
    stereo=False,
    filter_empty_gt=False,
    img_info_prototype='bevdet',
)

test_data_config = dict(
    pipeline=test_pipeline,
    ann_file=data_root + 'bevdetv2-nuscenes_infos_val.pkl')

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        data_root=data_root,
        ann_file=data_root + 'bevdetv2-nuscenes_infos_train.pkl',
        pipeline=train_pipeline),
    val=test_data_config,
    test=test_data_config)

for key in ['val', 'train', 'test']:
    data[key].update(share_data_config)

model = dict(
    img_view_transformer=dict(grid_config=grid_config),
    img_backbone=dict(with_cp=False),
    occ_head=dict(
        Dz=2,
        num_classes=3,
        use_mask=True,
        class_balance=False,
        loss_occ=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            ignore_index=255,
            loss_weight=1.0,
            class_weight=[1.0, 15.0, 1.0]),
    ),
)

find_unused_parameters = True

runner = dict(type='EpochBasedRunner', max_epochs=12)
evaluation = dict(interval=999, start=999)
