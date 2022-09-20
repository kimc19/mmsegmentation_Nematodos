norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='CascadeEncoderDecoder',
    num_stages=2,
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=4),
    decode_head=[
        dict(
            type='FPNHead',
            in_channels=[256, 256, 256, 256],
            in_index=[0, 1, 2, 3],
            feature_strides=[4, 8, 16, 32],
            channels=128,
            dropout_ratio=-1,
            num_classes=2,
            norm_cfg=dict(type='BN', requires_grad=True),
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='PointHead',
            in_channels=[256],
            in_index=[0],
            channels=256,
            num_fcs=3,
            coarse_pred_each_layer=True,
            dropout_ratio=-1,
            num_classes=2,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
    ],
    train_cfg=dict(
        num_points=2048, oversample_ratio=3, importance_sample_ratio=0.75),
    test_cfg=dict(
        mode='whole',
        subdivision_steps=2,
        subdivision_num_points=8196,
        scale_factor=2))
dataset_type = 'NematodosDataset'
data_root = '../data/nematodos'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (768, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1024, 768), ratio_range=(0.5, 1.5)),
    dict(type='RandomRotate', prob=0.75, degree=30),
    dict(type='RandomCrop', crop_size=(768, 1024), cat_max_ratio=0.25),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(768, 1024), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 768),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='NematodosDataset',
        data_root='../data/nematodos',
        img_dir='images',
        ann_dir='annotations',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(type='Resize', img_scale=(1024, 768), ratio_range=(0.5, 1.5)),
            dict(type='RandomRotate', prob=0.75, degree=30),
            dict(type='RandomCrop', crop_size=(768, 1024), cat_max_ratio=0.25),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size=(768, 1024), pad_val=0, seg_pad_val=255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_semantic_seg'])
        ],
        split='splits/train.txt'),
    val=dict(
        type='NematodosDataset',
        data_root='../data/nematodos',
        img_dir='images',
        ann_dir='annotations',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 768),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        split='splits/val.txt'),
    test=dict(
        type='NematodosDataset',
        data_root='../data/nematodos',
        img_dir='images',
        ann_dir='annotations',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1024, 768),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        split='splits/test.txt'),
    val_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1, shuffle=False))
log_config = dict(
    interval=10000,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False, interval=5000),
        dict(
            type='MMSegWandbHook',
            with_step=False,
            init_kwargs=dict(
                entity='seg_nematodos',
                project='Nematodos',
                name='pointrend_pretrain',
                id='pointrend_pretrain',
                resume='allow',
                notes=
                'Entrenamiento modelo pointrend pretrain, lr=0.01, m=0.9, A1, 80k iteraciones, batch=2'
            ),
            log_checkpoint=True,
            log_checkpoint_metadata=True,
            num_eval_images=100)
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'https://download.openmmlab.com/mmsegmentation/v0.5/point_rend/pointrend_r101_512x1024_80k_cityscapes/pointrend_r101_512x1024_80k_cityscapes_20200711_170850-d0ca84be.pth'
resume_from = None
workflow = [('train', 1), ('val', 1)]
cudnn_benchmark = True
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    power=0.9,
    min_lr=0.0001,
    by_epoch=False,
    warmup='linear',
    warmup_iters=200)
runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=10000, max_keep_ckpts=2)
evaluation = dict(
    interval=10000, metric=['mIoU', 'mDice', 'mFscore'], pre_eval=True)
work_dir = '../work_dirs/PointRend_pretrain'
seed = 0
gpu_ids = range(0, 1)
device = 'cuda'
