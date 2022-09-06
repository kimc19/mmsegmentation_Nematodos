from mmcv import mkdir_or_exist
from mmcv import Config
from mmseg.apis import set_random_seed
from mmseg.utils import get_device
import argparse

parser = argparse.ArgumentParser(description='Generador de máscaras a partir de anotaciones VIA.')

parser.add_argument('--data_root', type=str, help='Dirección de la carpeta con los datos.')

args = parser.parse_args()

if args.data_root == None:
	data_root = '../data/nematodos'
else:
	data_root = args.data_root
	
img_dir = 'images'
ann_dir = 'annotations'

cfg = Config.fromfile('../configs/stdc/stdc1_512x1024_80k_cityscapes.py')

# Modify num classes of the model in decode/auxiliary head
norm_cfg = dict(type='BN', requires_grad=True)
cfg.model.decode_head.num_classes=2
cfg.model.auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=128,
            channels=64,
            num_convs=1,
            num_classes=2,
            in_index=2,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=10000),
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='FCNHead',
            in_channels=128,
            channels=64,
            num_convs=1,
            num_classes=2,
            in_index=1,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=10000),
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='STDCHead',
            in_channels=256,
            channels=64,
            num_convs=1,
            num_classes=2,
            boundary_threshold=0.1,
            in_index=0,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=True,
            loss_decode=[
                dict(
                    type='CrossEntropyLoss',
                    loss_name='loss_ce',
                    use_sigmoid=True,
                    loss_weight=1.0),
                dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0)
            ]),
    ]

# Modify dataset type and path
cfg.dataset_type = 'NematodosDataset'
cfg.data_root = data_root

cfg.data.samples_per_gpu = 8
cfg.data.workers_per_gpu = 2 #Wand support

cfg.img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
cfg.crop_size = (512, 1024)
cfg.train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(1024, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomRotate', prob=0.75, degree=180),
    dict(type='RandomCrop', crop_size=cfg.crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **cfg.img_norm_cfg),
    dict(type='Pad', size=cfg.crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

cfg.test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **cfg.img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

cfg.data.train.type = cfg.dataset_type
cfg.data.train.data_root = cfg.data_root
cfg.data.train.img_dir = img_dir
cfg.data.train.ann_dir = ann_dir
cfg.data.train.pipeline = cfg.train_pipeline
cfg.data.train.split = 'splits/train.txt'

cfg.data.val.type = cfg.dataset_type
cfg.data.val.data_root = cfg.data_root
cfg.data.val.img_dir = img_dir
cfg.data.val.ann_dir = ann_dir
cfg.data.val.pipeline = cfg.test_pipeline
cfg.data.val.split = 'splits/val.txt'

cfg.data.test.type = cfg.dataset_type
cfg.data.test.data_root = cfg.data_root
cfg.data.test.img_dir = img_dir
cfg.data.test.ann_dir = ann_dir
cfg.data.test.pipeline = cfg.test_pipeline
cfg.data.test.split = 'splits/test.txt'

cfg.work_dir = '../work_dirs/stdc_pretrain'

#Set iterations, and interval of iterations save
cfg.checkpoint_config.interval = 10000
cfg.checkpoint_config.max_keep_ckpts = 2

# Set evaluations metrics
cfg.evaluation.interval=10000
cfg.evaluation.metric=['mIoU','mDice','mFscore']

# Set checkpoint file for pretraining
#cfg.load_from = '../checkpoints/stdc1_in1k-pre_512x1024_80k_cityscapes_20220224_141648-3d4c2981.pth'

# Set seed to facitate reproducing the result
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)
cfg.device = get_device()

# Set hooks: Text, Wandb
cfg.log_config = dict(
    interval=10000,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='MMSegWandbHook',
             init_kwargs={
                 'entity': 'kimc19',
                 'project': 'Nematodos_STDC',
                 'name': 'stdc_base_2',
                 'id': 'stdc_base_2',
                 'resume': 'allow',
                 'notes':'Entrenamiento con modelo stdc base preentrenado con 80k iteraciones, batch=8'
                 },
             log_checkpoint=True,
             log_checkpoint_metadata=True,
             num_eval_images=100)
        ])

# Look at the final config used for training
print(f'Config:\n{cfg.pretty_text}')

# Save config file
mkdir_or_exist("../configs/_nematodos_/stdc")
cfg.dump("../configs/_nematodos_/stdc/stdc_nematodos_base2.py")