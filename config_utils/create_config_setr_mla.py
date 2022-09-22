from mmcv import mkdir_or_exist
from mmcv import Config
from mmseg.apis import set_random_seed
from mmseg.utils import get_device
import argparse

parser = argparse.ArgumentParser(description='Generador de config files para el modelo setr mla.')

parser.add_argument('--data_root', type=str, help='Dirección de la carpeta con los datos.')

args = parser.parse_args()

if args.data_root == None:
	data_root = '../data/nematodos'
else:
	data_root = args.data_root

img_dir = 'images'
ann_dir = 'annotations'

cfg = Config.fromfile('../configs/setr/setr_mla_512x512_160k_b8_ade20k.py')

# Since we use only one GPU, BN is used instead of SyncBN
cfg.norm_cfg = dict(type='BN', requires_grad=True)
cfg.model.decode_head.norm_cfg = cfg.norm_cfg
cfg.model.neck.norm_cfg = cfg.norm_cfg

# Modify num classes of the model in decode/auxiliary head
cfg.model.decode_head.num_classes = 2
cfg.model.auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=256,
            channels=256,
            in_index=0,
            dropout_ratio=0,
            norm_cfg=cfg.norm_cfg,
            act_cfg=dict(type='ReLU'),
            num_convs=0,
            kernel_size=1,
            concat_input=False,
            num_classes=2,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='FCNHead',
            in_channels=256,
            channels=256,
            in_index=1,
            dropout_ratio=0,
            norm_cfg=cfg.norm_cfg,
            act_cfg=dict(type='ReLU'),
            num_convs=0,
            kernel_size=1,
            concat_input=False,
            num_classes=2,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='FCNHead',
            in_channels=256,
            channels=256,
            in_index=2,
            dropout_ratio=0,
            norm_cfg=cfg.norm_cfg,
            act_cfg=dict(type='ReLU'),
            num_convs=0,
            kernel_size=1,
            concat_input=False,
            num_classes=2,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='FCNHead',
            in_channels=256,
            channels=256,
            in_index=3,
            dropout_ratio=0,
            norm_cfg=cfg.norm_cfg,
            act_cfg=dict(type='ReLU'),
            num_convs=0,
            kernel_size=1,
            concat_input=False,
            num_classes=2,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    ]

# Modify dataset type and path
cfg.dataset_type = 'NematodosDataset'
cfg.data_root = data_root

#Batch size
#cfg.data.samples_per_gpu = 10
#cfg.data.workers_per_gpu = 2 #Wand support
cfg.data.val_dataloader=dict(samples_per_gpu=1, workers_per_gpu=1, shuffle=False)

cfg.img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
cfg.crop_size = (768, 1024)
cfg.train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1024, 768), ratio_range=(0.5, 1.5)),
    dict(type='RandomRotate', prob=0.75, degree=30),
    dict(type='RandomCrop', crop_size=cfg.crop_size, cat_max_ratio=0.25),
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
        img_scale=(1024, 768),
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

cfg.work_dir = '../work_dirs/SetrMLA'

#Set iterations, and interval of iterations save
#cfg.runner.max_iters = 80000
#cfg.checkpoint_config.interval = 10000
cfg.checkpoint_config.max_keep_ckpts = 2

# Set evaluations metrics
#cfg.evaluation.interval=10000
cfg.evaluation.metric=['mIoU','mDice','mFscore']

# Set validation loss
cfg.workflow = [('train', 1), ('val', 1)]

# Set checkpoint file for pretraining
#cfg.load_from = '../checkpoints/

# Set seed to facitate reproducing the result
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)
cfg.device = get_device()

# Set hooks: Text, Wandb
cfg.log_config = dict(
    interval=16000,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False, interval=8000),
        dict(type='MMSegWandbHook',
             with_step=False,
             init_kwargs={
                 'entity': 'kimc19',
                 'project': 'SetrMLA_Nematodos',
                 'name': 'SetrMLA_base',
                 'id': 'SetrMLA_base',
                 'resume': 'allow',
                 'notes':'Entrenamiento modelo SetrMLA'
                 },
             log_checkpoint=True,
             log_checkpoint_metadata=True,
             num_eval_images=100)
        ])

# Look at the final config used for training
print(f'Config:\n{cfg.pretty_text}')

# Save config file
mkdir_or_exist("../configs/_nematodos_/setr")
cfg.dump("../configs/_nematodos_/setr/SetrMLA_base.py")