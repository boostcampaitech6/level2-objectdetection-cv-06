### custom 한 config.py 를 만들 때 다음의 format 을 따르시면 됩니다. 공식문서를 토대로 만들었습니다.
### mmdetection 3.x
# model, dataset pipeline, optimizer, schedule 말고는 바꿀 것 없음.

## - 0. 상속 불러오기
# mmdetection/configs 안에 있는 사용하려는 모델에 있는 기준
# 이름 format : {model}_[model setting]_{backbone}_{neck}_[norm setting]_[misc]_[gpu x batch_per_gpu]_{schedule}_{dataset}
# {} 안에 있는 항목은 필수, [] 안에 있는 항목은 optional
_base_ = [
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

## - model setting : 원하는 부분만 dict type 으로 수정할 수 있음.
# model settings
pretrained_ckpt = 'https://github.com/shinya7y/weights/releases/download/v1.0.2/res2net50_v1b_26w_4s-3cf99910_mmdetv2-92ed3313.pth'  # noqa
model = dict(
    type='GFL',
    # data normalize
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.27423945, 117.1344846, 109.78107765],
        std=[54.2332011, 53.670966899999996, 55.038554850000004],
        bgr_to_rgb=True,
        pad_size_divisor=32),

    backbone=dict(
        type='Res2Net',
        depth=101,
        scales=4,
        base_width=26,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://res2net101_v1d_26w_4s')),
    neck=[
        dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            start_level=1,
            add_extra_convs='on_output',
            num_outs=5),
        dict(
            type='SEPC',
            out_channels=256,
            stacked_convs=4,
            pconv_deform=True,
            lcconv_deform=True,
            ibn=True,  # please set imgs/gpu >= 4
            pnorm_eval=False,
            lcnorm_eval=False,
            lcconv_padding=1)
    ],
    bbox_head=dict(
        type='GFLSEPCHead',
        num_classes=10,
        in_channels=256,
        stacked_convs=0,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        reg_max=16,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))



## - dataset and evaluator

# transform
dataset_type = 'CocoDataset'
data_root = '/data/ephemeral/home/dataset'
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomResize', scale=[(1333, 480), (1333, 960)], keep_ratio=True),
    dict(type='Brightness', prob=0.25),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# dataloader
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='pseudo_dino_train3.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='pseudo_dino_val3.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'test.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=test_pipeline))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'pseudo_dino_val3.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)

test_evaluator = val_evaluator


## - Training and testing config (model train_cfg, test_cfg 와 별도)
train_cfg = dict(
    type='EpochBasedTrainLoop',  # The training loop type. Refer to https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py
    max_epochs=24,  # Maximum training epochs
    val_begin=1,
    val_interval=1)  # Validation intervals. Run validation every epoch.
val_cfg = dict(type='ValLoop')  # The validation loop type
test_cfg = dict(type='TestLoop')  # The testing loop type


## - Optimization config
# optimizer
optim_wrapper = dict(  # Optimizer wrapper config
    type='AmpOptimWrapper',  # Optimizer wrapper type, switch to AmpOptimWrapper to enable mixed precision training.
    loss_scale=512.,
    optimizer=dict(  # Optimizer config. Support all kinds of optimizers in PyTorch. Refer to https://pytorch.org/docs/stable/optim.html#algorithms
        type='SGD',  # Stochastic gradient descent optimizer
        lr=0.02,  # The base learning rate
        momentum=0.9,  # Stochastic gradient descent with momentum
        weight_decay=0.0001),  # Weight decay of SGD
    clip_grad=dict(max_norm=35, norm_type=2),  # Gradient clip option. Set None to disable gradient clip. Find usage in https://mmengine.readthedocs.io/en/latest/tutorials/optimizer.html
    )

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=19,
        by_epoch=True,
        milestones=[16, 19],
        gamma=0.1)
]

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)


## - hook
default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, out_suffix='.json'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3, save_best='coco/bbox_mAP'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook', draw=True, interval=100))

# custom hook - early stopping 등
custom_hooks = [dict(
    type="EarlyStoppingHook",
    min_delta=0.005,
    monitor='coco/bbox_mAP',
    patience=6),
    dict(type="NumClassCheckHook"),
    dict(type='EmptyCacheHook', after_epoch=True),
]

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend'),
                dict(type='WandbVisBackend')]
visualizer = dict(
    # type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
    type="Visualizer", vis_backends=vis_backends
    )

log_config = dict(
    hooks = [
    dict(type='TextLoggerHook')
    ]
)

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'

# pretrained model 불러오기
load_from = 'https://github.com/shinya7y/UniverseNet/releases/download/20.10/universenet101_2008d_fp16_4x4_mstrain_480_960_20e_coco_20201023_epoch_20-3e0d236a.pth'

# 다시 학습 재개하려는 pth 주소
resume = False
