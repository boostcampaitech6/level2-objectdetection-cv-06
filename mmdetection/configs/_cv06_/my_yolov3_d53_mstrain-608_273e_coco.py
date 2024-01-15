# _base_ = "../yolo/yolov3_d53_mstrain-608_273e_coco.py"
_base_ = "../_base_/default_runtime.py"

# model settings
model = dict(
    type="YOLOV3",
    backbone=dict(
        type="Darknet",
        depth=53,
        out_indices=(3, 4, 5),
        init_cfg=dict(type="Pretrained", checkpoint="open-mmlab://darknet53"),
    ),
    neck=dict(
        type="YOLOV3Neck",
        num_scales=3,
        in_channels=[1024, 512, 256],
        out_channels=[512, 256, 128],
    ),
    bbox_head=dict(
        type="YOLOV3Head",
        num_classes=10,
        in_channels=[512, 256, 128],
        out_channels=[1024, 512, 256],
        anchor_generator=dict(
            type="YOLOAnchorGenerator",
            base_sizes=[
                [(116, 90), (156, 198), (373, 326)],
                [(30, 61), (62, 45), (59, 119)],
                [(10, 13), (16, 30), (33, 23)],
            ],
            strides=[32, 16, 8],
        ),
        bbox_coder=dict(type="YOLOBBoxCoder"),
        featmap_strides=[32, 16, 8],
        loss_cls=dict(
            type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0, reduction="sum"
        ),
        loss_conf=dict(
            type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0, reduction="sum"
        ),
        loss_xy=dict(
            type="CrossEntropyLoss", use_sigmoid=True, loss_weight=2.0, reduction="sum"
        ),
        loss_wh=dict(type="MSELoss", loss_weight=2.0, reduction="sum"),
    ),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type="GridAssigner", pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0
        )
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        conf_thr=0.005,
        nms=dict(type="nms", iou_threshold=0.45),
        max_per_img=100,
    ),
)

data_root = "data/coco/"
dataset_type = "CocoDataset"

img_norm_cfg = dict(
    mean=[123.27423945, 117.1344846, 109.78107765],
    std=[54.2332011, 53.670966899999996, 55.038554850000004],
    to_rgb=True,
)
# albu_train_transforms = [
#     # dict(type="CLAHE", clip_limit=23.0, tile_grid_size=(8, 8), always_apply=False, p=0.5),
#     # dict(type="Emboss", alpha=(0.2, 1.0), strength=(0.5, 2.0), always_apply=False, p=0.5),
#     # dict(
#     #     type="ShiftScaleRotate", shift_limit=0.0625, scale_limit=0.0, rotate_limit=0, interpolation=1, p=0.5
#     # ),
#     # dict(type="RandomBrightnessContrast", brightness_limit=[0.1, 0.3], contrast_limit=[0.1, 0.3], p=0.2),
#     # dict(
#     #     type="OneOf",
#     #     transforms=[
#     #         dict(type="RGBShift", r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=1.0),
#     #         dict(
#     #             type="HueSaturationValue", hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0
#     #         ),
#     #     ],
#     #     p=0.1,
#     # ),
#     # dict(type="JpegCompression", quality_lower=85, quality_upper=95, p=0.2),
#     # dict(type="ChannelShuffle", p=0.1),
#     # dict(
#     #     type="OneOf",
#     #     transforms=[dict(type="Blur", blur_limit=3, p=1.0), dict(type="MedianBlur", blur_limit=3, p=1.0)],
#     #     p=0.1,
#     # ),
# ]


img_scale = (608, 608)  # height, width

train_pipeline = [
    dict(type="Mosaic", img_scale=img_scale, pad_val=114.0),
    dict(
        type="RandomAffine",
        scaling_ratio_range=(0.1, 2),
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
    ),
    dict(type="MixUp", img_scale=img_scale, ratio_range=(0.8, 1.6), pad_val=114.0),
    # dict(type="YOLOXHSVRandomAug"),
    # dict(type="Resize", img_scale=img_scale, keep_ratio=True),
    dict(type="RandomFlip", flip_ratio=0.5),
    # According to the official implementation, multi-scale
    # training is not considered here but in the
    # 'mmdet/models/detectors/yolox.py'.
    dict(type="PhotoMetricDistortion"),
    dict(type="Normalize", **img_norm_cfg),
    # dict(
    #     type="Pad",
    #     pad_to_square=True,
    #     # If the image is three-channel, the pad value needs
    #     # to be set separately for each channel.
    #     pad_val=dict(img=(114.0, 114.0, 114.0)),
    # ),
    dict(type="Pad", size_divisor=32),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]


# train_pipeline = [
#     dict(type="LoadImageFromFile", to_float32=True),
#     dict(type="LoadAnnotations", with_bbox=True),
#     dict(
#         type="Expand",
#         mean=img_norm_cfg["mean"],
#         to_rgb=img_norm_cfg["to_rgb"],
#         ratio_range=(1, 2),
#     ),
#     dict(
#         type="MinIoURandomCrop",
#         min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
#         min_crop_size=0.3,
#     ),
#     dict(type="Resize", img_scale=[(512, 512), (1024, 1024)], keep_ratio=True),
#     dict(type="RandomFlip", flip_ratio=0.5),
#     dict(type="PhotoMetricDistortion"),
#     # dict(
#     #     type="Albu",
#     #     transforms=albu_train_transforms,
#     #     bbox_params=dict(
#     #         type="BboxParams",
#     #         format="coco",
#     #         label_fields=["gt_labels"],
#     #         min_visibility=0.0,
#     #         filter_lost_elements=True,
#     #     ),
#     #     keymap={"img": "image", "gt_masks": "masks", "gt_bboxes": "bboxes"},
#     #     update_pad_shape=False,
#     #     skip_img_without_anno=True,
#     # ),
#     dict(type="Normalize", **img_norm_cfg),
#     dict(type="Pad", size_divisor=32),
#     dict(type="DefaultFormatBundle"),
#     dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
# ]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(608, 608),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]


train_dataset = dict(
    # _delete_=True,
    type="MultiImageMixDataset",
    dataset=dict(
        type=dataset_type,
        ann_file="/data/ephemeral/home/dataset/new_train.json",
        img_prefix="/data/ephemeral/home/dataset/",
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="LoadAnnotations", with_bbox=True),
        ],
        filter_empty_gt=False,
    ),
    pipeline=train_pipeline,
)

data = dict(
    # _delete_=True,
    samples_per_gpu=8,
    workers_per_gpu=4,
    # persistent_workers=True,
    train=train_dataset,
    val=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/instances_val2017.json",
        img_prefix=data_root + "val2017/",
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/instances_val2017.json",
        img_prefix=data_root + "val2017/",
        pipeline=test_pipeline,
    ),
)


# optimizer
optimizer = dict(type="SGD", lr=0.001, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy="step",
    warmup="linear",
    warmup_iters=2000,  # same as burn-in in darknet
    warmup_ratio=0.1,
    step=[218, 246],
)
# runtime settings
runner = dict(type="EpochBasedRunner", max_epochs=40)
evaluation = dict(interval=1, metric=["bbox"])

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(base_batch_size=64)
