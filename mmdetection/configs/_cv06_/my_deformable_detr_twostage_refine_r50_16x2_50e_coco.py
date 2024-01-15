_base_ = "../_base_/default_runtime.py"
model = dict(
    type="DeformableDETR",
    backbone=dict(
        type="ResNeXt",
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=True),
        style="pytorch",
        init_cfg=dict(type="Pretrained", checkpoint="open-mmlab://resnext101_64x4d"),
    ),
    neck=dict(
        type="ChannelMapper",
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type="GN", num_groups=32),
        num_outs=4,
    ),
    bbox_head=dict(
        type="DeformableDETRHead",
        num_query=300,
        num_classes=10,
        in_channels=2048,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=True,
        transformer=dict(
            type="DeformableDetrTransformer",
            encoder=dict(
                type="DetrTransformerEncoder",
                num_layers=6,
                transformerlayers=dict(
                    type="BaseTransformerLayer",
                    attn_cfgs=dict(
                        type="MultiScaleDeformableAttention", embed_dims=256
                    ),
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=("self_attn", "norm", "ffn", "norm"),
                ),
            ),
            decoder=dict(
                type="DeformableDetrTransformerDecoder",
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type="DetrTransformerDecoderLayer",
                    attn_cfgs=[
                        dict(
                            type="MultiheadAttention",
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1,
                        ),
                        dict(type="MultiScaleDeformableAttention", embed_dims=256),
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=(
                        "self_attn",
                        "norm",
                        "cross_attn",
                        "norm",
                        "ffn",
                        "norm",
                    ),
                ),
            ),
        ),
        positional_encoding=dict(
            type="SinePositionalEncoding", num_feats=128, normalize=True, offset=-0.5
        ),
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0
        ),
        loss_bbox=dict(type="L1Loss", loss_weight=5.0),
        loss_iou=dict(type="GIoULoss", loss_weight=2.0),
    ),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type="HungarianAssigner",
            cls_cost=dict(type="FocalLossCost", weight=2.0),
            reg_cost=dict(type="BBoxL1Cost", weight=5.0, box_format="xywh"),
            iou_cost=dict(type="IoUCost", iou_mode="giou", weight=2.0),
        )
    ),
    test_cfg=dict(max_per_img=100),
)

data_root = "data/coco/"
dataset_type = "CocoDataset"

img_norm_cfg = dict(
    mean=[123.27423945, 117.1344846, 109.78107765],
    std=[54.2332011, 53.670966899999996, 55.038554850000004],
    to_rgb=True,
)

# img_scale = (608, 608)

albu_train_transforms = [
    dict(
        type="CLAHE", clip_limit=23.0, tile_grid_size=(8, 8), always_apply=False, p=0.5
    ),
    dict(
        type="Sharpen",
        alpha=(0.2, 1.0),
        lightness=(0.5, 1.0),
        always_apply=False,
        p=0.5,
    ),
    dict(
        type="Emboss", alpha=(0.2, 1.0), strength=(0.5, 2.0), always_apply=False, p=0.5
    ),
    # dict(
    #     type="ShiftScaleRotate", shift_limit=0.0625, scale_limit=0.0, rotate_limit=0, interpolation=1, p=0.5
    # ),
    # dict(type="RandomBrightnessContrast", brightness_limit=[0.1, 0.3], contrast_limit=[0.1, 0.3], p=0.2),
    # dict(
    #     type="OneOf",
    #     transforms=[
    #         dict(type="RGBShift", r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=1.0),
    #         dict(
    #             type="HueSaturationValue", hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0
    #         ),
    #     ],
    #     p=0.1,
    # ),
    # dict(type="JpegCompression", quality_lower=85, quality_upper=95, p=0.2),
    # dict(type="ChannelShuffle", p=0.1),
    dict(
        type="OneOf",
        transforms=[
            dict(type="Blur", blur_limit=3, p=1.0),
            dict(type="MedianBlur", blur_limit=3, p=1.0),
            dict(type="MotionBlur", blur_limit=3, p=1.0),
        ],
        p=0.1,
    ),
]


train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    # dict(
    #     type="Expand",
    #     mean=img_norm_cfg["mean"],
    #     to_rgb=img_norm_cfg["to_rgb"],
    #     ratio_range=(1, 2),
    # ),
    # dict(
    #     type="MinIoURandomCrop",
    #     min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    #     min_crop_size=0.3,
    # ),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(
        type="AutoAugment",
        policies=[
            [
                dict(
                    type="Resize",
                    img_scale=[
                        (360, 1024),
                        (392, 1024),
                        (424, 1024),
                        (456, 1024),
                        (488, 1024),
                        (520, 1024),
                        (552, 1024),
                        (584, 1024),
                        (616, 1024),
                        (648, 1024),
                        (680, 1024),
                    ],
                    multiscale_mode="value",
                    keep_ratio=True,
                )
            ],
            [
                dict(
                    type="Resize",
                    # The radio of all image in train dataset < 7
                    # follow the original impl
                    img_scale=[(300, 1024), (400, 1024), (500, 1024)],
                    multiscale_mode="value",
                    keep_ratio=True,
                ),
                dict(
                    type="RandomCrop",
                    crop_type="absolute_range",
                    crop_size=(320, 500),
                    allow_negative_crop=True,
                ),
                dict(
                    type="Resize",
                    img_scale=[
                        (360, 1024),
                        (392, 1024),
                        (424, 1024),
                        (456, 1024),
                        (488, 1024),
                        (520, 1024),
                        (552, 1024),
                        (584, 1024),
                        (616, 1024),
                        (648, 1024),
                        (680, 1024),
                    ],
                    multiscale_mode="value",
                    override=True,
                    keep_ratio=True,
                ),
            ],
        ],
    ),
    # dict(type="PhotoMetricDistortion"),
    # dict(
    #     type="Albu",
    #     transforms=albu_train_transforms,
    #     bbox_params=dict(
    #         type="BboxParams",
    #         format="coco",
    #         label_fields=["gt_labels"],
    #         min_visibility=0.0,
    #         filter_lost_elements=True,
    #     ),
    #     keymap={"img": "image", "gt_masks": "masks", "gt_bboxes": "bboxes"},
    #     update_pad_shape=False,
    #     skip_img_without_anno=True,
    # ),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=1),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]
# test_pipeline, NOTE the Pad's size_divisor is different from the default
# setting (size_divisor=32). While there is little effect on the performance
# whether we use the default setting or use size_divisor=1.
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1024, 615),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=1),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

# train_dataset = dict(
#     # _delete_=True,
#     type="MultiImageMixDataset",
#     dataset=dict(
#         type=dataset_type,
#         ann_file="/data/ephemeral/home/dataset/new_train.json",
#         img_prefix="/data/ephemeral/home/dataset/",
#         pipeline=[
#             dict(type="LoadImageFromFile"),
#             dict(type="LoadAnnotations", with_bbox=True),
#         ],
#         filter_empty_gt=False,
#     ),
#     pipeline=train_pipeline,
# )

data = dict(
    # _delete_=True,
    samples_per_gpu=2,
    workers_per_gpu=2,
    # persistent_workers=True,
    train=dict(
        type=dataset_type,
        ann_file=data_root
        + "annotations/instances_val2017.json",  # Path of annotation file
        img_prefix=data_root + "val2017/",
        pipeline=train_pipeline,
    ),
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
optimizer = dict(
    type="AdamW",
    lr=2e-4,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            "backbone": dict(lr_mult=0.1),
            "sampling_offsets": dict(lr_mult=0.1),
            "reference_points": dict(lr_mult=0.1),
        }
    ),
)
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
lr_config = dict(policy="step", step=[40, 80])
runner = dict(type="EpochBasedRunner", max_epochs=100)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (16 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=32)
