_base_ = "../deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_coco.py"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

albu_train_transforms = [
    dict(
        type="CLAHE", clip_limit=23.0, tile_grid_size=(8, 8), always_apply=False, p=0.5
    ),
    # dict(type="Emboss", alpha=(0.2, 1.0), strength=(0.5, 2.0), always_apply=False, p=0.5),
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
    # dict(
    #     type="OneOf",
    #     transforms=[dict(type="Blur", blur_limit=3, p=1.0), dict(type="MedianBlur", blur_limit=3, p=1.0)],
    #     p=0.1,
    # ),
]

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="Expand",
        mean=img_norm_cfg["mean"],
        to_rgb=img_norm_cfg["to_rgb"],
        ratio_range=(1, 2),
    ),
    dict(
        type="MinIoURandomCrop",
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3,
    ),
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
    dict(type="PhotoMetricDistortion"),
    dict(
        type="Albu",
        transforms=albu_train_transforms,
        bbox_params=dict(
            type="BboxParams",
            format="coco",
            label_fields=["gt_labels"],
            min_visibility=0.0,
            filter_lost_elements=True,
        ),
        keymap={"img": "image", "gt_masks": "masks", "gt_bboxes": "bboxes"},
        update_pad_shape=False,
        skip_img_without_anno=True,
    ),
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

lr_config = dict(policy="step", step=[40, 80])
runner = dict(type="EpochBasedRunner", max_epochs=100)
