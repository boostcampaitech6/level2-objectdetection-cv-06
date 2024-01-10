_base_ = "../deformable_detr/deformable_detr_twostage_refine_r50_16x2_50e_coco.py"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
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
