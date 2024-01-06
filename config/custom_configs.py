def get_custom_cfgs(cfg_str: str):
    if cfg_str == "faster_rcnn_r50_fpn":
        cfg_dir = "faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py"
        custum_cfg = {
            "model.roi_head.bbox_head.num_classes": 10,
            "evaluation": dict(interval=1, metric="bbox", save_best="bbox_mAP"),
            "optimizer_config.grad_clip": {"max_norm": 35, "norm_type": 2},
            "checkpoint_config": {"max_keep_ckpts": 3, "interval": 1},
            # max_keep_ckpts 값으로 지정한 개수만큼만 epoch pth 보관
        }
        custom_pipeline = {
            "train": [[2, "img_scale", (512, 512)]],
            "val": [[1, "img_scale", (512, 512)]],
            "test": [[1, "img_scale", (512, 512)]],
        }
    elif cfg_str == "yolov3_d53":
        cfg_dir = "yolo/yolov3_d53_mstrain-608_273e_coco.py"
        custum_cfg = {
            "model.bbox_head.num_classes": 10,
            "evaluation": dict(interval=1, metric="bbox", save_best="bbox_mAP"),
            "checkpoint_config": {"max_keep_ckpts": 3, "interval": 1},
        }
        custom_pipeline = {
            "train": [[4, "img_scale", [(256, 256), (512, 512)]]],
            "val": [[1, "img_scale", (512, 512)]],
            "test": [[1, "img_scale", (512, 512)]],
        }
    elif cfg_str == "detr_r50":
        cfg_dir = "detr/detr_r50_8x2_150e_coco.py"
        custum_cfg = {
            "model.bbox_head.num_classes": 10,
            "evaluation": dict(interval=1, metric="bbox", save_best="bbox_mAP"),
            "checkpoint_config": {"max_keep_ckpts": 3, "interval": 1},
        }
        custom_pipeline = {
            "train": [
                [
                    3,
                    "policies",
                    [
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
                ]
            ],
            "val": [[1, "img_scale", (1024, 615)]],
            "test": [[1, "img_scale", (1024, 615)]],
        }

    return cfg_dir, custum_cfg, custom_pipeline
