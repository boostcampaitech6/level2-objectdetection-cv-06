# mmdetection
based on [mmdetection](https://github.com/open-mmlab/mmdetection) <br>
made by 김시웅, 백광현

## pipeline
**train**<br>
<br>
train_mmd.py >> yamls/train에 사용될 .yaml >> mmdetection/configs/\_cv06_ 내 config.py

**test**<br>
<br>
test_mmd.py >> yamls/test/test_mmd.yaml >> results/train/train_serial/.yaml


## 사용설명
### train
1. config 파일 제작<br>
train 할 모델에 대한 config.py 를 제작하여 mmdetection/configs/\_cv06_ 에 위치시킨다. 이 때, custom 한 config.py 의 이름은 ```{model}_[model setting]_{backbone}_{neck}_[norm setting]_[misc]_[gpu x batch_per_gpu]_{schedule}_{dataset}.py``` 를 따른다.
```python
## - 0. 상속 불러오기
# mmdetection/configs 안에 있는 모델 기준
_base_ = [
    "../_base_/models/faster_rcnn_r50_fpn.py",
    "../_base_/datasets/coco_detection.py",
    "../_base_/schedules/schedule_1x.py",
    "../_base_/default_runtime.py",
]

## - 1. dataset custom setting

# transform 정의
img_norm_cfg = dict(
    mean=[123.27423945, 117.1344846, 109.78107765],
    std=[54.2332011, 53.670966899999996, 55.038554850000004],
    to_rgb=True,
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="LoadAnnotations", with_bbox=True
    ),
    dict(type="Resize", img_scale=(512, 512), keep_ratio=True),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(512, 512),
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

# evaluator & dataset and dataloader 정의
data = dict(
    samples_per_gpu=2,  # Batch size of a single GPU
    workers_per_gpu=2,  # Worker to pre-fetch data for each single GPU
    train=dict(
        type=dataset_type,
        ann_file=data_root + "new_train.json",  # Path of annotation file
        img_prefix=data_root,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + "new_val.json",
        img_prefix=data_root,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + "test.json",
        img_prefix=data_root,
        pipeline=test_pipeline,
        samples_per_gpu=2,  # Batch size of a single GPU used in testing
    ),
)
evaluation = dict(interval=1, metric="bbox")


## - 2. model custom setting
# __base__ 에서 불러온 모델에 덮어씌울 부분을 아래에 정의.
# 모델 별로 backbone, neck, rpn 부분이 다르니 유의. 모델이 달라지면 dataset 에서 classes 설정을 또 다르게 해줘야 함. -> 모델 별로 config 를 각각 만들어야 하는 이유.
# backbone, neck, rpn, train_cfg, test_cfg 를 수정하면 위 0 단계 __base__ 에서 가져온 모델에 overwrite 된다.

model = dict(
    # Backbone
    backbone=dict(
        type="ResNeXt",
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=True),
        style="pytorch",
        init_cfg=dict(type="Pretrained", checkpoint="open-mmlab://resnext101_64x4d"),
    ),
    # model training and testing settings
    # 이 부분은 model에서 train, test 에 사용되는 설정.
    train_cfg=dict(
        rpn=dict(
            assigner=dict(  # Config of assigner
                type="MaxIoUAssigner",
                pos_iou_thr=0.7,  # IoU >= threshold 0.7 will be taken as positive samples
                neg_iou_thr=0.3,  # IoU < threshold 0.3 will be taken as negative samples
                min_pos_iou=0.3,  # The minimal IoU threshold to take boxes as positive samples
                match_low_quality=True,  # Whether to match the boxes under low quality
                ignore_iof_thr=-1,
            ),  # IoF threshold for ignoring bboxes
            sampler=dict(  # Config of positive/negative sampler
                type="RandomSampler",
                num=256,  # Number of samples
                pos_fraction=0.5,  # The ratio of positive samples in the total samples.
                neg_pos_ub=-1,  # The upper bound of negative samples based on the number of positive samples.
                add_gt_as_proposals=False,
            ),  # Whether add GT as proposals after sampling.
            allowed_border=-1,  # The border allowed after padding for valid anchors.
            pos_weight=-1,  # The weight of positive samples during training.
            debug=False,
        ),  # Whether to set the debug mode
        rpn_proposal=dict(  # The config to generate proposals during training
            nms_across_levels=False,  # Whether to do NMS for boxes across levels. Only work in `GARPNHead`, naive rpn does not support do nms cross levels.
            nms_pre=2000,  # The number of boxes before NMS
            nms_post=1000,  # The number of boxes to be kept by NMS. Only work in `GARPNHead`.
            max_per_img=1000,  # The number of boxes to be kept after NMS.
            nms=dict(  # Config of NMS
                type="nms", iou_threshold=0.7  # Type of NMS  # NMS threshold
            ),
            min_bbox_size=0,
        ),  # The allowed minimal box size
        rcnn=dict(  # The config for the roi rpns.
            assigner=dict(  # Config of assigner for second stage, this is different for that in rpn
                type="MaxIoUAssigner",  # Type of assigner, MaxIoUAssigner is used for all roi rpns for now. Refer to https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/task_modules/assigners/max_iou_assigner.py#L14 for more details.
                pos_iou_thr=0.5,  # IoU >= threshold 0.5 will be taken as positive samples
                neg_iou_thr=0.5,  # IoU < threshold 0.5 will be taken as negative samples
                min_pos_iou=0.5,  # The minimal IoU threshold to take boxes as positive samples
                match_low_quality=False,  # Whether to match the boxes under low quality (see API doc for more details).
                ignore_iof_thr=-1,
            ),  # IoF threshold for ignoring bboxes
            sampler=dict(
                type="RandomSampler",  # Type of sampler, PseudoSampler and other samplers are also supported. Refer to https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/task_modules/samplers/random_sampler.py#L14 for implementation details.
                num=512,  # Number of samples
                pos_fraction=0.25,  # The ratio of positive samples in the total samples.
                neg_pos_ub=-1,  # The upper bound of negative samples based on the number of positive samples.
                add_gt_as_proposals=True,
            ),  # Whether add GT as proposals after sampling.
            mask_size=28,  # Size of mask
            pos_weight=-1,  # The weight of positive samples during training.
            debug=False,
        ),
    ),  # Whether to set the debug mode
    # 여기서 nms, wbf 를 적용 가능
    test_cfg=dict(  # Config for testing hyperparameters for rpn and rcnn
        rpn=dict(  # The config to generate proposals during testing
            nms_across_levels=False,  # Whether to do NMS for boxes across levels. Only work in `GARPNHead`, naive rpn does not support do nms cross levels.
            nms_pre=1000,  # The number of boxes before NMS
            nms_post=1000,  # The number of boxes to be kept by NMS. Only work in `GARPNHead`.
            max_per_img=1000,  # The number of boxes to be kept after NMS.
            nms=dict(  # Config of NMS
                type="nms", iou_threshold=0.7  # Type of NMS  # NMS threshold
            ),
            min_bbox_size=0,
        ),  # The allowed minimal box size
        rcnn=dict(  # The config for the roi rpns.
            score_thr=0.05,  # Threshold to filter out boxes
            nms=dict(  # Config of NMS in the second stage
                type="nms", iou_thr=0.5  # Type of NMS
            ),  # NMS threshold
            max_per_img=100,  # Max number of detections of each image
            mask_thr_binary=0.5,
        ),
    ),
)  # Threshold of mask prediction


## - 3. schedule custom
# optimizer
optimizer = dict(type="SGD", lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy="step", warmup="linear", warmup_iters=500, warmup_ratio=0.001, step=[8, 11]
)

# trainer
runner = dict(type="EpochBasedRunner", max_epochs=12)


## - 4 .runtime custom
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        # dict(type='TensorboardLoggerHook')
    ],
)
# yapf:enable
custom_hooks = [dict(type="NumClassCheckHook")]

dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
workflow = [("train", 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = "fork"

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
```
2.  yaml 파일 제작<br>
   위 config 파일의 이름과 같은 이름으로 .yaml 파일을 만든다. 특정 모델의 config 파일을 하나 만들어 놓고, epoch, loss, optimizer, scheduler 등을 다르게 할 때 사용.<br>
   ```yaml
gpu_num : 0
seed : 666
train_dir : "dataset_path"
train_annotation_dir : "train_annotation_path"
val_annotation_dir : "val_annotation_path"

#wandb
wandb : True
wandb_project: "CV06-OD"
wandb_entity: "innovation-vision-tech"
wandb_run : "my_faster_rcnn_r50_fpn_1x_coco"
wandb_note : wandb_note

classes: ["General trash", "Paper", "Paper pack", "Metal", "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]

custom_batch_size: True
samples_per_gpu: 8
workers_per_gpu: 4

# mmdetection/configs 안의 경로
py_path: "_cv06_/my_faster_rcnn_r50_fpn_1x_coco.py"

# cfg 변경할 옵션들을 여기에 입력
cfg_options:

  # evaluation
  evaluation.interval: 1
  evaluation.metric: "bbox"
  evaluation.save_best: "bbox_mAP"
  checkpoint_config.max_keep_ckpts: 3
  checkpoint_config.interval: 1

  # early-stopping
  default_hooks.early_stopping:
    type: "EarlyStoppingHook"
    monitor: "coco/bbox_mAP"
    patience: 6
    min_delta: 0.005
```
<br>

3. train_mmd.py 실행
   ```bash
   python train_mmd.py -yaml {yaml 파일 이름}
   ```

4. wandb Artifacts - evaluation 탭에서 자신이 학습시킨 모델의 validation 결과를 확인

### test
1. test_mmd.py 실행
```bash
python test_mmd.py --yaml {train에 사용한 yaml 파일 이름}
```
2. results/pred/train_serial/images 에서 prediction 결과 확인<br>

3. results/pred/train_serial/analysis 에서 mAP 등 그래프 확인
