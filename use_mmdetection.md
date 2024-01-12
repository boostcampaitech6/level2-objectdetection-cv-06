# mmdetection
based on [mmdetection](https://github.com/open-mmlab/mmdetection) <br>
mmdetection 3.x 기준으로 작성<br>
made by 김시웅, 백광현<br><br>
_documentation_<br>
mmdetection documentation : [mmdetection](https://mmdetection.readthedocs.io/en/latest/user_guides/index.html) <br>
mmengine documentation : [mmengine](https://mmengine.readthedocs.io/en/latest/get_started/introduction.html) <br>
mmcv documentation : [mmcv](https://mmcv.readthedocs.io/en/2.x/get_started/introduction.html)

## pipeline
*mmdetection에서 config registry(OpenMMLab) 를 활용하여 구조를 짜 놓았고, 대부분 mmengine 에 소스코드가 있습니다. transform 같은 이미지 처리에는 mmcv 에 소스코드가 있습니다.*

**train**<br>
<br>
train_mmd.py >> yamls/train에 사용될 .yaml >> mmdetection/configs/\_cv06_ 내 config.py

**test**<br>
<br>
test_mmd.py >> results/train/train_serial/.yaml >> results/pred/train_serial/submission.csv


## Details
### train
- config 파일 제작<br>
train 할 모델에 대한 config.py 를 제작하여 mmdetection/configs/\_cv06_ 에 위치시킨다. 이 때, custom 한 config.py 의 이름은 ```{model}_[model setting]_{backbone}_{neck}_[norm setting]_[misc]_[gpu x batch_per_gpu]_{schedule}_{dataset}.py``` 를 따른다.<br><br>
    -  \_\_base__<br>
    `mmdetection/configs` 디렉토리 아래에 있는 여러 모델들을 기준으로 다른 세팅들이 적용된 `.py` 파일 중 하나 선택 (학습시킬 모델)<br>
    각 모델들은 가장 기본의 세팅을 가진 `.py` 파일을 가지는데, 그 파일을 기준으로 \_\_base__ 를 가져오면 편하다.
        ```python
        ## - 0. 상속 불러오기
        # mmdetection/configs 안에 있는 모델 기준
        _base_ = [
            "../_base_/models/faster_rcnn_r50_fpn.py",
            "../_base_/datasets/coco_detection.py",
            "../_base_/schedules/schedule_1x.py",
            "../_base_/default_runtime.py",
        ]
        ```
        <br>
    - model 정의
      - \_\_base__ 에서 불러온 모델에 덮어씌울 부분을 정의. <br>
      - 모델 별로 backbone, neck, rpn 부분이 다르니 유의. 모델이 달라지면 `num_classes`에 class 수 설정을 우리에 맞게 세팅해줘야 한다.(모델 별로 config 를 각각 만들어야 하는 이유)<br>
      - custom config 에서 수정하면 \_\_base__ 에서 가져온 모델 위에 overwrite 된다.<br>
      - mmdetection 3.x 버전에서 model 안에 들어가야 하는 요소는 `data preprocessor`, `backbone`, `neck`, `bbox_head` 등이고, 모델이 train 할 때와 test 할 때의 setting을 `train_cfg`, `test_cfg` 에서 준다.(`model.eval()` 과 동일)
        ```python
        model = dict(
        type='FasterRCNN',
        # data normalize
        data_preprocessor=dict(
            type='DetDataPreprocessor',
            mean=[123.27423945, 117.1344846, 109.78107765],
            std=[54.2332011, 53.670966899999996, 55.038554850000004],
            bgr_to_rgb=True,
            pad_size_divisor=32),

        # backbone
        backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch',
            init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),

        # neck
        neck=dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            num_outs=5),

        # heads -> num_classes 변경
        rpn_head=dict(
            type='RPNHead',
            in_channels=256,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                scales=[8],
                ratios=[0.5, 1.0, 2.0],
                strides=[4, 8, 16, 32, 64]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[.0, .0, .0, .0],
                target_stds=[1.0, 1.0, 1.0, 1.0]),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        roi_head=dict(
            type='StandardRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32]),
            bbox_head=dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=10,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='L1Loss', loss_weight=1.0))),

        # model training and testing settings
        train_cfg=dict(
            rpn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    match_low_quality=True,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
            rpn_proposal=dict(
                nms_pre=2000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)),
        test_cfg=dict(
            rpn=dict(
                nms_pre=1000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.05,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=100)
            # soft-nms is also supported for rcnn testing
            # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
        ))
        ```
        <br>
    - dataset & dataloader & evaluator
      - transform
        - `torchvision.transform` 과 동일
        ```python
        train_pipeline = [
            dict(type='LoadImageFromFile', backend_args=backend_args),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', scale=(512, 512), keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PackDetInputs')
        ]
        test_pipeline = [
            dict(type='LoadImageFromFile', backend_args=backend_args),
            dict(type='Resize', scale=(512, 512), keep_ratio=True),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                        'scale_factor'))
        ]
        ```
        <br>

        - multi-scale 적용
        ```python
        train_pipeline = [
            dict(type='LoadImageFromFile', backend_args=backend_args),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='RandomResize',
                scale=[(1333, 480), (1333, 960)],
                keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PackDetInputs')
        ]
        ```
        <br>

        - autoaugment 적용
        ```python
        train_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='AutoAugment',
                policies=[[
                    dict(
                        type='Resize',
                        img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                                (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                                (736, 1333), (768, 1333), (800, 1333)],
                        multiscale_mode='value',
                        keep_ratio=True)
                ],
                        [
                            dict(
                                type='Resize',
                                img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                                multiscale_mode='value',
                                keep_ratio=True),
                            dict(
                                type='RandomCrop',
                                crop_type='absolute_range',
                                crop_size=(384, 600),
                                allow_negative_crop=True),
                            dict(
                                type='Resize',
                                img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                            (576, 1333), (608, 1333), (640, 1333),
                                            (672, 1333), (704, 1333), (736, 1333),
                                            (768, 1333), (800, 1333)],
                                multiscale_mode='value',
                                override=True,
                                keep_ratio=True)
                        ]]),
                dict(type='PackDetInputs',
                     meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape','scale_factor'))]
        ```
        <br>

        - TTA
        ```python
        tta_pipeline = [
                    dict(type='LoadImageFromFile'),
                    dict(
                        type='TestTimeAug',
                        transforms=[
                            [dict(type='Resize', img_scale=(1333, 800), keep_ratio=True)],
                            [dict(type='RandomFlip', flip_ratio=0.),
                            dict(type='RandomFlip', flip_ratio=1.)],
                            [dict(type='PackXXXInputs', keys=['img'])],
                        ])
        ]
        ```
      - dataloader
        - train, valid, test loader 정의
        - batch_size, num_workers 정의
        - class, annotation 정보는 `dataloader.dataset` 에 있어야 하며 `train_mmd.py`, `test_mmd.py` 에서 넘겨준다.
        - sampler, batch_sampler 등 정보는 공식문서 확인 [mmdetection](https://mmdetection.readthedocs.io/en/latest/user_guides/index.html)
        ```python
        train_dataloader = dict(
                batch_size=2,
                num_workers=2,
                persistent_workers=True,
                sampler=dict(type='DefaultSampler', shuffle=True),
                batch_sampler=dict(type='AspectRatioBatchSampler'),
                dataset=dict(
                    type=dataset_type,
                    data_root=data_root,
                    ann_file='new_train.json',
                    data_prefix=dict(img=''),
                    filter_cfg=dict(filter_empty_gt=True, min_size=32),
                    pipeline=train_pipeline,
                    backend_args=backend_args))
        ```
      - evaluator
        - val 단게에서 bbox mAP, class wise precision 등을 계산
        ```python
        val_evaluator = dict(
            type='CocoMetric',
            ann_file=data_root + 'new_val.json',
            metric='bbox',
            format_only=False,
            backend_args=backend_args)

        test_evaluator = val_evaluator
        ```
    - Training and testing config
      - model train_cfg, test_cfg 와 별도로 epoch, loop 방식 등을 지정합니다.
      ```python
      train_cfg = dict(
          type='EpochBasedTrainLoop',  # The training loop type. Refer to https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py
          max_epochs=12,  # Maximum training epochs
          val_begin=1,
          val_interval=1)  # Validation intervals. Run validation every epoch.
      val_cfg = dict(type='ValLoop')  # The validation loop type
      test_cfg = dict(type='TestLoop')  # The testing loop type
      ```
    - Optimization config
      - optimizer, learning_rate 을 지정합니다.
      ```python
      # optimizer
      optim_wrapper = dict(  # Optimizer wrapper config
          type='OptimWrapper',  # Optimizer wrapper type, switch to AmpOptimWrapper to enable mixed precision training.
          optimizer=dict(  # Optimizer config. Support all kinds of optimizers in PyTorch.
              type='SGD',  # Stochastic gradient descent optimizer
              lr=0.02,  # The base learning rate
              momentum=0.9,  # Stochastic gradient descent with momentum
              weight_decay=0.0001),  # Weight decay of SGD
          clip_grad=None,  # Gradient clip option. Set None to disable gradient clip.
          )

      # learning rate
      param_scheduler = [
          dict(
              type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
          dict(
              type='MultiStepLR',
              begin=0,
              end=12,
              by_epoch=True,
              milestones=[8, 11],
              gamma=0.1)
      ]

      # Default setting for scaling LR automatically
      #   - `enable` means enable scaling LR automatically
      #       or not by default.
      #   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
      auto_scale_lr = dict(enable=False, base_batch_size=16)
      ```
      - average mixed precision training
      ```python
      optim_wrapper = dict(
          type='AmpOptimWrapper', # Average Mixed Precision
          loss_scale=512.,
          optimizer=dict(
              type='SGD',
              lr=0.01,
              momentum=0.9,
              weight_decay=0.0001),
          clip_grad=dict(max_norm=35, norm_type=2)
          )
      ```
    - hook
      - mmdetection/engine/hooks 내부에 custom hook 을 만들 수 있습니다. `@HOOKS.register_module()` 으로 데코레이터를 달아주면 됩니다.
      ```python
      default_scope = 'mmdet'

      default_hooks = dict(
          timer=dict(type='IterTimerHook'),
          logger=dict(type='LoggerHook', interval=50, out_suffix='.json'),
          param_scheduler=dict(type='ParamSchedulerHook'),
          checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3, save_best='coco/bbox_mAP'),
          sampler_seed=dict(type='DistSamplerSeedHook'),
          visualization=dict(type='DetVisualizationHook', draw=True, interval=100))

      # custom hook - early stopping 등. test 때는 SubmissionHook 을 추가(test_mmd.py)
      custom_hooks = [dict(
          type="EarlyStoppingHook",
          min_delta=0.005,
          monitor='coco/bbox_mAP',
          patience=6),
          dict(type="NumClassCheckHook"),
          dict(type='EmptyCacheHook', after_epoch=True),
      ]
      ```
    - 기타 env
      - gpu, log, visualization 관련 설정
      ```python
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
      load_from = None

      # 다시 학습 재개하려는 pth 주소
      resume = False
      ```
      <br>

- train, test 시 config 정보 갱신
  - config 파일을 읽어온 후 `train_mmd.py`, `test_mmd.py` 에서 config 파일 정보를 원하는 setting에 맞춰 갱신합니다. 아래는 train 예시
  ```python
  cfg = Config.fromfile(mmdconfig_dir)

  # dataset train
  # 여기서 palette 로 나중에 특정한 것만 눈에 띄게 표시 가능
  cfg.train_dataloader.dataset.metainfo = dict(classes=yaml["classes"])
  cfg.train_dataloader.dataset.data_root = data_dir
  cfg.train_dataloader.dataset.ann_file = train_annotation_dir

  # dataset val
  cfg.val_dataloader.dataset.metainfo = dict(classes=yaml["classes"])
  cfg.val_dataloader.dataset.data_root = data_dir
  cfg.val_dataloader.dataset.ann_file = val_annotation_dir

  # batch_size, num_workers
  cfg.train_dataloader.batch_size = yaml["batch_size"]
  cfg.train_dataloader.num_workers = yaml["num_workers"]

  # evaluator
  cfg.val_evaluator.ann_file = val_annotation_dir
  cfg.val_evaluator.classwise = True

  # visualization -> prediction 결과를 저장하려면 draw=True.
  cfg.default_hooks.visualization = dict(
      type="DetVisualizationHook", draw=True, interval=100
  )

  # 추가 config 수정
  for key, value in yaml["cfg_options"].items():
      if isinstance(value, list):
          yaml["cfg_options"][key] = tuple(value)
  cfg.merge_from_dict(yaml["cfg_options"])

  # 기타 설정
  cfg.randomness = dict(seed=yaml["seed"], deterministic=False, diff_rank_seed=False)
  cfg.gpu_ids = [0]
  cfg.work_dir = train_result_dir

  # wandb -> local 에도 저장하려면 LocalVisBackend 까지 vis_backends에 넣으면 됩니다.
  cfg.visualizer = dict(
      type="DetLocalVisualizer",
      vis_backends=[
          # dict(type='LocalVisBackend'),
          dict(
              type="WandbVisBackend",
              init_kwargs={
                  "project": yaml["wandb_project"],
                  "entity": yaml["wandb_entity"],
                  "name": yaml["wandb_run"],
                  "config": {
                      "optimizer": cfg.optim_wrapper.optimizer.type,
                      "learning_rate": cfg.optim_wrapper.optimizer.lr,
                      "architecture": cfg.model.type,
                      "dataset": cfg.train_dataloader.dataset.type,
                      "n_epochs": cfg.train_cfg.max_epochs,
                      "notes": yaml["wandb_note"],
                  },
              },
              commit=True,
          )
      ],
  )
  ```

- yaml 파일 제작<br>
   위 config 파일의 이름과 같은 이름으로 .yaml 파일을 만든다. 특정 모델의 config 파일을 하나 만들어 놓고 그 모델과 같은 이름으로 yaml 파일을 만들어 줍니다.<br>
   ```yaml
    ## config.py 에 지정되지 않은 것을 지정하거나 수정

    gpu_num : 0
    seed : 666
    train_dir : "/data/ephemeral/home/dataset/"
    train_annotation_dir : "/data/ephemeral/home/dataset/new_train.json"
    val_annotation_dir : "/data/ephemeral/home/dataset/new_val.json"

    # class
    classes: ["General trash", "Paper", "Paper pack", "Metal", "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]

    # dataloader batch
    batch_size: 8
    num_workers: 4

    # mmdetection/configs 안의 경로
    py_path: "_cv06_/custom_format.py"

    #wandb
    wandb : True
    wandb_project: "CV06-OD"
    wandb_entity: "innovation-vision-tech"
    wandb_run : "test_mmdetection_3.x"
    wandb_note : wandb_note

    # 학습 재개(resume), 모델 경로(resume_from) specify checkpoint
    resume: "auto"
    resume_from: None

    # cfg 변경할 옵션들을 여기에 입력
    cfg_options:

    # epoch
    train_cfg.max_epochs: 2

    # evaluation
    train_cfg.val_interval: 1
    val_evaluator.metric: "bbox"
    ```

- `train_mmd.py` 실행
  - `train_mmd.py` 에 config 를 수정하는 항목들이 있으니 유의. (결과 저장, 시각화 등등)
   ```bash
   python train_mmd.py --yaml {yaml 파일 이름}
   ```

### test
- test_mmd.py 실행
    ```bash
    python test_mmd.py --dir {train 후 생긴 train_serial 이름}
    ```

### analyze
- results/pred/train_serial/images 에서 prediction 결과 확인<br>

- `tools/` 안에 있는 결과분석 코드 활용
  - plot_curve(loss, bbox_mAP), confusion matrix, flops 등 계산
  ```bash
  tools {mode} --yaml {/data/ephemeral/home/level2-objectdetection-cv-06/yamls/analyze.yaml}
  ```
