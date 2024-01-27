# Level 2 Object Detection
cv-06 <br>
김시웅 이동형 조형서 백광현 최수진 박정민
- - -
### How to use
1. `mmdetection/configs/_cv06_` 안에 사용하고자 하는 model, dataset, scheduler, runtime 이 담긴 custom config.py 를 넣는다.<br>
   - 자세한 것은 `use_mmdetection.md` 참고
    ```python
    ### mmdetection 3.x

    ## - 0. __base__
    # {model}_[model setting]_{backbone}_{neck}_[norm setting]_[misc]_[gpu x batch_per_gpu]_{schedule}_{dataset}
    _base_ = [
        '../_base_/models/faster-rcnn_r50_fpn.py',
        '../_base_/datasets/coco_detection.py',
        '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
    ]

    # model settings
    model = dict(
        type='FasterRCNN',

        # data normalize
        data_preprocessor=dict(),

        # backbone
        backbone=dict(),

        # neck
        neck=dict(),

        # heads -> num_classes 변경
        rpn_head=dict(),
        roi_head=dict(),

        # model training and testing settings
        train_cfg=dict(),
        test_cfg=dict())


    ## - dataset and evaluator

    # transform
    train_pipeline = []
    test_pipeline = []

    # dataloader
    train_dataloader = dict()

    val_dataloader = dict()

    test_dataloader = dict()

    val_evaluator = dict()

    test_evaluator = val_evaluator


    ## - Training and testing config (model train_cfg, test_cfg 와 별도)
    train_cfg = dict()
    val_cfg = dict(type='ValLoop')
    test_cfg = dict(type='TestLoop')


    ## - Optimization config
    # optimizer
    optim_wrapper = dict()

    # learning rate
    param_scheduler = []

    # Default setting for scaling LR automatically
    auto_scale_lr = dict(enable=False, base_batch_size=16)


    ## - hook
    default_scope = 'mmdet'

    default_hooks = dict()

    # custom hook - early stopping 등
    custom_hooks = []

    # env
    env_cfg = dict()

    # visualization
    vis_backends = [dict(type='LocalVisBackend'),
                    dict(type='WandbVisBackend')]
    visualizer = dict(
        # type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer'
        type="Visualizer", vis_backends=vis_backends
        )

    # log
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

2. `{project_dir}/yamls` 에 train에 사용될 인자들이 담긴 `.yaml` 파일을 저장<br><br>
3. **train**<br>
    ```bash
    python train_mmd.py --yaml {yaml file name}
    ```
4. **test**<br>
    ```bash
    python test_mmd.py --dir {pred dir name}
    ```
5. result analyze<br>
    ```bash
    python tool {mode} --yaml {analyze.yaml path}
    ```
