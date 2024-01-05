import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# from torch.optim.lr_scheduler import StepLR

# from torchvision import transforms, utils

import os, sys, shutil, itertools, random

import matplotlib.pyplot as plt

from datetime import datetime
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm

import wandb

from modules.utils import load_yaml, save_yaml


prj_dir = os.path.dirname(os.path.abspath(__file__))
# os.path.abspath(__file__)로 이 py파일의 절대 경로 찾기
# os.path.dirname()는 폴더 경로

sys.path.append(prj_dir)

sys.path.append("./mmdetection/")
# mmdetection 폴더 경로 추가


from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor
from mmdet.utils import get_device

import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    train_serial = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_result_dir = os.path.join(prj_dir, "results", "train", train_serial)
    os.makedirs(train_result_dir, exist_ok=True)
    # data dir

    # Load config
    config_path = os.path.join(prj_dir, "config", "train_mmd.yaml")
    config = load_yaml(config_path)
    shutil.copy(config_path, os.path.join(train_result_dir, "train_mmd.yaml"))

    annotation_dir = config["annotation_dir"]
    data_dir = config["train_dir"]
    # data_gen_dir = config['train_gen_dir']

    mmdconfig_dir = os.path.join(
        prj_dir, "mmdetection", "configs", config["config_dir"]
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu_num"])

    # config file 들고오기
    cfg = Config.fromfile(mmdconfig_dir)

    # dataset config 수정
    cfg.data.train.classes = config["classes"]
    cfg.data.train.img_prefix = data_dir
    cfg.data.train.ann_file = data_dir + "train.json"  # train json 정보

    # cfg.data.val.classes = config["classes"]
    # cfg.data.val.img_prefix = data_dir
    # cfg.data.val.ann_file = data_dir + 'test.json' # test json 정보
    # validation set 구성 후 이부분 수정?

    # cfg.data.test.classes = config["classes"]
    # cfg.data.test.img_prefix = data_dir
    # cfg.data.test.ann_file = data_dir + "test.json"  # test json 정보

    cfg.data.samples_per_gpu = config["samples_per_gpu"]
    cfg.seed = config["seed"]
    cfg.gpu_ids = [0]
    cfg.work_dir = train_result_dir
    cfg.device = get_device()

    for keys, value in config["custom_cfg"].items():
        keys = keys.split(".")
        temp = cfg
        for key in keys[:-1]:
            temp = temp[key]
        temp[keys[-1]] = value

    for v in config["custom_pipeline"]["train"]:
        if isinstance(v[2], list):
            cfg.data.train.pipeline[v[0]][v[1]] = tuple(v[2])
            # yaml은 tuple 표현이 안되는데 pipeline resize는 tuple로 받지 않으면 에러 발생
        else:
            cfg.data.train.pipeline[v[0]][v[1]] = v[2]
    # for v in config["custom_pipeline"]["val"]:
    #     if isinstance(v[2], list):
    #         cfg.data.val.pipeline[v[0]][v[1]] = tuple(v[2])
    #     else:
    #         cfg.data.val.pipeline[v[0]][v[1]] = v[2]
    # cfg.data.test.pipeline[1]["img_scale"] = [512, 512]

    # wandb
    cfg.log_config.hooks = [
        dict(type="TextLoggerHook"),
        dict(
            type="MMDetWandbHook",
            init_kwargs={
                "project": config["wandb_project"],
                "entity": config["wandb_entity"],
                "name": config["wandb_run"],
                "config": {
                    "optimizer": cfg.optimizer.type,
                    "learning_rate": cfg.optimizer.lr,
                    "architecture": cfg.model.type,
                    "dataset": cfg.dataset_type,
                    "n_epochs": cfg.runner.max_epochs,
                    # "loss": [cfg.model.rpn_head.loss_cls.type, cfg.model.rpn_head.loss_bbox.type, cfg.model.roi_head.bbox_head.loss_cls.type, cfg.model.roi_head.bbox_head.loss_cls.type],
                    "notes": config["wandb_note"],
                },
            },
            interval=10,
            log_checkpoint=True,
            log_checkpoint_metadata=True,
            num_eval_images=0,
        ),
    ]

    # build_dataset
    datasets = [build_dataset(cfg.data.train)]

    print(datasets[0])

    # 모델 build 및 pretrained network 불러오기
    model = build_detector(cfg.model)
    model.init_weights()

    # 모델 학습
    train_detector(model, datasets[0], cfg, distributed=False, validate=False)
    # @@ validate = True이면 지정한 validation set으로 validation 진행
