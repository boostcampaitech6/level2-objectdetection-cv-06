import os
import sys
import shutil

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm

from datetime import datetime
from utils import load_yaml, save_yaml

import argparse
import wandb

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
    # 학습 결과 파일로 results/train/학습시작시간 dir
    train_serial = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_result_dir = os.path.join(prj_dir, "results", "train", train_serial)
    os.makedirs(train_result_dir, exist_ok=True)

    # Load yaml
    parser = argparse.ArgumentParser(description="Train MMDetection model")
    parser.add_argument("--yaml", type=str, help="yaml file name for this train")

    args = parser.parse_args()

    yaml_name = args.yaml
    yaml_path = os.path.join(prj_dir, "yamls", yaml_name)
    yaml = load_yaml(yaml_path)
    shutil.copy(yaml_path, os.path.join(train_result_dir, "train_mmd.yaml"))

    train_annotation_dir = yaml["train_annotation_dir"]
    val_annotation_dir = yaml["val_annotation_dir"]
    data_dir = yaml["train_dir"]
    # data_gen_dir = yaml['train_gen_dir']

    os.environ["CUDA_VISIBLE_DEVICES"] = str(yaml["gpu_num"])

    mmdconfig_dir = os.path.join(prj_dir, "mmdetection", "configs", yaml["py_path"])

    cfg = Config.fromfile(mmdconfig_dir)

    # dataset train
    cfg.data.train.classes = yaml["classes"]
    cfg.data.train.img_prefix = data_dir
    cfg.data.train.ann_file = train_annotation_dir  # train json 정보

    # mosaic 사용시만 이 코드 사용?
    # cfg.data.train.dataset.classes = yaml["classes"]
    # cfg.data.train.dataset.img_prefix = data_dir
    # cfg.data.train.dataset.ann_file = train_annotation_dir  # train json 정보

    # dataset val
    cfg.data.val.classes = yaml["classes"]
    cfg.data.val.img_prefix = data_dir
    cfg.data.val.ann_file = val_annotation_dir

    # batch_size, num_workers
    if yaml["custom_batch_size"]:
        cfg.data.samples_per_gpu = yaml["samples_per_gpu"]
        cfg.data.workers_per_gpu = yaml["workers_per_gpu"]

    # 기타 설정
    cfg.seed = yaml["seed"]
    cfg.gpu_ids = [0]
    cfg.work_dir = train_result_dir
    cfg.device = get_device()

    for key, value in yaml["cfg_options"].items():
        if isinstance(value, list):
            yaml["cfg_options"][key] = tuple(value)
            # source code에 assert isinstance(img_scale, tuple)와 같이
            # tuple이 아니면 에러가 발생하는 부분들이 있는데 yaml은 tuple을 지원안해서 추가한 코드

        cfg.merge_from_dict(yaml["cfg_options"])

    # wandb
    cfg.log_config.hooks = [
        dict(type="TextLoggerHook"),
        dict(
            type="MMDetWandbHook",
            init_kwargs={
                "project": yaml["wandb_project"],
                "entity": yaml["wandb_entity"],
                "name": yaml["wandb_run"],
                "config": {
                    "optimizer": cfg.optimizer.type,
                    "learning_rate": cfg.optimizer.lr,
                    "architecture": cfg.model.type,
                    "dataset": cfg.dataset_type,
                    "n_epochs": cfg.runner.max_epochs,
                    # "loss": [cfg.model.rpn_head.loss_cls.type, cfg.model.rpn_head.loss_bbox.type, cfg.model.roi_head.bbox_head.loss_cls.type, cfg.model.roi_head.bbox_head.loss_cls.type],
                    "notes": yaml["wandb_note"],
                },
            },
            interval=10,
            # 이미지 시각화 부분인데 0이 아닌 다른값을 두면 버그 발생해서 validation 진행 불가능
            num_eval_images=100,
            bbox_score_thr=0.05,
        ),
    ]

    # config 저장
    with open(os.path.join(train_result_dir, "config.txt"), "w") as file:
        file.write(cfg.pretty_text)

    # build_dataset
    datasets = [build_dataset(cfg.data.train)]

    print(datasets[0])

    # 모델 build 및 pretrained network 불러오기
    model = build_detector(cfg.model)
    model.init_weights()

    # 모델 학습
    # @@ validate = True이면 지정한 validation set으로 validation 진행
    train_detector(model, datasets[0], cfg, distributed=False, validate=True)
