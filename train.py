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

from model.optimizers import get_optimizer
from model.losses import get_loss_function
from model.models import get_model

from modules.schedulers import get_scheduler

# from modules.datasets import BaseDataset
from modules.metrics import get_metric_function
from modules.datasets import get_dataset_function
from modules.transforms import get_transform_function
from modules.utils import load_yaml, save_yaml, mean_average_precision, val_get_bboxes

# from modules.logger import MetricAverageMeter, LossAverageMeter

prj_dir = os.path.dirname(os.path.abspath(__file__))
# os.path.abspath(__file__)로 이 py파일의 절대 경로 찾기
# os.path.dirname()는 폴더 경로

sys.path.append(prj_dir)

import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    train_serial = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_result_dir = os.path.join(prj_dir, "results", "train", train_serial)
    os.makedirs(train_result_dir, exist_ok=True)
    # data dir

    # Load config
    config_path = os.path.join(prj_dir, "config", "train.yaml")
    config = load_yaml(config_path)
    shutil.copy(config_path, os.path.join(train_result_dir, "train.yaml"))

    annotation_dir = config["annotation_dir"]
    data_dir = config["train_dir"]
    # data_gen_dir = config['train_gen_dir']

    # seed
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(config["seed"])
    random.seed(config["seed"])

    # wandb
    if config["wandb"]:
        wandb.init(
            project=config["wandb_project"],
            entity="innovation-vision-tech",
            config={
                "learning_rate": config["optimizer"]["args"]["lr"],
                "architecture": config["model"]["architecture"],
                "dataset": "GarbageDataset",
                "n_epochs": config["n_epochs"],
                "loss": config["loss"]["name"],
                "notes": config["wandb_note"],
            },
            name=config["wandb_run"],
        )
    else:
        wandb.init(mode="online")

    # wandb.run.name = config["wandb_run"]
    wandb.run.save()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu_num"])
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("device : ", device)

    transform = get_transform_function(config["transform"], config)
    print(transform)

    if config["dataset"] == "BaseDataset":
        dataset = get_dataset_function(config["dataset"])
        dataset = dataset(annotation_dir, data_dir, transforms=transform, val_ratio=config["val_size"])

        train_dataset, val_dataset = dataset.split_dataset()

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            drop_last=config["drop_last"],
            num_workers=config["num_workers"],
            pin_memory=use_cuda,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            drop_last=config["drop_last"],
            num_workers=config["num_workers"],
            pin_memory=use_cuda,
        )
    else:
        dataset = get_dataset_function(config["dataset"])
        dataset = dataset(
            data_dir,
            transform,
            val_ratio=config["val_size"],
            seed=config["seed"],
            drop_age_mode=config["drop_age_mode"],
            drop_age=config["drop_age"],
        )

        train_dataset, val_dataset = dataset.split_dataset()

        # train_sampler = dataset.get_sampler("train")
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            drop_last=config["drop_last"],
            num_workers=config["num_workers"],
            shuffle=config["shuffle"],
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            drop_last=config["drop_last"],
            num_workers=config["num_workers"],
            shuffle=False
        )

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    num_classes = dataset.num_classes
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    if config["model_custom"]:
        model = get_model(config["model"]["architecture"])
        model = model(**config["model"]["args"])
    else:
        model = get_model(config["model"]["architecture"])
        model = model(config["model"]["architecture"], **config["model"]["args"])
    model = model.to(device)
    print(f"Load model architecture: {config['model']['architecture']}")
    wandb.watch(model)

    optimizer = get_optimizer(optimizer_str=config["optimizer"]["name"])
    optimizer = optimizer(model.parameters(), **config["optimizer"]["args"])

    scheduler = get_scheduler(scheduler_str=config["scheduler"]["name"])
    scheduler = scheduler(optimizer=optimizer, **config["scheduler"]["args"])

    loss_func = get_loss_function(loss_function_str=config["loss"]["name"])
    loss_func = loss_func(**config["loss"]["args"])
    # loss_func = loss_func()

    metric_funcs = {metric_name: get_metric_function(metric_name) for metric_name in config["metrics"]}
    max_mean_ap_score = 0

    model.train()

    score_lst = ["mean_ap"]

    for epoch_id in range(config["n_epochs"]):
        tic = time()
        train_loss = 0
        train_scores = {metric_name: 0 for metric_name, _ in metric_funcs.items() if metric_name in score_lst}

        for iter, (img, label) in enumerate(tqdm(train_dataloader)):
            img = img.to(device)
            label = label.to(device)

            batch_size = img.shape[0]

            pred_value = model(img)

            loss = loss_func(pred_value, label)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item() / len(train_dataloader)

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # val mean_ap 계산 방식처럼 변경할지 고민 필요
        # mean_ap 계산
        train_scores["mean_ap"] = metric_funcs["mean_ap"](train_dataloader, model)
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        scheduler.step()

        # Validation
        valid_loss = 0
        valid_scores = {metric_name: 0 for metric_name, _ in metric_funcs.items() if metric_name in score_lst}

        # if (iter % 20 == 0) or (iter == len(qd_train_dataloader)-1):
        model.eval()
        toc = time()
        train_time = toc - tic

        val_idx = 0
        val_pred_boxes = []
        val_target_boxes = []
        for img, label in val_dataloader:
            ##fill##
            img = img.to(device)
            label = label.to(device)
            batch_size = img.shape[0]
            with torch.no_grad():
                pred_value = model(img)

            loss = loss_func(pred_value, label)

            valid_loss += loss.item() / len(val_dataloader)

            val_idx = val_get_bboxes(
                val_pred_boxes,
                val_target_boxes,
                pred_value,
                label,
                batch_size,
                val_idx,
                iou_threshold=0.5,
                threshold=0.4,
            )

        # val_mean_ap 계산
        val_mean_ap = mean_average_precision(
            val_pred_boxes, val_target_boxes, iou_threshold=0.5, box_format="midpoint"
        )

        valid_scores["mean_ap"] = val_mean_ap

        # print("Epoch [%4d/%4d] | Train Loss %.4f | Train Acc %.4f | Valid Loss %.4f | Valid Acc %.4f" %
        #     (epoch_id, config['n_epochs'], train_loss, train_acc, valid_loss, valid_acc))
        print(
            "Epoch [%4d/%4d] | Train Loss %.4f | Train mAP %.4f | Valid Loss %.4f | Valid mAP %.4f"
            % (
                epoch_id,
                config["n_epochs"],
                train_loss,
                train_scores["mean_ap"],
                valid_loss,
                valid_scores["mean_ap"],
            )
        )

        new_wandb_metric_dict = {
            "train_time": train_time,
            "train_loss": train_loss,
            "train_mean_ap": train_scores["mean_ap"],
            "valid_loss": valid_loss,
            "valid_mean_ap": valid_scores["mean_ap"],
        }
        wandb.log(new_wandb_metric_dict)

        if max_mean_ap_score < valid_scores["mean_ap"]:
            print(f"New best model for val mean_ap : {valid_scores['mean_ap']:2.4}! saving the best model..")
            check_point = {
                "epoch": epoch_id + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
            }

            torch.save(check_point, os.path.join(train_result_dir, f"model_{epoch_id}.pt"))
            torch.save(check_point, os.path.join(train_result_dir, f"best_model.pt"))
            early_stopping_count = 0
            max_mean_ap_score = valid_scores["mean_ap"]
        else:
            early_stopping_count += 1

        if early_stopping_count >= config["early_stopping_count"]:
            exit()

    # print(model)
