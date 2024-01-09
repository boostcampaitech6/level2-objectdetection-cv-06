import torch
from torchvision import transforms

import numpy as np
import pandas as pd
import os, sys, random
from tqdm import tqdm
from datetime import datetime

from modules.utils import load_yaml, save_yaml

# from yamls.custom_config_handler import get_custom_cfgs

prj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(prj_dir)

sys.path.append("./mmdetection/")
# mmdetection 폴더 경로 추가

import mmcv
from mmcv import Config
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor
from mmdet.models import build_detector
from mmdet.apis import single_gpu_test
from mmdet.utils import get_device
from mmcv.runner import load_checkpoint

from mmcv.parallel import MMDataParallel

from pycocotools.coco import COCO

import argparse
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # Load yaml
    parser = argparse.ArgumentParser(description="Test MMDetection model")
    parser.add_argument("--yaml", type=str, help="yaml file name for this test")

    args = parser.parse_args()

    yaml_name = args.yaml

    yaml = load_yaml(os.path.join(prj_dir, "yamls", "test_mmd.yaml"))
    train_yaml = load_yaml(
        os.path.join(prj_dir, "results", "train", yaml["train_serial"], f"{yaml_name}")
    )

    pred_serial = yaml["train_serial"] + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    # Device set
    os.environ["CUDA_VISIBLE_DEVICES"] = str(yaml["gpu_num"])
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # result_dir
    pred_result_dir = os.path.join(prj_dir, "results", "pred", pred_serial)
    img_result_dir = os.path.join(pred_result_dir, "images")
    os.makedirs(pred_result_dir, exist_ok=True)

    annotation_dir = yaml["annotation_dir"]
    data_dir = yaml["test_dir"]

    check_point_path = os.path.join(
        prj_dir, "results", "train", yaml["train_serial"], yaml["pth_name"]
    )

    mmdconfig_dir = os.path.join(
        prj_dir, "mmdetection", "configs", train_yaml["py_path"]
    )

    cfg = Config.fromfile(mmdconfig_dir)

    cfg.data.test.classes = yaml["classes"]
    cfg.data.test.img_prefix = data_dir
    cfg.data.test.ann_file = annotation_dir
    cfg.data.test.test_mode = True

    if yaml["custom_batch_size"]:
        cfg.data.samples_per_gpu = yaml["samples_per_gpu"]
        cfg.data.workers_per_gpu = yaml["workers_per_gpu"]

    cfg.seed = yaml["seed"]
    cfg.gpu_ids = [0]
    cfg.work_dir = pred_result_dir
    cfg.device = get_device()

    for key, value in train_yaml["cfg_options"].items():
        if isinstance(value, list):
            train_yaml["cfg_options"][key] = tuple(value)
            # source code에 assert isinstance(img_scale, tuple)와 같이
            # tuple이 아니면 에러가 발생하는 부분들이 있는데 yaml은 tuple을 지원안해서 추가한 코드

    cfg.merge_from_dict(train_yaml["cfg_options"])

    # Save yaml
    save_yaml(os.path.join(pred_result_dir, "train_mmd.yaml"), train_yaml)
    save_yaml(os.path.join(pred_result_dir, "predict_mmd.yaml"), yaml)

    # build dataset & dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    print(dataset)

    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))  # build detector
    checkpoint = load_checkpoint(
        model, check_point_path, map_location="cpu"
    )  # ckpt load
    # cpu로 한번 보내는 이유?

    # print(f"==>> model.CLASSES: {model.CLASSES}")
    model.CLASSES = dataset.CLASSES
    print(f"==>> model.CLASSES: {model.CLASSES}")

    print(cfg.device)
    model = MMDataParallel(model.to(cfg.device), device_ids=[0])
    output = single_gpu_test(
        model, data_loader, show=True, out_dir=img_result_dir, show_score_thr=0.05
    )  # output 계산

    mmcv.dump(output, os.path.join(pred_result_dir, "result.pkl"))

    # submission 양식에 맞게 output 후처리
    prediction_strings = []
    file_names = []
    coco = COCO(cfg.data.test.ann_file)
    img_ids = coco.getImgIds()

    class_num = 10
    for i, out in enumerate(output):
        prediction_string = ""
        image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
        for j in range(class_num):
            for o in out[j]:
                prediction_string += (
                    str(j)
                    + " "
                    + str(o[4])
                    + " "
                    + str(o[0])
                    + " "
                    + str(o[1])
                    + " "
                    + str(o[2])
                    + " "
                    + str(o[3])
                    + " "
                )

        prediction_strings.append(prediction_string)
        file_names.append(image_info["file_name"])

    submission = pd.DataFrame()
    submission["PredictionString"] = prediction_strings
    submission["image_id"] = file_names

    save_path = os.path.join(pred_result_dir, f"output_{pred_serial}.csv")
    submission.to_csv(save_path, index=None)
    print(f"Inference Done! Inference result saved at {save_path}")
    print(submission.head())
