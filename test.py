import torch
from torchvision import transforms

import numpy as np
import pandas as pd
import os, sys, random
from tqdm import tqdm
from datetime import datetime

from modules.transforms import get_transform_function
from modules.utils import load_yaml, save_yaml, cellboxes_to_boxes, non_max_suppression
from modules.datasets import TestDataset
from model.models import get_model

prj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(prj_dir)

import warnings

warnings.filterwarnings("ignore")
if __name__ == "__main__":
    # Load Yaml
    config = load_yaml(os.path.join(prj_dir, "config", "test.yaml"))
    train_config = load_yaml(
        os.path.join(prj_dir, "results", "train", config["train_serial"], "train.yaml")
    )

    pred_serial = (
        config["train_serial"] + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    # Set random seed, deterministic
    torch.cuda.manual_seed(train_config["seed"])
    torch.manual_seed(train_config["seed"])
    np.random.seed(train_config["seed"])
    random.seed(train_config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Device set
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config["gpu_num"])
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # result_dir
    pred_result_dir = os.path.join(prj_dir, "results", "pred", pred_serial)
    os.makedirs(pred_result_dir, exist_ok=True)

    annotation_dir = config["annotation_dir"]
    data_dir = config["test_dir"]

    transform = get_transform_function(config["transform"], config)

    test_dataset = TestDataset(annotation_dir, data_dir, transform)
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=False,
        drop_last=False,
    )

    if train_config["model_custom"]:
        model = get_model(train_config["model"]["architecture"])
        model = model(**train_config["model"]["args"])
    else:
        model = get_model(train_config["model"]["architecture"])
        model = model(
            train_config["model"]["architecture"], **train_config["model"]["args"]
        )
    model = model.to(device)

    print(f"Load model architecture: {train_config['model']['architecture']}")

    check_point_path = os.path.join(
        prj_dir, "results", "train", config["train_serial"], "best_model.pt"
    )
    # check_point = torch.load(check_point_path,map_location=torch.device("cpu"))
    # cpu로 두는 이유?
    check_point = torch.load(check_point_path, map_location=device)
    model.load_state_dict(check_point["model"])

    iou_threshold = 0.5
    threshold = 0.4
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # score_threshold = 0.05
    score_threshold = 0.4
    # confidence 점수가 이값보다 낮으면 삭제
    # => output.csv에 포함할지 말지 설정
    # ==> 값은 상황에 따라 바꾸는게 좋다
    # nms에서 threshold값보다 작은 박스를 삭제하고 있으므로 확인 필요
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    box_format = "midpoint"

    # Save config
    save_yaml(os.path.join(pred_result_dir, "train.yaml"), train_config)
    save_yaml(os.path.join(pred_result_dir, "predict.yaml"), config)

    model.eval()
    test_pred_boxes = []
    test_idx = 0

    with torch.no_grad():
        for iter, img in enumerate(tqdm(test_dataloader)):
            img = img.to(device)

            batch_size = img.shape[0]
            pred_value = model(img)
            bboxes = cellboxes_to_boxes(pred_value)

            for idx in range(batch_size):
                nms_boxes = non_max_suppression(
                    bboxes[idx],
                    iou_threshold=iou_threshold,
                    threshold=threshold,
                    box_format=box_format,
                )

                for nms_box in nms_boxes:
                    test_pred_boxes.append([test_idx] + nms_box)
                    # [test_idx, class번호, confidence, x, y, width, height]

                test_idx += 1

    prediction_strings = []
    file_names = []
    coco = test_dataset.coco

    cur_filename = ""
    prediction_string = ""

    # submission 파일 생성
    for i, output in enumerate(test_pred_boxes):
        # prediction_string = ''
        image_info = coco.loadImgs(coco.getImgIds(imgIds=output[0]))[0]
        if image_info["file_name"] != cur_filename:
            if i != 0:
                file_names.append(cur_filename)
                prediction_strings.append(prediction_string)
            cur_filename = image_info["file_name"]
            prediction_string = ""

        if output[2] > score_threshold:
            # label[1~10] -> label[0~9]
            xmin = (output[3] - output[5] / 2) * 1024
            ymin = (output[4] - output[6] / 2) * 1024
            xmax = (output[3] + output[5] / 2) * 1024
            ymax = (output[4] + output[6] / 2) * 1024
            prediction_string += (
                str(int(output[1]))
                + " "
                + str(output[2])
                + " "
                + str(xmin)
                + " "
                + str(ymin)
                + " "
                + str(xmax)
                + " "
                + str(ymax)
                + " "
            )
    submission = pd.DataFrame()
    submission["image_id"] = file_names
    submission["PredictionString"] = prediction_strings

    save_path = os.path.join(pred_result_dir, "output.csv")
    submission.to_csv(save_path, index=None)
    print(f"Inference Done! Inference result saved at {save_path}")
    print(submission.head())
