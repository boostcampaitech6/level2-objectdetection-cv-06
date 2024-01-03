import torch
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

import numpy as np
import pandas as pd


import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from modules.utils import mean_average_precision, get_bboxes


def get_metric_function(metric_function_str):
    """
    Add metrics, weights for weighted score
    """

    if metric_function_str == "mean_ap":
        return mean_ap


def mean_ap(loader, model, iou_threshold=0.5, threshold=0.4):
    # 학습된 model로 test dataset(== train_dataset)의 prediction box와 target box 생성
    pred_boxes, target_boxes = get_bboxes(loader, model, iou_threshold, threshold)

    # model이 얼마나 정확히 예측하였는지 mAP계산
    mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold, box_format="midpoint")
    return mean_avg_prec
