import torch
import torch.nn as nn
from torch.nn import functional as F

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from modules.utils import intersection_over_union


def get_loss_function(loss_function_str: str):
    if loss_function_str == "YoloV1Loss":
        return YoloV1Loss


class YoloV1Loss(nn.Module):
    """
    Calculate the loss for yolo (v1) model
    """

    def __init__(self, S=7, B=2, C=10):
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")

        """
        S is split size of image (in paper 7),
        B is number of boxes (in paper 2),
        C is number of classes (Custom dataset is 10),
        """
        self.S = S  # 그리드 크기
        self.B = B  # bounding box 수
        self.C = C  # class 수
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        predictions = predictions.reshape(
            -1, self.S, self.S, self.C + self.B * 5
        )  # 7x7x20 feature map flatten

        # 예측한 2개의 bounding box의 IoU 계산
        # 첫번째 bounding box와 target과 iou 계산
        iou_b1 = intersection_over_union(
            predictions[..., 11:15], target[..., 11:15]
        )  # [..., 11:15] 첫번째 bounding box
        # 두번째 bounding box와 target과 iou 계산
        iou_b2 = intersection_over_union(
            predictions[..., 16:20], target[..., 11:15]
        )  # [..., 16:20] 두번째 bounding box
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        iou_maxes, bestbox = torch.max(ious, dim=0)  # bounding box 두개 중 더 큰 IoU를 가진 박스
        exists_box = target[..., 10].unsqueeze(3)  # 해당 grid cell에 ground-truth가 존재하는지 여부 (1 : 존재, 0 : 존재x)

        # ======================== #
        #     Localization Loss    #
        # ======================== #

        # box_predictions : IoU 더 큰 값의 bounding box
        box_predictions = exists_box * (  # ground-truth가 존재하면 예측
            (
                bestbox * predictions[..., 16:20]  # IoU가 더 큰 박스가 두번째 박스일 때
                + (1 - bestbox) * predictions[..., 11:15]  # IoU가 더 큰 박스가 첫번째 박스일 때
            )
        )

        box_targets = exists_box * target[..., 11:15]

        # width, height 루트 씌우기
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        # MSE loss
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        # ======================== #
        #      Confidence Loss     #
        # ======================== #

        # confidence loss는 object가 있을 때, 없을 때 나눠서 계산 (exists_box: object 존재 유무)

        ### For Object Loss ###

        # pred_box : IoU가 큰 box의 confidence score
        pred_box = bestbox * predictions[..., 15:16] + (1 - bestbox) * predictions[..., 10:11]
        # MSE Loss
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 10:11]),
        )

        ### For No Object Loss ###
        # object가 없을 때는 두개의 bounding box 모두 계산

        # 첫번째 bounding box의 MSE loss
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 10:11], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 10:11], start_dim=1),
        )
        # 두번째 bounding box의 MSE loss
        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 15:16], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 10:11], start_dim=1),
        )

        # ======================== #
        #    Classification Loss   #
        # ======================== #

        # MSE loss
        class_loss = self.mse(
            torch.flatten(
                exists_box * predictions[..., :10],
                end_dim=-2,
            ),
            torch.flatten(
                exists_box * target[..., :10],
                end_dim=-2,
            ),
        )

        # ======================== #
        #         Final Loss       #
        # ======================== #

        loss = (
            self.lambda_coord * box_loss  # localization loss
            + object_loss  # confidence loss (object 있을 때)
            + self.lambda_noobj * no_object_loss  # confidence loss (object 없을 때)
            + class_loss  # classification loss
        )

        return loss
