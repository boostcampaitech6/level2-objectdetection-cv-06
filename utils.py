import torch
import numpy as np
from collections import Counter
import os

import yaml


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def save_yaml(path, obj):
    with open(path, "w") as f:
        yaml.dump(obj, f, sort_keys=False)


# IOU 계산
def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        # boxes_preds[..., 0:4] = [...[[x_center, y_center, width, height], ...]...]
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        # 왼쪽 위 코너 좌표 = (x_center - width/2, y_center - height/2)
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        # 오른쪽 아래 좌표 = (x_center + width/2, y_center + height/2)
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]  # (N, 1)
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)
    # intersection의 왼쪽 위 코너(x1, y1), 오른쪽 아래 코너 (x2, y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    # intersection이 없을 때 x2 < x1, y2 < y1이 되어 음수값이 나올 수 있으므로
    # torch.clamp로 하한을 0으로 지정

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


# NMS 계산
def non_max_suppression(bboxes, iou_threshold, threshold, box_format="midpoint"):
    """
    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    # box의 confidence score가 threshold보다 작으면 제거
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    # box들을 confidence score 기준 내림차순 정렬
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        # 다른 클래스의 박스와 특정 threshold 이하의 박스
        # 다시말해서, 같은 클래스의 특정 threshold 이상의 박스를 없앤다.
        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]  # 다른 클래스의 박스이거나
            or intersection_over_union(  # IOU 계산은 0:class_pred,1:confidence를 제외한 bbox 4개값만 필요
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold  # 많이 안 겹치는 박스면 남긴다
            # ==> 같은 클래스이면서 많이 겹치는 박스는 제거
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


# MAP 계산
# https://ctkim.tistory.com/entry/mAPMean-Average-Precision-%EC%A0%95%EB%A6%AC
def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=10
):
    """
    Calculates mean average precision

    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes

    Returns:
        float: mAP value across all classes given a specific IoU threshold
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                # [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
                # [1]은 class_prediction
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])
        # gt는 [train_idx, class_prediction, prob_score, x1, y1, x2, y2] 꼴
        # => gt[0]는 train index
        # ==> train_idx:#of gt boxes in img[train_idx]

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        # Precision-recall 곡선은 confidence(box probabilities 0~100%) threshold 값이
        # 100부터 0까지 바뀔 때 변하는 recall과 그에 따른 precision 값을 확인한다
        # (x=recall, y=precision인 그래프)
        # 따라서 detections을 confidence 값 기준 내림차순으로 정렬하고
        # confidence 값이 높은 detection부터 순서대로
        # True Positive인지 False Positive인지 확인하고 그 결과의 누적합을 본다
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]
            # bbox[0]은 train_idx => 현재 detection과 동일한 이미지에서 나온 것만 남기기

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(  # IOU 계산은 0:train_idx,1:class_pred,2:confidence를 제외한 bbox 4개값만 필요
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # amount_bboxes는 train_idx:[0,0,...,0] 형태를 담은 dict
                    # []안의 0 갯수는 img[train_idx]의 gt box 갯수

                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        TP_cumsum = torch.cumsum(TP, dim=0)
        # TP는 confidence 점수가 높은 detection 부터 TP면 1 아니면 0이 들어있다
        # => cumsum(누적합)을 구하면 cumsum i번째에는
        # TP의 i번째 confidence 점수를 threshold로 둘 때의 누적 TP갯수가 들어있다
        FP_cumsum = torch.cumsum(FP, dim=0)
        # FP_cumsum은 threshold 별 누적 FP 갯수
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        # recall = TP / (TP+FN) = TP / total_true_bboxes
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        # precision = TP / (TP+FP)
        # recalls와 precisions의 i번째 자리에는
        # 동일한 confidence threshold 값일 때의 recall, precision값이 각각 들어있다

        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # x=recall, y=precision 인 precision-recall 곡선에 (x=0, y=1) 시작점을 torch.cat으로 추가

        average_precisions.append(torch.trapz(precisions, recalls))
        # torch.trapz for numerical integration
        # 그래프의 면적을 사다리꼴의 조합으로 근사
        # https://pytorch.org/docs/stable/generated/torch.trapezoid.html#torch.trapezoid
        # https://en.wikipedia.org/wiki/Trapezoidal_rule

    return sum(average_precisions) / len(average_precisions)
    # mean AP 계산


# 이미지의 예측된 box와 ground truth box들 얻는 함수
def get_bboxes(
    loader,
    model,
    iou_threshold,
    threshold,
    pred_format="cells",
    box_format="midpoint",
    device="cuda",
):
    all_pred_boxes = []
    all_true_boxes = []

    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0
    # loader에 들어있는 순서대로 이미지의 번호를 지정

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)
            # predictions에는 각 그리드마다 2개의 bbox

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)
        # bboxes는 batch_size 개의 list들을 담은 list
        # 원소인 각 list들은 49개의 [class번호, confidence, x, y, width, height]를 담고 있다
        # (cellboxes_to_boxes함수는 predictions과 달리 각 그리드마다 1개의 bbox만)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )
            # 동일 클래스인 bbox들끼리 비교해 일정 비율이상 겹치면서 confidence가 낮은 박스는 제거

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)
                # [class번호, confidence, x, y, width, height]를
                # [train_dix, class번호, confidence, x, y, width, height]로 변경하여 저장

            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes


# 이미지의 예측된 box와 ground truth box들 얻는 함수
def val_get_bboxes(
    all_pred_boxes,
    all_true_boxes,
    predictions,
    labels,
    batch_size,
    val_idx,
    iou_threshold,
    threshold,
    pred_format="cells",
    box_format="midpoint",
    device="cuda",
):
    # batch_size = predictions.shape[0]

    true_bboxes = cellboxes_to_boxes(labels)
    bboxes = cellboxes_to_boxes(predictions)
    # bboxes는 batch_size 개의 list들을 담은 list
    # 원소인 각 list들은 49개의 [class번호, confidence, x, y, width, height]를 담고 있다
    # (cellboxes_to_boxes함수는 predictions과 달리 각 그리드마다 1개의 bbox만)

    for idx in range(batch_size):
        nms_boxes = non_max_suppression(
            bboxes[idx],
            iou_threshold=iou_threshold,
            threshold=threshold,
            box_format=box_format,
        )
        # 동일 클래스인 bbox들끼리 비교해 일정 비율이상 겹치면서 confidence가 낮은 박스는 제거

        for nms_box in nms_boxes:
            all_pred_boxes.append([val_idx] + nms_box)
            # [class번호, confidence, x, y, width, height]를
            # [train_dix, class번호, confidence, x, y, width, height]로 변경하여 저장

        for box in true_bboxes[idx]:
            # many will get converted to 0 pred
            if box[1] > threshold:
                all_true_boxes.append([val_idx] + box)

        val_idx += 1

    return val_idx


def convert_cellboxes(predictions, S=7):
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios. Tried to do this
    vectorized, but this resulted in quite difficult to read
    code... Use as a black box? Or implement a more intuitive,
    using 2 for loops iterating range(S) and convert them one
    by one, resulting in a slower but more readable implementation.
    """

    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7, 7, 20)
    # 0~9는 class_preds, 10, 15는 각 박스 confidence score
    bboxes1 = predictions[..., 11:15]
    bboxes2 = predictions[..., 16:20]
    # 11~14, 16~19는 bboxes

    scores = torch.cat(
        (predictions[..., 10].unsqueeze(0), predictions[..., 15].unsqueeze(0)), dim=0
    )
    # predictions[..., 10]과 predictions[..., 15]은 (batch_size,7,7) 형태
    # (1,batch_size,7,7) 두개를 dim=0으로 합쳐서 (2,batch_size,7,7) 형태
    best_box = scores.argmax(0).unsqueeze(-1)
    # dim=0을 기준으로 argmax => batch안의 각 표본의 7x7개의 각 칸마다 2개의 confidence score를 비교
    # => 칸에서 bbox1의 confidence가 높으면 0, bbox2의 confidence가 높으면 1
    # ==> (batch_size,7,7) 형태를 unsqueeze(-1)로 (batch_size,7,7,1)로 변경
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # predictions[..., 10:11]로 표현하면 (batch_size,7,7)이 아니라 (batch_size,7,7,1) 형태가 되므로
    # best_box = scores.argmax(0)만 해도 동일한 결과가 나온다
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    # 각 칸에서 bbox1의 confidence가 더 크면 best_box = 0 아니면 1 => confidence 값이 더 큰 bbox만 남고 아닌 bbox는 * 0
    # best_box와 1은 broadcasting으로 bboxes1, bboxes2와 동일한 size로 변경된다
    # => [batch_size,7,7,4] 꼴

    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    # tensor.repeat(sizes)는 지정한 size만큼 tensor를 반복한 새 tensor 생성
    # => [0,1,2,3,4,5,6]을 하나의 원소e 처럼 보고 [batch_size, 7, 1] tensor에 입력
    # ==> 7x1꼴 [[e],[e],[e],[e],[e],[e],[e]]이 batch_size 개
    # ===> [[[0,1,2,3,4,5,6],[0,1,2,3,4,5,6],...,[0,1,2,3,4,5,6]], ...]로 (batch_size, 7, 7) 형태
    # ====> unsqueeze(-1) 하면 [[[[0],[1],[2],[3],[4],[5],[6]], [[0],[1],[2],[3],[4],[5],[6]], ...], ...]로 (batch_size, 7, 7, 1) 형태

    # x,y,width,height 모두 칸(cell)의 가로 세로 길이가 1이라고 볼 때 칸 내부에서 0~1 범위 값이므로
    # 이미지 전체의 가로와 세로가 1인 기준으로 바꾸려면 한 칸의 가로 세로를 1/7로 바꾸는 등의 변환이 필요하다
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    # => 칸 하나의 bestbox의 x좌표는 칸 안에서의 상대적인 0~1범위
    # => 열(column) 칸(cell) 번호를 더해주어 0~7 범위로 변경
    # ==> 1/7로 다시 곱해주면 범위는 0~1로 돌아오지만 이미지 가로 전체 범위에서 x좌표로 변환된다
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # best_boxes[..., 0]은 (batch_size, 7, 7)형태가 되므로
    # cell_indices의 (batch_size, 7, 7, 1)와 동일한 형태가 되려면
    # best_boxes[..., :1]을 써야한다
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    # cell_indices.permute(0, 2, 1, 3))는
    # [[[[0],[0],[0],[0],[0],[0],[0]], [[1],[1],[1],[1],[1],[1],[1]], [[2],[2],[2],[2],[2],[2],[2]],...], ...]
    # 여기서도 y좌표를 행(row) 칸 번호를 더해주어 0~7 범위로 변경 후 1/7을 곱해주어
    # 이미지 세로 전체 범위(0~1)에서의 y좌표로 변환
    w_y = 1 / S * best_boxes[..., 2:4]
    # width, height도 칸 내부에서의 0~1범위이므로 이미지 전체를 0~1범위로 볼 때는 1/7을 곱해주어야한다

    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    # (batch_size, 7, 7, 1)인 x,y와 (batch_size, 7, 7, 2)인 w_y를 dim=-1로 concatenate
    # => (batch_size, 7, 7, 4) 형태
    predicted_class = predictions[..., :10].argmax(-1).unsqueeze(-1)
    # (batch_size, 7, 7, 10)을 argmax(-1)로 확률이 가장 높은 class만 남기면
    # (batch_size, 7, 7) 형태, 즉 각 칸마다 class 한개 예측
    # unsqueeze로 (batch_size, 7, 7, 1) 형태로 변경
    best_confidence = torch.max(predictions[..., 10], predictions[..., 15]).unsqueeze(
        -1
    )
    # predictions[..., 10]과 predictions[..., 15]은 confidence score이고 (batch_size,7,7) 형태
    # 더 큰 confidence score 쪽을 저장
    # unsqueeze로 (batch_size, 7, 7, 1) 형태로 변경
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )
    # (batch_size, 7, 7, 1)형태인 predicted_class, best_confidence와 (batch_size, 7, 7, 4) 형태인 converted_bboxes를
    # dim=-1로 concatenate하면 (batch_size, 7, 7, 6) 형태가 된다

    return converted_preds


def cellboxes_to_boxes(out, S=7):
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    # bbox의 cell 내부 상대좌표를 이미지 전체 좌표로 변경한 (batch_size, 7, 7, 6) 형태의 tensor를
    # (batch_size, 7*7, 6) 로 형태를 변경
    # 0:class번호, 1:bbox confidence score, 2,3,4,5: x,y,width,height
    converted_pred[..., 0] = converted_pred[..., 0].long()
    # 0번 자리에는 class번호가 들어있음
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
            # bboxes의 원소 하나는 [class번호, confidence, x, y, width, height] 형태
            # len(bboxes) == s*s == 49
        all_bboxes.append(bboxes)
        # len(all_bboxes) == out.shape[0] == batch_size

    return all_bboxes
