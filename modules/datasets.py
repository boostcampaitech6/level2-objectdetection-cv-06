import torch
from torch.utils.data import Dataset, Subset, random_split, WeightedRandomSampler
from torchvision.transforms import (
    Resize,
    ToTensor,
    Normalize,
    Compose,
    CenterCrop,
    ColorJitter,
)
import numpy as np
from collections import defaultdict
from typing import Tuple, List
from enum import Enum
from PIL import Image
import os, random

from pycocotools.coco import COCO

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import pandas as pd


def get_dataset_function(dataset_function_str):
    if dataset_function_str == "BaseDataset":
        return BaseDataset
    elif dataset_function_str == "StratifiedShuffleSplitDataset":
        return StratifiedShuffleSplitDataset


class BaseDataset(torch.utils.data.Dataset):
    # @@@@@@@@@@@@@@@@@@
    num_classes = 10
    # @@@@@@@@@@@@@@@@@@

    def __init__(self, annotation, data_dir, S=7, B=2, C=10, transforms=None, val_ratio=0.2):
        super().__init__()
        self.data_dir = data_dir

        # coco annotation 불러오기 (coco API)
        self.coco = COCO(annotation)

        # S x S grid 영역
        self.S = S

        # 각 그리드별 bounding box 개수
        self.B = B

        # class num
        self.C = C
        self.transforms = transforms

        self.val_ratio = val_ratio

    def __len__(self):
        return len(self.coco.getImgIds())

    def __getitem__(self, index):
        # 이미지 아이디 가져오기
        image_id = self.coco.getImgIds(imgIds=index)

        # 이미지 정보 가져오기
        image_info = self.coco.loadImgs(image_id)[0]

        # 이미지 로드
        img_path = os.path.join(self.data_dir, image_info["file_name"])
        image = Image.open(img_path)

        # 어노테이션 파일 로드
        ann_ids = self.coco.getAnnIds(imgIds=image_info["id"])
        anns = self.coco.loadAnns(ann_ids)

        # 박스 가져오기
        bbox = np.array([x["bbox"] for x in anns])

        # 레이블 가져오기
        labels = np.array([x["category_id"] for x in anns])

        # 박스 단위를 0~1로 조정
        boxes = []
        for box, label in zip(bbox, labels):
            boxes.append(
                [
                    label,
                    (box[0] + (box[2] / 2)) / 1024,
                    (box[1] + (box[3] / 2)) / 1024,
                    (box[2]) / 1024,
                    (box[3]) / 1024,
                ]
            )  # (x_mid, y_mid , width, height)

        boxes = torch.tensor(boxes)

        if self.transforms:
            image, boxes = self.transforms(image, boxes)

        # 그리드 단위로 변환
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            # i,j 는 박스가 위치하는 row, column을 의미
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i
            # x,y는 이미지 전체를 0~1 범위로 표현
            # self.S * x, self.S * y 는 이미지 전체를 0~7 범위로 표현
            # => x_cell, y_cell은 속하는 해당 칸(그리드)안에서의 위치를 0~1로 표현

            """
            Calculating the width and height of cell of bounding box,
            relative to the cell is done by the following, with
            width as the example:
            
            width_pixels = (width*self.image_width)
            cell_pixels = (self.image_width)
            
            Then to find the width relative to the cell is simply:
            width_pixels/cell_pixels, simplification leads to the
            formulas below.
            """
            # 높이, 너비 그리드
            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )
            # 전체 이미지에 비해 각 칸의 가로, 세로는 1/7
            # width, height는 전체 이미지 대비 값이므로 * 7을 해서 칸 대비 값으로 변경

            # If no object already found for specific cell i,j
            # Note: This means we restrict to ONE object
            # per cell!
            # 각 그리드당 박스 개수 하나로 제한
            if label_matrix[i, j, self.C] == 0:
                # 해당 그리드에 박스가 존재한다는 표시
                label_matrix[i, j, self.C] = 1

                # 박스 좌표 (그리드 단위)
                box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])

                label_matrix[i, j, self.C + 1 : self.C + 5] = box_coordinates

                # class label을 one-hot encoding으로 처리
                label_matrix[i, j, class_label] = 1

                # label_matrix[i,j, : ]은 self.C + 5 * self.B 차원
                # 0~ self.C-1까진 class label one-hot vector
                # self.C는 첫번째 박스 confidence score 자리 (label일 때는 0 또는 1만)
                # self.C+1~self.C+4 까지는 x_cell, y_cell, width_cell, height_cell
                # self.C+5는 두번째 박스 confidence score이지만 label일때는 사용안함
                # self.C+6~self.C+9도 동일하게 미사용

        return image, label_matrix

    def split_dataset(self) -> Tuple[Subset, Subset]:
        """데이터셋을 학습과 검증용으로 나누는 메서드
        데이터셋을 train 과 val 로 나눕니다,
        pytorch 내부의 torch.utils.data.random_split 함수를 사용하여 torch.utils.data.Subset 클래스 둘로 나눕니다.
        """
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        return train_set, val_set

class StratifiedShuffleSplitDataset(torch.utils.data.Dataset):
    num_classes = 10

    def __init__(self, annotation, data_dir, S=7, B=2, C=10, transforms=None, val_ratio=0.2):
        super().__init__()
        self.data_dir = data_dir

        # coco annotation 불러오기 (coco API)
        self.coco = COCO(annotation)

        # S x S grid 영역
        self.S = S
        # 각 그리드별 bounding box 개수
        self.B = B
        # class num
        self.C = C
        self.transforms = transforms

        self.val_ratio = val_ratio

    def __len__(self):
        return len(self.coco.getImgIds())

    def __getitem__(self, index):
        # 이미지 아이디 가져오기
        image_id = self.coco.getImgIds(imgIds=index)
        # 이미지 정보 가져오기
        image_info = self.coco.loadImgs(image_id)[0]
        # 이미지 로드
        img_path = os.path.join(self.data_dir, image_info["file_name"])
        image = Image.open(img_path)

        # 어노테이션 파일 로드
        ann_ids = self.coco.getAnnIds(imgIds=image_info["id"])
        anns = self.coco.loadAnns(ann_ids)

        # 박스 가져오기
        bbox = np.array([x["bbox"] for x in anns])

        # 레이블 가져오기
        labels = np.array([x["category_id"] for x in anns])

        # 박스 단위를 0~1로 조정
        boxes = []
        for box, label in zip(bbox, labels):
            boxes.append(
                [
                    label,
                    (box[0] + (box[2] / 2)) / 1024,
                    (box[1] + (box[3] / 2)) / 1024,
                    (box[2]) / 1024,
                    (box[3]) / 1024,
                ]
            )  # (x_mid, y_mid , width, height)

        boxes = torch.tensor(boxes)

        if self.transforms:
            image, boxes = self.transforms(image, boxes)

        # 그리드 단위로 변환
        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            # i,j 는 박스가 위치하는 row, column을 의미
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i
            # x,y는 이미지 전체를 0~1 범위로 표현
            # self.S * x, self.S * y 는 이미지 전체를 0~7 범위로 표현
            # => x_cell, y_cell은 속하는 해당 칸(그리드)안에서의 위치를 0~1로 표현

            """
            Calculating the width and height of cell of bounding box,
            relative to the cell is done by the following, with
            width as the example:
            
            width_pixels = (width*self.image_width)
            cell_pixels = (self.image_width)
            
            Then to find the width relative to the cell is simply:
            width_pixels/cell_pixels, simplification leads to the
            formulas below.
            """
            # 높이, 너비 그리드
            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )
            # 전체 이미지에 비해 각 칸의 가로, 세로는 1/7
            # width, height는 전체 이미지 대비 값이므로 * 7을 해서 칸 대비 값으로 변경

            # If no object already found for specific cell i,j
            # Note: This means we restrict to ONE object
            # per cell!
            # 각 그리드당 박스 개수 하나로 제한
            if label_matrix[i, j, self.C] == 0:
                # 해당 그리드에 박스가 존재한다는 표시
                label_matrix[i, j, self.C] = 1

                # 박스 좌표 (그리드 단위)
                box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])

                label_matrix[i, j, self.C + 1 : self.C + 5] = box_coordinates

                # class label을 one-hot encoding으로 처리
                label_matrix[i, j, class_label] = 1

                # label_matrix[i,j, : ]은 self.C + 5 * self.B 차원
                # 0~ self.C-1까진 class label one-hot vector
                # self.C는 첫번째 박스 confidence score 자리 (label일 때는 0 또는 1만)
                # self.C+1~self.C+4 까지는 x_cell, y_cell, width_cell, height_cell
                # self.C+5는 두번째 박스 confidence score이지만 label일때는 사용안함
                # self.C+6~self.C+9도 동일하게 미사용

        return image, label_matrix

    def split_dataset(self) -> Tuple[Subset, Subset]:
        """데이터셋을 학습과 검증용으로 나누는 메서드
        데이터셋을 train 과 val 로 나눕니다,
        train_test_split을 사용하여 클래스 비율을 유지하면서 데이터를 나눕니다.
        """

        var = [(ann['image_id'], ann['category_id']) for ann in self.coco.dataset['annotations']]
        X = np.array([v[0] for v in var]).reshape(-1, 1)  # image ids
        y = np.array([v[1] for v in var])  # category ids

        # train_test_split을 사용하여 데이터셋을 분할합니다.
        train_index, val_index = train_test_split(
            [i for i in range(len(self))], 
            test_size=self.val_ratio, 
            random_state=666, 
            stratify=y
        )

        # Subset을 생성하기 위해 인덱스를 list로 변환합니다.
        train_index = train_index.tolist()
        val_index = val_index.tolist()

        train_set = Subset(self, train_index)
        val_set = Subset(self, val_index)

        return train_set, val_set


class StratifiedDataset(torch.utils.data.Dataset):
    num_classes = 10

    def __init__(self, annotation, data_dir, S=7, B=2, C=10, transforms=None, val_ratio=0.2):
        super().__init__()
        self.data_dir = data_dir
        self.coco = COCO(annotation)
        self.S = S
        self.B = B
        self.C = C
        self.transforms = transforms
        self.val_ratio = val_ratio

    def __len__(self):
        return len(self.coco.getImgIds())

    def __getitem__(self, index):
        # 이미지 아이디 가져오기
        image_id = self.coco.getImgIds(imgIds=index)
        image_info = self.coco.loadImgs(image_id)[0]
        img_path = os.path.join(self.data_dir, image_info["file_name"])
        image = Image.open(img_path)

        ann_ids = self.coco.getAnnIds(imgIds=image_info["id"])
        anns = self.coco.loadAnns(ann_ids)
        bbox = np.array([x["bbox"] for x in anns])
        labels = np.array([x["category_id"] for x in anns])

        boxes = []
        for box, label in zip(bbox, labels):
            boxes.append(
                [
                    label,
                    (box[0] + (box[2] / 2)) / 1024,
                    (box[1] + (box[3] / 2)) / 1024,
                    (box[2]) / 1024,
                    (box[3]) / 1024,
                ]
            )

        boxes = torch.tensor(boxes)

        if self.transforms:
            image, boxes = self.transforms(image, boxes)

        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i
            width_cell, height_cell = (
                width * self.S,
                height * self.S,
            )

            if label_matrix[i, j, self.C] == 0:
                label_matrix[i, j, self.C] = 1
                box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                label_matrix[i, j, self.C + 1 : self.C + 5] = box_coordinates
                label_matrix[i, j, class_label] = 1

        return image, label_matrix

    def split_dataset(self) -> Tuple[Subset, Subset]:
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val

        # 이미지 아이디를 기준으로 훈련 데이터와 검증 데이터를 나눕니다.
        image_ids = [i for i in range(len(self))]
        train_ids, val_ids = train_test_split(image_ids, test_size=self.val_ratio, random_state=666, stratify=self.labels)

        # 훈련 데이터셋과 검증 데이터셋을 Subset으로 나눠서 반환합니다.
        train_set = Subset(self, train_ids)
        val_set = Subset(self, val_ids)
        
        return train_set, val_set

class TestDataset(Dataset):
    '''
      data_dir: data가 존재하는 폴더 경로
      transforms: data transform (resize, crop, Totensor, etc,,,)
    '''

    def __init__(self, annotation, data_dir, transforms=None):
        super().__init__()
        self.data_dir = data_dir
        # coco annotation 불러오기 (coco API)
        self.coco = COCO(annotation)

        self.transforms = transforms

    def __getitem__(self, index: int):
        
        image_id = self.coco.getImgIds(imgIds=index)

        image_info = self.coco.loadImgs(image_id)[0]
        
        # image = cv2.imread(os.path.join(self.data_dir, image_info['file_name']))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # image /= 255.0
        img_path = os.path.join(self.data_dir, image_info["file_name"])
        image = Image.open(img_path)

        ann_ids = self.coco.getAnnIds(imgIds=image_info['id'])
        anns = self.coco.loadAnns(ann_ids)

        # image = torch.tensor(image, dtype=torch.float32).permute(2,0,1)

        if self.transforms:
            image = self.transforms(image)
        
        # print(f"==>> image.shape: {image.shape}")

        return image
    
    def __len__(self) -> int:
        return len(self.coco.getImgIds())
