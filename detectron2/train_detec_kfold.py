
# Run only on the first execution
# !python setup.py build develop

import datetime
import json
import os
import copy
import torch
import detectron2
from detectron2.data import detection_utils as utils
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator
from detectron2.data import build_detection_test_loader, build_detection_train_loader
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import load_coco_json

from pycocotools.coco import COCO
from torch.utils.data import Dataset, Subset, random_split
from typing import Tuple, List
from PIL import Image
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from detectron2.structures import BoxMode
from skmultilearn.model_selection import iterative_train_test_split
from sklearn.model_selection import StratifiedKFold
import detectron2.data.transforms as T

global kfold
kfold = True

class Detectron2COCODataset(Dataset):
    def __init__(self, annotation, data_dir, val_ratio=0.2, kfold=True, num_folds=5):
        super().__init__()
        self.data_dir = data_dir
        self.coco = COCO(annotation)
        self.val_ratio = val_ratio
        self.labels = self.get_labels()
        self.kfold = kfold
        self.num_folds = num_folds

    def __len__(self):
        return len(self.coco.getImgIds())

    def __getitem__(self, index):
        image_id = self.coco.getImgIds(imgIds=index)
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.data_dir, image_info["file_name"])
        height, width = image_info["height"], image_info["width"]
        ann_ids = self.coco.getAnnIds(imgIds=image_info["id"])
        anns = self.coco.loadAnns(ann_ids)
        objs = []
        for ann in anns:
            obj = {
                "bbox": ann["bbox"],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": ann["category_id"],
            }
            objs.append(obj)
        record = {
            "file_name": image_path,
            "image_id": int(image_id[0]),
            "height": height,
            "width": width,
            "annotations": objs,
        }
        return record

    def get_labels(self):
        labels = []
        for image_id in self.coco.getImgIds():
            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            anns = self.coco.loadAnns(ann_ids)
            category_ids = [ann["category_id"] for ann in anns]
            most_common_id, _ = Counter(category_ids).most_common(1)[0]
            labels.append(most_common_id)
        return labels

    def split_dataset(self):
        if self.kfold:
            self.cv = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=666)

            for fold_ind, (train_idx, val_idx) in enumerate(self.cv.split([i for i in range(len(self))], self.labels)):
                train_set = Subset(self, train_idx)
                val_set = Subset(self, val_idx)
                yield train_set, val_set, fold_ind
        
        else:
            train_ids, val_ids = train_test_split(
                [i for i in range(len(self))], 
                test_size=self.val_ratio, 
                random_state=666, 
                stratify=self.labels
            )
            train_set = Subset(self, train_ids)
            val_set = Subset(self, val_ids)
            return train_set, val_set


# Example training loop
stratified_dataset = Detectron2COCODataset(annotation='../../dataset/train.json',
                                           data_dir='../../dataset/', val_ratio=0.2, kfold=True, num_folds=5)

# config 불러오기
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml'))

cfg.SEED = 666

cfg.DATALOADER.NUM_WORKERS = 8

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml')

cfg.SOLVER.IMS_PER_BATCH = 8
cfg.SOLVER.BASE_LR = 0.01

cfg.SOLVER.MAX_ITER = 13500
cfg.SOLVER.STEPS = (10500,12500)

cfg.SOLVER.CHECKPOINT_PERIOD = 4000

model_name = cfg.MODEL.WEIGHTS.split('/')[-1].split('.')[0]
dir = os.path.join('./output/', f"{model_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
os.makedirs(dir, exist_ok=True)

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10

cfg.TEST.EVAL_PERIOD = 1000

output_folder = os.path.join('./output_eval/', f"{model_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
model_name = cfg.MODEL.WEIGHTS.split('/')[-1].split('.')[0]


# Split the dataset
for fold_ind, (train_set, val_set, _) in enumerate(stratified_dataset.split_dataset()):

    # Set metadata for your classes
    thing_classes = ["General trash", "Paper", "Paper pack", "Metal", 
                    "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]

    train_name = f"coco_trash_train_{fold_ind}"
    val_name = f"coco_trash_val_{fold_ind}"
    
    if train_name not in DatasetCatalog.list():
        DatasetCatalog.register(train_name, lambda: train_set)
    if val_name not in DatasetCatalog.list():
        DatasetCatalog.register(val_name, lambda: val_set)

    MetadataCatalog.get(train_name).set(thing_classes=thing_classes)
    MetadataCatalog.get(val_name).set(thing_classes=thing_classes)

    # config 수정하기
    cfg.DATASETS.TRAIN = (train_name,)
    cfg.DATASETS.TEST = (val_name,)


    cfg.OUTPUT_DIR = f"{dir}/fold_{fold_ind}"  # original_output_dir은 원래의 출력 디렉토리 경로입니다.


    # mapper - input data를 어떤 형식으로 return할지 (따라서 augmnentation 등 데이터 전처리 포함 됨)

    def MyMapper(dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        # image_path, label = dataset_dict
        # image = utils.read_image(image_path, format='BGR')  
        image = utils.read_image(dataset_dict['file_name'], format='BGR')

        transform_list = [
            T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
            T.RandomBrightness(0.8, 1.8),
            T.RandomContrast(0.6, 1.3)
        ]
        
        image, transforms = T.apply_transform_gens(transform_list, image)
        
        dataset_dict['image'] = torch.as_tensor(image.transpose(2,0,1).astype('float32'))
        
        annos = [
            utils.transform_instance_annotations(obj, transforms, image.shape[:2])
            for obj in dataset_dict.pop('annotations')
            if obj.get('iscrowd', 0) == 0
        ]
        
        instances = utils.annotations_to_instances(annos, image.shape[:2])
        dataset_dict['instances'] = utils.filter_empty_instances(instances)

        return dataset_dict

    # trainer - DefaultTrainer를 상속
    class MyTrainer(DefaultTrainer):
        
        @classmethod
        def build_train_loader(cls, cfg, sampler=None):
            return build_detection_train_loader(
            cfg, mapper = MyMapper, sampler = sampler
            )
        
        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_folder=output_folder):
            if output_folder is None:
                os.makedirs(output_folder, exist_ok=True)
                
            return COCOEvaluator(dataset_name, cfg, False, output_folder)

    # train
    os.makedirs(cfg.OUTPUT_DIR, exist_ok = True)

    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
