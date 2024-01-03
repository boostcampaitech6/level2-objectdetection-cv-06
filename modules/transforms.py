from torchvision import transforms

# from facenet_pytorch import MTCNN
# from rembg import remove
from PIL import Image
import numpy as np

# import mediapipe as mp
import cv2
# from torchvision.transforms import v2

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transform_function(transform_function_str, config):
    if transform_function_str == "baseTransform":
        return baseTransform(config)
    elif transform_function_str == "testTransform":
        return testTransform(config)
    # elif transform_function_str == "centerCrop_transform":
    #     return CCTransform(config)
    # elif transform_function_str == "faceCrop_transform":
    #     return FCTransform(config)
    # elif transform_function_str == "CCHFTransform":
    #     return CCHFTransform(config)
    # elif transform_function_str == "CCASTransform":
    #     return CCASTransform(config)
    # elif transform_function_str == "CCACTransform":
    #     return CCACTransform(config)

# 사용할 transform 정의
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


def baseTransform(config):
    return Compose(
        [
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
        ]
    )

def testTransform(config):
    return transforms.Compose(
        [
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
        ]
    )