import cv2
import torch
import json
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch

exp_name = 'train01'

model = torch.hub.load('.', 'custom', path= f'/data/ephemeral/home/yolov5/runs/train/exp10/weights/best.pt', source='local', force_reload=True)
# model.conf = 0.001  # confidence threshold (0-1)
# model.iou = 0.6  # NMS IoU threshold (0-1)

prediction_string = ['']  * 4871
image_id = [f'test/{i:04}.jpg' for i in range(4871)]
for i in tqdm(range(4871)):
    img = Image.open(f'./data/images/{i:04}.jpg')

    results = model(img, size=1024, augment=True)
    for bbox in results.pandas().xyxy[0].values:
        xmin, ymin, xmax, ymax, confidence, cls, name = bbox
        prediction_string[i] += f'{cls} {confidence} {xmin} {ymin} {xmax} {ymax} '

raw_data ={
    'PredictionString' : prediction_string,
    'image_id' : image_id
}

dataframe = pd.DataFrame(raw_data)

dataframe.to_csv(f'./runs/val/submission_tta_cleaned_{exp_name}.csv', na_rep='NaN',index=None)