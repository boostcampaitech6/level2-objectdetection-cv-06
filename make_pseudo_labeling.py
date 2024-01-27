import pandas as pd
import json
import argparse
import os

# read json
def read_coco_json(json_path):
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    return coco_data

# csv 에서 iamge_id, pred, bbox 정보를 coco foramt 으로 가져오기
def extract_df_info(df, coco_data):
    annotations = []

    for idx, row in df.iterrows():
        image_id = None
        for coco_image in coco_data['images']:
            if coco_image['file_name'] == row['image_id']:
                image_id = coco_image['id']
                break

        if image_id is not None:
            bbox_info_list = row['PredictionString']
            for bbox_info in bbox_info_list:
                
                cls, xmin, ymin, xmax, ymax = bbox_info

                bbox_width = float(xmax) - float(xmin)
                bbox_height = float(ymax) - float(ymin)

                annotation = {
                    "image_id": image_id,
                    "category_id": int(cls),
                    "area": bbox_width * bbox_height,
                    "bbox": [round(float(xmin),2), round(float(ymin),2), round(bbox_width,2), round(bbox_height,2)],
                    "iscrowd": 0,
                    "id": len(annotations),
                }

                annotations.append(annotation)

    return annotations

# test.json 에 annotations 추가
def add_annotations_to_coco(coco_data, annotations):
    if 'annotations' not in coco_data:
        coco_data['annotations'] = []

    coco_data['annotations'].extend(annotations)
    return coco_data

# 결과를 새로운 json 파일로 저장
def save_to_json(output_path, coco_data):
    with open(output_path, 'w') as f:
        json.dump(coco_data, f)
        
# csv 에서 score thr 줘서 걸러내기
def process_row(row):
    split_values = row.split()
    thr = 0.3
    split_groups = [[split_values[i]]+split_values[i+2:i+6] for i in range(0, len(split_values), 6) if float(split_values[i+1]) > thr]

    return split_groups

# pseudo labeling + train.json
def merge_json(train_json_path, test_json_path, output_json_path):
    # Read train.json
    with open(train_json_path, 'r') as f:
        train_data = json.load(f)

    # Read test.json
    with open(test_json_path, 'r') as f:
        test_data = json.load(f)

    # Get the last image ID and annotation ID in train.json
    last_image_id = train_data['images'][-1]['id']
    last_annotation_id = train_data['annotations'][-1]['id']

    # Update image IDs and add images from test.json to train.json
    for image in test_data['images']:
        last_image_id += 1
        image['id'] = last_image_id
        train_data['images'].append(image)
        
    check_image_id = train_data['annotations'][-1]['image_id'] + 1

    # Update annotation IDs and image IDs in annotations and add annotations from test.json to train.json
    for annotation in test_data['annotations']:
        last_annotation_id += 1
        annotation['id'] = last_annotation_id
        annotation['image_id'] += check_image_id
        train_data['annotations'].append(annotation)

    # Save the merged data to output.json
    with open(output_json_path, 'w') as f:
        json.dump(train_data, f)

def main():
    parser = argparse.ArgumentParser(description='Make Pseudo labeling annotations')
    parser.add_argument('--csv', type=str, help='Path to the csv file')
    parser.add_argument('--test_json', type=str, help='Path to the test.json file')
    parser.add_argument('--train_json', type=str, required=False, help='Path to the test.json file')
    parser.add_argument('--output', type=str, help='Path to the output.json file')
    parser.add_argument('--merge', type=bool, default=False, help='Path to the output.json file')
    args = parser.parse_args()

    df = pd.read_csv(args.csv, index_col=False)

    df['PredictionString'] = df['PredictionString'].apply(process_row)

    coco_data = read_coco_json(args.test_json)

    annotations = extract_df_info(df, coco_data)

    updated_coco_data = add_annotations_to_coco(coco_data, annotations)

    save_to_json(args.output, updated_coco_data)
    
    if args.merge:
        merge_json(args.train_json, args.output, os.path.join(os.path.splitext(args.output)[0], '_train_merge.json'))
    
    
if __name__ == "__main__":
    main()