import argparse
import json
import os
import shutil
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tqdm import tqdm


# -- train.json 내의 정보를 이미지 각각의 xml 로 만들어주기 (coco -> pascal)
# -- images, categories, annotation 정보를 보존
def json_to_xml(json_path, xml_output_folder, class_mapping=None):
    with open(json_path, "r") as json_file:
        json_data = json.load(json_file)

    os.makedirs(xml_output_folder, exist_ok=True)

    for image_info in tqdm(json_data["images"]):
        image_id = image_info["id"]
        filename = image_info["file_name"]
        image_width = image_info["width"]
        image_height = image_info["height"]

        root = ET.Element("annotation")

        folder = ET.SubElement(root, "folder")
        folder.text = "images"

        filename_elem = ET.SubElement(root, "filename")
        filename_elem.text = filename

        path = ET.SubElement(root, "path")
        path.text = os.path.join("images", filename)

        source = ET.SubElement(root, "source")
        database = ET.SubElement(source, "database")
        database.text = "Unknown"

        size = ET.SubElement(root, "size")
        width = ET.SubElement(size, "width")
        height = ET.SubElement(size, "height")
        depth = ET.SubElement(size, "depth")
        width.text = str(image_width)
        height.text = str(image_height)
        depth.text = "3"  # RGB

        annotations = [
            ann for ann in json_data["annotations"] if ann["image_id"] == image_id
        ]

        for obj in annotations:
            obj_elem = ET.SubElement(root, "object")
            name = ET.SubElement(obj_elem, "name")
            ## labelImg 내에서 data cleansing 효율을 위해서 label encoding -> 클래스 이름 변경
            name.text = class_mapping.get(obj["category_id"], "Unknown")
            pose = ET.SubElement(obj_elem, "pose")
            pose.text = "Unspecified"
            truncated = ET.SubElement(obj_elem, "truncated")
            truncated.text = "0"
            difficult = ET.SubElement(obj_elem, "difficult")
            difficult.text = "0"

            bbox = obj["bbox"]
            xmin, ymin, w, h = bbox
            xmax, ymax = xmin + w, ymin + h

            bndbox = ET.SubElement(obj_elem, "bndbox")
            xmin_elem = ET.SubElement(bndbox, "xmin")
            ymin_elem = ET.SubElement(bndbox, "ymin")
            xmax_elem = ET.SubElement(bndbox, "xmax")
            ymax_elem = ET.SubElement(bndbox, "ymax")

            xmin_elem.text = str(float(xmin))
            ymin_elem.text = str(float(ymin))
            xmax_elem.text = str(float(xmax))
            ymax_elem.text = str(float(ymax))

            attribute = ET.SubElement(obj_elem, "attribute")
            attribute.text = class_mapping.get(obj["category_id"], "Unknown")

        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
        output_path = os.path.join(xml_output_folder, filename.replace(".jpg", ".xml"))

        with open(output_path, "w") as xml_file:
            xml_file.write(xml_str)


# -- 수정한 xml 파일을 하나의 json 파일로 만들기 (pascal -> coco)
def xml_to_json(xml_folder, json_output_path, label_encoding_mapping):
    json_data = {"images": [], "categories": [], "annotations": []}

    image_id_mapping = {}
    category_id_mapping = {}
    annotation_id_counter = 0

    xml_files = os.listdir(xml_folder)
    xml_files.sort(key=lambda x: int(x.split(".")[0]))

    for xml_file in tqdm(xml_files):
        if xml_file.endswith(".xml"):
            xml_path = os.path.join(xml_folder, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            image_info = {
                "width": int(root.find("size/width").text),
                "height": int(root.find("size/height").text),
                "file_name": "/".join(root.find("path").text.split("/")[-2:]),
                "license": 0,
                "flickr_url": None,
                "coco_url": None,
                "date_captured": None,
                "id": int(
                    os.path.splitext(root.find("filename").text)[0].split("/")[-1]
                ),
            }
            json_data["images"].append(image_info)
            image_id_mapping[image_info["id"]] = image_info["file_name"]

            for object_elem in root.findall("object"):
                category_name = object_elem.find("name").text
                if category_name not in category_id_mapping:
                    category_id = label_encoding_mapping[category_name]
                    category_id_mapping[category_name] = category_id
                    category_info = {
                        "id": category_id,
                        "name": category_name,
                        "supercategory": category_name,
                    }
                    json_data["categories"].append(category_info)

                category_id = category_id_mapping[category_name]

                annotation_info = {
                    "image_id": image_info["id"],
                    "category_id": category_id,
                    "area": (
                        float(root.find("object/bndbox/xmax").text)
                        - float(root.find("object/bndbox/xmin").text)
                    )
                    * (
                        float(root.find("object/bndbox/ymax").text)
                        - float(root.find("object/bndbox/ymin").text)
                    ),
                    "bbox": [
                        float(root.find("object/bndbox/xmin").text),
                        float(root.find("object/bndbox/ymin").text),
                        float(root.find("object/bndbox/xmax").text)
                        - float(root.find("object/bndbox/xmin").text),
                        float(root.find("object/bndbox/ymax").text)
                        - float(root.find("object/bndbox/ymin").text),
                    ],
                    "iscrowd": int(object_elem.find("difficult").text),
                    "id": annotation_id_counter,
                }

                annotation_id_counter += 1
                json_data["annotations"].append(annotation_info)

    # - sort
    json_data["images"] = sorted(json_data["images"], key=lambda x: x["id"])
    json_data["categories"] = sorted(json_data["categories"], key=lambda x: x["id"])
    json_data["annotations"] = sorted(json_data["annotations"], key=lambda x: x["id"])

    with open(json_output_path, "w") as json_file:
        json.dump(json_data, json_file, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Process JSON and XML files.")
    parser.add_argument(
        "--function",
        choices=["json_to_xml", "update_class_names", "xml_to_json"],
        help="Specify the function to run.",
    )
    parser.add_argument("--json_path", type=str, help="Path to the JSON file.")
    parser.add_argument(
        "--xml_output_folder", type=str, help="Path to the folder to save XML files."
    )
    parser.add_argument(
        "--xml_folder", type=str, help="Path to the folder to save XML files."
    )
    parser.add_argument(
        "--json_output_path", type=str, help="Path to save the final JSON file."
    )
    args = parser.parse_args()

    if args.function == "json_to_xml":
        class_mapping = {
            0: "General trash",
            1: "Paper",
            2: "Paper pack",
            3: "Metal",
            4: "Glass",
            5: "Plastic",
            6: "Styrofoam",
            7: "Plastic bag",
            8: "Battery",
            9: "Clothing",
        }
        json_to_xml(args.json_path, args.xml_output_folder, class_mapping)

    elif args.function == "xml_to_json":
        label_encoding_mapping = {
            "General trash": 0,
            "Paper": 1,
            "Paper pack": 2,
            "Metal": 3,
            "Glass": 4,
            "Plastic": 5,
            "Styrofoam": 6,
            "Plastic bag": 7,
            "Battery": 8,
            "Clothing": 9,
        }
        xml_to_json(args.xml_folder, args.json_output_path, label_encoding_mapping)


if __name__ == "__main__":
    main()
