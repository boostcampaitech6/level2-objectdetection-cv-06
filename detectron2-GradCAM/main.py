import os
import math
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from detectron2_gradcam import Detectron2GradCAM

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo

from gradcam import GradCAM, GradCamPlusPlus


############### 확인할 이미지, 학습된 yaml, pth 파일 주소를 수정해야 합니다! ###############
img_number = "4291.jpg"
img_path = os.path.join("/data/ephemeral/home/dataset/train", img_number)
config_file = "/data/ephemeral/home/sample_code/baseline/detectron2/trash_faster_rcnn_R_101_FPN_3x_test.yaml"
model_file = (
    "/data/ephemeral/home/sample_code/baseline/detectron2/output/model_final.pth"
)
##########################################################################################


config_list = [
    "MODEL.ROI_HEADS.SCORE_THRESH_TEST",
    "0.5",
    "MODEL.ROI_HEADS.NUM_CLASSES",
    "10",  # 각 task에 맞게 class 수를 수정해줘야 합니다.
    "MODEL.WEIGHTS",
    model_file,
]

layer_name = "backbone.bottom_up.res5.2.conv3"
instance = 0  # CAM is generated per object instance, not per class!


def main():
    cam_extractor = Detectron2GradCAM(
        config_file,
        config_list,
        img_path=img_path,
        root_dir="/data/ephemeral/home/dataset/",
        custom_dataset="coco_trash_train",
    )
    grad_cam = GradCamPlusPlus

    ex_output, ex_cam_orig = cam_extractor.get_cam(
        target_instance=instance, layer_name=layer_name, grad_cam_instance=grad_cam
    )
    pred_cnt = len(ex_output["output"]["instances"].pred_classes)  # 예측한 class 개수

    row_num = math.ceil(pred_cnt / 3)
    col_num = 3
    plt.rcParams["figure.figsize"] = (col_num * 4, row_num * 4)

    for i in tqdm(range(pred_cnt)):
        image_dict, cam_orig = cam_extractor.get_cam(
            target_instance=i, layer_name=layer_name, grad_cam_instance=grad_cam
        )

        v = Visualizer(
            image_dict["image"],
            MetadataCatalog.get(cam_extractor.cfg.DATASETS.TRAIN[0]),
            scale=1.0,
        )
        out = v.draw_instance_predictions(
            image_dict["output"]["instances"][i].to("cpu")
        )

        plt.subplot(row_num, 3, i + 1)
        plt.imshow(out.get_image(), interpolation="none")
        plt.imshow(image_dict["cam"], cmap="jet", alpha=0.5)
        plt.gca().axes.xaxis.set_visible(False)
        plt.gca().axes.yaxis.set_visible(False)
        plt.title(f"{i} : class ({image_dict['label']})")

    plt.suptitle(f"Grad CAM for a test image")

    plt.savefig(f"{img_number[:4]}_cam.jpg", dpi=200)

    plt.show()


if __name__ == "__main__":
    main()
