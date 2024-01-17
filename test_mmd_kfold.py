import pandas as pd
import os, sys
import os.path as osp
from datetime import datetime
from copy import deepcopy

from utils import load_yaml, save_yaml

from mmengine import ConfigDict
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

prj_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(prj_dir)
sys.path.append("./mmdetection/")

from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmdet.evaluation import DumpDetResults
from mmdet.registry import RUNNERS
from mmdet.utils import setup_cache_size_limit_of_dynamo

from pycocotools.coco import COCO

import argparse
import warnings

from utils import load_yaml, save_yaml


warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # Reduce the number of repeated compilations and improve testing speed.
    setup_cache_size_limit_of_dynamo()

    # Load yaml
    parser = argparse.ArgumentParser(description="Test MMDetection model")
    parser.add_argument("--dir", type=str, help="dirname of train serial")
    parser.add_argument("--tta", type=bool, default=False, help="weather use tta")

    args = parser.parse_args()

    pred_serial = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for fold_index in range(5):

        dir_name = args.dir
        target_dir = os.path.join(prj_dir, "results/train", dir_name)
        train_yaml = [y for y in os.listdir(target_dir) if y.endswith("yaml")][0]
        yaml = load_yaml(os.path.join(target_dir, train_yaml))
        # print(yaml)


        # Device set
        os.environ["CUDA_VISIBLE_DEVICES"] = str(yaml["gpu_num"])

        
        # result_dir
        pred_result_dir = os.path.join(prj_dir, "results", "pred", pred_serial, f"fold_{fold_index}")
        img_result_dir = os.path.join(pred_result_dir, "images")
        submission_file = os.path.join(pred_result_dir, f"{pred_serial}fold_{fold_index}.csv")
        os.makedirs(pred_result_dir, exist_ok=True)

        annotation_dir = yaml["test_annotation_dir"]
        data_dir = yaml["test_dir"]

        new_target_dir = os.path.join(target_dir, f"_fold_{fold_index}")

        check_point = [p for p in os.listdir(new_target_dir) if p.startswith("best")][0]
        check_point_path = os.path.join(new_target_dir, check_point)

        mmdconfig_dir = os.path.join(prj_dir, "mmdetection", "configs", yaml["py_path"])

        cfg = Config.fromfile(mmdconfig_dir)

        # test evaluator
        cfg.test_evaluator.format_only = True
        cfg.test_evaluator.ann_file = annotation_dir
        cfg.test_evaluator.outfile_prefix = osp.join(pred_result_dir, "result_")
        cfg.test_dataloader.dataset.metainfo = dict(classes=yaml["classes"])
        cfg.test_dataloader.dataset.data_root = data_dir
        cfg.test_dataloader.dataset.ann_file = annotation_dir

        cfg.test_dataloader.batch_size = 1
        cfg.test_dataloader.num_workers = 8

        # 기타 설정
        cfg.randomness = dict(seed=yaml["seed"], deterministic=False, diff_rank_seed=False)
        cfg.gpu_ids = [0]
        cfg.work_dir = pred_result_dir

        cfg.custom_hooks = [
            dict(type="SubmissionHook", output_file=submission_file),
        ]

        # wandb log 해제
        cfg.visualizer = dict(
            type="DetLocalVisualizer",
            vis_backends=[
                dict(type="LocalVisBackend"),
            ],
            name="visualizer",
        )

        # visualization
        cfg.default_hooks.visualization = dict(
            type="DetVisualizationHook", draw=False, interval=1
        )

        # load check_point
        cfg.load_from = check_point_path

        # tta
        if args.tta:
            if "tta_model" not in cfg:
                warnings.warn(
                    "Cannot find ``tta_model`` in config, " "we will set it as default."
                )
                cfg.tta_model = dict(
                    type="DetTTAModel",
                    tta_cfg=dict(nms=dict(type="nms", iou_threshold=0.5), max_per_img=100),
                )
            if "tta_pipeline" not in cfg:
                warnings.warn(
                    "Cannot find ``tta_pipeline`` in config, " "we will set it as default."
                )
                test_data_cfg = cfg.test_dataloader.dataset
                while "dataset" in test_data_cfg:
                    test_data_cfg = test_data_cfg["dataset"]
                cfg.tta_pipeline = deepcopy(test_data_cfg.pipeline)
                flip_tta = dict(
                    type="TestTimeAug",
                    transforms=[
                        [
                            dict(type="RandomFlip", prob=1.0),
                            dict(type="RandomFlip", prob=0.0),
                        ],
                        [
                            dict(
                                type="PackDetInputs",
                                meta_keys=(
                                    "img_id",
                                    "img_path",
                                    "ori_shape",
                                    "img_shape",
                                    "scale_factor",
                                    "flip",
                                    "flip_direction",
                                ),
                            )
                        ],
                    ],
                )
                cfg.tta_pipeline[-1] = flip_tta

        # Save yaml
        save_yaml(os.path.join(pred_result_dir, train_yaml), yaml)

        # build the runner from config
        if "runner_type" not in cfg:
            # build the default runner
            runner = Runner.from_cfg(cfg)
        else:
            # build customized runner from the registry
            # if 'runner_type' is set in the cfg
            runner = RUNNERS.build(cfg)

        # save result output
        output_path = os.path.join(pred_result_dir, "result.pkl")
        runner.test_evaluator.metrics.append(DumpDetResults(out_file_path=output_path))

        # start testing
        runner.test()
