import argparse
import os
import os.path as osp
import sys
import shutil

from datetime import datetime
from utils import load_yaml, save_yaml

import wandb

prj_dir = os.path.dirname(os.path.abspath(__file__))
# os.path.abspath(__file__)로 이 py파일의 절대 경로 찾기
# os.path.dirname()는 폴더 경로

sys.path.append(prj_dir)
sys.path.append("./mmdetection/")
# mmdetection 폴더 경로 추가

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet.utils import setup_cache_size_limit_of_dynamo

from utils import load_yaml, save_yaml

import warnings

warnings.filterwarnings("ignore")


def main():
    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    # 학습 결과 파일로 results/train/학습시작시간 dir
    train_serial = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_result_dir = os.path.join(prj_dir, "results", "train", train_serial)
    os.makedirs(train_result_dir, exist_ok=True)

    # Load yaml
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("--yaml", type=str, help="yaml file name for this train")

    args = parser.parse_args()

    yaml_name = args.yaml
    yaml_path = os.path.join(prj_dir, "yamls", yaml_name)
    yaml = load_yaml(yaml_path)
    shutil.copy(yaml_path, os.path.join(train_result_dir, yaml_name))

    # dataset 경로
    train_annotation_dir = yaml["train_kfold_annotation_dir"]
    val_annotation_dir = yaml["val_kfold_annotation_dir"]
    data_dir = yaml["train_dir"]

    # gpu 설정
    os.environ["CUDA_VISIBLE_DEVICES"] = str(yaml["gpu_num"])

    for fold_index in range(yaml["kfold_splits"]):
        # load config
        mmdconfig_dir = os.path.join(prj_dir, "mmdetection", "configs", yaml["py_path"])
        cfg = Config.fromfile(mmdconfig_dir)
        # dataset train
        # 여기서 palette 로 나중에 특정한 것만 눈에 띄게 표시 가능
        cfg.train_dataloader.dataset.metainfo = dict(classes=yaml["classes"])
        cfg.train_dataloader.dataset.data_root = data_dir
        cfg.train_dataloader.dataset.ann_file = os.path.join(
            train_annotation_dir, f"train_fold_{fold_index}.json"
        )

        # dataset val
        cfg.val_dataloader.dataset.metainfo = dict(classes=yaml["classes"])
        cfg.val_dataloader.dataset.data_root = data_dir
        cfg.val_dataloader.dataset.ann_file = os.path.join(
            val_annotation_dir, f"val_fold_{fold_index}.json"
        )

        # batch_size, num_workers
        cfg.train_dataloader.batch_size = yaml["batch_size"]
        cfg.train_dataloader.num_workers = yaml["num_workers"]

        # evaluator
        cfg.val_evaluator.ann_file = os.path.join(
            val_annotation_dir, f"val_fold_{fold_index}.json"
        )
        cfg.val_evaluator.classwise = True

        # visualization
        cfg.default_hooks.visualization = dict(
            type="DetVisualizationHook", draw=True, interval=100
        )

        # 추가 config 수정
        for key, value in yaml["cfg_options"].items():
            if isinstance(value, list):
                yaml["cfg_options"][key] = tuple(value)
        cfg.merge_from_dict(yaml["cfg_options"])

        # 기타 설정
        cfg.randomness = dict(
            seed=yaml["seed"], deterministic=False, diff_rank_seed=False
        )
        cfg.gpu_ids = [0]
        # fold_dir = os.path.join(train_result_dir, f"fold_{fold_index}")
        # cfg.work_dir = fold_dir
        # cfg.work_dir = os.path.join(train_result_dir, f"fold_{fold_index}")
        fold_dir_name = f"fold_{fold_index}"
        cfg.work_dir = os.path.join(train_result_dir, fold_dir_name)

        # cfg.work_dir = os.path.join(train_result_dir, f"/fold_{fold_index}")

        # wandb
        cfg.visualizer = dict(
            type="DetLocalVisualizer",
            vis_backends=[
                # dict(type='LocalVisBackend'),
                dict(
                    type="WandbVisBackend",
                    init_kwargs={
                        "project": yaml["wandb_project"],
                        "entity": yaml["wandb_entity"],
                        "name": yaml["wandb_run"] + f"_fold_{fold_index}",
                        "config": {
                            "optimizer": cfg.optim_wrapper.optimizer.type,
                            "learning_rate": cfg.optim_wrapper.optimizer.lr,
                            "architecture": cfg.model.type,
                            "dataset": cfg.train_dataloader.dataset.type,
                            "n_epochs": cfg.train_cfg.max_epochs,
                            "notes": yaml["wandb_note"],
                        },
                    },
                    commit=True,
                )
            ],
        )

        # config 저장
        with open(os.path.join(train_result_dir, "config.txt"), "w") as file:
            file.write(cfg.pretty_text)

        # resume is determined in this priority: resume from > auto_resume
        if yaml["resume"] == "auto":
            cfg.resume = False
        elif yaml["resume"] is not None:
            cfg.resume = True
            cfg.load_from = yaml["resume_from"]

        # build the runner from config
        if "runner_type" not in cfg:
            # build the default runner
            runner = Runner.from_cfg(cfg)
        else:
            # build customized runner from the registry
            # if 'runner_type' is set in the cfg
            runner = RUNNERS.build(cfg)

        # start training
        runner.train()


if __name__ == "__main__":
    main()
