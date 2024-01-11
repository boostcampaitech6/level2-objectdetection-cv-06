import argparse
import yaml
from types import SimpleNamespace

from pathlib import Path
import analyze_logs
import confusion_matrix
import eval_metric
import fuse_results
import get_flops


def main():
    parser = argparse.ArgumentParser(description="Analysis")
    parser.add_argument(
        "mode",
        metavar="Select one analyze_logs, confusion_matrix, eval_metric, fuse_results, get_flops",
        type=str,
        help="""
        analyze_logs - plot_curve of bbox_mAP, loss etc in logs
        confusion_matrix - save confusion matrix
        eval_metric - Evaluate metric of the results saved in pkl format
        fuse_results - Fusion image prediction results using Weighted Boxes Fusion from multiple models.
        get_flops - Get a detector flops
        """,
    )
    parser.add_argument("--yaml", help="Path to the configuration YAML file")
    global_args = parser.parse_args()

    with open(global_args.yaml, "r") as y:
        yaml_file = yaml.safe_load(y)

    args = SimpleNamespace(**yaml_file)

    if global_args.mode == "analyze_logs":
        analyze_logs.main(args)
    elif global_args.mode == "confusion_matrix":
        confusion_matrix.main(args)
    elif global_args.mode == "eval_metric":
        eval_metric.main(args)
    elif global_args.mode == "fuse_results":
        fuse_results.main(args)
    elif global_args.mode == "get_flops":
        get_flops.main(args)
    else:
        print("Invalid task specified in the configuration.")


if __name__ == "__main__":
    main()
