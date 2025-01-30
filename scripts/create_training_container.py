import argparse
import os
from os.path import exists, dirname
import warnings

from os.path import join
from tqdm import tqdm

from usam.datasets.storage import Storage
from usam.datasets.training_container import TrainingContainer, save


def parse_directory(directory):
    result_files = list()
    for root, dirs, files in os.walk(directory):
        for file in files:
            f = join(root, file)
            f = f.replace(directory, "")
            if f.startswith("/"):
                f = f[1:]
            if f.endswith(".pkl"):
                result_files.append(f)
    return result_files


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--out-file", type=str, required=True)
    parser.add_argument("--no-augmentations", action='store_true')
    parser.add_argument("--offload", action='store_true')
    return parser.parse_args()


def main():
    args = get_arguments()
    print(args.root)
    assert exists(args.root), f"Prediction root {args.root} does not exist."
    assert exists(dirname(args.out_file))

    files = parse_directory(args.root)

    container = TrainingContainer()
    pbar = tqdm(files, desc="Create Container")
    for file_idx, file in enumerate(pbar):
        print(file)
        storage = Storage(args.root, file)
        model_types = storage.get_models()
        for m in ["tiny", "small", "base_plus", "large"]:
            if m not in model_types:
                warnings.warn(f"Model {m} not found in {file} (Position {file_idx})")
        for m in model_types:
            for i in storage.get_ids():
                # Comparable ious
                ious = dict()
                sam_scores = dict()
                for _m in storage.get_models():
                    pred = storage.get_prediction(_m, i)
                    metrics = pred.metrics.__dict__
                    ious["iou_"+_m] = metrics["iou"]
                    ious["unsupervised_iou_"+_m] = metrics["unsupervised_iou"]
                    sam_scores["sam_score_" + _m] = metrics["sam_score"]
                # Load original embedding
                pred = storage.get_prediction(m, i)
                metrics = pred.metrics.__dict__
                metrics["gt_size"] = storage.get_ground_truth_mask_sizes(i)
                metrics["image_width"] = storage.content["image_width"]
                metrics["image_height"] = storage.content["image_height"]
                metrics["image_quality"] = 100
                metrics["is_augmented"] = False
                for x in ["tiny", "small", "base_plus", "large"]:
                    metrics["is_" + x] = x == m
                for k, v in ious.items():
                    metrics[k] = v
                for k, v in sam_scores.items():
                    metrics[k] = v
                metrics["model_gap"] = (
                        metrics["iou_large"] - metrics["iou_tiny"])
                metrics["task_gap"] = (
                        metrics["iou"] - metrics["unsupervised_iou"])
                metrics["prompt_gap"] = (
                        metrics["iou_dense_prompt"] - metrics["iou"])

                container.append(
                    iou_token=pred.iou_token[0].clone(),
                    mask_token=pred.mask_token.clone(),
                    metrics=metrics,
                    image_name=storage.file_name,
                    model=m,
                )

    save(args.out_file, container, offload=args.offload)


if __name__ == "__main__":
    main()
