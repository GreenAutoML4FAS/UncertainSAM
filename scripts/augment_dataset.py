import argparse

import torch
import tqdm
from os import makedirs
from os.path import exists, join
import numpy as np
from torch.utils.data import DataLoader

import usam.datasets as data
from usam.patch_sam2 import (
    patch_sam2, build_sam2, SAM2ImagePredictor, read_mask_tokens,
    read_iou_token,
)
from usam.training.scheduler import SimpleAugmentationScheduler
import pycocotools.mask as mask_utils
from usam.datasets.storage import Storage, Metrics


MODELS = {
    'tiny': {"checkpoint": "sam2_hiera_tiny.pt", "config": "sam2_hiera_t.yaml"},
    'small': {"checkpoint": "sam2_hiera_small.pt", "config": "sam2_hiera_s.yaml"},
    'base_plus': {"checkpoint": "sam2_hiera_base_plus.pt", "config": "sam2_hiera_b+.yaml"},
    'large': {"checkpoint": "sam2_hiera_large.pt", "config": "sam2_hiera_l.yaml"},
}


def to_rle(mask):
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))


def rle_area(rle):
    return mask_utils.area(rle)


def rle_ious(preds, gt):
    return mask_utils.iou(preds, [gt], pyiscrowd=[False])


def entropy(mask):
    mask = np.clip(mask, 1e-6, 1 - 1e-6)
    e = -mask * np.log2(mask) - (1 - mask) * np.log2(1 - mask)
    e = mask * e
    e = np.sum(e) / np.sum(mask)
    return e


def load_dataset(
        dataset,
        root,
        split,
        start,
        end,
        sample,
        train_subsets,
        augmentations
):
    """
    Loads the dataset
    :param dataset: One of the supported datasets: mose, davis, sav ade20k, coco
    :param root: root directory of the dataset
    :param split: one of "train", "val", "test"
    :param start: Start frame
    :param end: end frame
    :param sample: subsample by this factor, i.e. 1 means no subsampling
    :param train_subsets: Only necessary for SAV dataset
    :param augmentations: A set of augmentations
    :return:
    """
    if dataset == "mose":
        dataset = data.MOSEImageDataset
        dataset = dataset(
            root=root,
            split=split,
            augmentations=augmentations,
            start_frame=start,
            end_frame=end,
            sample=sample,
        )
    elif dataset == "davis":
        dataset = data.DavisImageDataset
        dataset = dataset(
            root=root,
            split=split,
            augmentations=augmentations,
            start_frame=start,
            end_frame=end,
            sample=sample,
        )
    elif dataset == "sav":
        dataset = data.SAVImageDataset
        dataset = dataset(
            root=root,
            split=split,
            augmentations=augmentations,
            start_frame=start,
            end_frame=end,
            train_subsets=train_subsets,
            sample=sample,
        )
    elif dataset == "ade20k":
        dataset = data.ADE20KImageDataset
        dataset = dataset(
            root=root,
            split=split,
            augmentations=augmentations,
            start_frame=start,
            end_frame=end,
            sample=sample,
        )
    elif dataset == "coco":
        dataset = data.COCOImageDataset
        dataset = dataset(
            root=root,
            split=split,
            augmentations=augmentations,
            start_frame=start,
            end_frame=end,
            sample=sample,
        )
    else:
        raise NotImplementedError

    return dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--root", required=True, help="Dataset Root")
    parser.add_argument("--split", required=True, help="Split of the dataset")
    parser.add_argument("--model-root", required=True, help="Model root")
    parser.add_argument("--config-root", required=True, help="Model config root")
    parser.add_argument("--start", type=int, default=None, help="Start frame")
    parser.add_argument("--end", type=int, default=None, help="End frame")
    parser.add_argument("--sample", type=int, default=None, help="Sample rate")
    parser.add_argument("--train-subsets", nargs="+", type=int, default=None)
    parser.add_argument("--skip-augs", action="store_true", help="Skip augmentations")
    args = parser.parse_args()
    assert exists(args.root)
    makedirs(args.output_dir, exist_ok=True)
    assert exists(args.output_dir)
    args.dataset = args.dataset.lower()
    assert args.dataset in ["mose", "davis", "sav", "ade20k", "coco"]
    assert args.split in ["train", "val", "test"]
    assert exists(args.root)

    return args


def predict(
        predictors: dict,
        img: np.ndarray,
        gt_rles: dict,
        cnts: list,
        smpls: list,
        aug_imgs: list,
        aug_cnts: list,
        aug_smpls: list,
        augs: list,
        storage: Storage,
        only_first_prompt=False
):
    augs = [None] + augs
    imgs = [img] + aug_imgs
    cnts = [cnts] + aug_cnts
    smpls = [smpls] + aug_smpls
    # Iterate over all instances
    for i, idx in enumerate(list(gt_rles.keys())):
        # Iterate overa all models
        results = dict()
        results_merged = dict()
        for model, predictor in predictors.items():
            # Iterate over augmentations
            results[model] = dict()
            results_merged[model] = dict()
            for a, (img, aug, _cnts, _smpls) in enumerate(zip(imgs, augs, cnts, smpls)):
                # Iterate over prompt samples
                results[model][a] = list()
                predictor.set_image(img)
                for smpl in np.concatenate([_cnts[i:i+1], _smpls[i]]):
                    masks, scores, logits, _ = predictor.predict(
                        point_coords=[smpl],
                        point_labels=[1],
                        multimask_output=True,
                    )
                    masks = masks[0:3]
                    if aug is not None:
                        _, _, masks = aug(masks=masks, backward=True)
                    rles = []
                    for mask in masks:
                        rles.append(to_rle(mask))
                    ious = rle_ious(rles, gt_rles[idx])
                    iou_token = read_iou_token(predictor).clone()
                    mask_tokens = read_mask_tokens(predictor)[0].clone()
                    logits = 1 / (1 + np.exp(-logits))
                    inds = np.argsort(ious[:, 0])[::-1]
                    res = {
                        "masks": masks[inds], "scores": scores[inds],
                        "logits": logits[inds], "ious": ious[inds],
                        "rles": [rles[x] for x in inds],
                        "iou_token": iou_token,
                        "mask_tokens": mask_tokens[0:3][list(inds)]
                    }
                    results[model][a].append(res)
                    if only_first_prompt:
                        break
                # Create prediction with all samples
                merged_coordinates = np.concatenate([_cnts[i:i+1], _smpls[i]])
                masks, scores, logits, _ = predictor.predict(
                    point_coords=merged_coordinates,
                    point_labels=np.ones(len(merged_coordinates)),
                    multimask_output=True,
                )
                if aug is not None:
                    _, _, masks = aug(masks=masks, backward=True)

                rles = []
                for mask in masks:
                    rles.append(to_rle(mask))
                ious = rle_ious(rles, gt_rles[idx])
                iou_token = read_iou_token(predictor).clone()
                mask_tokens = read_mask_tokens(predictor)[0].clone()
                logits = 1 / (1 + np.exp(-logits))
                inds = np.argsort(ious[:, 0])[::-1]
                res = {
                    "masks": masks[inds], "scores": scores[inds],
                    "logits": logits[inds], "ious": ious[inds],
                    "rles": [rles[x] for x in inds],
                    "iou_token": iou_token,
                    "mask_tokens": mask_tokens[0:3][list(inds)]
                }
                results_merged[model][a] = res

         # Calculate the overall and sample entropy
        probability_map = list()
        sample_entropies = list()
        sample_probabilities = list()
        for model in predictors.keys():
            for a in results[model].keys():
                for p in results[model][a]:
                    p_task = p["scores"] / np.sum(p["scores"])
                    for j in range(3):
                        # Overall entropy
                        probability_map.append(p["logits"][j])
                        sample_entropies.append(entropy(p["logits"][j]))
                        sample_probabilities.append(
                            1/len(predictors.keys()) *
                            1/len(results[model].keys()) *
                            1/len(results[model][a]) *
                            p_task[j]
                        )
        probability_map = np.stack(probability_map)
        sample_probabilities = np.asarray(sample_probabilities)
        sample_probabilities /= np.sum(sample_probabilities)
        h_all = entropy(np.sum(probability_map*sample_probabilities[:, None, None], axis=0))

        # Create output results
        for model in predictors.keys():
            p = results[model]
            p_merged = results_merged[model]
            p = p[list(p.keys())[0]][0]
            p_merged = p_merged[list(p_merged.keys())[0]]

            best = np.argmax(p["ious"])
            best_dense = np.argmax(p_merged["ious"])
            uns_best = np.argmax(p["scores"])

            # Store values
            m = Metrics()
            m.iou = float(p["ious"][best])
            m.sam_score = float(p["scores"][best])
            m.entropy = entropy(p["logits"][best])
            m.unsupervised_sam_score = float(p["scores"][uns_best])
            m.unsupervised_iou = float(p["ious"][uns_best])
            m.iou_dense_prompt = float(p_merged["ious"][best_dense])
            m.h_all = float(h_all)

            # Calculate and store other measures
            scores = p["scores"] / np.sum(p["scores"])
            h_task = np.sum(p["logits"] * scores[:, None, None], axis=0)
            m.h_task = float(entropy(h_task))

            p_aug = results[model]
            p_aug = [p_aug[x][0]["logits"][best] for x in p_aug.keys()]
            h_augmentation = np.sum(np.asarray(p_aug) * 1 / len(p_aug), axis=0)
            m.h_augmentation = float(entropy(h_augmentation))

            p_prompt = results[model]
            p_prompt = p_prompt[list(p_prompt.keys())[0]]
            p_prompt = [x["logits"][best] for x in p_prompt]
            h_prompt = np.sum(np.asarray(p_prompt) * 1 / len(p_prompt), axis=0)
            m.h_prompt = float(entropy(h_prompt))

            p_theta = [list(x.values())[0][0] for x in results.values()]
            p_theta = [x["logits"][best] for x in p_theta]
            h_theta = np.sum(np.asarray(p_theta) * 1 / len(p_theta), axis=0)
            m.h_theta = float(entropy(h_theta))

            # Store the prediction
            if storage is not None:
                print(model)
                storage.set_prediction(
                    model, idx, p["scores"], p["iou_token"],
                    p["mask_tokens"][best], m,
                )




def collate_fn(batch):
    """
    Necessary for the dataloader
    :param batch:
    :return:
    """
    return batch[0]


def main(args):
    """
    Augments data and stores the results that is necessary for model training
    :param args:
    :return:
    """
    # Load dataset
    augmentations = SimpleAugmentationScheduler()
    if args.skip_augs:
        augmentations = None
    dataset = load_dataset(
        args.dataset, args.root, args.split, args.start, args.end,
        args.sample, args.train_subsets, augmentations
    )


    dataloader = DataLoader(
        dataset, batch_size=1, num_workers=8, collate_fn=collate_fn)

    # Iterate over dataset
    models = ["tiny", "small", "base_plus", "large"]

    # Load predictors
    predictors = {}
    for model in models:
        model_cfg = join(args.config_root, MODELS[model]["config"])
        model_checkpoint = join(args.model_root, MODELS[model]["checkpoint"])
        sam2_model = build_sam2(model_cfg, model_checkpoint, device="cuda")
        predictor = SAM2ImagePredictor(sam2_model)
        patch_sam2(predictor)
        predictors[model] = predictor

    # Iterate over dataset
    for batch in tqdm.tqdm(dataloader, desc=f"Iterate over {args.dataset}"):
        if not args.skip_augs:
            img, gt, cnts, smpls, img_name, aug_imgs, aug_cnts, aug_smpls, augs \
                = batch
        else:
            img, gt, cnts, smpls, img_name = batch
            aug_imgs, aug_cnts, aug_smpls, augs = [], [], [], []
        # Skip empty images
        if gt is None:
            continue
        # Create storage containers
        storage = Storage(args.output_dir, img_name)
        h, w, c = img.shape
        storage.set_image_width(w)
        storage.set_image_height(h)
        # Predict and store data
        gt_rles = {idx: to_rle(mask) for idx, mask in gt.items()}
        for idx, rle in gt_rles.items():
            storage.set_ground_truth_mask_sizes(idx, rle_area(rle))
        predict(
            predictors, img, gt_rles, cnts, smpls, aug_imgs, aug_cnts,
            aug_smpls, augs, storage, only_first_prompt=args.skip_augs
        )
        print(storage.get_models())
        storage.save()



if __name__ == "__main__":
    args = parse_args()
    main(args)
