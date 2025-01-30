import numpy as np
import pycocotools.mask as mask_utils

from usam.data.storage import Prediction, Storage
from usam.utils.rle import rle_ious


def iou_to_main_prediction(
        original_masks,
        original_scores,
        pred_masks,
        pred_scores
):
    scores = list()
    ious = list()
    for o, os, p, ps in zip(
            original_masks, original_scores, pred_masks, pred_scores
    ):
        ious.append(rle_ious([o], p)[0])
        scores.append(os * ps)
    ious = np.concatenate(ious)
    final_score = np.sum(ious * scores) / np.sum(scores)
    return final_score


def epistemic_uncertainty(masks: list, scores: list):
    m, s = masks, scores
    total_score = np.sum(s)
    w = s[0] * m[0] + s[1] * m[1] + s[2] * m[2]
    inds = w != 0
    w = w[inds] / total_score
    entropy = -(
            w * np.log(w) +
            (1 - w) * np.log(np.maximum(0.00001, 1 - w))
    )
    entropy = np.sum(w * entropy) / w.sum()
    ambiguity = 1 - np.sum(w ** 2) / w.sum()
    return float(entropy), float(ambiguity)


def aleatoric_uncertainty(masks: list, scores: list):
    total_scores = [0, 0, 0]
    weights = [np.zeros_like(m).astype(np.float32) for m in masks[0]]
    for m, s in zip(masks, scores):
        for i in [0, 1, 2]:
            total_scores[i] += s[i].astype(np.float32)
            weights[i][m[i]] += s[i].astype(np.float32)
    aleatoric_entropy = list()
    aleatoric_ambiguity = list()
    for s, w in zip(total_scores, weights):
        inds = w != 0
        if len(inds) == 0:
            raise Exception("sdf")
        w = w[inds] / s
        entropy = -(
                w * np.log(w) +
                (1 - w) * np.log(np.maximum(0.00001, 1 - w))
        )
        entropy = np.sum(w * entropy) / w.sum()
        ambiguity = 1 - np.sum(w ** 2) / w.sum()
        aleatoric_ambiguity.append(float(ambiguity))
        aleatoric_entropy.append(float(entropy))

    return aleatoric_ambiguity, aleatoric_entropy


def model_differences(iou_tiny, iou_small, iou_base_plus, iou_large):
    # Calc metric for main prediction
    results = {
        "tiny_vs_small": iou_tiny - iou_small,
        "tiny_vs_base_plus": iou_tiny - iou_base_plus,
        "tiny_vs_large": iou_tiny - iou_large,
        "small_vs_tiny": iou_small - iou_tiny,
        "small_vs_base_plus": iou_small - iou_base_plus,
        "small_vs_large": iou_small - iou_large,
        "base_plus_vs_tiny": iou_base_plus - iou_tiny,
        "base_plus_vs_small": iou_base_plus - iou_small,
        "base_plus_vs_large": iou_base_plus - iou_large,
        "large_vs_tiny": iou_large - iou_tiny,
        "large_vs_small": iou_large - iou_small,
        "large_vs_base_plus": iou_large - iou_base_plus,
    }
    return results


def mean_entropy(mask):
    # Calc the mean entropy of the mask
    if np.max(mask) > 1 or np.min(mask) < 0:
        # Apply sigmoid to the mask
        mask = 1 / (1 + np.exp(-mask))
    entropy = -(
            mask * np.log(np.maximum(0.00001, mask)) +
            (1 - mask) * np.log(np.maximum(0.00001, 1 - mask))
    )
    entropy = np.sum(mask * entropy) / np.maximum(mask.sum(), 1)
    return entropy


def mean_perplexity(mask):
    # Calc the mean perplexity of the mask
    if np.max(mask) > 1 or np.min(mask) < 0:
        # Apply sigmoid to the mask
        mask = 1 / (1 + np.exp(-mask))
    perplexity = -(
            mask * np.log2(np.maximum(0.00001, mask)) +
            (1 - mask) * np.log2(np.maximum(0.00001, 1 - mask))
    )
    perplexity = np.sum(mask * 2 ** perplexity) / np.maximum(mask.sum(), 1)
    return perplexity

