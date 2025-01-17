import types
from typing import List, Optional, Tuple

import numpy as np
import torch
import os
from os.path import exists, join

from sam2.build_sam import build_sam2 as build_sam2_original
from sam2.sam2_image_predictor import SAM2ImagePredictor as SAM2ImagePredictor_original
from sam2.sam2_image_predictor import SAM2Base

from sam2.utils.misc import get_sdpa_settings

from usam.MLP import MLP

OLD_GPU, USE_FLASH_ATTN, MATH_KERNEL_ON = get_sdpa_settings()

build_sam2 = build_sam2_original
SAM2ImagePredictor = SAM2ImagePredictor_original


def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        repeat_image: bool,
        high_res_features: Optional[List[torch.Tensor]] = None,
        custom_token: dict = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Predicts masks. See 'forward' for more details."""
    # Concatenate output tokens
    s = 0
    if self.pred_obj_scores:
        output_tokens = torch.cat(
            [
                self.obj_score_token.weight,
                self.iou_token.weight,
                self.mask_tokens.weight,
            ],
            dim=0,
        )
        s = 1
    else:
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0
        )
    output_tokens = output_tokens.unsqueeze(0).expand(
        sparse_prompt_embeddings.size(0), -1, -1
    )
    tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

    # Expand per-image data in batch direction to be per-mask
    if repeat_image:
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0],
                                          dim=0)
    else:
        assert image_embeddings.shape[0] == tokens.shape[0]
        src = image_embeddings
    src = src + dense_prompt_embeddings
    assert (
            image_pe.size(0) == 1
    ), "image_pe should have size 1 in batch dim (from `get_dense_pe()`)"
    pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
    b, c, h, w = src.shape

    # Run the transformer
    hs, src = self.transformer(src, pos_src, tokens)
    iou_token_out = hs[:, s, :]
    mask_tokens_out = hs[:, s + 1: (s + 1 + self.num_mask_tokens), :]

    # ### Modification for Certainty estimation ###
    if custom_token is not None:
        #iou_token_out = custom_token["iou_token"].to(hs.device)
        mask_tokens_out = custom_token["mask_tokens"].to(hs.device)
    else:
        self.iou_token_out = iou_token_out.detach().cpu()
        self.mask_tokens_out = mask_tokens_out.detach().cpu()
        # Predict scores
        if iou_token_out.shape[0] == 1:
            inputs = torch.cat((iou_token_out.expand(3, -1), mask_tokens_out[0, 0:3]), dim=1)
        else:
            raise NotImplementedError("Batch size > 1 not supported")
            inputs = torch.cat((iou_token_out[0], mask_tokens_out[0]), dim=1)

        for model in self.regression_models:
            self.scores[model] = self.regression_models[model](inputs).detach().cpu().float().numpy()
            if model.endswith("gap"):
                self.scores[model] = self.scores[model] - 0.5

    # ###

    # Upscale mask embeddings and predict masks using the mask tokens
    src = src.transpose(1, 2).view(b, c, h, w)
    if not self.use_high_res_features:
        upscaled_embedding = self.output_upscaling(src)
    else:
        dc1, ln1, act1, dc2, act2 = self.output_upscaling
        feat_s0, feat_s1 = high_res_features
        upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
        upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

    hyper_in_list: List[torch.Tensor] = []
    for i in range(self.num_mask_tokens):
        hyper_in_list.append(
            self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
        )
    hyper_in = torch.stack(hyper_in_list, dim=1)
    b, c, h, w = upscaled_embedding.shape
    masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h,
                                                                       w)
    # Generate mask quality predictions
    iou_pred = self.iou_prediction_head(iou_token_out)
    if self.pred_obj_scores:
        assert s == 1
        object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
    else:
        # Obj scores logits - default to 10.0, i.e. assuming the object is present, sigmoid(10)=1
        object_score_logits = 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)

    return masks, iou_pred, mask_tokens_out, object_score_logits


def predict(
    self,
    point_coords: Optional[np.ndarray] = None,
    point_labels: Optional[np.ndarray] = None,
    box: Optional[np.ndarray] = None,
    mask_input: Optional[np.ndarray] = None,
    multimask_output: bool = True,
    return_logits: bool = False,
    normalize_coords=True,
    custom_token: dict = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    if not self._is_image_set:
        raise RuntimeError(
            "An image must be set with .set_image(...) before mask prediction."
        )
    mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
        point_coords, point_labels, box, mask_input, normalize_coords
    )
    masks, iou_predictions, low_res_masks = self._predict(
        unnorm_coords,
        labels,
        unnorm_box,
        mask_input,
        multimask_output,
        return_logits=return_logits,
        custom_token=custom_token,
    )
    mlp_scores = self.model.sam_mask_decoder.scores

    masks_np = masks.squeeze(0).float().detach().cpu().numpy()
    iou_predictions_np = iou_predictions.squeeze(0).float().detach().cpu().numpy()
    low_res_masks_np = low_res_masks.squeeze(0).float().detach().cpu().numpy()
    return masks_np, iou_predictions_np, low_res_masks_np, mlp_scores


@torch.no_grad()
def _predict(
    self,
    point_coords: Optional[torch.Tensor],
    point_labels: Optional[torch.Tensor],
    boxes: Optional[torch.Tensor] = None,
    mask_input: Optional[torch.Tensor] = None,
    multimask_output: bool = True,
    return_logits: bool = False,
    img_idx: int = -1,
    custom_token: dict = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not self._is_image_set:
        raise RuntimeError(
            "An image must be set with .set_image(...) before mask prediction."
        )
    if point_coords is not None:
        concat_points = (point_coords, point_labels)
    else:
        concat_points = None
    if True:
        # Embed prompts
        if boxes is not None:
            box_coords = boxes.reshape(-1, 2, 2)
            box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=boxes.device)
            box_labels = box_labels.repeat(boxes.size(0), 1)
            # we merge "boxes" and "points" into a single "concat_points" input (where
            # boxes are added at the beginning) to sam_prompt_encoder
            if concat_points is not None:
                concat_coords = torch.cat([box_coords, concat_points[0]], dim=1)
                concat_labels = torch.cat([box_labels, concat_points[1]], dim=1)
                concat_points = (concat_coords, concat_labels)
            else:
                concat_points = (box_coords, box_labels)

        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=concat_points,
            boxes=None,
            masks=mask_input,
        )

        # Predict masks
        batched_mode = (
            concat_points is not None and concat_points[0].shape[0] > 1
        )  # multi object prediction
        high_res_features = [
            feat_level[img_idx].unsqueeze(0)
            for feat_level in self._features["high_res_feats"]
        ]
        low_res_masks, iou_predictions, _, _ = self.model.sam_mask_decoder(
            image_embeddings=self._features["image_embed"][img_idx].unsqueeze(0),
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
            custom_token=custom_token,
        )

        # Upscale the masks to the original image resolution
        masks = self._transforms.postprocess_masks(
            low_res_masks, self._orig_hw[img_idx]
        )
        low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)
        if not return_logits:
            masks = masks > self.mask_threshold

        return masks, iou_predictions, low_res_masks


def forward(
    self,
    image_embeddings: torch.Tensor,
    image_pe: torch.Tensor,
    sparse_prompt_embeddings: torch.Tensor,
    dense_prompt_embeddings: torch.Tensor,
    multimask_output: bool,
    repeat_image: bool,
    high_res_features: Optional[List[torch.Tensor]] = None,
    custom_token: dict = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    masks, iou_pred, mask_tokens_out, object_score_logits = self.predict_masks(
        image_embeddings=image_embeddings,
        image_pe=image_pe,
        sparse_prompt_embeddings=sparse_prompt_embeddings,
        dense_prompt_embeddings=dense_prompt_embeddings,
        repeat_image=repeat_image,
        high_res_features=high_res_features,
        custom_token=custom_token
    )
    if multimask_output:
        masks = masks[:, 1:, :, :]
        iou_pred = iou_pred[:, 1:]
    elif self.dynamic_multimask_via_stability and not self.training:
        masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
    else:
        masks = masks[:, 0:1, :, :]
        iou_pred = iou_pred[:, 0:1]
    if multimask_output and self.use_multimask_token_for_obj_ptr:
        sam_tokens_out = mask_tokens_out[:, 1:]
    else:
        sam_tokens_out = mask_tokens_out[:, 0:1]
    return masks, iou_pred, sam_tokens_out, object_score_logits


# def attention_forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
#     # Input projections
#     q = self.q_proj(q)
#     k = self.k_proj(k)
#     v = self.v_proj(v)
#
#     # Separate into heads
#     q = self._separate_heads(q, self.num_heads)
#     k = self._separate_heads(k, self.num_heads)
#     v = self._separate_heads(v, self.num_heads)
#
#     dropout_p = self.dropout_p if self.training else 0.0
#     # Read out environment variable "DROPOUT_P" to set dropout_p
#     if "DROPOUT_P" in os.environ:
#         dropout_p = float(os.environ["DROPOUT_P"])
#
#     # Attention
#     with torch.backends.cuda.sdp_kernel(
#             enable_flash=USE_FLASH_ATTN,
#             # if Flash attention kernel is off, then math kernel needs to be enabled
#             enable_math=(OLD_GPU and dropout_p > 0.0) or MATH_KERNEL_ON,
#             enable_mem_efficient=OLD_GPU,
#     ):
#         out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
#
#     out = self._recombine_heads(out)
#     out = self.out_proj(out)
#
#     return out


def patch_sam2(instance, model_dir=None):
    """ Patches Sam2 to predict uncertainty quantification methods """
    if isinstance(instance, SAM2ImagePredictor):
        instance.predict = types.MethodType(predict, instance)
        instance._predict = types.MethodType(_predict, instance)
        instance = instance.model.sam_mask_decoder
    elif isinstance(instance, SAM2Base):
        instance = instance.sam_mask_decoder
    else:
        raise ValueError(f"Unknown instance type: {type(instance)}")

    # Add token storage
    instance.__setattr__("iou_token_out", None)
    instance.__setattr__("mask_tokens_out", None)
    instance.predict_masks = types.MethodType(predict_masks, instance)
    instance.forward = types.MethodType(forward, instance)

    # Load models if existing
    instance.__setattr__("model_dir", model_dir)
    instance.__setattr__("regression_models", dict())
    instance.__setattr__("scores", dict())

    if model_dir is not None:
        for model in os.listdir(model_dir):
            weights = join(model_dir, model, "model.pth")
            if exists(weights):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                state_dict = torch.load(weights, map_location=device)
                mlp = MLP().to(device)
                mlp.load_state_dict(state_dict)
                mlp.eval()
                instance.regression_models[model] = mlp
                instance.scores[model] = None

    # # list all nn.Module in the instance
    # modules = instance.named_modules()
    # for name, module in modules:
    #     if isinstance(module, Attention):
    #         module.forward = types.MethodType(attention_forward, module)
