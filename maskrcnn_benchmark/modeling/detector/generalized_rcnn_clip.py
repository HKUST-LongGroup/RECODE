# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads


class GeneralizedRCNNClip(nn.Module):
    """
    baseline for zero-shot SGG.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNNClip, self).__init__()
        self.cfg = cfg.clone()
        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)
        self.type = self.cfg.TYPE
        self.feat_type = self.cfg.FEAT_TYPE

    def forward(self, images, targets=None, logger=None, obj_features=None, proposals=None, spatials=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.feat_type == 'offline':
            images = None
            features = None
            proposals = proposals
            proposal_losses = {}
        else:
            images = to_image_list(images)
            features = self.backbone(images.tensors)
            proposals, proposal_losses = self.rpn(images, features, targets)

        # images = to_image_list(images)
        # features = self.backbone(images.tensors)
        # proposals, proposal_losses = self.rpn(images, features, targets)

        if self.roi_heads:
            if self.type == 'extract_clip_feature':
                clip_obj_features = self.roi_heads(images=images, features=features, proposals=proposals, targets=targets, logger=logger)
                return clip_obj_features
            if self.type == 'extract_proposal':
                extract_proposals = self.roi_heads(images=images, features=features, proposals=proposals, targets=targets, logger=logger)
                return extract_proposals

            x, result, detector_losses = self.roi_heads(images, features, proposals, targets, logger, obj_features=obj_features, spatials=spatials)
            
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}
        
        if self.training:
            losses = {}
            losses.update(detector_losses)
            if not self.cfg.MODEL.RELATION_ON:
                # During the relationship training stage, the rpn_head should be fixed, and no loss. 
                losses.update(proposal_losses)
            return losses

        return result
