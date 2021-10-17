import torch
import torch.nn.functional as F

from mmdet.core import (bbox2result, bbox2roi, bbox_mapping, merge_aug_bboxes,
                        merge_aug_masks, multiclass_nms)
from ..builder import HEADS, build_head, build_roi_extractor
from .cascade_roi_head import CascadeRoIHead


@HEADS.register_module()
class SCNetRoIHead(CascadeRoIHead):
    """RoIHead for `SCNet <https://arxiv.org/abs/2012.10150>`_.

    Args:
        num_stages (int): number of cascade stages.
        stage_loss_weights (list): loss weight of cascade stages.
        semantic_roi_extractor (dict): config to init semantic roi extractor.
        semantic_head (dict): config to init semantic head.
        feat_relay_head (dict): config to init feature_relay_head.
        glbctx_head (dict): config to init global context head.
    """

    def __init__(self,
                 num_stages,
                 stage_loss_weights,
                 semantic_roi_extractor=None,
                 semantic_head=None,
                 feat_relay_head=None,
                 glbctx_head=None,
                 **kwargs):
        super(SCNetRoIHead, self).__init__(num_stages, stage_loss_weights,
                                           **kwargs)

        assert self.with_bbox
        assert not self.with_shared_head  # shared head is not supported


    def init_weights(self, pretrained):
        """Initialize the weights in head.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        for i in range(self.num_stages):
            if self.with_bbox:
                self.bbox_roi_extractor[i].init_weights()
                self.bbox_head[i].init_weights()
        if self.with_glbctx:
            self.glbctx_head.init_weights()

    @property
    def with_glbctx(self):
        """bool: whether the head has global context head"""
        return hasattr(self, 'glbctx_head') and self.glbctx_head is not None

    def _fuse_glbctx(self, roi_feats, glbctx_feat, rois):
        """Fuse global context feats with roi feats."""
        assert roi_feats.size(0) == rois.size(0)
        img_inds = torch.unique(rois[:, 0].cpu(), sorted=True).long()
        fused_feats = torch.zeros_like(roi_feats)
        for img_id in img_inds:
            inds = (rois[:, 0] == img_id.item())
            fused_feats[inds] = roi_feats[inds] + glbctx_feat[img_id]
        return fused_feats

    def _slice_pos_feats(self, feats, sampling_results):
        """Get features from pos rois."""
        num_rois = [res.bboxes.size(0) for res in sampling_results]
        num_pos_rois = [res.pos_bboxes.size(0) for res in sampling_results]
        inds = torch.zeros(sum(num_rois), dtype=torch.bool)
        start = 0
        for i in range(len(num_rois)):
            start = 0 if i == 0 else start + num_rois[i - 1]
            stop = start + num_pos_rois[i]
            inds[start:stop] = 1
        sliced_feats = feats[inds]
        return sliced_feats

    def _bbox_forward(self,
                      stage,
                      x,
                      rois,
                      semantic_feat=None,
                      glbctx_feat=None):
        """Box head forward function used in both training and testing."""
        bbox_roi_extractor = self.bbox_roi_extractor[stage]
        bbox_head = self.bbox_head[stage]
        bbox_feats = bbox_roi_extractor(
            x[:len(bbox_roi_extractor.featmap_strides)], rois)
        if self.with_glbctx and glbctx_feat is not None:
            bbox_feats = self._fuse_glbctx(bbox_feats, glbctx_feat, rois)
        cls_score, bbox_pred, relayed_feat = bbox_head(
            bbox_feats, return_shared_feat=True)

        bbox_results = dict(
            cls_score=cls_score,
            bbox_pred=bbox_pred,
            relayed_feat=relayed_feat)
        return bbox_results

    def _bbox_forward_train(self,
                            stage,
                            x,
                            sampling_results,
                            gt_bboxes,
                            gt_labels,
                            rcnn_train_cfg,
                            semantic_feat=None,
                            glbctx_feat=None):
        """Run forward function and calculate loss for box head in training."""
        bbox_head = self.bbox_head[stage]
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(
            stage,
            x,
            rois,
            glbctx_feat=glbctx_feat)

        bbox_targets = bbox_head.get_targets(sampling_results, gt_bboxes,
                                             gt_labels, rcnn_train_cfg)
        loss_bbox = bbox_head.loss(bbox_results['cls_score'],
                                   bbox_results['bbox_pred'], rois,
                                   *bbox_targets)

        bbox_results.update(
            loss_bbox=loss_bbox, rois=rois, bbox_targets=bbox_targets)
        return bbox_results

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      gt_semantic_seg=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            proposal_list (list[Tensors]): list of region proposals.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None, list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None, Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            gt_semantic_seg (None, list[Tensor]): semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = dict()

        semantic_feat = None

        # global context branch
        if self.with_glbctx:
            mc_pred, glbctx_feat = self.glbctx_head(x)
            loss_glbctx = self.glbctx_head.loss(mc_pred, gt_labels)
            losses['loss_glbctx'] = loss_glbctx
        else:
            glbctx_feat = None

        for i in range(self.num_stages):
            self.current_stage = i
            rcnn_train_cfg = self.train_cfg[i]
            lw = self.stage_loss_weights[i]

            # assign gts and sample proposals
            sampling_results = []
            bbox_assigner = self.bbox_assigner[i]
            bbox_sampler = self.bbox_sampler[i]
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]

            for j in range(num_imgs):
                assign_result = bbox_assigner.assign(proposal_list[j],
                                                     gt_bboxes[j],
                                                     gt_bboxes_ignore[j],
                                                     gt_labels[j])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[j],
                    gt_bboxes[j],
                    gt_labels[j],
                    feats=[lvl_feat[j][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

            bbox_results = \
                self._bbox_forward_train(
                    i, x, sampling_results, gt_bboxes, gt_labels,
                    rcnn_train_cfg, semantic_feat, glbctx_feat)
            roi_labels = bbox_results['bbox_targets'][0]

            for name, value in bbox_results['loss_bbox'].items():
                losses[f's{i}.{name}'] = (
                    value * lw if 'loss' in name else value)

            # refine boxes
            if i < self.num_stages - 1:
                pos_is_gts = [res.pos_is_gt for res in sampling_results]
                with torch.no_grad():
                    proposal_list = self.bbox_head[i].refine_bboxes(
                        bbox_results['rois'], roi_labels,
                        bbox_results['bbox_pred'], pos_is_gts, img_metas)


        return losses

    def simple_test(self, x, proposal_list, img_metas, rescale=False):
        """Test without augmentation."""
#         if self.with_semantic:
#             _, semantic_feat = self.semantic_head(x)
#         else:
#             semantic_feat = None

        if self.with_glbctx:
            mc_pred, glbctx_feat = self.glbctx_head(x)
        else:
            glbctx_feat = None

        num_imgs = len(proposal_list)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # "ms" in variable names means multi-stage
        ms_scores = []
        rcnn_test_cfg = self.test_cfg

        rois = bbox2roi(proposal_list)
        for i in range(self.num_stages):
            bbox_head = self.bbox_head[i]
            bbox_results = self._bbox_forward(
                i,
                x,
                rois,
#                 semantic_feat=semantic_feat,
                glbctx_feat=glbctx_feat)
            # split batch bbox prediction back to each image
            cls_score = bbox_results['cls_score']
            bbox_pred = bbox_results['bbox_pred']
            num_proposals_per_img = tuple(len(p) for p in proposal_list)
            rois = rois.split(num_proposals_per_img, 0)
            cls_score = cls_score.split(num_proposals_per_img, 0)
            bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            ms_scores.append(cls_score)

            if i < self.num_stages - 1:
                bbox_label = [s[:, :-1].argmax(dim=1) for s in cls_score]
                rois = torch.cat([
                    bbox_head.regress_by_class(rois[i], bbox_label[i],
                                               bbox_pred[i], img_metas[i])
                    for i in range(num_imgs)
                ])

        # average scores of each image by stages
        cls_score = [
            sum([score[i] for score in ms_scores]) / float(len(ms_scores))
            for i in range(num_imgs)
        ]

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(num_imgs):
            det_bbox, det_label = self.bbox_head[-1].get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        det_bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head[-1].num_classes)
            for i in range(num_imgs)
        ]


        return det_bbox_results

    def aug_test(self, img_feats, proposal_list, img_metas, rescale=False):
        
        
#         if self.with_semantic:
#             semantic_feats = [
#                 self.semantic_head(feat)[1] for feat in img_feats
#             ]
#         else:
#             semantic_feats = [None] * len(img_metas)

        if self.with_glbctx:
            glbctx_feats = [self.glbctx_head(feat)[1] for feat in img_feats]
        else:
            glbctx_feats = [None] * len(img_metas)

        rcnn_test_cfg = self.test_cfg
        aug_bboxes = []
        aug_scores = []
#         for x, img_meta, semantic_feat, glbctx_feat in zip(
#                 img_feats, img_metas, semantic_feats, glbctx_feats):
        for x, img_meta, glbctx_feat in zip(image_feats, img_metas, glbctx_feats):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']

            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip)
            # "ms" in variable names means multi-stage
            ms_scores = []

            rois = bbox2roi([proposals])
            for i in range(self.num_stages):
                bbox_head = self.bbox_head[i]
                bbox_results = self._bbox_forward(
                    i,
                    x,
                    rois,
#                     semantic_feat=semantic_feat,
                    glbctx_feat=glbctx_feat)
                ms_scores.append(bbox_results['cls_score'])
                if i < self.num_stages - 1:
                    bbox_label = bbox_results['cls_score'].argmax(dim=1)
                    rois = bbox_head.regress_by_class(
                        rois, bbox_label, bbox_results['bbox_pred'],
                        img_meta[0])

            cls_score = sum(ms_scores) / float(len(ms_scores))
            bboxes, scores = self.bbox_head[-1].get_bboxes(
                rois,
                cls_score,
                bbox_results['bbox_pred'],
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores,
                                                rcnn_test_cfg.score_thr,
                                                rcnn_test_cfg.nms,
                                                rcnn_test_cfg.max_per_img)

        det_bbox_results = bbox2result(det_bboxes, det_labels,
                                       self.bbox_head[-1].num_classes)

        return [det_bbox_results]
