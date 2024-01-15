import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..model_utils.model_nms_utils import class_agnostic_nms
from ..model_utils.unsupervised_regression_utils import single_frame_geometric_consistency_loss, \
    single_frame_boundary_loss, single_frame_contrast_loss, single_frame_boundary_residual_loss
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import common_utils, loss_utils


class PointHeadTemplate(nn.Module):
    def __init__(self, model_cfg, num_class):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class

        self.build_losses(self.model_cfg.LOSS_CONFIG)
        self.forward_ret_dict = None

    def build_losses(self, losses_cfg):
        self.add_module(
            'cls_loss_func',
            loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
            # loss_utils.SigmoidFocalClassificationLoss(alpha=0.875, gamma=2.0)
            # loss_utils.SigmoidFocalClassificationLoss(alpha=0.1, gamma=2.0)
        )
        if self.model_cfg.LOSS_CONFIG.get('USE_UNSUPERVISED', False):
            self.p2_supervision = losses_cfg.get(
                "P2_SUPERVISION", None)
        else:
            reg_loss_type = losses_cfg.get('LOSS_REG', None)
            if reg_loss_type == 'smooth-l1':
                self.reg_loss_func = F.smooth_l1_loss
            elif reg_loss_type == 'l1':
                self.reg_loss_func = F.l1_loss
            elif reg_loss_type == 'WeightedSmoothL1Loss':
                self.reg_loss_func = loss_utils.WeightedSmoothL1Loss(
                    code_weights=losses_cfg.LOSS_WEIGHTS.get('code_weights', None)
                )
            else:
                self.reg_loss_func = F.smooth_l1_loss

    @staticmethod
    def make_fc_layers(fc_cfg, input_channels, output_channels):
        fc_layers = []
        c_in = input_channels
        for k in range(0, fc_cfg.__len__()):
            fc_layers.extend([
                nn.Linear(c_in, fc_cfg[k], bias=False),
                nn.BatchNorm1d(fc_cfg[k]),
                nn.ReLU(),
            ])
            c_in = fc_cfg[k]
        fc_layers.append(nn.Linear(c_in, output_channels, bias=True))
        return nn.Sequential(*fc_layers)
    
    def get_nms_proposals(self, batch_size, batch_index, batch_cls_scores, 
                                batch_box_preds, nms_config):
        """
        Args:
            batch_size:
            batch_index: 
            batch_cls_scores: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
            batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
            nms_config:

        Returns:
            batch_dict:
                rois: (B, num_rois, 7+C)
                roi_scores: (B, num_rois)
                roi_labels: (B, num_rois)

        """
        # batch_size = batch_dict['batch_size']
        # batch_box_preds = batch_dict['batch_box_preds']
        # batch_cls_preds = batch_dict['batch_cls_preds']
        rois = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE, batch_box_preds.shape[-1]))
        roi_scores = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE))
        # roi_labels = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE), dtype=torch.long)

        for index in range(batch_size):
            # assert batch_cls_preds.shape.__len__() == 2
            batch_mask = (batch_index == index)
            box_preds = batch_box_preds[batch_mask]
            # cls_preds = batch_cls_preds[batch_mask]
            cur_roi_scores = batch_cls_scores[batch_mask]

            # cur_roi_scores, cur_roi_labels = torch.max(cls_preds, dim=1)

            with torch.no_grad():  # get the nms without backprop
                selected, selected_scores = class_agnostic_nms(
                    box_scores=cur_roi_scores, box_preds=box_preds, nms_config=nms_config
                )

            rois[index, :len(selected), :] = box_preds[selected]
            roi_scores[index, :len(selected)] = cur_roi_scores[selected]
            # roi_labels[index, :len(selected)] = cur_roi_labels[selected]
        return rois, roi_scores
    

    def get_foreground_proposals(self, batch_size, batch_index, point_box_preds, p2_scores, fg_threshold, num_box_samples, point_int_cls_preds=None):
        """
        Args:
            batch_size:
            batch_index: 
            point_box_preds: (B * npoints, 7+C)
            p2_scores: (B * npoints)
            nms_config:

        Returns:
            batch_dict:
                rois: (B, num_rois, 7+C)
                roi_scores: (B, num_rois)
                roi_labels: (B, num_rois)

        """
        boxes = point_box_preds.new_zeros((batch_size, num_box_samples, point_box_preds.shape[-1]))
        # fg_points = ptc[pp_score < fg_threshold]
        # idx = np.random.randint(fg_points.shape[0], size=num_samples)
        intermediate_cls = point_int_cls_preds.new_zeros((batch_size, num_box_samples, point_int_cls_preds.shape[-1])) if point_int_cls_preds is not None else None

        for index in range(batch_size):
            # assert batch_cls_preds.shape.__len__() == 2
            batch_mask = (batch_index == index)
            box_preds = point_box_preds[batch_mask]
            batch_p2_scores = p2_scores[batch_mask]

            fg_mask = batch_p2_scores < fg_threshold
            selected = np.random.choice(np.where(fg_mask.cpu().numpy())[0], num_box_samples) 

            boxes[index, :len(selected), :] = box_preds[selected]
            # roi_labels[index, :len(selected)] = cur_roi_labels[selected]
            if intermediate_cls is not None:
                intermediate_cls[index, :len(selected), :] = point_int_cls_preds[batch_mask][selected]
        return boxes, intermediate_cls
    
    def get_foreground_proposals_sample(self, fg_threshold, num_box_samples):
        """
        Args:
            batch_size:
            batch_index: 
            point_box_preds: (B * npoints, 7+C)
            p2_scores: (B * npoints)
            nms_config:

        Returns:
            batch_dict:
                rois: (B, num_rois, 7+C)
                roi_scores: (B, num_rois)
                roi_labels: (B, num_rois)

        """
        points = self.forward_ret_dict['points']
        batch_index = self.forward_ret_dict['batch_index']
        batch_size = self.forward_ret_dict['batch_size']
        # point_cls_scores = self.forward_ret_dict['point_cls_scores']
        # point_pred_classes = self.forward_ret_dict['point_pred_classes']
        point_pred_classes = self.forward_ret_dict['point_cls_preds'].max(dim=-1).indices
        point_box_preds = self.forward_ret_dict['point_box_preds']
        p2_scores = self.forward_ret_dict['p2_score']
        point_boxes3d = self.box_coder.decode_torch(point_box_preds, points, point_pred_classes + 1)[:, :7]  # total_n_points x 7
        boxes = point_boxes3d.new_zeros((batch_size, num_box_samples, point_boxes3d.shape[-1]))

        for index in range(batch_size):
            # assert batch_cls_preds.shape.__len__() == 2
            batch_mask = (batch_index == index)
            box_preds = point_boxes3d[batch_mask]
            batch_p2_scores = p2_scores[batch_mask]

            fg_mask = batch_p2_scores < fg_threshold
            if fg_mask.sum() > 0:
                selected = np.random.choice(np.where(fg_mask.cpu().numpy())[0], num_box_samples)
            else:
                selected = []

            boxes[index, :len(selected), :] = box_preds[selected]
        return boxes
    

    def assign_stack_targets(self, points, gt_boxes, extend_gt_boxes=None,
                             ret_box_labels=False, ret_part_labels=False,
                             set_ignore_flag=True, use_ball_constraint=False, central_radius=2.0):
        """
        Args:
            points: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
            gt_boxes: (B, M, 8)
            extend_gt_boxes: [B, M, 8]
            ret_box_labels:
            ret_part_labels:
            set_ignore_flag:
            use_ball_constraint:
            central_radius:

        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_box_labels: (N1 + N2 + N3 + ..., code_size)

        """
        assert len(points.shape) == 2 and points.shape[1] == 4, 'points.shape=%s' % str(points.shape)
        assert len(gt_boxes.shape) == 3 and gt_boxes.shape[2] == 8, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert extend_gt_boxes is None or len(extend_gt_boxes.shape) == 3 and extend_gt_boxes.shape[2] == 8, \
            'extend_gt_boxes.shape=%s' % str(extend_gt_boxes.shape)
        assert set_ignore_flag != use_ball_constraint, 'Choose one only!'
        batch_size = gt_boxes.shape[0]
        bs_idx = points[:, 0]
        point_cls_labels = points.new_zeros(points.shape[0]).long()
        point_box_labels = gt_boxes.new_zeros((points.shape[0], 8)) if ret_box_labels else None
        point_part_labels = gt_boxes.new_zeros((points.shape[0], 3)) if ret_part_labels else None
        for k in range(batch_size):
            bs_mask = (bs_idx == k)
            points_single = points[bs_mask][:, 1:4]
            point_cls_labels_single = point_cls_labels.new_zeros(bs_mask.sum())
            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                points_single.unsqueeze(dim=0), gt_boxes[k:k + 1, :, 0:7].contiguous()
            ).long().squeeze(dim=0)
            box_fg_flag = (box_idxs_of_pts >= 0)
            if set_ignore_flag:
                extend_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    points_single.unsqueeze(dim=0), extend_gt_boxes[k:k+1, :, 0:7].contiguous()
                ).long().squeeze(dim=0)
                fg_flag = box_fg_flag
                ignore_flag = fg_flag ^ (extend_box_idxs_of_pts >= 0)
                point_cls_labels_single[ignore_flag] = -1
            elif use_ball_constraint:
                box_centers = gt_boxes[k][box_idxs_of_pts][:, 0:3].clone()
                box_centers[:, 2] += gt_boxes[k][box_idxs_of_pts][:, 5] / 2
                ball_flag = ((box_centers - points_single).norm(dim=1) < central_radius)
                fg_flag = box_fg_flag & ball_flag
            else:
                raise NotImplementedError

            gt_box_of_fg_points = gt_boxes[k][box_idxs_of_pts[fg_flag]]
            point_cls_labels_single[fg_flag] = 1 if self.num_class == 1 else gt_box_of_fg_points[:, -1].long()
            point_cls_labels[bs_mask] = point_cls_labels_single

            if ret_box_labels and gt_box_of_fg_points.shape[0] > 0:
                point_box_labels_single = point_box_labels.new_zeros((bs_mask.sum(), 8))
                fg_point_box_labels = self.box_coder.encode_torch(
                    gt_boxes=gt_box_of_fg_points[:, :-1], points=points_single[fg_flag],
                    gt_classes=gt_box_of_fg_points[:, -1].long()
                )
                point_box_labels_single[fg_flag] = fg_point_box_labels
                point_box_labels[bs_mask] = point_box_labels_single

            if ret_part_labels:
                point_part_labels_single = point_part_labels.new_zeros((bs_mask.sum(), 3))
                transformed_points = points_single[fg_flag] - gt_box_of_fg_points[:, 0:3]
                transformed_points = common_utils.rotate_points_along_z(
                    transformed_points.view(-1, 1, 3), -gt_box_of_fg_points[:, 6]
                ).view(-1, 3)
                offset = torch.tensor([0.5, 0.5, 0.5]).view(1, 3).type_as(transformed_points)
                point_part_labels_single[fg_flag] = (transformed_points / gt_box_of_fg_points[:, 3:6]) + offset
                point_part_labels[bs_mask] = point_part_labels_single

        targets_dict = {
            'point_cls_labels': point_cls_labels,
            'point_box_labels': point_box_labels,
            'point_part_labels': point_part_labels
        }
        return targets_dict

    def get_cls_layer_loss(self, tb_dict=None):
        point_cls_labels = self.forward_ret_dict['point_cls_labels'].view(-1)
        point_cls_preds = self.forward_ret_dict['point_cls_preds'].view(-1, self.num_class)

        # encourage exploring foreground points
        if self.model_cfg.get('RL', False) and self.model_cfg.get('P2_CLS_EXPLORATION', True):
            p2_score = self.forward_ret_dict['p2_score']
            points = self.forward_ret_dict['points']

            point_cls_p2_labels = (p2_score.view(-1) < 0.2).int()
            point_cls_p2_labels[points[:, 2] > 2.5] = 0
            point_cls_labels = torch.logical_or(point_cls_labels, point_cls_p2_labels).int()

        positives = (point_cls_labels > 0)
        negative_cls_weights = (point_cls_labels == 0) * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        pos_normalizer = positives.sum(dim=0).float()
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)

        one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), self.num_class + 1)
        one_hot_targets.scatter_(-1, (point_cls_labels * (point_cls_labels >= 0).long()).unsqueeze(dim=-1).long(), 1.0)
        one_hot_targets = one_hot_targets[..., 1:]
        cls_loss_src = self.cls_loss_func(point_cls_preds, one_hot_targets, weights=cls_weights)
        point_loss_cls = cls_loss_src.sum()

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_cls = point_loss_cls * loss_weights_dict['point_cls_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({
            'point_loss_cls': point_loss_cls.item(),
            'point_pos_num': pos_normalizer.item()
        })
        return point_loss_cls, tb_dict

    def get_part_layer_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict['point_cls_labels'] > 0
        pos_normalizer = max(1, (pos_mask > 0).sum().item())
        point_part_labels = self.forward_ret_dict['point_part_labels']
        point_part_preds = self.forward_ret_dict['point_part_preds']
        point_loss_part = F.binary_cross_entropy(torch.sigmoid(point_part_preds), point_part_labels, reduction='none')
        point_loss_part = (point_loss_part.sum(dim=-1) * pos_mask.float()).sum() / (3 * pos_normalizer)

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_part = point_loss_part * loss_weights_dict['point_part_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'point_loss_part': point_loss_part.item()})
        return point_loss_part, tb_dict

    def get_box_layer_loss(self, tb_dict=None):
        pos_mask = self.forward_ret_dict['point_cls_labels'] > 0
        point_box_labels = self.forward_ret_dict['point_box_labels']
        point_box_preds = self.forward_ret_dict['point_box_preds']

        reg_weights = pos_mask.float()
        pos_normalizer = pos_mask.sum().float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        point_loss_box_src = self.reg_loss_func(
            point_box_preds[None, ...], point_box_labels[None, ...], weights=reg_weights[None, ...]
        )
        point_loss_box = point_loss_box_src.sum()

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_box = point_loss_box * loss_weights_dict['point_box_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({'point_loss_box': point_loss_box.item()})
        return point_loss_box, tb_dict
    
    def get_unsupervised_cls_loss(self, tb_dict=None):
        # from rote-da
        num_fg_cls = 1
        assert num_fg_cls == 1, "only support 1 class."
        p2_score = self.forward_ret_dict['p2_score']
        points = self.forward_ret_dict['points']
        point_cls_preds = self.forward_ret_dict['point_cls_preds'].view(-1, num_fg_cls)  # npoints x num_class = 1
        point_cls_labels = (p2_score.view(-1, num_fg_cls) < self.p2_supervision.FG_THRESHOLD).int()  # NOTE: this is assuming a single class, torch.zeros(point_cls_preds.shape)

        # filter points that are too high
        if self.p2_supervision.get('MAX_HEIGHT', None) is not None:
            # print("filtering points that are too high")
            point_cls_labels[points[:, 2] > self.p2_supervision.MAX_HEIGHT] = 0

        self.forward_ret_dict['point_cls_labels'] = point_cls_labels
        ignore_mask = (p2_score >= self.p2_supervision.FG_THRESHOLD) & \
            (p2_score < self.p2_supervision.BG_THRESHOLD)

        preserved_mask = torch.logical_not(ignore_mask)
        point_cls_labels = point_cls_labels[preserved_mask]  # npoints x 1
        point_cls_preds = point_cls_preds[preserved_mask]
        p2_score = p2_score[preserved_mask]

        one_hot_targets = point_cls_labels  # npoints x 1

        positives = point_cls_labels > 0
        pos_normalizer = positives.sum(dim=0).float()
        cls_weights = torch.ones_like(p2_score)  # npoints
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_loss_src = self.cls_loss_func(
            point_cls_preds, one_hot_targets, weights=cls_weights)
        point_loss_cls = cls_loss_src.sum()

        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({
            'ignored_cls_num': ignore_mask.sum().item(),
        })
        for _c in range(num_fg_cls):
            tb_dict.update({
                f'class_{_c+1}_pos': (point_cls_labels == (_c + 1)).sum().item(),
            })
        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        total_loss = point_loss_cls * loss_weights_dict['point_cls_weight']

        tb_dict.update({
            'dense_point_loss_cls': point_loss_cls.item(),
            # 'dense_point_total_loss': total_loss.item(),
            'dense_point_pos_num': pos_normalizer.item()
        })
        return total_loss, tb_dict
    
    def get_unsupervised_reg_loss(self, tb_dict=None, use_intermediate_cls=False):
        points = self.forward_ret_dict['points']
        batch_index = self.forward_ret_dict['batch_index']
        batch_size = self.forward_ret_dict['batch_size']
        # point_cls_scores = self.forward_ret_dict['point_cls_scores']
        point_pred_classes = self.forward_ret_dict['point_pred_classes']
        p2_scores = self.forward_ret_dict['p2_score']
        if use_intermediate_cls:
            point_int_cls_preds = self.forward_ret_dict['point_int_cls_preds']
            _, point_pred_classes = point_int_cls_preds.max(dim=-1)
        else:
            point_int_cls_preds = None
        
        # decode box into global coordinates
        point_box_preds = self.forward_ret_dict['point_box_preds']
        # _, pred_classes = point_cls_scores.max(dim=-1)
        if self.model_cfg.LOSS_CONFIG.get('REG_LOSS_TYPE', 'boundary') == 'residual':
            point_boxes3d = self.box_coder.decode_torch_soft(point_box_preds, points, pred_classes=point_int_cls_preds)[:, :7]
        else:
            point_boxes3d = self.box_coder.decode_torch(point_box_preds, points, point_pred_classes + 1)[:, :7]
        
        # # NMS
        # nms_config = self.model_cfg.NMS_CONFIG
        # pred_boxes_3d, _ = self.get_nms_proposals(batch_size, batch_index, point_cls_scores, 
        #                         point_boxes3d, nms_config)  # B x nboxes x 7

        # Select loss only at foreground points
        pred_boxes_3d, intermediate_cls_preds = self.get_foreground_proposals(batch_size, batch_index, point_boxes3d, p2_scores, 
                                                      self.p2_supervision.FG_THRESHOLD, self.p2_supervision.NUM_BOX_SAMPLES, point_int_cls_preds=point_int_cls_preds)
        
        # Get proper shapes
        xyz_batch_cnt = points.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (batch_index == bs_idx).sum()

        assert xyz_batch_cnt.min() == xyz_batch_cnt.max(), "xyz_batch_cnt: " + str(xyz_batch_cnt)
        ptc = points.view(batch_size, -1, 3)  # B x npoints x 3
        p2_scores = p2_scores.view(batch_size, -1)  # B x npoints
        
        # get loss from single frame (p2 guided foreground)
        # foreground_inside_loss, background_inside_loss, foreground_outside_loss = \
        #     single_frame_geometric_consistency_loss(ptc, p2_scores, pred_boxes_3d, batch_size, self.model_cfg.LOSS_CONFIG)
        # unsup_loss_reg, foreground_loss, background_loss, size_loss = \
        #     single_frame_boundary_loss(ptc, p2_scores, pred_boxes_3d, batch_size, self.model_cfg.LOSS_CONFIG, cls_int_logits=intermediate_cls_preds)
        if self.model_cfg.LOSS_CONFIG.get('REG_LOSS_TYPE', 'boundary') == 'contrast':
            unsup_loss_reg, foreground_loss, background_loss, size_loss = \
                single_frame_contrast_loss(ptc, p2_scores, pred_boxes_3d, batch_size, self.model_cfg.LOSS_CONFIG)
        elif self.model_cfg.LOSS_CONFIG.get('REG_LOSS_TYPE', 'boundary') == 'residual':
            batch_box_residual_size = point_box_preds[..., 3:6]
            unsup_loss_reg, foreground_loss, background_loss, size_loss = \
                single_frame_boundary_residual_loss(ptc, p2_scores, pred_boxes_3d, batch_box_residual_size, batch_size, self.model_cfg.LOSS_CONFIG)
        else:
            unsup_loss_reg, foreground_loss, background_loss, size_loss = \
                single_frame_boundary_loss(ptc, p2_scores, pred_boxes_3d, batch_size, self.model_cfg.LOSS_CONFIG, cls_int_logits=intermediate_cls_preds)
        
        if tb_dict is None:
            tb_dict = {}
        # tb_dict.update({
        #     "dense_foreground_inside_loss": foreground_inside_loss.item(),
        #     "dense_background_inside_loss": background_inside_loss.item(),
        #     "dense_foreground_outside_loss": foreground_outside_loss.item()
        # })
        tb_dict.update({
            "dense_foreground_loss": foreground_loss,
            "dense_background_loss": background_loss,
            "dense_size_loss": size_loss,
        })

        # unsup_loss_reg = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['foreground_inside_loss'] * foreground_inside_loss + \
        #                  self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['background_inside_loss'] * background_inside_loss + \
        #                  self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['foreground_outside_loss'] * foreground_outside_loss
        
        tb_dict.update({
            "dense_total_unsup_loss_reg": unsup_loss_reg.item(),
        })

        return unsup_loss_reg, tb_dict
    

    def generate_predicted_boxes(self, points, point_cls_preds, point_box_preds):
        """
        Args:
            points: (N, 3)
            point_cls_preds: (N, num_class)
            point_box_preds: (N, box_code_size)
        Returns:
            point_cls_preds: (N, num_class)
            point_box_preds: (N, box_code_size)

        """
        _, pred_classes = point_cls_preds.max(dim=-1)
        point_box_preds = self.box_coder.decode_torch(point_box_preds, points, pred_classes + 1)

        return point_cls_preds, point_box_preds

    def forward(self, **kwargs):
        raise NotImplementedError
