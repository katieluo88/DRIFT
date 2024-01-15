import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils import box_coder_utils, common_utils, loss_utils
from ..model_utils.model_nms_utils import class_agnostic_nms
from ..model_utils.unsupervised_regression_utils import single_frame_geometric_consistency_loss, \
    single_frame_boundary_loss, single_frame_contrast_loss, filter_class_labels, \
    single_frame_boundary_residual_loss
from .target_assigner.proposal_target_layer import ProposalTargetLayer


class RoIHeadTemplate(nn.Module):
    def __init__(self, num_class, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.box_coder = getattr(box_coder_utils, self.model_cfg.TARGET_CONFIG.BOX_CODER)(
            **self.model_cfg.TARGET_CONFIG.get('BOX_CODER_CONFIG', {})
        )
        self.proposal_target_layer = ProposalTargetLayer(roi_sampler_cfg=self.model_cfg.TARGET_CONFIG)
        self.build_losses(self.model_cfg.LOSS_CONFIG)
        self.forward_ret_dict = None

    def build_losses(self, losses_cfg):
        if not self.model_cfg.LOSS_CONFIG.get('USE_UNSUPERVISED', False):
            self.add_module(
                'reg_loss_func',
                loss_utils.WeightedSmoothL1Loss(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'])
            )

    def make_fc_layers(self, input_channels, output_channels, fc_list):
        fc_layers = []
        pre_channel = input_channels
        for k in range(0, fc_list.__len__()):
            fc_layers.extend([
                nn.Conv1d(pre_channel, fc_list[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(fc_list[k]),
                nn.ReLU()
            ])
            pre_channel = fc_list[k]
            if self.model_cfg.DP_RATIO >= 0 and k == 0:
                fc_layers.append(nn.Dropout(self.model_cfg.DP_RATIO))
        fc_layers.append(nn.Conv1d(pre_channel, output_channels, kernel_size=1, bias=True))
        fc_layers = nn.Sequential(*fc_layers)
        return fc_layers

    @torch.no_grad()
    def proposal_layer(self, batch_dict, nms_config):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
            nms_config:

        Returns:
            batch_dict:
                rois: (B, num_rois, 7+C)
                roi_scores: (B, num_rois)
                roi_labels: (B, num_rois)

        """
        batch_size = batch_dict['batch_size']
        batch_box_preds = batch_dict['batch_box_preds']
        batch_cls_preds = batch_dict['batch_cls_preds']
        rois = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE, batch_box_preds.shape[-1]))
        roi_scores = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE))
        roi_labels = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE), dtype=torch.long)
        if 'batch_int_cls_preds' in batch_dict:
            batch_int_cls_preds = batch_dict['batch_int_cls_preds']
            roi_int_cls_logits = batch_int_cls_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE, batch_int_cls_preds.shape[-1]))
        else:
            roi_int_cls_logits = None

        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_cls_preds.shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_cls_preds'].shape.__len__() == 3
                batch_mask = index
            box_preds = batch_box_preds[batch_mask]
            cls_preds = batch_cls_preds[batch_mask]

            cur_roi_scores, cur_roi_labels = torch.max(cls_preds, dim=1)

            if nms_config.MULTI_CLASSES_NMS:
                raise NotImplementedError
            else:
                selected, selected_scores = class_agnostic_nms(
                    box_scores=cur_roi_scores, box_preds=box_preds, nms_config=nms_config
                )

            rois[index, :len(selected), :] = box_preds[selected]
            roi_scores[index, :len(selected)] = cur_roi_scores[selected]
            if roi_int_cls_logits is not None:
                cls_preds = batch_int_cls_preds[batch_mask]
                roi_int_cls_logits[index, :len(selected), :] = cls_preds[selected]
                cur_roi_scores, cur_roi_labels = torch.max(cls_preds, dim=1)
                roi_labels[index, :len(selected)] = cur_roi_labels[selected]
            else:
                roi_labels[index, :len(selected)] = cur_roi_labels[selected]

        batch_dict['rois'] = rois
        batch_dict['roi_scores'] = roi_scores
        batch_dict['roi_labels'] = roi_labels + 1
        batch_dict['has_class_labels'] = True if batch_cls_preds.shape[-1] > 1 or ('batch_int_cls_preds' in batch_dict and roi_int_cls_logits.shape[-1] > 1) else False
        if roi_int_cls_logits is not None:
            batch_dict['roi_int_cls_logits'] = roi_int_cls_logits
        batch_dict.pop('batch_index', None)
        return batch_dict

    def assign_targets(self, batch_dict):
        batch_size = batch_dict['batch_size']
        with torch.no_grad():
            targets_dict = self.proposal_target_layer.forward(batch_dict)

        rois = targets_dict['rois']  # (B, N, 7 + C)
        gt_of_rois = targets_dict['gt_of_rois']  # (B, N, 7 + C + 1)
        targets_dict['gt_of_rois_src'] = gt_of_rois.clone().detach()

        # canonical transformation
        roi_center = rois[:, :, 0:3]
        roi_ry = rois[:, :, 6] % (2 * np.pi)
        gt_of_rois[:, :, 0:3] = gt_of_rois[:, :, 0:3] - roi_center
        gt_of_rois[:, :, 6] = gt_of_rois[:, :, 6] - roi_ry

        # transfer LiDAR coords to local coords
        gt_of_rois = common_utils.rotate_points_along_z(
            points=gt_of_rois.view(-1, 1, gt_of_rois.shape[-1]), angle=-roi_ry.view(-1)
        ).view(batch_size, -1, gt_of_rois.shape[-1])

        # flip orientation if rois have opposite orientation
        heading_label = gt_of_rois[:, :, 6] % (2 * np.pi)  # 0 ~ 2pi
        opposite_flag = (heading_label > np.pi * 0.5) & (heading_label < np.pi * 1.5)
        heading_label[opposite_flag] = (heading_label[opposite_flag] + np.pi) % (2 * np.pi)  # (0 ~ pi/2, 3pi/2 ~ 2pi)
        flag = heading_label > np.pi
        heading_label[flag] = heading_label[flag] - np.pi * 2  # (-pi/2, pi/2)
        heading_label = torch.clamp(heading_label, min=-np.pi / 2, max=np.pi / 2)

        gt_of_rois[:, :, 6] = heading_label
        targets_dict['gt_of_rois'] = gt_of_rois
        return targets_dict
    
    # def assign_targets_unsupervised(self, batch_dict, targets_dict):
    #     targets_dict['batch_size'] = batch_dict['batch_size']
    #     targets_dict['points'] = batch_dict['points']
    #     targets_dict['p2_score'] = batch_dict['p2_score']
    #     return targets_dict

    def get_box_reg_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        code_size = self.box_coder.code_size
        reg_valid_mask = forward_ret_dict['reg_valid_mask'].view(-1)
        gt_boxes3d_ct = forward_ret_dict['gt_of_rois'][..., 0:code_size]
        gt_of_rois_src = forward_ret_dict['gt_of_rois_src'][..., 0:code_size].view(-1, code_size)
        rcnn_reg = forward_ret_dict['rcnn_reg']  # (rcnn_batch_size, C)
        roi_boxes3d = forward_ret_dict['rois']
        rcnn_batch_size = gt_boxes3d_ct.view(-1, code_size).shape[0]

        fg_mask = (reg_valid_mask > 0)
        fg_sum = fg_mask.long().sum().item()

        tb_dict = {}

        if loss_cfgs.REG_LOSS == 'smooth-l1':
            rois_anchor = roi_boxes3d.clone().detach().view(-1, code_size)
            rois_anchor[:, 0:3] = 0
            rois_anchor[:, 6] = 0
            reg_targets = self.box_coder.encode_torch(
                gt_boxes3d_ct.view(rcnn_batch_size, code_size), rois_anchor
            )

            rcnn_loss_reg = self.reg_loss_func(
                rcnn_reg.view(rcnn_batch_size, -1).unsqueeze(dim=0),
                reg_targets.unsqueeze(dim=0),
            )  # [B, M, 7]
            rcnn_loss_reg = (rcnn_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(fg_sum, 1)
            rcnn_loss_reg = rcnn_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight']
            tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item()

            if loss_cfgs.CORNER_LOSS_REGULARIZATION and fg_sum > 0:
                # TODO: NEED to BE CHECK
                fg_rcnn_reg = rcnn_reg.view(rcnn_batch_size, -1)[fg_mask]
                fg_roi_boxes3d = roi_boxes3d.view(-1, code_size)[fg_mask]

                fg_roi_boxes3d = fg_roi_boxes3d.view(1, -1, code_size)
                batch_anchors = fg_roi_boxes3d.clone().detach()
                roi_ry = fg_roi_boxes3d[:, :, 6].view(-1)
                roi_xyz = fg_roi_boxes3d[:, :, 0:3].view(-1, 3)
                batch_anchors[:, :, 0:3] = 0
                rcnn_boxes3d = self.box_coder.decode_torch(
                    fg_rcnn_reg.view(batch_anchors.shape[0], -1, code_size), batch_anchors
                ).view(-1, code_size)

                rcnn_boxes3d = common_utils.rotate_points_along_z(
                    rcnn_boxes3d.unsqueeze(dim=1), roi_ry
                ).squeeze(dim=1)
                rcnn_boxes3d[:, 0:3] += roi_xyz

                loss_corner = loss_utils.get_corner_loss_lidar(
                    rcnn_boxes3d[:, 0:7],
                    gt_of_rois_src[fg_mask][:, 0:7]
                )
                loss_corner = loss_corner.mean()
                loss_corner = loss_corner * loss_cfgs.LOSS_WEIGHTS['rcnn_corner_weight']

                rcnn_loss_reg += loss_corner
                tb_dict['rcnn_loss_corner'] = loss_corner.item()
        else:
            raise NotImplementedError

        return rcnn_loss_reg, tb_dict


    def break_up_pc(self, pc):
        batch_idx = pc[:, 0]
        xyz = pc[:, 1:4].contiguous()
        features = (pc[:, 4:].contiguous() if pc.size(-1) > 4 else None)
        return batch_idx, xyz, features


    def get_unsupervised_box_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        rois = forward_ret_dict['rois']  # B x n_box x 7 (x, y, z, w, l, h, ry)
        rcnn_reg = forward_ret_dict['rcnn_reg']  # B x C
        rcnn_cls = forward_ret_dict['rcnn_cls']  # B x 1
        batch_size = forward_ret_dict['batch_size']
        points = forward_ret_dict['points']
        p2_scores = forward_ret_dict['p2_score']
        rcnn_int_cls = forward_ret_dict['roi_int_cls_logits'] if 'roi_int_cls_logits' in forward_ret_dict else None # B x Class
        batch_idx, xyz, _ = self.break_up_pc(points)
        tb_dict = {}

        # get boxes3d from rois
        batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_size, rois=rois, cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
        # if rcnn_int_cls is not None:
        #     rcnn_int_cls = rcnn_int_cls.view(batch_size, -1, rcnn_int_cls.shape[-1])
        
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (batch_idx == bs_idx).sum()

        assert xyz_batch_cnt.min() == xyz_batch_cnt.max(), "xyz_batch_cnt: " + str(xyz_batch_cnt)
        ptc = xyz.view(batch_size, -1, 3)  # B x npoints x 3
        p2_scores = p2_scores.view(batch_size, -1)  # B x npoints

        # get loss from single frame (p2 guided foreground)
        # foreground_inside_loss, background_inside_loss, foreground_outside_loss, box_point_mask = \
        #     single_frame_geometric_consistency_loss(ptc, p2_scores, batch_box_preds, batch_size, loss_cfgs, return_box_point_mask=True)
        if loss_cfgs.get('REG_LOSS_TYPE', 'boundary') == 'contrast':
            unsup_loss_reg, foreground_loss, background_loss, size_loss, box_scores, box_fgpoint_counts = \
                single_frame_contrast_loss(ptc, p2_scores, batch_box_preds, batch_size, loss_cfgs, return_box_scores=True)
        elif loss_cfgs.get('REG_LOSS_TYPE', 'boundary') == 'residual':
            batch_box_residual_size = rcnn_reg[..., 3:6]
            # print("using residual", batch_box_residual_size.shape)
            unsup_loss_reg, foreground_loss, background_loss, size_loss, box_scores, box_fgpoint_counts = \
                single_frame_boundary_residual_loss(ptc, p2_scores, batch_box_preds, batch_box_residual_size, batch_size, loss_cfgs, cls_int_logits=rcnn_int_cls, return_box_scores=True)
        else:
            unsup_loss_reg, foreground_loss, background_loss, size_loss, box_scores, box_fgpoint_counts = \
                single_frame_boundary_loss(ptc, p2_scores, batch_box_preds, batch_size, loss_cfgs, cls_int_logits=rcnn_int_cls, return_box_scores=True)
        
        # tb_dict["roi_foreground_inside_loss"] = foreground_inside_loss.item()
        # tb_dict["roi_background_inside_loss"] = background_inside_loss.item()
        # tb_dict["roi_foreground_outside_loss"] = foreground_outside_loss.item()
        tb_dict["roi_foreground_loss"] = foreground_loss
        tb_dict["roi_background_loss"] = background_loss
        tb_dict["roi_size_loss"] = size_loss

        # unsup_loss_reg = loss_cfgs.LOSS_WEIGHTS['foreground_inside_loss'] * foreground_inside_loss + \
        #                  loss_cfgs.LOSS_WEIGHTS['background_inside_loss'] * background_inside_loss + \
        #                  loss_cfgs.LOSS_WEIGHTS['foreground_outside_loss'] * foreground_outside_loss
        
        # box_point_mask: B x Nbox x Npoints, mask denoting point is inside box
        # import ipdb; ipdb.set_trace()
        assert batch_cls_preds.shape[-1] == 1, "Only supporting 1 class for now"

        # batch_cls_labels = torch.zeros_like(batch_cls_preds)
        # for batch_i in range(batch_size):
        #     for box_j in range(batch_box_preds.shape[1]):
        #         box_p2 = p2_scores[batch_i][box_point_mask[batch_i, box_j, :]]
        #         if len(box_p2) == 0:
        #             batch_cls_labels[batch_i, box_j, :] = 0.  # assign as background if no points inside box
        #         else:
        #             box_p2_quant = torch.quantile(box_p2, 0.2)
        #             batch_cls_labels[batch_i, box_j, :] = \
        #                 1. if box_p2_quant < 0.5 else 0. if box_p2_quant > 0.8 else -1.  # assign for all classes
        # batch_cls_labels = box_scores.detach()  # remove from graph
        batch_cls_labels = torch.exp(box_scores.detach())  # remove from graph

        # import ipdb; ipdb.set_trace()

        # apply commonsense filtering, as in MODEST
        if loss_cfgs.get('FILTER_CONFIG', None) is not None:
            with torch.no_grad():
                filtered_cls_mask = filter_class_labels(batch_box_preds, box_fgpoint_counts, loss_cfgs.FILTER_CONFIG)
                batch_cls_labels[~filtered_cls_mask] = 0  # TODO: check value
        
        # forward_ret_dict["roi_head_cls_labels"] = batch_cls_labels

        forward_ret_dict["roi_head_cls_labels"] = torch.clamp(batch_cls_labels, min=0.)
        
        # ipdb.set_trace()
        # rcnn_cls_flat = batch_cls_preds[batch_cls_labels >= 0].view(-1)
        # rcnn_cls_labels = batch_cls_labels[batch_cls_labels >= 0].view(-1)
        # unsup_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), rcnn_cls_labels, reduction='none')
        # unsup_loss_cls = unsup_loss_cls.sum() / torch.clamp(rcnn_cls_labels.sum(), min=1.0)

        unsup_loss_cls = F.l1_loss(batch_cls_preds.view(-1), batch_cls_labels.view(-1), reduction='mean')

        tb_dict["roi_unsup_loss_cls"] = unsup_loss_cls.item()

        return unsup_loss_reg, unsup_loss_cls, tb_dict        


    def get_box_cls_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        rcnn_cls = forward_ret_dict['rcnn_cls']
        rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'].view(-1)
        if loss_cfgs.CLS_LOSS == 'BinaryCrossEntropy':
            rcnn_cls_flat = rcnn_cls.view(-1)
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), rcnn_cls_labels.float(), reduction='none')
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        elif loss_cfgs.CLS_LOSS == 'CrossEntropy':
            batch_loss_cls = F.cross_entropy(rcnn_cls, rcnn_cls_labels, reduction='none', ignore_index=-1)
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        else:
            raise NotImplementedError

        rcnn_loss_cls = rcnn_loss_cls * loss_cfgs.LOSS_WEIGHTS['rcnn_cls_weight']
        tb_dict = {'rcnn_loss_cls': rcnn_loss_cls.item()}
        return rcnn_loss_cls, tb_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        # import ipdb; ipdb.set_trace()
        rcnn_loss = torch.tensor(0).cuda().float()
        if self.model_cfg.LOSS_CONFIG.get('USE_UNSUPERVISED', False):
            unsup_loss_reg, unsup_loss_cls, unsup_tb_dict = self.get_unsupervised_box_loss(self.forward_ret_dict)
            tb_dict.update(unsup_tb_dict)
            rcnn_loss += unsup_loss_reg + unsup_loss_cls
        else:
            rcnn_loss_cls, cls_tb_dict = self.get_box_cls_layer_loss(self.forward_ret_dict)
            rcnn_loss += rcnn_loss_cls
            tb_dict.update(cls_tb_dict)

            rcnn_loss_reg, reg_tb_dict = self.get_box_reg_layer_loss(self.forward_ret_dict)
            rcnn_loss += rcnn_loss_reg
            tb_dict.update(reg_tb_dict)
        tb_dict['rcnn_loss'] = rcnn_loss.item()
        return rcnn_loss, tb_dict

    def generate_predicted_boxes(self, batch_size, rois, cls_preds, box_preds):
        """
        Args:
            batch_size:
            rois: (B, N, 7)
            cls_preds: (BN, num_class)
            box_preds: (BN, code_size)

        Returns:

        """
        code_size = self.box_coder.code_size
        # batch_cls_preds: (B, N, num_class or 1)
        batch_cls_preds = cls_preds.view(batch_size, -1, cls_preds.shape[-1])
        batch_box_preds = box_preds.view(batch_size, -1, code_size)

        roi_ry = rois[:, :, 6].view(-1)
        roi_xyz = rois[:, :, 0:3].view(-1, 3)
        local_rois = rois.clone().detach()
        local_rois[:, :, 0:3] = 0

        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, local_rois).view(-1, code_size)

        batch_box_preds = common_utils.rotate_points_along_z(
            batch_box_preds.unsqueeze(dim=1), roi_ry
        ).squeeze(dim=1)
        batch_box_preds[:, 0:3] += roi_xyz
        batch_box_preds = batch_box_preds.view(batch_size, -1, code_size)
        return batch_cls_preds, batch_box_preds
