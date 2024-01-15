import torch
from .detector3d_template import Detector3DTemplate
from ..model_utils.unsupervised_regression_utils import single_frame_boundary_loss, \
    filter_class_labels, sample_size_offset_near_box, single_frame_p2_loss, \
    single_frame_contrast_loss, alignment_reward
from ..model_utils.rewards import reward_mix
from ..model_utils import model_nms_utils


reward_fn_definitions_ = {
    'reward_mix': reward_mix,
}


class PointRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def score_predictions(self, pred_dicts, batch_dict, reward_fn, sample_fn, rl_cfg):
        # get top k predictions
        batch_size = batch_dict['batch_size']
        points = batch_dict['points']
        _, xyz, _ = self.roi_head.break_up_pc(points)
        ptc = xyz.view(batch_size, -1, 3)  # B x npoints x 3
        p2_scores = batch_dict['p2_score']
        p2_scores = p2_scores.view(batch_size, -1)  # B x npoints
        if self.model_cfg.RL.get('USE_SEED_LABELS', False):
            seed_reward_weight = batch_dict['seed_reward_weight']
            seed_labels = batch_dict['gt_boxes']  # B x max_nboxes x 8
        else:
            seed_labels = None

        if self.model_cfg.RL.get('USE_FG_SAMPLING', False):
            fg_sampling_weight = self.model_cfg.RL.USE_FG_SAMPLING.WEIGHT
            with torch.no_grad():
                fg_box_preds = self.point_head.get_foreground_proposals_sample(0.2, 100)
        else:
            fg_box_preds, fg_sampling_weight = None, None

        gt_boxes = []
        gt_names = []
        max_gt = 0
        pred_rewards = []

        for batch_i in range(batch_size):

            box_preds = pred_dicts[batch_i]['pred_boxes']
            box_names = pred_dicts[batch_i]['pred_labels']

            # sample boxes near predicted boxes
            box_preds_new, box_names_new = sample_fn(box_preds, box_names, 200)

            box_preds = torch.cat([box_preds, box_preds_new], dim=0)
            box_preds[:, 3:6] = torch.clamp(box_preds[:, 3:6], min=0.01)
            box_preds = box_preds.unsqueeze(0)  # 1 x num_boxes x 7
            box_names = torch.cat([box_names, box_names_new], dim=0)

            # # compute box scores
            box_scores, box_fgpoint_counts = reward_fn(ptc[batch_i], p2_scores[batch_i], box_preds[0])
            box_scores = box_scores.unsqueeze(0)

            if seed_labels is not None:
                seed_box_mask = seed_labels[batch_i, :, -1] > 0  # max_nboxes
                seed_preds = seed_labels[batch_i, :, :7][seed_box_mask]  # nboxes x 7
                seed_names = seed_labels[batch_i, :, 7][seed_box_mask]  # nboxes
                seed_preds_new, seed_names_new = sample_fn(seed_preds, seed_names, 100)
                seed_preds = torch.cat([seed_preds, seed_preds_new], dim=0)
                seed_preds[:, 3:6] = torch.clamp(seed_preds[:, 3:6], min=0.01)
                seed_names = torch.cat([seed_names, seed_names_new], dim=0)

                seed_scores, seed_fgpoint_counts = reward_fn(ptc[batch_i], p2_scores[batch_i], seed_preds)

                box_scores = torch.cat([box_scores, seed_reward_weight * seed_scores.unsqueeze(0)], dim=1) # b x (num_pred + num_seed)
                box_preds = torch.cat([box_preds, seed_preds.unsqueeze(0)], dim=1)  # b x (num_pred + num_seed) x 7
                box_names = torch.cat([box_names, seed_names], dim=0)  # b x (num_pred + num_seed)
                box_fgpoint_counts = torch.cat([box_fgpoint_counts, seed_fgpoint_counts], dim=0)  # b x (num_pred + num_seed)
            
            if fg_box_preds is not None:
                fg_box_pred = fg_box_preds[batch_i]  # num_fg x 7
                fg_box_pred[:, 3:6] = torch.clamp(fg_box_pred[:, 3:6], min=0.01)
                fg_names = fg_box_pred.new_ones(fg_box_pred.shape[0])

                fg_scores, fg_fgpoint_counts = reward_fn(ptc[batch_i], p2_scores[batch_i], fg_box_pred)

                box_scores = torch.cat([box_scores, fg_sampling_weight * fg_scores.unsqueeze(0)], dim=1) # b x (num_pred + num_seed)
                box_preds = torch.cat([box_preds, fg_box_pred.unsqueeze(0)], dim=1)  # b x (num_pred + num_seed) x 7
                box_names = torch.cat([box_names, fg_names], dim=0)  # b x (num_pred + num_seed)
                box_fgpoint_counts = torch.cat([box_fgpoint_counts, fg_fgpoint_counts], dim=0)  # b x (num_pred + num_seed)

            # apply NMS with scores
            selected, selected_scores = model_nms_utils.class_agnostic_nms(
                box_scores=box_scores[0], box_preds=box_preds[0],
                nms_config=rl_cfg.NMS_CONFIG,
                score_thresh=1e-5,
            )
            if selected.shape[0] == 0:
                selected = torch.tensor([box_fgpoint_counts.argmax()]).to(selected.device).to(selected.dtype)
                # print('No boxes selected, using box with most foreground points', box_fgpoint_counts.max())
            box_preds = box_preds[:, selected, :]
            box_names = box_names[selected]
            box_scores = box_scores[:, selected]
            num_selection = min(max(int(box_preds.shape[1] * rl_cfg.top_k / 100.), 1), 10)

            # get top k predictions
            top_scores = box_scores.topk(num_selection, dim=1).indices  # 1 x num_boxes --> 1 x top_k
            top_boxes = box_preds[0, top_scores[0], :]  # top_k x 7
            top_labels = box_names[top_scores[0]]  # top_k
            gt_boxes.append(top_boxes)
            gt_names.append(top_labels)
            if top_boxes.shape[0] > max_gt:
                max_gt = top_boxes.shape[0]
            pred_rewards.append(box_scores[0, top_scores[0]].mean())
        
        # pad gt_boxes and gt_names
        batch_box_topk = torch.zeros(batch_size, max_gt, 8).to(gt_boxes[0].device)
        for batch_i in range(batch_size):
            batch_box_topk[batch_i, :gt_boxes[batch_i].shape[0], :-1] = gt_boxes[batch_i]
            batch_box_topk[batch_i, :gt_boxes[batch_i].shape[0], -1] = gt_names[batch_i]

        return_dict = {
            'boxes': batch_box_topk,
            'rewards': torch.stack(pred_rewards).mean()
        }
        return return_dict

    def forward(self, batch_dict):
        if self.model_cfg.get('RL', None) is not None and self.training:
            # get reward function from config
            reward_fn = reward_fn_definitions_[self.model_cfg.RL.get('REWARD_FN', 'reward_kl4')]
            self.eval()
            with torch.no_grad():
                for cur_module in self.module_list:
                    batch_dict = cur_module(batch_dict)
                
                # get top k predictions
                pred_dicts, recall_dicts = self.post_processing(batch_dict)
                top_k_pred = self.score_predictions(pred_dicts, batch_dict, 
                                                    reward_fn=reward_fn, #single_frame_p2_loss, #single_frame_boundary_loss, reward_mix
                                                    sample_fn=sample_size_offset_near_box,
                                                    rl_cfg=self.model_cfg.RL)
            # store into data as targets
            batch_dict['gt_boxes'] = top_k_pred['boxes'].detach()
            self.train()
        
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            if self.model_cfg.get('RL', None) is not None:
                tb_dict['num_max_gt_boxes'] = batch_dict['gt_boxes'].shape[1]
                tb_dict['rewards'] = top_k_pred['rewards']
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss_point, tb_dict = self.point_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss_point + loss_rcnn
        return loss, tb_dict, disp_dict
