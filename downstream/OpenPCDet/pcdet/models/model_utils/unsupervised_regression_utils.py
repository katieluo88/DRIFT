import torch
import torch.nn.functional as F
import numpy as np
import ipdb
import types
import copy
from ...utils import common_utils


def get_pdf(x, mean, std):
    y = (x - mean) / std
    return torch.exp(-y * y / 2) / np.sqrt(2 * np.pi) / std


def sample_size_offset_near_box(boxes, labels, num_samples, noise_xyz=0.3, noise_lwh=0.3, noise_angle=0.3):
    """
    Sample from a gaussian noise the size and offset near the given boxes.
    :param boxes: (M, 7)
    :param labels: (M)
    :param noise_xyz: (float)
    :param noise_lwh: (float)
    """
    assert boxes.shape[-1] == 7
    num_boxes = boxes.shape[0]
    noise_xyz = torch.randn(num_samples, 3).to(boxes.dtype).to(boxes.device) * noise_xyz
    noise_lwh = torch.randn(num_samples, 3).to(boxes.dtype).to(boxes.device) * noise_lwh
    noise_angle = (2. * torch.rand(num_samples) - 1.).to(boxes.dtype).to(boxes.device) * noise_angle

    boxes_new = boxes.clone()
    sample_idx = np.random.choice(num_boxes, num_samples, replace=True)
    boxes_new = boxes_new[sample_idx]
    boxes_new[:, 3:6] += noise_lwh
    boxes_new[:, :3] += noise_xyz
    boxes_new[:, 6] += noise_angle
    labels_new = labels.clone()[sample_idx]
    return boxes_new, labels_new


def sample_box_near_fg(ptc, pp_score, num_samples, noise_xyz=0.3, noise_lwh=0.3, noise_angle=0.3, fg_thresh=0.2):
    """
    Sample boxes near foreground points, defined as points having pp_score < fg_thresh.
    :param ptc: (N, 3)
    :param pp_score: (N)
    :param num_samples: (int)
    """
    pass


def size_score(boxes):
    mean = torch.tensor([5.12985166, 1.9629637, 1.88716147]).view(1, 3).to(boxes.dtype).to(boxes.device)
    std = torch.tensor([2.07506971, 0.44118937, 0.58281559]).view(1, 3).to(boxes.dtype).to(boxes.device)
    y = (boxes[..., 3:6] - mean) / std
    return torch.prod(torch.exp(-y * y / 2) / np.sqrt(2 * np.pi) / std, dim=-1)


def foreground_score(
    scale: torch.Tensor,  # (B, M, N_points,)
    pp_score: torch.Tensor,  # (B, N_points,)
    foreground: torch.Tensor,  # (B, N_points)
    scale_range: float = 1.5,
    alpha: float = 0.5,
) -> torch.Tensor:  # (N_points,)
    assert scale_range > 0.01
    
    pp_score = torch.clamp(pp_score, min=0.2, max=0.8)
    with torch.no_grad():
        log_p = alpha * torch.log(1 - pp_score) - np.log(alpha + 1)
        log_p = log_p.unsqueeze(dim=1).repeat(1, scale.shape[1], 1)  # (B, M, N_points,)
        valid = (scale < scale_range).float()  # (B, M, N_points,)
        count = torch.sum(valid, dim=-1)  # (B, M，)
        in_1xbox = scale < 1.0
        valid_fg = torch.logical_and(foreground.unsqueeze(dim=1), in_1xbox).float()  # [B, n_box, N_points]
        box_fgpoint_counts = torch.sum(valid_fg, dim=-1)  # [B, n_box]
    
    try:
        score = log_p + torch.distributions.normal.Normal(torch.tensor([0.88]).to(scale.device), torch.tensor([0.16]).to(scale.device)).log_prob(scale)  # (M, N_points,)
    except:
        ipdb.set_trace()

    score = score * valid
    score = torch.where(
        count > 0,
        (torch.sum(score, dim=-1) / torch.clamp(count, min=1) - torch.log(torch.sum(torch.exp(score) * valid, dim=-1)).detach() + torch.log(torch.clamp(count, min=1))) / torch.clamp(count, min=1),
        torch.zeros_like(count))
    # score = score * valid
    # score = torch.where(
    #     count > 0,
    #     (torch.sum(score, dim=-1) - torch.log(torch.sum(torch.exp(score) * valid, dim=-1)).detach() + torch.log(torch.clamp(count, min=1))) / torch.clamp(count, min=1),
    #     torch.zeros_like(count))
    return score, box_fgpoint_counts


def get_scale(
    ptc: torch.Tensor,  # (B, N_points, 4)
    boxes_3d: torch.Tensor,  # (B, M, 7)
) -> torch.Tensor:
    batch_size, num_boxes = boxes_3d.shape[0], boxes_3d.shape[1]
    trs = common_utils.get_transform_to_box(boxes_3d)  # [B x nbox, 4, 4]
    ptc_in_box = common_utils.transform_points_torch(ptc, trs)  # B x n_box x npoints x 3
    ptc_in_box = torch.abs(ptc_in_box)
    scale = ptc_in_box / torch.clamp(boxes_3d[..., 3:6].view(batch_size, num_boxes, 1, 3) * 0.5, min=1e-3)
    scale = torch.max(scale, dim=-1).values

    return scale


def size_score_logprob(boxs, prior_mean, prior_std):
    mean = torch.tensor(prior_mean).to(boxs.dtype).to(boxs.device)
    std = torch.tensor(prior_std).to(boxs.dtype).to(boxs.device)
    m = torch.distributions.normal.Normal(mean, std, validate_args=None)
    scale = m.log_prob(boxs[..., 3:6]).sum(dim=-1) # + np.log(np.sqrt(np.prod(prior_std))) + np.log((2 * np.pi) ** 1.5)
    return scale

def gaussian_shape_score(boxs, prior_mean, prior_std):
    mean = torch.tensor(prior_mean).view(1, 3).to(boxs.dtype).to(boxs.device)
    std = torch.tensor(prior_std).view(1, 3).to(boxs.dtype).to(boxs.device)
    y = (boxs[..., 3:6] - mean) / std
    score = torch.prod(torch.exp(-y * y / 2) / np.sqrt(2 * np.pi) / std, dim=-1)
    return score


def size_score_multiclass_logprob(boxs):
    means = [torch.tensor([4.74451989, 1.91059287, 1.71107344]),
             torch.tensor([0.79702832, 0.77995906, 1.74490618]),
             torch.tensor([9.40333292, 2.83213669, 3.29912471]),
             torch.tensor([1.75168967, 0.61337013, 1.36374118])]
    stds = [torch.tensor([0.55945502, 0.16199086, 0.24788235]),
            torch.tensor([0.18178839, 0.15324556, 0.1771094 ]),
            torch.tensor([3.14533891, 0.27833338, 0.43030043]),
            torch.tensor([0.32637668, 0.25591983, 0.34314765])]
    weights = [0.1498824052196492, 0.07024215684742731, 0.6137651955260903, 0.16929808460478427]
    dists = [torch.distributions.normal.Normal(
        mean.view(1, 3).to(boxs.dtype).to(boxs.device), 
        std.view(1, 3).to(boxs.dtype).to(boxs.device), validate_args=None) for mean, std in zip(means, stds)]
    scales = [2 * np.log(w) + m.log_prob(boxs[..., 3:6]).sum(dim=-1) for w, m in zip(weights, dists)]
    scale = torch.logsumexp(torch.stack(scales, dim=0), dim=0)
    return scale


def size_score_multiclass_learned_logprob(boxs, cls_logits, 
                                            prior_means = [[4.74451989, 1.91059287, 1.71107344],
                                                            [0.79702832, 0.77995906, 1.74490618],
                                                            [9.40333292, 2.83213669, 3.29912471],
                                                            [1.75168967, 0.61337013, 1.36374118]], 
                                            prior_stds = [[0.3, 0.3, 0.3],
                                                          [0.3, 0.3, 0.3],
                                                          [0.3, 0.3, 0.3],
                                                          [0.3, 0.3, 0.3]],
                                            temperature=0.5):
    """
    Args:
        boxs: (B, num_boxes, 7) [x, y, z, dx, dy, dz, heading]
        cls_logits: (B, num_boxes, num_classes)
    """
    dists = [torch.distributions.normal.Normal(
        torch.tensor(mean).to(boxs.dtype).to(boxs.device), 
        torch.tensor(std).to(boxs.dtype).to(boxs.device), validate_args=None) for mean, std in zip(prior_means, prior_stds)]
    cls_logits = F.log_softmax(cls_logits / temperature, dim=-1)  # B x num_boxes x num_classes
    scales = [m.log_prob(boxs[..., 3:6]).sum(dim=-1) + cls_logits[..., i] for i, m in enumerate(dists)]  # 4, B x num_boxes
    scale = torch.logsumexp(torch.stack(scales, dim=0), dim=0)
    return scale


def size_score_volume(boxes):
    return -torch.prod(boxes[..., 3:6], dim=-1)


def filter_class_labels(batch_box_preds, box_fgpoint_counts, filter_cfgs):
    """
    Args:
        batch_box_preds: B x n_box x 7
        box_fgpoint_counts: B x n_box
        filter_cfgs: dict of configs

    Returns:
        filtered_cls_mask: B x n_box
    """
    # filter by volume
    volumes = torch.prod(batch_box_preds[..., 3:6], dim=-1)
    valid_volumes = torch.logical_and(volumes > filter_cfgs['volume_min'], volumes < filter_cfgs['volume_max'])

    # filter by height
    center_z = batch_box_preds[..., 2]
    valid_heights = torch.logical_and(center_z > filter_cfgs['height_min'], center_z < filter_cfgs['height_max'])

    # filter by fg point counts
    if 'fgpoint_count_min' in filter_cfgs:
        valid_fgpoint_counts = box_fgpoint_counts > filter_cfgs['fgpoint_count_min']
        valid_volumes = torch.logical_and(valid_volumes, valid_fgpoint_counts)
    return torch.logical_and(valid_volumes, valid_heights)


def single_frame_contrast_loss(ptc, p2_scores, boxes_3d, batch_size, loss_cfgs, return_box_scores=False):
    # _shape = boxs.shape[:-1]
    foreground = torch.logical_and(p2_scores < loss_cfgs.UNSUPERVISED_P2_FOREGROUND_THRESHOLD, ptc[..., -1] < 3.)

    scale = get_scale(ptc, boxes_3d)
    fg_score, box_fgpoint_counts = foreground_score(
        scale,
        p2_scores,
        foreground,
        scale_range=1.5,
        alpha=0.5)
    
    sz_score = size_score_logprob(boxes_3d, prior_mean=[5.12985166, 1.9629637, 1.88716147], prior_std=[2.07506971, 0.44118937, 0.58281559])
    score = fg_score + sz_score

    loss = -torch.mean(score)

    if return_box_scores:
        return loss, fg_score.mean().item(), 0, sz_score.mean().item(), score, box_fgpoint_counts
    else:
        return loss, fg_score.mean().item(), 0, sz_score.mean().item()

def single_frame_p2_loss(ptc, p2_scores, boxes_3d, batch_size, loss_cfgs, cls_int_logits=None, return_box_scores=False):
    unsup_loss_reg, foreground_loss, background_loss, size_loss = None, None, None, None

    if torch.isnan(boxes_3d).any():
        print('boxes_3d has nan')
    batch_size, num_boxes = boxes_3d.shape[0], boxes_3d.shape[1]

    trs = common_utils.get_transform_to_box(boxes_3d)  # [B x nbox, 4, 4]
    ptc_in_box = common_utils.transform_points_torch(ptc, trs)  # B x n_box x npoints x 3
    ptc_in_box = torch.abs(ptc_in_box)
    in_1xbox = torch.all(ptc_in_box < boxes_3d[..., 3:6].unsqueeze(dim=2) * 0.5, dim=-1) # B x n_box x n_points

    foreground = torch.logical_and(p2_scores < loss_cfgs.UNSUPERVISED_P2_FOREGROUND_THRESHOLD, ptc[..., -1] < 3.).unsqueeze(dim=1)
    valid_fg = torch.logical_and(foreground, in_1xbox).float()  # [B, n_box, N_points]
    box_fgpoint_counts = torch.sum(valid_fg, dim=-1)  # [B, n_box]

    batch_box_scores = torch.zeros((batch_size, num_boxes))
    for batch_i in range(batch_size):
        for box_j in range(num_boxes):
            box_p2 = p2_scores[batch_i][in_1xbox[batch_i][box_j]]
            if len(box_p2) > 0 and torch.quantile(box_p2, 0.2) < 0.7:
                batch_box_scores[batch_i][box_j] = 1.
    if return_box_scores:
        return unsup_loss_reg, foreground_loss, background_loss, size_loss, batch_box_scores, box_fgpoint_counts
    return unsup_loss_reg, foreground_loss, background_loss, size_loss

def closeness_rectangle(cluster_ptc, delta=0.1, d0=1e-2):
    max_beta = -float("inf")
    choose_angle = None
    for angle in np.arange(0, 90 + delta, delta):
        angle = angle / 180.0 * np.pi
        components = np.array(
            [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
        )
        projection = cluster_ptc @ components.T
        min_x, max_x = projection[:, 0].min(), projection[:, 0].max()
        min_y, max_y = projection[:, 1].min(), projection[:, 1].max()
        Dx = np.vstack((projection[:, 0] - min_x, max_x - projection[:, 0])).min(axis=0)
        Dy = np.vstack((projection[:, 1] - min_y, max_y - projection[:, 1])).min(axis=0)
        beta = np.vstack((Dx, Dy)).min(axis=0)
        beta = np.maximum(beta, d0)
        beta = 1 / beta
        beta = beta.sum()
        if beta > max_beta:
            max_beta = beta
            choose_angle = angle
    angle = choose_angle
    components = np.array(
        [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
    )
    projection = cluster_ptc @ components.T
    min_x, max_x = projection[:, 0].min(), projection[:, 0].max()
    min_y, max_y = projection[:, 1].min(), projection[:, 1].max()

    if (max_x - min_x) < (max_y - min_y):
        angle = choose_angle + np.pi / 2
        components = np.array(
            [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
        )
        projection = cluster_ptc @ components.T
        min_x, max_x = projection[:, 0].min(), projection[:, 0].max()
        min_y, max_y = projection[:, 1].min(), projection[:, 1].max()

    area = (max_x - min_x) * (max_y - min_y)

    rval = np.array(
            [
                [max_x, min_y],
                [min_x, min_y],
                [min_x, max_y],
                [max_x, max_y],
            ]
        )
    rval = rval @ components
    return rval, angle, area

def convert_obj(obj, Tr):
    loc_lidar = Tr[:3, :3].T @ obj.t  # rect
    loc_lidar[2] += obj.h / 2
    return np.array([loc_lidar[0], loc_lidar[1], loc_lidar[2], obj.w, obj.l, obj.h, obj.ry ])

def convert_corners(ptc_box, corners, ry, area):
    ry *= -1
    ry = -(np.pi / 2 + ry)
    l = np.linalg.norm(corners[0] - corners[1])
    w = np.linalg.norm(corners[0] - corners[-1])
    c = (corners[0] + corners[2]) / 2
    bottom = ptc_box[:, 2].min()
    h = ptc_box[:, 2].max() - bottom
    obj = types.SimpleNamespace()
    obj.t = np.array([c[0], c[1], bottom])
    obj.l = l
    obj.w = w
    obj.h = h
    obj.ry = ry
    obj.volume = area * h
    Tr = np.eye(4)

    box_fit = convert_obj(obj, Tr)
    return box_fit

def normalize_box(box):
    box = copy.deepcopy(box)
    if box[3] < box[4]:
        _tmp = box[3]
        box[3] = box[4]
        box[4] = _tmp
        box[-1] += np.pi / 2
    if box[-1] > np.pi:
        box[-1] -= np.pi
    if box[-1] < 0:
        box[-1] += np.pi
    return box

def alignment_reward(ptc, p2_scores, boxes_3d, batch_size, loss_cfgs, cls_int_logits=None, return_box_scores=False, include_range=1.5, default_val=-float('Inf')):
    unsup_loss_reg, foreground_loss, background_loss, size_loss = None, None, None, None
    if torch.isnan(boxes_3d).any():
        print('boxes_3d has nan')
    batch_size, num_boxes = boxes_3d.shape[0], boxes_3d.shape[1]
    foreground = torch.logical_and(p2_scores < loss_cfgs.UNSUPERVISED_P2_FOREGROUND_THRESHOLD, ptc[..., -1] < 3.).unsqueeze(dim=1)
    trs = common_utils.get_transform_to_box(boxes_3d)  # [B x nbox, 4, 4]
    ptc_in_box = common_utils.transform_points_torch(ptc, trs)  # B x n_box x npoints x 3
    ptc_in_box = torch.abs(ptc_in_box)

    in_range = torch.all(ptc_in_box < boxes_3d[..., 3:6].unsqueeze(dim=2) * 0.5 * include_range, dim=-1) # B x n_box x n_points
    valid_fg = torch.logical_and(foreground, in_range)  # [B, n_box, N_points]
    
    # get the angle
    rotation_reward = torch.zeros((batch_size, num_boxes)).fill_(default_val) 
    for batch_i in range(batch_size):
        for box_j in range(num_boxes):
            # tightest fit over the original ptc
            fg_in_box = ptc[batch_i][valid_fg[batch_i, box_j]].clone().detach().cpu().numpy() # num_fg_pts x 3
            if len(fg_in_box) > 0: 
                corners, opt_angle, area = closeness_rectangle(fg_in_box[:,:2])
                norm_opt_angle = normalize_box(convert_corners(fg_in_box, corners, opt_angle, area))[-1]
                norm_pred_angle = normalize_box(boxes_3d[batch_i, box_j])[-1]
                rotation_reward[batch_i, box_j] = -1 * torch.abs(norm_pred_angle - norm_opt_angle)
    rotation_reward = torch.exp(rotation_reward) # make it positive?
    
    # return fgpoint counts for breaking ties
    in_1xbox = torch.all(ptc_in_box < boxes_3d[..., 3:6].unsqueeze(dim=2) * 0.5, dim=-1)
    valid_fg = torch.logical_and(foreground, in_1xbox).float()  # [B, n_box, N_points]
    box_fgpoint_counts = torch.sum(valid_fg, dim=-1)  # [B, n_box]
    assert return_box_scores, 'must return box scores'
    if return_box_scores:
        return unsup_loss_reg, foreground_loss, background_loss, size_loss, rotation_reward, box_fgpoint_counts
    return unsup_loss_reg, foreground_loss, background_loss, size_loss

def single_frame_boundary_loss(ptc, p2_scores, boxes_3d, batch_size, loss_cfgs, cls_int_logits=None, return_box_scores=False):
    """
    Args:
        ptc: B x npoints x 3        - points in lidar coordinates, single frame
        p2_scores: B x npoints      - p2 scores of points
        boxes_3d: B x n_box x 7      - boxes in lidar coordinates
        batch_size: int             - B
        loss_cfgs: dict of configs

    Returns:
        foreground_inside_loss: float tensor
        background_inside_loss: float tensor
        foreground_outside_loss: float tensor
    """
    if torch.isnan(boxes_3d).any():
        print('boxes_3d has nan')
    num_boxes = boxes_3d.shape[1]

    # foreground = (p2_scores < loss_cfgs.UNSUPERVISED_P2_FOREGROUND_THRESHOLD).unsqueeze(dim=1)
    foreground = torch.logical_and(p2_scores < loss_cfgs.UNSUPERVISED_P2_FOREGROUND_THRESHOLD, ptc[..., -1] < 3.).unsqueeze(dim=1)

    trs = common_utils.get_transform_to_box(boxes_3d)  # [B x nbox, 4, 4]
    ptc_in_box = common_utils.transform_points_torch(ptc, trs)  # B x n_box x npoints x 3
    ptc_in_box = torch.abs(ptc_in_box)

    distances_box_to_points = torch.linalg.norm(ptc_in_box, dim=-1)  # B x n_box x npoints
    distances_box_to_points[~foreground.repeat(1, num_boxes, 1)] = 100.  # arbitrary large number

    assos_bound = torch.clamp(boxes_3d[..., 3:6].unsqueeze(dim=2), min=1.)
    in_2xbox = torch.all(ptc_in_box < assos_bound, dim=-1)
    valid = torch.logical_and(foreground, in_2xbox).float()  # [B, n_box, npoints]

    in_1xbox = torch.all(ptc_in_box < boxes_3d[..., 3:6].unsqueeze(dim=2) * 0.5, dim=-1)
    valid_bg = torch.logical_and(~foreground, in_1xbox).float()  # [B, n_box, N_points]

    valid_fg = torch.logical_and(foreground, in_1xbox).float()  # [B, n_box, N_points]
    box_fgpoint_counts = torch.sum(valid_fg, dim=-1)  # [B, n_box]
    # box_inv_p2_scores = 1. - p2_scores.unsqueeze(dim=1).repeat(1, num_boxes, 1)  # [B, n_box, N_points]
    # box_inv_p2_scores = torch.sum(box_inv_p2_scores * valid_fg, dim=-1) / torch.clamp(box_fgpoint_counts, min=1)  # [B, n_box]

    scale = ptc_in_box / torch.clamp(boxes_3d[..., 3:6].view(batch_size, num_boxes, 1, 3) * 0.5, min=1e-3)
    scale = torch.max(scale, dim=-1).values  # [B, n_box, N_points]

    m = torch.distributions.normal.Normal(loss_cfgs.BOX_BOUND_MEAN, loss_cfgs.BOX_BOUND_STD, validate_args=None)
    scale = m.log_prob(scale)
    # scale = m.log_prob(scale) + np.log(loss_cfgs.BOX_BOUND_STD) + np.log((2 * np.pi) ** 0.5)

    fg = torch.mean(scale * valid, dim=-1) * loss_cfgs.LOSS_WEIGHTS['fg_loss_weight']

    fg[box_fgpoint_counts == 0] = -distances_box_to_points[box_fgpoint_counts == 0].min(dim=-1).values * 0.1

    if loss_cfgs.LOSS_WEIGHTS['bg_loss_weight'] > 0:
        bg = torch.sum(scale * valid_bg, dim=-1) / torch.clamp(torch.sum(valid_bg, dim=-1), min=1) * loss_cfgs.LOSS_WEIGHTS['bg_loss_weight']
    else:
        bg = torch.zeros_like(fg)

    # sz = size_score_logprob(boxes_3d) * loss_cfgs.LOSS_WEIGHTS['sz_loss_weight']
    if loss_cfgs.get('MIXTURE_SIZE_PRIOR', False) == 'even':
        sz = size_score_multiclass_logprob(boxes_3d) * loss_cfgs.LOSS_WEIGHTS['sz_loss_weight']
    elif loss_cfgs.get('MIXTURE_SIZE_PRIOR', False) == 'learned':
        assert cls_int_logits is not None
        sz = size_score_multiclass_learned_logprob(boxes_3d, cls_int_logits) * loss_cfgs.LOSS_WEIGHTS['sz_loss_weight']
    else:
        sz = size_score_logprob(boxes_3d, 
                                loss_cfgs.LOSS_WEIGHTS['size_prior_mean'], 
                                loss_cfgs.LOSS_WEIGHTS['size_prior_std']) * loss_cfgs.LOSS_WEIGHTS['sz_loss_weight']
    
    score = fg + bg + sz
    loss = -torch.mean(score)

    if return_box_scores:
        return loss, fg.mean().item(), bg.mean().item(), sz.mean().item(), score, box_fgpoint_counts
    else:
        return loss, fg.mean().item(), bg.mean().item(), sz.mean().item()



def single_frame_boundary_residual_loss(ptc, p2_scores, boxes_3d, batch_box_residual_size, batch_size, loss_cfgs, cls_int_logits=None, return_box_scores=False):
    """
    Args:
        ptc: B x npoints x 3        - points in lidar coordinates, single frame
        p2_scores: B x npoints      - p2 scores of points
        boxes_3d: B x n_box x 7      - boxes in lidar coordinates
        batch_box_residual_size: B x n_box x 3 - box residual size
        batch_size: int             - B
        loss_cfgs: dict of configs

    Returns:
        foreground_inside_loss: float tensor
        background_inside_loss: float tensor
        foreground_outside_loss: float tensor
    """
    if torch.isnan(boxes_3d).any():
        print('boxes_3d has nan')
    num_boxes = boxes_3d.shape[1]

    # foreground = (p2_scores < loss_cfgs.UNSUPERVISED_P2_FOREGROUND_THRESHOLD).unsqueeze(dim=1)
    foreground = torch.logical_and(p2_scores < loss_cfgs.UNSUPERVISED_P2_FOREGROUND_THRESHOLD, ptc[..., -1] < 3.).unsqueeze(dim=1)

    trs = common_utils.get_transform_to_box(boxes_3d)  # [B x nbox, 4, 4]
    ptc_in_box = common_utils.transform_points_torch(ptc, trs)  # B x n_box x npoints x 3
    ptc_in_box = torch.abs(ptc_in_box)

    distances_box_to_points = torch.linalg.norm(ptc_in_box, dim=-1)  # B x n_box x npoints
    distances_box_to_points[~foreground.repeat(1, num_boxes, 1)] = 100.  # arbitrary large number

    assos_bound = torch.clamp(boxes_3d[..., 3:6].unsqueeze(dim=2), min=1.)
    in_2xbox = torch.all(ptc_in_box < assos_bound, dim=-1)
    valid = torch.logical_and(foreground, in_2xbox).float()  # [B, n_box, npoints]

    in_1xbox = torch.all(ptc_in_box < boxes_3d[..., 3:6].unsqueeze(dim=2) * 0.5, dim=-1)
    valid_bg = torch.logical_and(~foreground, in_1xbox).float()  # [B, n_box, N_points]

    valid_fg = torch.logical_and(foreground, in_1xbox).float()  # [B, n_box, N_points]
    box_fgpoint_counts = torch.sum(valid_fg, dim=-1)  # [B, n_box]
    # box_inv_p2_scores = 1. - p2_scores.unsqueeze(dim=1).repeat(1, num_boxes, 1)  # [B, n_box, N_points]
    # box_inv_p2_scores = torch.sum(box_inv_p2_scores * valid_fg, dim=-1) / torch.clamp(box_fgpoint_counts, min=1)  # [B, n_box]

    scale = ptc_in_box / torch.clamp(boxes_3d[..., 3:6].view(batch_size, num_boxes, 1, 3) * 0.5, min=1e-3)
    scale = torch.max(scale, dim=-1).values  # [B, n_box, N_points]

    m = torch.distributions.normal.Normal(loss_cfgs.BOX_BOUND_MEAN, loss_cfgs.BOX_BOUND_STD, validate_args=None)
    scale = m.log_prob(scale)
    # scale = m.log_prob(scale) + np.log(loss_cfgs.BOX_BOUND_STD) + np.log((2 * np.pi) ** 0.5)

    fg = torch.mean(scale * valid, dim=-1) * loss_cfgs.LOSS_WEIGHTS['fg_loss_weight']

    fg[box_fgpoint_counts == 0] = -distances_box_to_points[box_fgpoint_counts == 0].min(dim=-1).values * 0.1

    if loss_cfgs.LOSS_WEIGHTS['bg_loss_weight'] > 0:
        bg = torch.sum(scale * valid_bg, dim=-1) / torch.clamp(torch.sum(valid_bg, dim=-1), min=1) * loss_cfgs.LOSS_WEIGHTS['bg_loss_weight']
    else:
        bg = torch.zeros_like(fg)

    score = fg + bg
    loss = -torch.mean(score)

    sz = (torch.abs(batch_box_residual_size) ** 2).mean(dim=-1) * loss_cfgs.LOSS_WEIGHTS['sz_loss_weight']
    loss += sz.mean()

    if return_box_scores:
        return loss, fg.mean().item(), bg.mean().item(), sz.mean().item(), score, box_fgpoint_counts
    else:
        return loss, fg.mean().item(), bg.mean().item(), sz.mean().item()



def single_frame_geometric_consistency_loss(ptc, p2_scores, roi_boxes3d, batch_size, loss_cfgs, return_box_point_mask=False):
    """
    Args:
        ptc: B x npoints x 3        - points in lidar coordinates, single frame
        p2_scores: B x npoints      - p2 scores of points
        roi_boxes3d: B x n_box x 7  - boxes in lidar coordinates
        batch_size: int             - B
        loss_cfgs: dict of configs

    Returns:
        foreground_inside_loss: float tensor
        background_inside_loss: float tensor
        foreground_outside_loss: float tensor
    """
    # Get assignments of point to boxes
    differences = (ptc.unsqueeze(2) - roi_boxes3d[:, :, :3].unsqueeze(1))  # B x npoints x n_box x 3
    distances = torch.sqrt(torch.sum(differences**2, dim=-1))
    # assignments = distances.argmin(axis=-1)  # B x npoints
    assign_dist, assignments = distances.min(dim=-1)  # B x npoints 
    # assignment_mask = distances.min(axis=-1) < loss_cfgs.UNSUPERVISED_ASSIGNMENT_THRESHOLD
    assignment_mask = assign_dist < loss_cfgs.UNSUPERVISED_ASSIGNMENT_THRESHOLD

    # Transform points to box coordinate system
    Tr_lidar_boxes = common_utils.get_transform_to_box(roi_boxes3d)  # B x Nbox x 4 x 4
    ptc_in_box = common_utils.transform_points_torch(ptc, Tr_lidar_boxes)  # B x n_box x npoints x 3

    # Compute distances inside (to edge of box) and outside (to closest center of edges)
    # B x n_box x npoints
    distances_inside = torch.stack([
        torch.minimum(torch.abs(ptc_in_box[..., ax] + (roi_boxes3d[..., ax + 3] / 2.).unsqueeze(-1)),
                        torch.abs(ptc_in_box[..., ax] - (roi_boxes3d[..., ax + 3] / 2.).unsqueeze(-1)))
        for ax in range(3)
    ])
    distances_inside = distances_inside.min(dim=0).values
    distances_inside_assignment = torch.gather(distances_inside, dim=1, index=assignments.unsqueeze(1)).squeeze(1)  # B x npoints
    # distances_inside_assignment[~assignment_mask] = 0.

    dim_diag_matrix = torch.diag_embed(roi_boxes3d[..., 3:6] / 2.)
    # B x n_box x npoints
    distances_outside = torch.stack([
        torch.minimum(torch.linalg.vector_norm(ptc_in_box + dim_diag_matrix[:, :, [ax], :], dim=-1),
                    torch.linalg.vector_norm(ptc_in_box - dim_diag_matrix[:, :, [ax], :], dim=-1))
        for ax in range(3)
    ])
    distances_outside = distances_outside.min(dim=0).values
    distances_outside_assignment = torch.gather(distances_outside, dim=1, index=assignments.unsqueeze(1)).squeeze(1)  # B x npoints
    # distances_outside_assignment[~assignment_mask] = 0.

    # get mask of assignment, fg/bg, inside
    outside_flag_x = torch.logical_or(ptc_in_box[..., 0] > (roi_boxes3d[..., [3]] / 2.), ptc_in_box[..., 0] < (-roi_boxes3d[..., [3]] / 2.))
    outside_flag_y = torch.logical_or(ptc_in_box[..., 1] > (roi_boxes3d[..., [4]] / 2.), ptc_in_box[..., 1] < (-roi_boxes3d[..., [4]] / 2.))
    outside_flag_z = torch.logical_or(ptc_in_box[..., 2] > (roi_boxes3d[..., [5]] / 2.), ptc_in_box[..., 2] < (-roi_boxes3d[..., [5]] / 2.))
    # B x Nbox x Npoints
    outside_box_mask = torch.logical_or(torch.logical_or(outside_flag_x, outside_flag_y), outside_flag_z)
    foreground_mask = p2_scores < loss_cfgs.UNSUPERVISED_P2_FOREGROUND_THRESHOLD  # B x Npoints
    background_mask = p2_scores > loss_cfgs.UNSUPERVISED_P2_BACKGROUND_THRESHOLD  # B x Npoints

    # foreground point, inside
    foreground_inside_mask = torch.logical_and(foreground_mask.unsqueeze(1), ~outside_box_mask)
    foreground_inside_mask = torch.logical_and(torch.gather(foreground_inside_mask, dim=1, 
                                                            index=assignments.unsqueeze(1)).squeeze(1), assignment_mask)  # B x Npoints
    
    foreground_inside_loss = distances_inside_assignment[foreground_inside_mask].sum() / torch.clamp(foreground_inside_mask.sum(), min=1.0)
    # foreground_inside_loss = distances_inside_assignment[foreground_inside_mask].mean()

    # background point, inside
    background_inside_mask = torch.logical_and(background_mask.unsqueeze(1), ~outside_box_mask)
    background_inside_mask = torch.logical_and(torch.gather(background_inside_mask, dim=1, 
                                                            index=assignments.unsqueeze(1)).squeeze(1), assignment_mask)  # B x Npoints
    background_inside_loss = distances_inside_assignment[background_inside_mask].sum() / torch.clamp(background_inside_mask.sum(), min=1.0)
    # background_inside_loss = distances_inside_assignment[background_inside_mask].mean()

    # foreground point, outside
    foreground_outside_mask = torch.logical_and(foreground_mask.unsqueeze(1), outside_box_mask)
    foreground_outside_mask = torch.logical_and(torch.gather(foreground_outside_mask, dim=1, 
                                                                index=assignments.unsqueeze(1)).squeeze(1), assignment_mask)  # B x Npoints
    foreground_outside_loss = distances_outside_assignment[foreground_outside_mask].sum() / torch.clamp(foreground_outside_mask.sum(), min=1.0)
    # foreground_outside_loss = distances_outside_assignment[foreground_outside_mask].mean()

    if return_box_point_mask:
        return foreground_inside_loss, background_inside_loss, foreground_outside_loss, ~outside_box_mask
    else:    
        return foreground_inside_loss, background_inside_loss, foreground_outside_loss
