import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import norm



def get_transform(boxs: torch.Tensor) -> torch.Tensor:
    # Extract box center coordinates, dimensions and rotation angle
    center_xyz = boxs[:, :3]
    dimensions = boxs[:, 3:6]
    rotation_xy = boxs[:, 6]

    # Compute rotation matrix around the z-axis
    cos, sin = torch.cos(rotation_xy), torch.sin(rotation_xy)
    zero, one = torch.zeros_like(cos), torch.ones_like(cos)
    rotation_z = torch.stack([cos, -sin, zero, zero,
                              sin, cos, zero, zero,
                              zero, zero, one, zero,
                              zero, zero, zero, one], dim=-1).view(-1, 4, 4)

    # Compute translation matrix to move the center of the box to the origin
    translation = torch.eye(4).to(boxs.device).unsqueeze(0).repeat(center_xyz.shape[0], 1, 1)
    translation[:, 3, :3] = -center_xyz
    return torch.matmul(translation, rotation_z)

def get_scale(
    ptc: torch.Tensor,  # (N_points, 4)
    boxs: torch.Tensor  # (M, 7)
) -> torch.Tensor:
    ptc = torch.cat([ptc[:, :3], torch.ones_like(ptc[:, :1])], dim=1)  # [N_points, 4]
    ptc = ptc.unsqueeze(dim=0).expand(boxs.shape[0], ptc.shape[0], 4)  # [M, N_points, 4]

    trs = get_transform(boxs)  # [M, 4, 4]
    ptc = torch.bmm(ptc, trs)[:, :, :3]  # [M, N_points, 3]
    ptc = torch.abs(ptc)  # [M, N_points, 3]

    scale = ptc / (boxs[:, 3:6].unsqueeze(dim=1) * 0.5)
    scale = torch.max(scale, dim=2).values
    return scale

def get_scale2(
    ptc: torch.Tensor,  # (N_points, 4)
    boxs: torch.Tensor,  # (M, 7)
) -> torch.Tensor:
    ptc = torch.cat([ptc[:, :3], torch.ones_like(ptc[:, :1])], dim=1)  # [N_points, 4]
    ptc = ptc.unsqueeze(dim=0).expand(boxs.shape[0], ptc.shape[0], 4)  # [M, N_points, 4]

    trs = get_transform(boxs)  # [M, 4, 4]
    ptc = torch.bmm(ptc, trs)[:, :, :3]  # [M, N_points, 3]
    ptc = torch.abs(ptc)  # [M, N_points, 3]

    scale = ptc / (boxs[:, 3:6].unsqueeze(dim=1) * 0.5)
    scale_xy = torch.max(scale[..., :2], dim=-1).values
    scale = torch.max(scale, dim=2).values
    return scale_xy, scale

def get_scale3(
    ptc: torch.Tensor,  # (N_points, 4)
    boxs: torch.Tensor,  # (M, 7)
) -> torch.Tensor:
    ptc = torch.cat([ptc[:, :3], torch.ones_like(ptc[:, :1])], dim=1)  # [N_points, 4]
    ptc = ptc.unsqueeze(dim=0).expand(boxs.shape[0], ptc.shape[0], 4)  # [M, N_points, 4]

    trs = get_transform(boxs)  # [M, 4, 4]
    ptc = torch.bmm(ptc, trs)[:, :, :3]  # [M, N_points, 3]
    ptc = torch.abs(ptc)  # [M, N_points, 3]

    scale = ptc / (boxs[:, 3:6].unsqueeze(dim=1) * 0.5)
    scale_xy = torch.max(scale[..., :2], dim=-1).values
    scale_z = scale[..., 2]
    scale = torch.max(scale, dim=2).values
    return scale_xy, scale_z, scale
        

def reward_p2_percentile(scale, p2_score, percentile=0.2):
    in_1xbox = scale < 1.0  # (M, N_points,)
    p2_score, sort_idx = torch.sort(p2_score)
    in_1xbox = in_1xbox[:, sort_idx]
    p2_score_in_1xbox = p2_score.expand(in_1xbox.shape)[in_1xbox]

    if p2_score_in_1xbox.shape[0] == 0:
        print("warning: p2_score_in_1xbox.shape[0] = 0!!!")
        return torch.ones(in_1xbox.shape[0], dtype=p2_score.dtype, device=p2_score.device)
    
    element_count = in_1xbox.sum(dim=1)
    start_index = F.pad(input=torch.cumsum(element_count,dim=0), pad=(1,0,), mode='constant', value=0)[:-1]
    twenty_pct_element_count = ((element_count - 1) * percentile).long()
    # twenty_pct_element_count = torch.minimum(twenty_pct_element_count, element_count-1)
    twenty_pct_index = start_index + twenty_pct_element_count
    # assert torch.all(twenty_pct_index < p2_score_in_1xbox.shape[0]), "twenty_pct_index >= p2_score_in_1xbox.shape[1]!!!"
    twenty_pct_index = torch.clamp(twenty_pct_index, max=p2_score_in_1xbox.shape[0]-1) # fixing the edge case
    twenty_pct = p2_score_in_1xbox[twenty_pct_index]
    
    # dealing with some edge cases:
    twenty_pct[element_count == 0] = 1
    return twenty_pct


def reward_num_fg_points(scale, ptc, p2_scores, fg_thresh=0.2):
    # scale = get_scale(ptc, box_preds)
    foreground = torch.logical_and(p2_scores < fg_thresh, ptc[..., -1] < 3.)
    in_1xbox = scale < 1.0  # (B, M, N_points,)
    valid_fg = torch.logical_and(foreground.unsqueeze(dim=0), in_1xbox).float()  # [n_box, N_points]
    box_fgpoint_counts = torch.sum(valid_fg, dim=-1)  # [n_box]
    return box_fgpoint_counts

def reward_num_bg_points(scale, ptc, p2_scores, bg_thresh=0.9):
    background = torch.logical_or(p2_scores > bg_thresh, ptc[..., -1] > 3.)
    in_1xbox = scale < 1.0  # (B, M, N_points,)
    valid_bg = torch.logical_and(background.unsqueeze(dim=0), in_1xbox).float()  # [n_box, N_points]
    box_bgpoint_counts = torch.sum(valid_bg, dim=-1)  # [B, n_box]
    return box_bgpoint_counts    

def reward_size_prior(boxs):
    """
    boxs: nbox x 7
    """
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


def reward_size_multivar_prior(boxs):
    """
    boxs: nbox x 7
    """
    means = [torch.tensor([4.74451989, 1.91059287, 1.71107344]),
             torch.tensor([0.79702832, 0.77995906, 1.74490618]),
             torch.tensor([9.40333292, 2.83213669, 3.29912471]),
             torch.tensor([1.75168967, 0.61337013, 1.36374118])]
    stds = [torch.tensor([0.55945502, 0.16199086, 0.24788235]),
            torch.tensor([0.18178839, 0.15324556, 0.1771094 ]),
            torch.tensor([3.14533891, 0.27833338, 0.43030043]),
            torch.tensor([0.32637668, 0.25591983, 0.34314765])]
    # weights = [0.1498824052196492, 0.07024215684742731, 0.6137651955260903, 0.16929808460478427]
    dist = [torch.distributions.multivariate_normal.MultivariateNormal(
        mean.to(boxs.dtype).to(boxs.device), 
        torch.diag(std.to(boxs.dtype).to(boxs.device)), validate_args=None)
            for mean, std in zip(means, stds)]
    scales = [m.log_prob(boxs[..., 3:6]) + torch.log(torch.sqrt(torch.prod(std.to(boxs.dtype).to(boxs.device)))) for m, std in zip(dist, stds)]
    # dists = [torch.distributions.normal.Normal(
    #     mean.view(1, 3).to(boxs.dtype).to(boxs.device), 
    #     std.view(1, 3).to(boxs.dtype).to(boxs.device), validate_args=None) for mean, std in zip(means, stds)]
    # scales = [2 * np.log(w) + m.log_prob(boxs[..., 3:6]).sum(dim=-1) for w, m in zip(weights, dists)]
    scale = torch.logsumexp(torch.stack(scales, dim=0), dim=0)
    return scale


def reward_size_prior_evenstd(boxs):
    """
    boxs: nbox x 7
    """
    means = [torch.tensor([4.74451989, 1.91059287, 1.71107344]),
             torch.tensor([0.79702832, 0.77995906, 1.74490618]),
             torch.tensor([9.40333292, 2.83213669, 3.29912471]),
             torch.tensor([1.75168967, 0.61337013, 1.36374118])]
    stds = [torch.tensor([0.5, 0.5, 0.5]),
            torch.tensor([0.5, 0.5, 0.5]),
            torch.tensor([0.5, 0.5, 0.5]),
            torch.tensor([0.5, 0.5, 0.5])]
    weights = [0.25, 0.25, 0.25, 0.25]
    dists = [torch.distributions.normal.Normal(
        mean.view(1, 3).to(boxs.dtype).to(boxs.device), 
        std.view(1, 3).to(boxs.dtype).to(boxs.device), validate_args=None) for mean, std in zip(means, stds)]
    scales = [np.log(w) + m.log_prob(boxs[..., 3:6]).sum(dim=-1) for w, m in zip(weights, dists)]
    scale = torch.logsumexp(torch.stack(scales, dim=0), dim=0)
    return scale


def fg_score_prev(
    scale: torch.Tensor,  # (M, N_points,)
    pp_score: torch.Tensor,  # (N_points,)
    scale_range: float,
    alpha: float,
) -> torch.Tensor:  # (N_points,)
    assert scale_range > 0.01
    with torch.no_grad():
        log_p = alpha * torch.log(1 - pp_score) - np.log(alpha + 1)
#         log_p -= alpha * torch.log(pp_score) - np.log(alpha + 1)
        log_p = log_p.unsqueeze(dim=0).expand(*scale.shape)  # (M, N_points,)
        valid = (scale < scale_range).to(scale.dtype)  # (M, N_points,)
        count = torch.sum(valid, dim=-1)  # (M，)
    score = log_p + torch.distributions.normal.Normal(torch.tensor([0.88]).to(scale.device), torch.tensor([0.16]).to(scale.device)).log_prob(scale)  # (M, N_points,)

    score = score * valid
    score = torch.where(
        count > 5,
        (torch.sum(score, dim=-1) / count - torch.log(torch.sum(torch.exp(score) * valid, dim=-1)).detach() + torch.log(count)) / count,
        torch.ones_like(count) * -15)
    return score

def fg_hard_score(
    scale,  # (M, N_points,)
    pp_score,  # (N_points,)
    ptc,  # (N_points, 3)
    scale_range=2.,
    fg_thresh=0.6,
) -> torch.Tensor:  # (N_points,)
    assert scale_range > 0.01
    with torch.no_grad():
        valid = (scale < scale_range).to(scale.dtype)  # (M, N_points,)
        # foreground = (pp_score < fg_thresh).to(scale.dtype)  # (N_points,)
        foreground = torch.logical_and(pp_score < fg_thresh, ptc[..., -1] < 3.).to(scale.dtype)
        valid_fg = torch.logical_and(valid, foreground.unsqueeze(dim=0))  # (M, N_points,)
    score = torch.distributions.normal.Normal(torch.tensor([0.8]).to(scale.device), torch.tensor([0.2]).to(scale.device)).log_prob(scale)  # (M, N_points,)

    score = torch.mean(score * valid_fg, dim=-1)  # (M, )
    return score


def reward_kl(
    ptc: torch.Tensor,  # (N_points, 4)
    pp_score: torch.Tensor,  # (N_points,)
    boxs: torch.Tensor,
) -> torch.tensor:  # (...,)
    scale = get_scale(ptc, boxs)
    fg = fg_score_prev(
        scale,
        pp_score,
        scale_range=2,
        alpha=0.5)
    sz = reward_size_prior(boxs)
    
    result = fg + sz
    return result


def reward_mix(ptc, pp_score, boxes, a=1., b=1., c=0., d=0.001, e=0.001, 
               min_ptc_count=3, 
               volume_min=0.1, 
               volume_max=315.,
               height_min=-5.,
               height_max=2.,):
    scale = get_scale(ptc, boxes)
    fg = torch.exp(fg_hard_score(scale, pp_score, ptc))
    sz = torch.exp(reward_size_multivar_prior(boxes))
    p2_perc = reward_p2_percentile(scale, pp_score)
    fg_cnt = reward_num_fg_points(scale, ptc, pp_score, fg_thresh=0.2)
    bg_cnt = reward_num_bg_points(scale, ptc, pp_score, bg_thresh=0.9)
    reward = a*fg + b*sz + d*fg_cnt - e*bg_cnt

    # filter by p2 percentile
    reward[p2_perc > 0.7 ] = 0
    reward[fg_cnt <= min_ptc_count] = 0
    
    # filter by volume
    volumes = torch.prod(boxes[:, 3:6], dim=-1)
    valid_volumes = torch.logical_and(volumes > volume_min, volumes < volume_max)

    # filter by height
    center_z = boxes[:, 2]
    valid_heights = torch.logical_and(center_z > height_min, center_z < height_max)
    valid_mask = torch.logical_and(valid_volumes, valid_heights)
    reward[~valid_mask] = 0

    return reward, fg_cnt

def fg_score(
    scale: torch.Tensor,  # (M, N_points,)
    pp_score: torch.Tensor,  # (N_points,)
    scale_range: float,
    alpha: float,
) -> torch.Tensor:  # (N_points,)
    assert scale_range > 0.01
    with torch.no_grad():
        log_p = alpha * torch.log(1 - pp_score) - np.log(alpha + 1)
        log_p = log_p.unsqueeze(dim=0).expand(*scale.shape)  # (M, N_points,)
        valid = (scale < scale_range).to(scale.dtype)  # (M, N_points,)
        valid = valid * (1 - pp_score).unsqueeze(dim=0)
        count = torch.sum(valid, dim=-1)  # (M，)
    score = log_p + torch.distributions.normal.Normal(torch.tensor([0.88]).to(scale.device), torch.tensor([0.16]).to(scale.device)).log_prob(scale)  # (M, N_points,)

    score = score * valid
    score = torch.where(
        count > 3.5,
        (torch.sum(score, dim=-1) / count - torch.log(torch.sum(torch.exp(score) * valid, dim=-1)).detach() + torch.log(count)) / count,
        torch.ones_like(count) * -1)
    return score

def get_cdf(scale_range, mu=0.88, sigma=0.16):
    return norm.cdf((scale_range - mu) / sigma) - norm.cdf(-mu / sigma)

