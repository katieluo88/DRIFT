import os
import os.path as osp
import pickle
import sys
import copy

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

import ray
import torch
from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from discovery_utils import kitti_util

def load_velo_scan(velo_filename):
    scan = np.fromfile(velo_filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    return scan

# def transform_points(pts_3d_ref, Tr):
#     pts_3d_ref = cart2hom(pts_3d_ref)  # nx4
#     return np.dot(pts_3d_ref, np.transpose(Tr)).reshape(-1, 4)[:, 0:3]

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def display_args(args):
    eprint("========== filtering info ==========")
    eprint("host: {}".format(os.getenv('HOSTNAME')))
    eprint(OmegaConf.to_yaml(args))
    eprint("====================================")

def filter_by_ppscore(pp_score, percentile=50, threshold=0.5):
    if len(pp_score) == 0 or np.percentile(pp_score, percentile) > threshold:
        return False
    return True


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_aum(cls_pred):
    cls_pred = sigmoid(cls_pred)
    cls_pred = np.sort(cls_pred, axis=1)[:, ::-1]
    return cls_pred[:, 0] - cls_pred[:, 1]

def crop_fov(ptc, calib, img_shape=(1024, 1224), border=0):
    # if isinstance(calib, dict):  # dataset == "ithaca365":
    #     # TODO: CHECK
    #     pts_cam = transform_points(ptc[:, 0:3], calib['ref_to_cam'])
    #     pts_2d = np.dot(pts_cam, calib['cam_intrinsic'].T)
    #     pts_img = (pts_2d[:, 0:2].T / pts_2d[:, 2]).T  # (N, 2)
    # else:  # dataset kitti
    pts_img = calib.project_velo_to_image(ptc)
    val_flag_1 = np.logical_and(
        pts_img[:, 0] >= border, pts_img[:, 0] < img_shape[1] - border)
    val_flag_2 = np.logical_and(
        pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    pts_valid_flag = np.logical_and(val_flag_merge, ptc[:, 0] >= 0)
    return pts_valid_flag


def update_pred(ptc, pp_score, calib, preds, previous_det_bbox, args, img_shape, confidences_per_class=None):
    preds = copy.deepcopy(preds)
    preds_boxes = preds['boxes_lidar']
    box_idxs_of_pts = points_in_boxes_gpu(
        torch.from_numpy(ptc).float().cuda().unsqueeze(dim=0),
        torch.from_numpy(preds_boxes).float().cuda().unsqueeze(dim=0).contiguous()
    ).long().squeeze(dim=0).cpu()
    mask_p2 = np.array([
        filter_by_ppscore(
            pp_score[box_idxs_of_pts == i],
            percentile=args.pp_score_percentile,
            threshold=args.pp_score_threshold)
         for i in range(preds_boxes.shape[0])
    ], dtype=bool)
    if args.confidence_score_threshold > 0 and len(preds['name']) > 0:
        if confidences_per_class is None:
            mask_score = np.array([score > args.confidence_score_threshold for score in preds['score']])
        else:
            mask_score = np.array([
                score > confidences_per_class[name] for name, score in zip(
                    preds['name'],
                    preds['score'] if not args.posterior_regularization.by_aum else
                compute_aum(preds['cls_pred']) )])
        mask_p2_soft = np.array([
            filter_by_ppscore(
                pp_score[box_idxs_of_pts == i],
                percentile=args.pp_score_percentile,
                threshold=args.soft_pp_score_threshold)
            for i in range(preds_boxes.shape[0])
        ])
        mask_score = np.logical_and(mask_score, mask_p2_soft)
        mask_score_soft = np.array([score > args.soft_confidence_score_threshold for score in preds['score']])
        mask_p2 = np.logical_and(mask_p2, mask_score_soft)
        if args.and_operation:
            mask = np.logical_and(mask_p2, mask_score)
        else:
            mask = np.logical_or(mask_p2, mask_score)
    else:
        mask = mask_p2
    if getattr(args, 'remove_near_fov', None) is not None:
        obj_centers = preds_boxes[mask][:, :3]
        sub_mask = crop_fov(
            obj_centers, calib, img_shape=img_shape, border=args.remove_near_fov.border)
        mask[mask] = sub_mask

    if getattr(args, 'check_aum', None) is not None:
        cls_pred = preds['cls_pred'][mask]
        if cls_pred.shape[0] > 0:
            aums = compute_aum(cls_pred)
            sub_mask = aums >= args.check_aum.threshold
            mask[mask] = sub_mask

    if getattr(args, 'check_consistency', None) is not None:
        preds_boxes = copy.deepcopy(preds_boxes[mask])
        preds_name = preds['name'][mask]
        if preds_boxes.shape[0] > 0 and len(previous_det_bbox) > 0:
            overlap_bevs = []
            for pre_det in previous_det_bbox:
                pre_det_bboxes = copy.deepcopy(pre_det['boxes_lidar'])
                if pre_det_bboxes.shape[0] > 0:
                    overlaps_bev = iou3d_nms_utils.boxes_iou_bev(
                        torch.from_numpy(pre_det_bboxes).float().cuda().contiguous(),
                        torch.from_numpy(preds_boxes).float().cuda().contiguous())
                    overlaps_bev = overlaps_bev.cpu().numpy()
                    max_overlap_bbox = np.argmax(overlaps_bev, axis=0)
                    max_overlap_name = pre_det['name'][max_overlap_bbox]
                    max_overlap = np.max(overlaps_bev, axis=0)
                    max_overlap[max_overlap_name != preds_name] = 0
                    overlap_bevs.append(max_overlap)
                else:
                    overlap_bevs.append(np.zeros(preds_boxes.shape[0]))
            overlap_bevs = np.array(overlap_bevs)
            is_consistent_det = overlap_bevs > args.check_consistency.threshold
            sub_mask = is_consistent_det.sum(axis=0) >= (args.check_consistency.ratio * len(previous_det_bbox))
            mask[mask] = sub_mask

    for k in preds:
        if not k in ["frame_id", "metadata"]:
            preds[k] = preds[k][mask]
    return preds


def process_one_scene(idx, args, det_bboxes, previous_det_bboxes=(), confidences_per_class=None):
    det_bbox = det_bboxes[idx]
    previous_det_bbox = [pre_result[idx] for pre_result in previous_det_bboxes]
    if args.dataset == "lyft":
        frame_id = int(det_bbox['frame_id'])
        pp_score_path = osp.join(args.data_paths.p2score_path, f"{frame_id:06d}.npy")
        lidar_path = osp.join(args.data_paths.scan_path, f"{frame_id:06d}.bin")
        calib = kitti_util.Calibration(
            osp.join(args.data_paths.calib_path, f"{frame_id:06d}.txt"))
    elif args.dataset == "ithaca365":
        # pp_score_path = osp.join(args.data_paths.p2score_path, f"{det_bbox['metadata']['token']}.npy")
        lidar_path = osp.join(args.data_paths.ptc_path, str(det_bbox['frame_id']))
        pp_score_path = osp.join(args.data_paths.pp_score_path, f"{det_bbox['frame_id'].stem}.npy")
        calib = None
    ptc = load_velo_scan(lidar_path)[:, :3]
    pp_score = np.load(pp_score_path)
    return update_pred(
        ptc, pp_score, calib, det_bbox,
        previous_det_bbox,
        args=args.det_filtering,
        img_shape=args.data_paths.image_shape,
        confidences_per_class=confidences_per_class
        )

@ray.remote(num_gpus=0.1, max_retries=0)
def process_batch_scene(idx_list, args, det_bboxes, previous_det_bboxes=(), confidences_per_class=None):
    return [
        process_one_scene(idx, args, det_bboxes, previous_det_bboxes, confidences_per_class)
        for idx in idx_list
    ]


@hydra.main(config_path="configs/", config_name="p2_score_filtering.yaml")
def main(args):
    display_args(args)
    ray.init(num_cpus=args.n_processes, num_gpus=4)
    det_bboxes = pickle.load(open(args.result_path, "rb"))
    previous_results = []
    for p in getattr(args, "previous_result_paths", []):
        previous_results.append(pickle.load(open(p, "rb")))
    det_bboxes_new = []
    det_bboxes_id = ray.put(det_bboxes)
    previous_results_id = ray.put(previous_results)

    # add confidence computation per class
    confidences_per_class = None
    # if getattr(args.det_filtering, "posterior_regularization", []):
    #     posterior_regularization = args.det_filtering.posterior_regularization
    #     # regularizations = pickle.load(open(args.posterior_regularization, "rb"))  # assume it is regularized
    #     args.det_filtering.confidence_score_threshold = 999.
    #     # loop through det. boxes
    #     confidences = {}
    #     total_det = 0
    #     # import ipdb; ipdb.set_trace()
    #     for frame_detection in det_bboxes:
    #         if len(frame_detection['name']) > 0:
    #             for name, score in zip(
    #                 frame_detection['name'],
    #                 frame_detection['score'] if not args.det_filtering.posterior_regularization.by_aum else
    #                 compute_aum(frame_detection['cls_pred'])):
    #                 if name not in confidences:
    #                     confidences[name] = []
    #                 confidences[name].append(score)
    #         total_det += frame_detection['name'].shape[0]

    #      # set threshold
    #     confidences_per_class = {
    #     } # some values diction

    #     # get threshold of confidence per class
    #     for name in confidences:
    #         reg_ratio = getattr(posterior_regularization, name.lower(), None)
    #         print(name.lower(), reg_ratio)
    #         percentile = getattr(posterior_regularization, "percentile", None)
    #         if percentile is not None: 
    #             print("using percentile!")
    #             num_box_per_class = max(int(len(confidences[name]) * (1 - percentile)), 
    #                 args.det_filtering.posterior_regularization.min_num_boxes)
    #         elif reg_ratio is None:
    #             num_box_per_class = 0
    #         else:
    #             reg_ratio *= args.det_filtering.posterior_regularization.alpha  # multiply by confidence constant
    #             num_box_per_class = int(reg_ratio * len(det_bboxes))  # int(len(confidences[name]) * (1 - percentile))
    #         sorted_confidence = np.sort(confidences[name])[::-1]
    #         confidences_per_class[name] = max(
    #             args.det_filtering.posterior_regularization.min_val,
    #             sorted_confidence[min(num_box_per_class, len(sorted_confidence) - 1)])
    #         print("Posterior regularization values:", reg_ratio, num_box_per_class, confidences_per_class[name])

    idx_list = np.arange(len(det_bboxes), dtype=int)
    for idx_sublist in np.array_split(idx_list, args.n_processes):
        det_bboxes_new.append(process_batch_scene.remote(
            idx_sublist, args, det_bboxes_id, previous_results_id, confidences_per_class))

    det_bboxes_new = ray.get(det_bboxes_new)
    det_bboxes_new = sum(det_bboxes_new, [])
    count_before = 0
    for det_bbox in det_bboxes:
        count_before += det_bbox['boxes_lidar'].shape[0]
    count_after = 0
    for det_bbox in det_bboxes_new:
        count_after += det_bbox['boxes_lidar'].shape[0]
    print(f"#bbox before: {count_before}, #bbox after: {count_after}")
    pickle.dump(det_bboxes_new, open(args.save_path, "wb"))


if __name__ == "__main__":
    main()
