import os
import os.path as osp
import sys
import warnings
import pickle

import hydra
import numpy as np
import sklearn
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from discovery_utils.pointcloud_utils import objs_lidar_nms, obs_within_fov

warnings.filterwarnings(
    "ignore", category=sklearn.exceptions.EfficiencyWarning)


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def display_args(args):
    eprint("========== kitti_label gen info ==========")
    eprint("host: {}".format(os.getenv('HOSTNAME')))
    eprint(OmegaConf.to_yaml(args))
    eprint("==========================================")

@hydra.main(config_path="configs/", config_name="generate_label_files_ithaca365.yaml")
def main(args: DictConfig):
    display_args(args)
    with open(args.data_paths.train_infos_path, 'rb') as f:
        train_infos = pickle.load(f)
    seed_detection_infos = {}
    for info in tqdm(train_infos):
        assert osp.exists(osp.join(args.data_paths.bbox_info_save_dst, info["timestamp"] + ".pkl")), "File missing."
        with open(osp.join(args.data_paths.bbox_info_save_dst, info["timestamp"] + ".pkl"), "rb") as f:
            seed_annos = pickle.load(f)
        if args.nms.enable and len(seed_annos['boxes_lidar']) > 0:
            seed_annos = objs_lidar_nms(seed_annos, nms_threshold=args.nms.threshold)
        # calib = kitti_util.Calibration(
        #     osp.join(args.calib_path, f"{idx:06d}.txt"))
        if args.fov_only and len(seed_annos['boxes_lidar']) > 0:
            seed_annos = obs_within_fov(seed_annos, info["ref_to_cam"], info["cam_intrinsic"], args.image_shape)
        seed_detection_infos[seed_annos["metadata"]["token"]] = seed_annos
    
    with open(args.data_paths.seed_labels_info_path, 'wb') as f:
        pickle.dump(seed_detection_infos, f)


if __name__ == "__main__":
    main()
