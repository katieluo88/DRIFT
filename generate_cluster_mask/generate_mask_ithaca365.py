import os
import os.path as osp
import sys
import warnings
import pickle

import hydra
import numpy as np
import sklearn
from omegaconf import DictConfig, OmegaConf
from sklearn import cluster
from tqdm.auto import tqdm

from discovery_utils.clustering_utils import filter_labels, precompute_affinity_matrix
from discovery_utils.pointcloud_utils import above_plane, estimate_plane, load_velo_scan, get_obj, transform_points
from discovery_utils import kitti_util

warnings.filterwarnings(
    "ignore", category=sklearn.exceptions.EfficiencyWarning)

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def display_args(args):
    eprint("========== clustering info ==========")
    eprint("host: {}".format(os.getenv('HOSTNAME')))
    eprint(OmegaConf.to_yaml(args))
    eprint("=====================================")


def convert_obj(obj, Tr):
    loc_lidar = Tr[:3, :3].T @ obj.t  # rect
    loc_lidar[2] += obj.h / 2
    return np.array([loc_lidar[0], loc_lidar[1], loc_lidar[2], obj.l, obj.w, obj.h, -(np.pi / 2 + obj.ry) ])

@hydra.main(config_path="configs/", config_name="generate_mask_ithaca365.yaml")
def main(args: DictConfig):
    display_args(args)
    # idx_list = [int(x) for x in open(args.data_paths.idx_list).readlines()]
    # idx_list = np.array(idx_list)
    with open(args.data_paths.train_infos_path, 'rb') as f:
        train_infos = pickle.load(f)

    if args.total_part > 1:
        num_infos_in_part = round(len(train_infos) / args.total_part)
        train_infos = train_infos[args.part * num_infos_in_part:(args.part + 1) * num_infos_in_part]

    if args.data_paths.get("bbox_info_save_dst", "None") is not None:
        os.makedirs(args.data_paths.bbox_info_save_dst, exist_ok=True)
        if not osp.exists(osp.join(args.data_paths.bbox_info_save_dst, "configs.yaml")):
            OmegaConf.save(config=args, f=osp.join(
                args.data_paths.bbox_info_save_dst, "configs.yaml"))
    # seed_infos = []
    for info in tqdm(train_infos):
        if osp.exists(osp.join(args.data_paths.bbox_info_save_dst, info["timestamp"] + ".pkl")):
            continue

        # load point cloud and p2 score
        ptc = np.fromfile(osp.join(args.data_paths.ptc_path, str(info["lidar_path"])), 
                          dtype=np.float32, count=-1).reshape([-1, 4])[:, :4]
        pp_score = np.load(
            osp.join(args.data_paths.pp_score_path, info["timestamp"] + ".npy"))
        
        # compute and filter ground plane, out of range points
        plane = estimate_plane(
            ptc[:, :3], max_hs=args.plane_estimate.max_hs, ptc_range=args.plane_estimate.range)
        plane_mask = above_plane(
            ptc[:, :3], plane,
            offset=args.plane_estimate.offset,
            only_range=args.plane_estimate.range)
        range_mask = (ptc[:, 0] <= args.limit_range[0][1]) * \
            (ptc[:, 0] > args.limit_range[0][0]) * \
            (ptc[:, 1] <= args.limit_range[1][1]) * \
            (ptc[:, 1] > args.limit_range[1][0])
        final_mask = plane_mask * range_mask

        # generate connectivity graph with p2 score and distance
        dist_knn_graph = precompute_affinity_matrix(
            ptc[final_mask],
            pp_score[final_mask],
            neighbor_type=args.graph.neighbor_type,
            affinity_type=args.graph.affinity_type,
            n_neighbors=args.graph.n_neighbors,
            radius=args.graph.radius,
        )

        if args.clustering.method == "DBSCAN":
            labels = np.zeros(ptc.shape[0], dtype=int) - 1
            labels[final_mask] = cluster.DBSCAN(
                metric='precomputed',
                eps=args.clustering.DBSCAN.eps,
                min_samples=args.clustering.DBSCAN.min_samples,
                n_jobs=-1).fit(dist_knn_graph).labels_
        else:
            raise NotImplementedError(args.clustering.method)
        
        # filtering + gen bounding box that are too small, too large, or too few points
        labels_filtered = filter_labels(
            ptc, pp_score, labels,
            **args.filtering)
        Tr = np.array([[0, -1, 0, 0],
                        [0, 0, -1, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 1]])
        ptc_in_rect = transform_points(ptc[:, :3], Tr)
        objs = []
        for i in range(1, labels_filtered.max()+1):
            obj = get_obj(ptc_in_rect[labels_filtered == i], ptc_in_rect,
                            fit_method=args.bbox_gen.fit_method)
            if obj.volume > args.filtering.min_volume and obj.volume < args.filtering.max_volume:
                objs.append(obj)
            else:
                labels_filtered[labels_filtered == i] = 0

        label_mapping = sorted(list(set(labels_filtered)))
        label_mapping = {x: i for i, x in enumerate(label_mapping)}
        for _i in label_mapping:
            labels_filtered[labels_filtered == _i] = label_mapping[_i]

        # convert obj to lidar coordinate
        seed_info = {
            "boxes_lidar": np.array([convert_obj(obj, Tr) for obj in objs]),
            "name": np.array(['Dynamic' for _ in range(len(objs))]),
            "metadata": {"token": info["lidar_token"]} 
        }
        # seed_infos.append(seed_info)
    
        with open(osp.join(args.data_paths.bbox_info_save_dst, info["timestamp"] + ".pkl"), "wb") as f:
            pickle.dump(seed_info, f)
        
        del dist_knn_graph


if __name__ == "__main__":
    main()
