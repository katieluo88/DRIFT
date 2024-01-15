import copy
import pickle
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm
import os.path as osp
import os
import torch

from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import common_utils
from ..dataset import DatasetTemplate
from .ithaca365_utils import map_name_from_general_to_detection

def cart2hom(pts_3d):
    n = pts_3d.shape[0]
    pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1), dtype=np.float32)))
    return pts_3d_hom


def transform_points(pts_3d_ref, Tr):
    pts_3d_ref = cart2hom(pts_3d_ref)  # nx4
    return np.dot(pts_3d_ref, np.transpose(Tr)).reshape(-1, 4)[:, 0:3]


class Ithaca365Dataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        root_path = (Path(root_path) if root_path is not None else Path(dataset_cfg.DATA_PATH)) / dataset_cfg.VERSION
        # root_path = (root_path if root_path is not None else Path(dataset_cfg.DATA_PATH))
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.infos = []
        self.include_ithaca365_data(self.mode)
        # self.ithaca_365_dataset = Ithaca365(
        #     version=self.dataset_cfg.VERSION,
        #     dataroot=str(self.root_path),
        #     verbose=True)
        if self.training and self.dataset_cfg.get('BALANCED_RESAMPLING', False):
            self.infos = self.balanced_infos_resampling(self.infos)

        if 'LOAD_HISTORY' in dataset_cfg:
            self.load_history = copy.deepcopy(dataset_cfg.LOAD_HISTORY)
            self.history_cache_dir = None
            if self.load_history.get('CACHE_ROOT', "none") != "none":
                if self.load_history.get("HISTORY_AUG", False):
                    self.history_cache_dir = osp.join(
                        self.load_history.CACHE_ROOT,
                        f"raw_points_fwonly={self.load_history.FORWARD_ONLY}"
                        f"_ithaca365_{dataset_cfg.VERSION}"
                        f"_history_scans_path={osp.basename(osp.normpath(self.load_history.DATA_PATH))}")
                else:
                    self.history_cache_dir = osp.join(
                        self.load_history.CACHE_ROOT,
                        f"fwonly={self.load_history.FORWARD_ONLY}_vs={self.load_history.VOXEL_SIZE:02f}"
                        f"_ithaca365_{dataset_cfg.VERSION}"
                        f"_history_scans_path={osp.basename(osp.normpath(self.load_history.DATA_PATH))}")
                os.makedirs(self.history_cache_dir, exist_ok=True)
                os.chmod(self.history_cache_dir, 0o777)

        if 'LOAD_P2_SCORE' in dataset_cfg:
            self.load_p2_score = copy.deepcopy(dataset_cfg.LOAD_P2_SCORE)
        self.contant_reflex = dataset_cfg.get('CONSTANT_REFLEX', False)

    def include_ithaca365_data(self, mode):
        self.logger.info('Loading Ithaca365 dataset')
        # import ipdb; ipdb.set_trace()
        ithaca365_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                ithaca365_infos.extend(infos)

        self.infos.extend(ithaca365_infos)
        self.logger.info('Total samples for Ithaca365 dataset: %d' % (len(ithaca365_infos)))

    def balanced_infos_resampling(self, infos):
        """
        Class-balanced sampling of nuScenes dataset from https://arxiv.org/abs/1908.09492
        """
        if self.class_names is None:
            return infos

        cls_infos = {name: [] for name in self.class_names}
        for info in infos:
            for name in set(info['gt_names']):
                if name in self.class_names:
                    cls_infos[name].append(info)

        duplicated_samples = sum([len(v) for _, v in cls_infos.items()])
        cls_dist = {k: len(v) / duplicated_samples for k, v in cls_infos.items()}

        sampled_infos = []

        frac = 1.0 / len(self.class_names)
        ratios = [frac / v for v in cls_dist.values()]

        for cur_cls_infos, ratio in zip(list(cls_infos.values()), ratios):
            sampled_infos += np.random.choice(
                cur_cls_infos, int(len(cur_cls_infos) * ratio)
            ).tolist()
        self.logger.info('Total samples after balanced resampling: %s' % (len(sampled_infos)))

        cls_infos_new = {name: [] for name in self.class_names}
        for info in sampled_infos:
            for name in set(info['gt_names']):
                if name in self.class_names:
                    cls_infos_new[name].append(info)

        cls_dist_new = {k: len(v) / len(sampled_infos) for k, v in cls_infos_new.items()}

        return sampled_infos

    def get_sweep(self, sweep_info):
        def remove_ego_points(points, center_radius=1.0):
            mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
            return points[mask]

        lidar_path = self.root_path / sweep_info['lidar_path']
        # lidar_path = sweep_info['lidar_path']
        points_sweep = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 4])[:, :4]
        # points_sweep[:, :3] = transform_points(points_sweep[:, :3], _LiDAR_adjust_mat)

        points_sweep = remove_ego_points(points_sweep).T
        if sweep_info['transform_matrix'] is not None:
            num_points = points_sweep.shape[1]
            points_sweep[:3, :] = sweep_info['transform_matrix'].dot(
                np.vstack((points_sweep[:3, :], np.ones(num_points))))[:3, :]

        cur_times = sweep_info['time_lag'] * np.ones((1, points_sweep.shape[1]))
        return points_sweep.T, cur_times.T

    def get_lidar_with_sweeps(self, index, max_sweeps=1):
        info = self.infos[index]
        lidar_path = self.root_path / info['lidar_path']
        # lidar_path = info['lidar_path']
        points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 4])[:, :4]
        if self.contant_reflex:
            points[:, 3] = self.contant_reflex
        # points[:, :3] = transform_points(points[:, :3], _LiDAR_adjust_mat)

        sweep_points_list = [points]
        sweep_times_list = [np.zeros((points.shape[0], 1))]

        for k in np.random.choice(len(info['sweeps']), max_sweeps - 1, replace=False):
            points_sweep, times_sweep = self.get_sweep(info['sweeps'][k])
            if self.contant_reflex:
                points_sweep[:, 3] = self.contant_reflex
            sweep_points_list.append(points_sweep)
            sweep_times_list.append(times_sweep)

        points = np.concatenate(sweep_points_list, axis=0)
        times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

        points = np.concatenate((points, times), axis=1)
        return points

    def get_p2_score(self, info):
        assert self.load_p2_score is not None
        token_key = 'token' if 'token' in info else 'lidar_token'
        lidar_path_token = info['lidar_path'].split('/')[-1].split('.')[0]
        # pp_score_path = osp.join(self.load_p2_score, f"{info['token']}.npy")
        # pp_score_path = osp.join(self.load_p2_score, f"{info[token_key]}.npy")
        pp_score_path = osp.join(self.load_p2_score, f"{lidar_path_token}.npy")
        return np.load(pp_score_path).astype(np.float32)

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs

        return len(self.infos)

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, cam_intrinsic):
        """
        Args:
            pts_rect:
            img_shape:
            cam_intrinsic:
        Returns:

        """
        # pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        pts_2d = np.dot(pts_rect, cam_intrinsic.T)
        pts_img = (pts_2d[:, 0:2].T / pts_2d[:, 2]).T  # (N, 2)
        pts_rect_depth = pts_rect[:, 2]
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[0])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[1])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag

    def get_history_raw(self, info):
        # sample = self.ithaca_365_dataset.get("sample",
        #                                           info['token'])
        # lidar_sd = self.ithaca_365_dataset.get("sample_data",
        #                                        sample['data']['LIDAR_TOP'])
        history_scans = None
        if self.history_cache_dir is not None and \
                osp.exists(osp.join(
                    self.history_cache_dir,
                    osp.basename(info['lidar_path']).split(".")[0]+".pkl")):
            try:
                target_save_path = osp.join(
                    self.history_cache_dir,
                    osp.basename(info['lidar_path']).split(".")[0]+".pkl")
                history_scans = pickle.load(
                    open(target_save_path, "rb"))
            except:
                print("reading error " + target_save_path)
        if history_scans is None:

            # history_scans = self.ithaca_365_dataset.get_other_traversals(
            #     lidar_sd['token'], sorted_by='pos_type',
            #     increasing_order=False, ranges=(0, 70.1),
            #     num_history=5, every_x_meter=5.
            # )
            source_path = osp.join(
                self.load_history.DATA_PATH,
                osp.basename(info['lidar_path']).split(".")[0]+".pkl")
            # print(f"reading {source_path}")
            history_scans = pickle.load(
                open(source_path, "rb"))
            assert len(history_scans.keys()) >= 1
            history_scans = list(history_scans.values())
            if self.load_history.FORWARD_ONLY:
                history_scans = [x[x[:, 0] > 0, :] for x in history_scans]
            if self.history_cache_dir is not None:
                target_save_path = osp.join(
                    self.history_cache_dir,
                    osp.basename(info['lidar_path']).split(".")[0]+".pkl")
                # print(f"saving {target_save_path}")
                pickle.dump(
                    history_scans,
                    open(target_save_path, "wb")
                )
                os.chmod(target_save_path, 0o777)

        if self.training and self.load_history.get("RANDOM_DROPOUT", False):
            num_scans = np.random.randint(1, high=len(history_scans)+1)
            _scan_num_choice = np.random.choice(
                len(history_scans), num_scans, replace=False)
            history_scans = [history_scans[i]
                             for i in _scan_num_choice]

        if self.load_history.LIMIT_NUM > 0 and len(history_scans) > self.load_history.LIMIT_NUM:
            if self.training:
                _scan_num_choice = np.random.choice(
                    len(history_scans), self.load_history.LIMIT_NUM, replace=False)
            else:
                _scan_num_choice = range(self.load_history.LIMIT_NUM)
            history_scans = [history_scans[i]
                                   for i in _scan_num_choice]

        return history_scans

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])
        # print(info['lidar_path'])
        points = self.get_lidar_with_sweeps(index, max_sweeps=self.dataset_cfg.MAX_SWEEPS)
        if self.dataset_cfg.FOV_POINTS_ONLY:
            imsize = info['imsize']
            pts_cam = transform_points(points[:, 0:3], info['ref_to_cam'])  # TODO: check if these paths are absolute or relative
            fov_flag = self.get_fov_flag(pts_cam, imsize, info['cam_intrinsic']) # TODO:  check if these paths are absolute or relative
            points = points[fov_flag]

        token_key = 'token' if 'token' in info else 'lidar_token'
        input_dict = {
            'points': points,
            'frame_id': Path(info['lidar_path']),
            'metadata': {'token': info[token_key]}
            # 'metadata': {'token': info['token']}
        }

        if 'gt_boxes' in info:
            if self.dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', False):
                mask = (info['num_lidar_pts'] > self.dataset_cfg.FILTER_MIN_POINTS_IN_GT - 1)
            else:
                mask = None

            input_dict.update({
                'gt_names': info['gt_names'] if mask is None else info['gt_names'][mask],
                'gt_boxes': info['gt_boxes'] if mask is None else info['gt_boxes'][mask]
            })

            # print
        if self.dataset_cfg.get('LOAD_HISTORY', False):
            input_dict["history_scans"] = self.get_history_raw(
                info)
        if self.dataset_cfg.get('LOAD_P2_SCORE', False):
            p2_scores = self.get_p2_score(
                info)
            if self.dataset_cfg.FOV_POINTS_ONLY:
                p2_scores = p2_scores[fov_flag]
            input_dict["p2_score"] = p2_scores
            assert input_dict["p2_score"].shape[0] == input_dict["points"].shape[0]

        # # load saved pseudo label for unlabeled data FOR ST3D
        # if self.dataset_cfg.get('USE_PSEUDO_LABEL', None) and self.training:
        #     # print("filling up pseudolabel")
        #     self.fill_pseudo_labels(input_dict)
        #     # print(input_dict['gt_classes'])

        data_dict = self.prepare_data(data_dict=input_dict)
        # print(info['lidar_path'])

        if self.dataset_cfg.get('SET_NAN_VELOCITY_TO_ZEROS', False):
            gt_boxes = data_dict['gt_boxes']
            gt_boxes[np.isnan(gt_boxes)] = 0
            data_dict['gt_boxes'] = gt_boxes

        if not self.dataset_cfg.PRED_VELOCITY and 'gt_boxes' in data_dict:
            data_dict['gt_boxes'] = data_dict['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6, -1]]
        # print(info['lidar_path'])
        return data_dict

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:
        Returns:
        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, 7]), 'pred_labels': np.zeros(num_samples)
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes
            pred_dict['pred_labels'] = pred_labels

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            single_pred_dict['metadata'] = batch_dict['metadata'][index]
            annos.append(single_pred_dict)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'token' not in det_annos[0]:
            return "Skipping Evaluation.\n", {}
        
        import json
        from ithaca365.ithaca365 import Ithaca365
        from . import ithaca365_utils
        nusc = Ithaca365(version=self.dataset_cfg.VERSION, dataroot=str(self.root_path), verbose=True)

        # *************** hack ****************
        # Set class name to be 'Dynamic' for all predictions
        eval_det_annos = copy.deepcopy(det_annos)
        for anno in eval_det_annos:
            anno.update({'name': np.array(["dynamic"] * anno['name'].shape[0])})
        # *************** hack ****************

        nusc_annos = ithaca365_utils.transform_det_annos_to_nusc_annos(eval_det_annos, nusc)
        nusc_annos['meta'] = {
            'use_camera': False,
            'use_lidar': True,
            'use_radar': False,
            'use_map': False,
            'use_external': False,
        }

        output_path = Path(kwargs['output_path'])
        output_path.mkdir(exist_ok=True, parents=True)
        res_path = str(output_path / 'results_nusc.json')
        with open(res_path, 'w') as f:
            json.dump(nusc_annos, f)

        self.logger.info(f'The predictions of NuScenes have been saved to {res_path}')

        if self.dataset_cfg.VERSION == 'v1.0-test':
            return 'No ground-truth annotations for evaluation', {}

        from ithaca365.eval.detection.config import config_factory
        from ithaca365.eval.detection.evaluate import NuScenesEval

        eval_set_map = {
            # 'v1.0-mini': 'mini_val',
            # 'v1.0-trainval': 'val',
            # 'v1.0-test': 'test'
            'v1.1': 'val',
            'v1.1-train': 'train'
        }
        eval_version = 'detection_by_range'
        eval_config = config_factory(eval_version)
        # try:
        #     eval_version = 'detection_cvpr_2019'
        #     eval_config = config_factory(eval_version)
        # except:
        #     eval_version = 'cvpr_2019'
        #     eval_config = config_factory(eval_version)

        nusc_eval = NuScenesEval(
            nusc,
            config=eval_config,
            result_path=res_path,
            # eval_set=eval_set_map[self.dataset_cfg.VERSION],
            eval_set=self.dataset_cfg.DATA_SPLIT.test,
            output_dir=str(output_path),
            verbose=True,
        )
        metrics_summary = nusc_eval.main(plot_examples=0, render_curves=False)

        with open(output_path / 'metrics_summary.json', 'r') as f:
            metrics = json.load(f)

        result_str, result_dict = ithaca365_utils.format_nuscene_results(metrics, self.class_names, version=eval_version)
        return result_str, result_dict

    def create_groundtruth_database(self, used_classes=None, max_sweeps=10):
        import torch

        database_save_path = self.root_path / f'gt_database_{max_sweeps}sweeps_withvelo'
        db_info_save_path = self.root_path / f'ithaca365_dbinfos_{max_sweeps}sweeps_withvelo.pkl'

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        for idx in tqdm(range(len(self.infos))):
            sample_idx = idx
            info = self.infos[idx]
            points = self.get_lidar_with_sweeps(idx, max_sweeps=max_sweeps)
            gt_boxes = info['gt_boxes']
            gt_names = info['gt_names']

            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
                torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
            ).long().squeeze(dim=0).cpu().numpy()

            for i in range(gt_boxes.shape[0]):
                filename = '%s_%s_%d.bin' % (sample_idx, gt_names[i], i)
                filepath = database_save_path / filename
                gt_points = points[box_idxs_of_pts == i]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or gt_names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': gt_names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}
                    if gt_names[i] in all_db_infos:
                        all_db_infos[gt_names[i]].append(db_info)
                    else:
                        all_db_infos[gt_names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    def update_groundtruth_database(self, source_info_path=None, det_info_path=None,
                                    used_classes=None, split='train', max_sweeps=1, use_gt_aug=False):

        # database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        # db_info_save_path = Path(self.root_path) / ('ithaca365_dbinfos_%s.pkl' % split)  # TODO: check if this is correct
        database_save_path = self.root_path / f'gt_database_{max_sweeps}sweeps_withvelo'
        db_info_save_path = self.root_path / f'ithaca365_dbinfos_{max_sweeps}sweeps_withvelo.pkl'

        if use_gt_aug:
            database_save_path.mkdir(parents=True, exist_ok=True)
            all_db_infos = {}

        with open(source_info_path, 'rb') as f:
            infos = pickle.load(f)

        # Set our current infos to be the new infos
        self.infos = infos

        with open(det_info_path, 'rb') as f:
            det_infos = pickle.load(f)

        # assert len(det_infos) == len(infos)
        for idx in range(len(infos)):
            sample_idx = idx
            info = infos[idx]
            points = self.get_lidar_with_sweeps(idx, max_sweeps=max_sweeps)
            # annotations = det_infos[idx]
            annotations = det_infos[idx] if isinstance(det_infos, list) else det_infos[info['lidar_token']]

            # assert ('token' in info and annotations['metadata']['token'] == info['token']) or annotations['metadata']['token'] == info['lidar_token']
            # del annos['frame_id']

            gt_names = annotations['name']
            gt_boxes = annotations['boxes_lidar']
            if gt_boxes.shape[0] == 0:  # HACK: to parse empty gt_boxes (remove this when the bug is fixed)
                gt_boxes = np.empty((0, 7), dtype=np.float32)

            velocity = np.empty((gt_boxes.shape[0],3))  # no velocity for ith365
            velocity[:] = np.NaN

            gt_boxes = np.concatenate([gt_boxes, velocity[:, :2]], axis=-1)
            info['gt_boxes'] = gt_boxes
            info['gt_boxes_velocity'] = velocity
            info['gt_names'] = np.array([map_name_from_general_to_detection[name] for name in gt_names])
            info['gt_boxes_token'] = np.array([np.nan for _ in gt_boxes])  # no tokens for detected GT

            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
                torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
            ).long().squeeze(dim=0).cpu().numpy()

            num_lidar_pts = []
            for i in range(gt_boxes.shape[0]):
                filename = '%s_%s_%d.bin' % (sample_idx, gt_names[i], i)
                filepath = database_save_path / filename
                gt_points = points[box_idxs_of_pts == i]
                num_lidar_pts.append(gt_points.shape[0])

                if use_gt_aug:
                    gt_points[:, :3] -= gt_boxes[i, :3]
                    with open(filepath, 'w') as f:
                        gt_points.tofile(f)

                    if (used_classes is None) or gt_names[i] in used_classes:
                        db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                        db_info = {'name': gt_names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                                'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}
                        if gt_names[i] in all_db_infos:
                            all_db_infos[gt_names[i]].append(db_info)
                        else:
                            all_db_infos[gt_names[i]] = [db_info]
            
            info['num_lidar_pts'] = np.array(num_lidar_pts)
            info['num_radar_pts'] = np.array([0 for _ in gt_boxes])  # no radar data

        if use_gt_aug:
            for k, v in all_db_infos.items():
                print('Database %s: %d' % (k, len(v)))

            with open(db_info_save_path, 'wb') as f:
                pickle.dump(all_db_infos, f)

        with open(Path(self.root_path) / osp.basename(source_info_path), 'wb') as f:
            pickle.dump(infos, f)


def create_modest_ithaca365_infos(version, data_path, save_path, max_sweeps=10):
    from ithaca365.ithaca365 import Ithaca365
    from ithaca365.utils import splits
    from . import ithaca365_utils
    # from ithaca365.eval.detection.config import config_factory
    # eval_version = 'detection_by_range'
    # eval_config = config_factory(eval_version)
    data_path = data_path / version
    save_path = save_path / version

    train_scenes = splits.train

    nusc = Ithaca365(version=version, dataroot=data_path, verbose=True)
    available_scenes = ithaca365_utils.get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    # only get the train scenes
    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    train_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in train_scenes])

    print('%s: train scene(%d)' % (version, len(train_scenes)))

    train_nusc_infos = ithaca365_utils.fill_modest_train_infos(
        data_path=data_path, nusc=nusc, train_scenes=train_scenes, max_sweeps=max_sweeps,
    )

    # with open(seed_det_path, 'rb') as f:
    #         seed_infos = pickle.load(f)

    # assert len(seed_infos) == len(train_nusc_infos)
    # for idx in range(len(train_nusc_infos)):
        # info = train_nusc_infos[idx]
        # points = nusc.get_lidar_with_sweeps(idx, max_sweeps=max_sweeps)
        # annotations = seed_infos[idx]
        # assert annotations['metadata']['token'] == info['token']  # NOTE: THIS IS THE SAMPLE_DATA TOKEN, NOT THE SAMPLE TOKEN

        # gt_names = annotations['name']
        # gt_boxes = annotations['boxes_lidar']

        # velocity = np.empty((gt_boxes.shape[0],3))  # no velocity for ith365
        # velocity[:] = np.NaN

        # gt_boxes = np.concatenate([gt_boxes, velocity[:, :2]], axis=-1)
        # info['gt_boxes'] = gt_boxes
        # info['gt_boxes_velocity'] = velocity
        # info['gt_names'] = np.array([map_name_from_general_to_detection[name] for name in gt_names])
        # info['gt_boxes_token'] = np.array([np.nan for _ in gt_boxes])  # no tokens for detected GT

        # box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
        #     torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
        #     torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
        # ).long().squeeze(dim=0).cpu().numpy()

        # num_lidar_pts = []
        # for i in range(gt_boxes.shape[0]):
        #     gt_points = points[box_idxs_of_pts == i]
        #     num_lidar_pts.append(gt_points.shape[0])

        # info['num_lidar_pts'] = np.array(num_lidar_pts)
        # info['num_radar_pts'] = np.array([0 for _ in gt_boxes])  # no radar data

    print('train sample: %d' % (len(train_nusc_infos)))
    info_save_path = save_path / f'ithaca365_modest_infos_{max_sweeps}sweeps_train.pkl'
    with open(info_save_path, 'wb') as f:
        pickle.dump(train_nusc_infos, f)
    return info_save_path


def create_ithaca365_info(version, data_path, save_path, max_sweeps=10,):
    from ithaca365.ithaca365 import Ithaca365
    from ithaca365.utils import splits
    from . import ithaca365_utils
    from ithaca365.eval.detection.config import config_factory
    eval_version = 'detection_by_range'
    eval_config = config_factory(eval_version)
    data_path = data_path / version
    save_path = save_path / version

    train_scenes = splits.train
    val_scenes = splits.val

    # assert version in ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    # if version == 'v1.0-trainval':
    #     train_scenes = splits.train
    #     val_scenes = splits.val
    # elif version == 'v1.0-test':
    #     train_scenes = splits.test
    #     val_scenes = []
    # elif version == 'v1.0-mini':
    #     train_scenes = splits.mini_train
    #     val_scenes = splits.mini_val
    # else:
    #     raise NotImplementedError

    nusc = Ithaca365(version=version, dataroot=data_path, verbose=True)
    available_scenes = ithaca365_utils.get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in train_scenes])
    val_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in val_scenes])

    print('%s: train scene(%d), val scene(%d)' % (version, len(train_scenes), len(val_scenes)))

    train_nusc_infos, val_nusc_infos = ithaca365_utils.fill_trainval_infos(
        data_path=data_path, nusc=nusc, train_scenes=train_scenes, val_scenes=val_scenes,
        test='test' in version, max_sweeps=max_sweeps,
        only_accurate_localization=eval_config.only_accurate_localization
    )

    # if version == 'v1.0-test':
    #     print('test sample: %d' % len(train_nusc_infos))
    #     with open(save_path / f'ithaca365_infos_{max_sweeps}sweeps_test.pkl', 'wb') as f:
    #         pickle.dump(train_nusc_infos, f)
    # else:
    print('train sample: %d, val sample: %d' % (len(train_nusc_infos), len(val_nusc_infos)))
    with open(save_path / f'ithaca365_infos_{max_sweeps}sweeps_train.pkl', 'wb') as f:
        pickle.dump(train_nusc_infos, f)
    with open(save_path / f'ithaca365_infos_{max_sweeps}sweeps_val.pkl', 'wb') as f:
        pickle.dump(val_nusc_infos, f)


if __name__ == '__main__':
    import yaml
    import argparse
    from pathlib import Path
    from easydict import EasyDict

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
    parser.add_argument('--func', type=str, default='create_ithaca365_infos', help='')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--version', type=str, default='v1.1', help='')
    parser.add_argument('--pseudo_labels', type=str, default=None, help='Path to pseudo label pickle')
    parser.add_argument('--info_path', type=str, default=None)
    args = parser.parse_args()

    dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
    ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
    data_path = ROOT_DIR / Path(args.data_path) if args.data_path is not None \
        else ROOT_DIR / 'data' / 'ithaca365'
    dataset_cfg.VERSION = args.version

    if args.func == 'create_ithaca365_infos':
        # dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
        # ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        # dataset_cfg.VERSION = args.version
        create_ithaca365_info(
            version=dataset_cfg.VERSION,
            data_path=data_path,
            # data_path=ROOT_DIR / 'data' / 'ithaca365',
            # data_path="/",
            save_path=data_path,
            # save_path=ROOT_DIR / 'data' / 'ithaca365',
            max_sweeps=dataset_cfg.MAX_SWEEPS
        )

        ithaca365_dataset = Ithaca365Dataset(
            dataset_cfg=dataset_cfg, class_names=None,
            root_path=data_path,
            # root_path=ROOT_DIR / 'data' / 'ithaca365',
            logger=common_utils.create_logger(), training=True
        )
        ithaca365_dataset.create_groundtruth_database(max_sweeps=dataset_cfg.MAX_SWEEPS)
    
    elif args.func == 'create_modest_ithaca365_infos':
        # assert args.pseudo_labels is not None, "Must have pseudo labels (seed labels), cannot be None."
        # dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
        # ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        # dataset_cfg.VERSION = args.version
        info_save_path = create_modest_ithaca365_infos(
            version=dataset_cfg.VERSION,
            data_path=data_path,
            save_path=data_path,
            max_sweeps=dataset_cfg.MAX_SWEEPS,
            # seed_det_path=args.pseudo_labels
        )

        ithaca365_dataset = Ithaca365Dataset(
            dataset_cfg=dataset_cfg, class_names=None,
            root_path=data_path,
            # root_path=ROOT_DIR / 'data' / 'ithaca365',
            logger=common_utils.create_logger(), training=True
        )
        if args.pseudo_labels is not None:
            # Needed for training MODEST only!
            ithaca365_dataset.update_groundtruth_database(
                source_info_path=info_save_path,
                det_info_path=args.pseudo_labels,
                split='train')
        # ithaca365_dataset.create_groundtruth_database(max_sweeps=dataset_cfg.MAX_SWEEPS)

    elif args.func == 'update_groundtruth_database':
        assert args.pseudo_labels is not None, "Must have pseudo labels, cannot be None."
        assert args.info_path is not None, "Must have source info path, cannot be None."

        ithaca365_dataset = Ithaca365Dataset(
            dataset_cfg=dataset_cfg, class_names=None,
            root_path=data_path,
            # root_path=ROOT_DIR / 'data' / 'ithaca365',
            logger=common_utils.create_logger(), training=True
        )
        ithaca365_dataset.update_groundtruth_database(
            source_info_path=args.info_path,
            det_info_path=args.pseudo_labels,
            split='train')
    
    else:
        raise NotImplementedError("Not a valid function to generate Ithaca365 infos.")
