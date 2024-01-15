#!/bin/bash
set -e

# formatstring="modest_adapt_round_%s"
formatstring="dynamic_%s_round_0"
# seed_label="/home/kzl6/adaptation/downstream/OpenPCDet/output/ithaca365_models/pointrcnn_eval/default/eval/epoch_no_number/train/eval_train/result.pkl"
seed_label=""
seed_ckpt=""
# seed_ckpt="/home/yy785/data_share/adaptation_ckpt/prcnn_kitti_pretrained_xyz.pth"
max_iter=10
no_filtering=false
filtering_arg=""
base_dataset="ithaca365"
save_ckpt_every=1
# model="pointrcnn_dense_point_small_lr_new_filter"
# model="pointrcnn_dense_point_load_p2_1ep_v4"
model="pointrcnn_dynamic_obj"
ithaca365_version="v1.1"
num_gpu=4

while getopts "M:F:b:f:m:s:aS:e:" opt
do
    case $opt in
        M) max_iter=$OPTARG ;;
        a) no_filtering=true ;;
        F) formatstring=$OPTARG ;;
        b) base_dataset=$OPTARG ;;
        f) filtering_arg=$OPTARG ;;
        m) model=$OPTARG ;;
        s) seed_label=$OPTARG ;;
        S) seed_ckpt=$OPTARG ;;
        e) save_ckpt_every=$OPTARG ;;
        *)
            echo "there is unrecognized parameter."
            exit 1
            ;;
    esac
done

set -x
proj_root_dir=$(pwd)

function generate_pl () {
    local result_path=${1}
    local target_path=${2}
    if [ ! -f ${target_path} ]; then
        if [ "$no_filtering" = true ]; then
            echo "Skipping filtering"
            if [ -L "${target_path}" ]; then
                rm ${target_path}
            fi
            ln -s ${result_path} ${target_path}
        else
            python ${proj_root_dir}/generate_cluster_mask/p2_score_filtering_lidar_consistency.py result_path=${result_path} save_path=${target_path} dataset="ithaca365" data_paths="ithaca365.yaml" ${filtering_arg} ${3}
        fi
        # touch ${target_path}/.finish_tkn
    else
        echo "=> Skipping generated ${target_path}"
    fi
}


if [ ! -d "${proj_root_dir}/generate_cluster_mask/intermediate_results" ]; then
    mkdir ${proj_root_dir}/generate_cluster_mask/intermediate_results/
fi

for ((i = 1 ; i <= ${max_iter} ; i++)); do
    iter_name=$(printf $formatstring ${i})
    pre_iter_name=$(printf $formatstring $((i-1)))

    # check if the iteration has been finished
    if [ -f "${proj_root_dir}/downstream/OpenPCDet/output/ithaca365_models/${model}/${iter_name}/eval/epoch_no_number/train/trainset_0/result.pkl" ]; then
        echo "${iter_name} has finished!"
        continue
    fi

    # filter with p2 score
    if [ ! -f "${proj_root_dir}/generate_cluster_mask/intermediate_results/pl_for_${iter_name}.pkl" ]
    then
        echo "=> Filtering ${iter_name} pseudo-labels"
        cd ${proj_root_dir}/generate_cluster_mask
        python ${proj_root_dir}/generate_cluster_mask/p2_score_filtering_lidar_consistency.py result_path=${proj_root_dir}/downstream/OpenPCDet/output/ithaca365_models/${model}/${pre_iter_name}/eval/epoch_no_number/train/trainset_0/result.pkl save_path=${proj_root_dir}/generate_cluster_mask/intermediate_results/pl_for_${iter_name}.pkl dataset="ithaca365" data_paths="ithaca365.yaml"
    else
        echo "=> Skipping filtering ${iter_name} pseudo-labels"
    fi

    # create the dataset
    if [ ! -d "${proj_root_dir}/downstream/OpenPCDet/data/ithaca_${iter_name}" ]
    then
        echo "=> Generating ${iter_name} dataset"
        cd ${proj_root_dir}/downstream/OpenPCDet/data
        mkdir ithaca_${iter_name}
        mkdir ithaca_${iter_name}/${ithaca365_version}
        ln -s /share/campbell/Skynet/nuScene_format/v1.1/data ./ithaca_${iter_name}/${ithaca365_version}/
        ln -s /share/campbell/Skynet/nuScene_format/v1.1/v1.1 ./ithaca_${iter_name}/${ithaca365_version}/
        cp ${proj_root_dir}/downstream/OpenPCDet/data/${base_dataset}/${ithaca365_version}/ithaca365_infos_1sweeps_val.pkl ./ithaca_${iter_name}/${ithaca365_version}/.
        cd ./ithaca_${iter_name}/${ithaca365_version}
    fi

    # run data pre-processing
    if [ ! -f "${proj_root_dir}/downstream/OpenPCDet/data/ithaca_${iter_name}/.finish_tkn" ]
    then
        echo "=> pre-processing ${iter_name} dataset"
        cd ${proj_root_dir}/downstream/OpenPCDet
        python -m pcdet.datasets.ithaca365.ithaca365_dataset --func update_groundtruth_database \
            --cfg_file tools/cfgs/dataset_configs/ithaca365_dataset_dynamic_obj.yaml \
            --data_path data/ithaca_${iter_name} \
            --pseudo_labels ${proj_root_dir}/generate_cluster_mask/intermediate_results/pl_for_${iter_name}.pkl \
            --info_path ${proj_root_dir}/downstream/OpenPCDet/data/ithaca365/${ithaca365_version}/ithaca365_modest_infos_1sweeps_train.pkl
        touch ${proj_root_dir}/downstream/OpenPCDet/data/ithaca_${iter_name}/.finish_tkn
    fi

    # start training
    cd ${proj_root_dir}/downstream/OpenPCDet/tools
    if [ ! -f "${proj_root_dir}/downstream/OpenPCDet/output/ithaca365_models/${model}/${iter_name}/ckpt/last_checkpoint.pth" ]
    then
        echo "=> ${iter_name} training"
        bash scripts/dist_train.sh ${num_gpu} --cfg_file cfgs/ithaca365_models/${model}.yaml \
            --wandb_project modest_pp_ithaca \
            --extra_tag ${iter_name} --merge_all_iters_to_one_epoch \
            --fix_random_seed \
            --set DATA_CONFIG.DATA_PATH ../data/ithaca_${iter_name}
    else
        echo "=> Skipping training ${iter_name} model; checkpoint already found."
    fi

    # generate the preditions on the training set
    if [ ! -f "${proj_root_dir}/downstream/OpenPCDet/output/ithaca365_models/${model}/${iter_name}/eval/epoch_no_number/train/trainset_0/result.pkl" ]
    then
        bash scripts/dist_test.sh ${num_gpu} --cfg_file cfgs/ithaca365_models/${model}.yaml \
            --extra_tag ${iter_name} --eval_tag trainset_0 \
            --ckpt ../output/ithaca365_models/${model}/${iter_name}/ckpt/last_checkpoint.pth \
            --set DATA_CONFIG.DATA_PATH ../data/ithaca_${iter_name} \
            DATA_CONFIG.DATA_SPLIT.test train DATA_CONFIG.INFO_PATH.test ithaca365_modest_infos_1sweeps_train.pkl
    else
        echo "=> Skipping eval on train"
    fi

done
