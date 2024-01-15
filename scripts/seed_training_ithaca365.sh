#!/bin/bash
set -e

formatstring="dynamic_%s_round_0"
base_dataset="ithaca365"
ithaca365_version="v1.1"
model="pointrcnn_dynamic_obj"

while getopts "F:b:m:" opt
do
    case $opt in
        F) formatstring=$OPTARG ;;
        b) base_dataset=$OPTARG ;;
        m) model=$OPTARG ;;
        *)
            echo "there is unrecognized parameter."
            exit 1
            ;;
    esac
done

set -x
proj_root_dir=$(pwd)

iter_name=$(printf $formatstring 0)
if [ ! -d "${proj_root_dir}/downstream/OpenPCDet/data/ithaca_${iter_name}" ]
then
    echo "=> Generating ${iter_name} dataset"
    cd ${proj_root_dir}/downstream/OpenPCDet/data
    mkdir ithaca_${iter_name}
    mkdir ithaca_${iter_name}/${ithaca365_version}
    ln -s /share/campbell/Skynet/nuScene_format/v1.1/data ./ithaca_${iter_name}/${ithaca365_version}/
    ln -s /share/campbell/Skynet/nuScene_format/v1.1/v1.1 ./ithaca_${iter_name}/${ithaca365_version}/
    cp ${proj_root_dir}/downstream/OpenPCDet/data/${base_dataset}/${ithaca365_version}/ithaca365_infos_1sweeps_val.pkl ./ithaca_${iter_name}/${ithaca365_version}/.
fi

# run data pre-processing
if [ ! -f "${proj_root_dir}/downstream/OpenPCDet/data/ithaca_${iter_name}/.finish_tkn" ]
then
    echo "=> pre-processing ${iter_name} dataset"
    cd ${proj_root_dir}/downstream/OpenPCDet
    python -m pcdet.datasets.ithaca365.ithaca365_dataset --func create_modest_ithaca365_infos \
        --cfg_file tools/cfgs/dataset_configs/ithaca365_dataset_dynamic_obj.yaml \
        --data_path data/ithaca_${iter_name} \
        --pseudo_labels /home/kzl6/modest_pp/generate_cluster_mask/intermediate_results/ithaca365_bbox_pp_score_fw70_5m_20hist_02filtering_knn.pkl 
    touch ${proj_root_dir}/downstream/OpenPCDet/data/ithaca_${iter_name}/.finish_tkn
fi

cd ${proj_root_dir}/downstream/OpenPCDet/tools

# start training
if [ ! -f "${proj_root_dir}/downstream/OpenPCDet/output/ithaca365_models/${model}/${iter_name}/ckpt/last_checkpoint.pth" ]
then
    echo "=> ${iter_name} training"
    bash scripts/dist_train.sh 4 --cfg_file cfgs/ithaca365_models/${model}.yaml \
        --wandb_project modest_pp_ithaca \
        --extra_tag ${iter_name} --merge_all_iters_to_one_epoch \
        --fix_random_seed \
        --set DATA_CONFIG.DATA_PATH ../data/ithaca_${iter_name}
else
    echo "=> Skipping training ${iter_name} model; checkpoint already found."
fi

# generate the preditions on the training set
bash scripts/dist_test.sh 4 --cfg_file cfgs/ithaca365_models/${model}.yaml \
    --extra_tag ${iter_name} --eval_tag trainset_0 \
    --ckpt ../output/ithaca365_models/${model}/${iter_name}/ckpt/last_checkpoint.pth \
    --set DATA_CONFIG.DATA_PATH ../data/ithaca_${iter_name} \
    DATA_CONFIG.DATA_SPLIT.test train DATA_CONFIG.INFO_PATH.test ithaca365_modest_infos_1sweeps_train.pkl
