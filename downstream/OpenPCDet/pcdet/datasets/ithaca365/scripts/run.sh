#!/bin/bash
#SBATCH -N 1                              # Total number of CPU nodes requested
#SBATCH --cpus-per-task 4
#SBATCH --mem=16G                         # CPU Memory pool for all cores
#SBATCH --partition=default_partition --gres=gpu:0
#SBATCH -o /home/yy785/projects/adaptation/downstream/OpenPCDet/pcdet/datasets/ithaca365/outputs/slurm/%x_%j_o.txt
#SBATCH -e /home/yy785/projects/adaptation/downstream/OpenPCDet/pcdet/datasets/ithaca365/outputs/slurm/%x_%j_e.txt
#SBATCH --requeue
#SBATCH -t 1-00:00:00

set -e
. /home/yy785/anaconda3/etc/profile.d/conda.sh
conda activate adaptation
set -x

cd /home/yy785/projects/adaptation/downstream/OpenPCDet/pcdet/datasets/ithaca365
${@}