#!/bin/bash
#SBATCH -N 1                              # Total number of CPU nodes requested
#SBATCH --cpus-per-task 4
#SBATCH --mem=256G                         # CPU Memory pool for all cores
#SBATCH --partition=default_partition --gres=gpu:0
#SBATCH -o /home/kzl6/modest_pp/generate_cluster_mask/outputs/slurm/%x_%j_o.txt
#SBATCH -e /home/kzl6/modest_pp/generate_cluster_mask/outputs/slurm/%x_%j_e.txt
#SBATCH --requeue
#SBATCH -t 1-00:00:00

set -e
. /home/kzl6/anaconda3/etc/profile.d/conda.sh
conda activate modestpp
set -x

cd /home/kzl6/modest_pp/generate_cluster_mask
${@}