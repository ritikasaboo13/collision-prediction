#!/bin/bash

## change the last two digits to your team id
#SBATCH --account=csci_ga_2572_001-2023fa-27

## change the partition number to use different number of GPUs
##SBATCH --partition=n1s8-v100-1
##SBATCH --gres=gpu:1
##SBATCH --cpus-per-task=8

##SBATCH --partition=n1s16-v100-2
##SBATCH --gres=gpu:2
##SBATCH --cpus-per-task=16

#SBATCH --partition=n1c24m128-v100-4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24

#SBATCH --time=12:00:00
#SBATCH --output=demo_%j.out
#SBATCH --error=demo_%j.err
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=rss9311@nyu.edu
#SBATCH --job-name=openstl

mkdir /tmp/$USER
export SINGULARITY_CACHEDIR=/tmp/$USER

singularity exec --nv \
--bind /scratch/rss9311 \
--overlay ../overlay-25GB-500K-copy.ext3:ro \
../cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif \
/bin/bash -c \
"source /ext3/env.sh;
export PYTHONPATH=\${PYTHONPATH}:/scratch/rss9311/pytorch-env-dir/simvpOpenStl; 
bash tools/dist_train.sh \
/scratch/rss9311/pytorch-env-dir/simvpOpenStl/configs/movingphy/simvp/SimVP_gSTA_Seg.py \
4 \
--dataname 'movingphy_segment' \
--data_root /scratch/rss9311/squashfs-root/dataset \
-b 4 \
-vb 4 \
--epoch 20 \
--ex_name exp_segment_epochs_20"
