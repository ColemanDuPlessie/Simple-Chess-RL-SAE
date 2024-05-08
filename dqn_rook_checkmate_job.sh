#!/bin/bash
#SBATCH --job-name=train_rook_checkmate_DQN # name
#SBATCH --nodes=1                    # nodes
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=64           # number of cores per tasks
#SBATCH --gres=gpu:4                 # number of gpus
#SBATCH --time 4:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=delta_logs/train_rook_checkmate_DQN.out           # output file name
#SBATCH --partition=gpuA100x4
#SBATCH --mem=208G
#SBATCH --constraint="scratch"
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --account=bbry-delta-gpu
#SBATCH --no-requeue

module reset # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)
             # $WORK and $SCRATCH are now set
module list  # job documentation and metadata

echo "job is starting on `hostname`"

module load python/3.11.6 # TODO is this right?
module load anaconda3_gpu
conda init # TODO is this necessary?
conda activate gymrl

export TORCH_SHOW_CPP_STACKTRACE=1

export OMP_NUM_THREADS=1  # if code is not multithreaded, otherwise set to 8 or 16
# srun -N 1 -n 4 ./a.out > myjob.out
# py-torch example, --ntasks-per-node=1 --cpus-per-task=64
source /u/cduplessie/huggingface-env/bin/activatels

export PYTHONPATH=$PYTHONPATH:/u/cduplessie/Simple-Chess-RL-SAE/

pip3 list

CUDA_VISIBLE_DEVICES=0,1,2,3 srun python3.6 train_dqn.py
