#!/bin/bash

#SBATCH --job-name=mri2mra-train
#SBATCH --mail-type=all
#SBATCH --mail-user=asagil@uw.edu
#SBATCH --requeue

#SBATCH --account=escience
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --mem=80G
#SBATCH --gpus=1
#SBATCH --time=05-01:00:00 # Max runtime in DD-HH:MM:SS format.


#SBATCH --chdir=/gscratch/escience/asagil/pytorch/t1-mra
#SBATCH --export=all
#SBATCH --output=mri2mratrain_%j.out # where STDOUT goes, %x is the job name, %j is the job ID
#SBATCH --error=mri2mratrain_%j.err # where STDERR goes, %x is the job name, %j is the job ID


# Your programs to run.
/gscratch/escience/asagil/miniconda3/envs/mri2mra/bin/python ./src/train.py --num_workers 1 --data_dir "/gscratch/escience/asagil/pytorch/data/split"
