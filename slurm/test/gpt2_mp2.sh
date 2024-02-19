#!/bin/bash
#SBATCH --job-name=mathlm
#SBATCH --partition=a40x
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8          # Crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=10          # Number of cores per tasks
#SBATCH --gres=gpu:8                 # Number of gpus
#SBATCH --output=gpt2_mp2_test_%j.out      # Set this dir where you want slurm outs to go
#SBATCH --error=gpt2_mp2_test_%j.out      # Set this dir where you want slurm outs to go
#SBATCH --exclusive      # Turn off node sharing
#SBATCH --account=neox
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --exclusive

source /admin/home-zhangir.azerbayev/miniconda3/bin/activate jaxdev

echo $CONDA_HOME

LEVANTER_HOME=/weka/proj-llemma-instruct/levanter
cd $LEVANTER_HOME

srun python -m levanter.main.train_lm \
    --config ${LEVANTER_HOME}/config/gpt2_small_fast_mp2.yaml 
