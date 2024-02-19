#!/bin/bash
#SBATCH --job-name=mathlm
#SBATCH --partition=a40x
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # Crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=10          # Number of cores per tasks
#SBATCH --mem-per-cpu=12g
#SBATCH --gres=gpu:1                 # Number of gpus
#SBATCH --output=llama_lora_%j.out      # Set this dir where you want slurm outs to go
#SBATCH --error=llama_lora_%j.out      # Set this dir where you want slurm outs to go
#SBATCH --exclusive      # Turn off node sharing
#SBATCH --account=neox
#SBATCH --open-mode=append
#SBATCH --requeue

source /admin/home-zhangir.azerbayev/miniconda3/bin/activate jaxdev

echo $CONDA_HOME

HF_HOME="/weka/proj-llemma-instruct/.cache"

LEVANTER_HOME=/weka/proj-llemma-instruct/levanter
cd $LEVANTER_HOME

srun python ${LEVANTER_HOME}/examples/gsm8k-lora/gsm8k_lora.py \
--config_path ${LEVANTER_HOME}/examples/gsm8k-lora/gsm8k-llama2.yaml \
--trainer.checkpointer.base_path ${LEVANTER_HOME}/checkpoints \
--hf_save_path ${LEVANTER_HOME}/hf-checkpoints \
--merged_hf_save_path ${LEVANTER_HOME}/merged-checkpoints \
--trainer.per_device_parallelism 2
