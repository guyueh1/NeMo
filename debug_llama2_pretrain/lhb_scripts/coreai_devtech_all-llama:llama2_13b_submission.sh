#!/bin/bash

# Parameters
#SBATCH --account=coreai_devtech_all
#SBATCH --dependency=singleton
#SBATCH --error=/lustre/fsw/devtech/hpc-devtech/hongbinl/nemo_megatron/scripts/support_llama/NeMo-Megatron-Launcher/launcher_scripts/results/llama2_13b/log-coreai_devtech_all-llama:llama2_13b_%j.err
#SBATCH --exclusive
#SBATCH --job-name=coreai_devtech_all-llama:llama2_13b
#SBATCH --mem=0
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --output=/lustre/fsw/devtech/hpc-devtech/hongbinl/nemo_megatron/scripts/support_llama/NeMo-Megatron-Launcher/launcher_scripts/results/llama2_13b/log-coreai_devtech_all-llama:llama2_13b_%j.out
#SBATCH --partition=luna
#SBATCH --time=0-01:00:00

# setup
export TRANSFORMERS_OFFLINE=1
export NCCL_AVOID_RECORD_STREAMS=1

# command 1
srun --output /lustre/fsw/devtech/hpc-devtech/hongbinl/nemo_megatron/scripts/support_llama/NeMo-Megatron-Launcher/launcher_scripts/results/llama2_13b/log-coreai_devtech_all-llama:llama2_13b_%j.out --error /lustre/fsw/devtech/hpc-devtech/hongbinl/nemo_megatron/scripts/support_llama/NeMo-Megatron-Launcher/launcher_scripts/results/llama2_13b/log-coreai_devtech_all-llama:llama2_13b_%j.err --container-image gitlab-master.nvidia.com/hongbinl/external_images:nemofw-23.07-ptl-2.0 --container-mounts /lustre/fsw/devtech/hpc-devtech/hongbinl/nemo_megatron/scripts/support_llama/NeMo-Megatron-Launcher/launcher_scripts:/lustre/fsw/devtech/hpc-devtech/hongbinl/nemo_megatron/scripts/support_llama/NeMo-Megatron-Launcher/launcher_scripts,/lustre/fsw/devtech/hpc-devtech/hongbinl/nemo_megatron/scripts/support_llama/NeMo-Megatron-Launcher/launcher_scripts/data:/lustre/fsw/devtech/hpc-devtech/hongbinl/nemo_megatron/scripts/support_llama/NeMo-Megatron-Launcher/launcher_scripts/data,/lustre/fsw/devtech/hpc-devtech/hongbinl/nemo_megatron/scripts/support_llama/NeMo-Megatron-Launcher/launcher_scripts/results:/lustre/fsw/devtech/hpc-devtech/hongbinl/nemo_megatron/scripts/support_llama/NeMo-Megatron-Launcher/launcher_scripts/results/lustre/fsw/devtech/hpc-devtech/hongbinl/nemo_megatron/nemo_repo/internal/NeMo:/lustre/fsw/devtech/hpc-devtech/hongbinl/nemo_megatron/nemo_repo/internal/NeMo,/lustre/fsw/devtech/hpc-devtech/hongbinl/nemo_megatron:/lustre/fsw/devtech/hpc-devtech/hongbinl/nemo_megatron --no-container-mount-home bash -c "
  wandb login 55b1e23693852385f4eae5b38c3c2bbcda88b585;
  cd /lustre/fsw/devtech/hpc-devtech/hongbinl/nemo_megatron/nemo_repo/internal/NeMo;
  git rev-parse HEAD;
  export PYTHONPATH=/lustre/fsw/devtech/hpc-devtech/hongbinl/nemo_megatron/Megatron-LM:/lustre/fsw/devtech/hpc-devtech/hongbinl/nemo_megatron/nemo_repo/TransformerEngine:/lustre/fsw/devtech/hpc-devtech/hongbinl/nemo_megatron/apex:/lustre/fsw/devtech/hpc-devtech/hongbinl/nemo_megatron/nemo_repo/internal/NeMo:\${PYTHONPATH};
  CUDA_DEVICE_MAX_CONNECTIONS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -u /lustre/fsw/devtech/hpc-devtech/hongbinl/nemo_megatron/nemo_repo/internal/NeMo/examples/nlp/language_modeling/megatron_gpt_pretraining.py  \
  --config-path=/lustre/fsw/devtech/hpc-devtech/hongbinl/nemo_megatron/scripts/support_llama/NeMo-Megatron-Launcher/launcher_scripts/results/llama2_13b \
  --config-name=llama2_13b_hydra.yaml \
  model.mcore_gpt=False \
  model.transformer_engine=False \
  model.sequence_parallel=False \
  exp_manager.wandb_logger_kwargs.name=llama2_13b_nemo \
  "
