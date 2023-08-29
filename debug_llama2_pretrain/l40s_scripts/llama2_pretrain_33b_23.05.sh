#!/bin/bash

# Parameters
#SBATCH --dependency=singleton
#SBATCH --error=/home/guyueh/llama2_l40s_profile/guyueh1_nemo/debug_llama2_pretrain/l40s_scripts/srun_logs/log-llama2_33b:debug_%j.err
#SBATCH --exclusive
#SBATCH --job-name=llama2_33b:debug
#SBATCH --mem=0
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --output=/home/guyueh/llama2_l40s_profile/guyueh1_nemo/debug_llama2_pretrain/l40s_scripts/srun_logs/log-llama2_33b:debug_%j.err
#SBATCH --partition=L40
#SBATCH --time=0-00:15:00

# setup

NEMO=/opt/NeMo
WORKD=/home/guyueh/llama2_l40s_profile/guyueh1_nemo/debug_llama2_pretrain/l40s_scripts
TP=${1:-1}
PP=${2:-1}
SP=${3:-"False"}
MICRO_BATCH_SIZE=${4:-1}
GLOBAL_BATCH_SIZE=${5:-128}
NUM_DEVICES=8
NUM_NODES=4
MODEL="33b"

# CONTAINER_IMAGE=nvcr.io/ea-bignlp/nemofw-training:23.05-py3 
# ENV_INSTALL="pip install git+https://github.com/NVIDIA/TransformerEngine.git@3b7b7c68fc310067567956d6f63f633e2012bcec; MAX_JOBS=20 pip install flash-attn==2.0.4  --no-build-isolation; "
CONTAINER_IMAGE=/home/guyueh/llama2_l40s_profile/nemo-llama_23.05-py3.sqsh
ENV_INSTALL=""

tag=${NUM_NODES}_nodes_${NUM_DEVICES}_devices_TP_${TP}_PP_${PP}_SP_${SP}_MBS_${MICRO_BATCH_SIZE}_GBS_${GLOBAL_BATCH_SIZE}_${MODEL}

# command 1
srun --output /home/guyueh/llama2_l40s_profile/guyueh1_nemo/debug_llama2_pretrain/l40s_scripts/srun_logs/log-llama2_33b:debug_%j.err --error /home/guyueh/llama2_l40s_profile/guyueh1_nemo/debug_llama2_pretrain/l40s_scripts/srun_logs/log-llama2_33b:debug_%j.err --container-image ${CONTAINER_IMAGE} --container-mounts "/home/guyueh/llama2_l40s_profile:/home/guyueh/llama2_l40s_profile" bash -c "
    ${ENV_INSTALL} cd /home/guyueh/llama2_l40s_profile/guyueh1_nemo/debug_llama2_pretrain/l40s_scripts/;
    git -C ${NEMO} rev-parse HEAD;
    nvidia-smi;
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python3 -u ${NEMO}/examples/nlp/language_modeling/megatron_gpt_pretraining.py  \
    --config-path ${WORKD} \
    --config-name llama2_${MODEL}_23.05.yaml \
    trainer.num_nodes=${NUM_NODES} \
    trainer.devices=${NUM_DEVICES} \
    model.global_batch_size=${GLOBAL_BATCH_SIZE} \
    model.micro_batch_size=${MICRO_BATCH_SIZE} \
    model.sequence_parallel=${SP} \
    model.pipeline_model_parallel_size=${PP} \
    model.tensor_model_parallel_size=${TP} \
    model.fp8=True \
    model.transformer_engine=True \
    model.fp8_e4m3=True \
    2>&1 > /home/guyueh/llama2_l40s_profile/guyueh1_nemo/debug_llama2_pretrain/l40s_scripts/srun_logs/llama2_pretrain_${tag}.log
  "
