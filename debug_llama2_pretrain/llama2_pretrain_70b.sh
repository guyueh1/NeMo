#!/bin/bash

# Parameters
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --dependency=singleton
#SBATCH --error=/lustre/fsw/joc/guyueh/llama2_a100_perf/nemo_jason_mcore_llama/debug_llama2_pretrain/llama2_70b/log-coreai_dlalgo_llm-llama2:70b_pretrain_profile_%j.err
#SBATCH --exclusive
#SBATCH --job-name=coreai_dlalgo_llm-llama2:70b_pretrain_profile
#SBATCH --mem=0
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --output=/lustre/fsw/joc/guyueh/llama2_a100_perf/nemo_jason_mcore_llama/debug_llama2_pretrain/llama2_70b/log-coreai_dlalgo_llm-llama2:70b_pretrain_profile_%j.err
#SBATCH --partition=luna
#SBATCH --time=0-00:30:00

# setup
export TRANSFORMERS_OFFLINE=1
export NCCL_AVOID_RECORD_STREAMS=1

MICRO_BATCH_SIZE=${1:-2}
TP=${2:-8}
PP=${3:-2}
SP=${4:-"True"}
GLOBAL_BATCH_SIZE=${5:-512}
NUM_DEVICES=8
NUM_NODES=8
MODEL="70b"

CONTAINER_IMAGE=/lustre/fsw/joc/guyueh/nemo_llama2.sqsh
# CONTAINER_IMAGE=nvcr.io/nvidian/bignlp-train:23.08-nemofw-nightly

tag=${NUM_NODES}_nodes_${NUM_DEVICES}_devices_TP_${TP}_PP_${PP}_SP_${SP}_MBS_${MICRO_BATCH_SIZE}_GBS_${GLOBAL_BATCH_SIZE}_${MODEL}

# command 1
srun --output /lustre/fsw/joc/guyueh/llama2_a100_perf/nemo_jason_mcore_llama/debug_llama2_pretrain/llama2_70b/log-coreai_dlalgo_llm-llama2:70b_pretrain_profile_%j.err --error /lustre/fsw/joc/guyueh/llama2_a100_perf/nemo_jason_mcore_llama/debug_llama2_pretrain/llama2_70b/log-coreai_dlalgo_llm-llama2:70b_pretrain_profile_%j.err --container-image ${CONTAINER_IMAGE} --container-mounts "/lustre/fsw/joc/guyueh/llama2_a100_perf:/lustre/fsw/joc/guyueh/llama2_a100_perf,/lustre/fsw/joc/guyueh/data:/lustre/fsw/joc/guyueh/data,/lustre/fsw/joc/big_nlp/nemo_gpt3:/lustre/fsw/joc/big_nlp/nemo_gpt3,/lustre/fsw/joc/big_nlp/nemo_ci_resources:/lustre/fsw/joc/big_nlp/nemo_ci_resources" --no-container-mount-home bash -c "
    cd /lustre/fsw/joc/guyueh/llama2_a100_perf/nemo_jason_mcore_llama/debug_llama2_pretrain;
    git rev-parse HEAD;
    export PYTHONPATH=/lustre/fsw/joc/guyueh/llama2_a100_perf/nemo_jason_mcore_llama:/lustre/fsw/joc/guyueh/llama2_a100_perf/TransformerEngine:/lustre/fsw/joc/guyueh/llama2_a100_perf/mlm-github:\${PYTHONPATH};
    CUDA_DEVICE_MAX_CONNECTIONS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -u /lustre/fsw/joc/guyueh/llama2_a100_perf/nemo_jason_mcore_llama/examples/nlp/language_modeling/megatron_gpt_pretraining.py  \
    --config-path=/lustre/fsw/joc/guyueh/llama2_a100_perf/nemo_jason_mcore_llama/debug_llama2_pretrain \
    --config-name=llama2_${MODEL}_hydra.yaml \
    trainer.num_nodes=${NUM_NODES} \
    trainer.devices=${NUM_DEVICES} \
    model.mcore_gpt=True \
    model.transformer_engine=True \
    model.micro_batch_size=${MICRO_BATCH_SIZE} \
    model.global_batch_size=${GLOBAL_BATCH_SIZE} \
    model.sequence_parallel=${SP} \
    model.pipeline_model_parallel_size=${PP} \
    model.tensor_model_parallel_size=${TP} \
    2>&1 > /lustre/fsw/joc/guyueh/llama2_a100_perf/nemo_jason_mcore_llama/debug_llama2_pretrain/llama2_pretrain_${tag}.log
  "
