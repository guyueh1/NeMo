NEMO=/lustre/fsw/joc/guyueh/llama2_a100_perf/nemo_jason_mcore_llama
# MLM=/lustre/fsw/joc/guyueh/llama2_a100_perf/mlm-github
# TE=/lustre/fsw/joc/guyueh/llama2_a100_perf/TransformerEngine
export PYTHONPATH=${NEMO}:${MLM}:${TE}:$PYTHONPATH
MICRO_BATCH_SIZE=${1:-2}
TP=${2:-1}
PP=${3:-1}
SP=${4:-"False"}
GLOBAL_BATCH_SIZE=${5:-128}
NUM_DEVICES=8
NUM_NODES=1
MODEL="7b"

version=$(git rev-parse HEAD)

tag=${NUM_NODES}_nodes_${NUM_DEVICES}_devices_TP_${TP}_PP_${PP}_SP_${SP}_MBS_${MICRO_BATCH_SIZE}_GBS_${GLOBAL_BATCH_SIZE}_${MODEL}_v_${version}

# OPTIM="distributed_fused_adam ++model.optim.bucket_cap_mb=125 ++model.optim.overlap_grad_sync=False"
# OPTIM="fused_adam"

# NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=0 \
CUDA_DEVICE_MAX_CONNECTIONS=1 \
torchrun --nproc_per_node=${NUM_DEVICES} ${NEMO}/examples/nlp/language_modeling/megatron_gpt_pretraining.py \
--config-path=${NEMO}/debug_llama2_sota \
--config-name llama2_${MODEL}_hydra.yaml \
++cluster_type=BCP \
trainer.num_nodes=${NUM_NODES} \
trainer.devices=${NUM_DEVICES} \
model.mcore_gpt=True \
model.transformer_engine=True \
model.micro_batch_size=${MICRO_BATCH_SIZE} \
model.global_batch_size=${GLOBAL_BATCH_SIZE} \
model.sequence_parallel=${SP} \
model.pipeline_model_parallel_size=${PP} \
model.tensor_model_parallel_size=${TP} \
model.encoder_seq_length=2048 \
2>&1 | tee llama2_pretrain_${tag}.log
