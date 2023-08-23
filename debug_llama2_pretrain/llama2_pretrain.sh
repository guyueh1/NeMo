NEMO=/lustre/fsw/joc/guyueh/llama2_a100_perf/nemo_jason_mcore_llama
export PYTHONPATH=$NEMO:$PYTHONPATH
MICRO_BATCH_SIZE=${1:-1}
TP=${2:-1}
SP=${3:-"False"}
NUM_DEVICES=${4:-8}
NUM_NODES=${5:-1}

version=$(git rev-parse HEAD)

tag=${NUM_NODES}_nodes_${NUM_DEVICES}_devices_TP_${TP}_SP_${SP}_MBS_${MICRO_BATCH_SIZE}_v_${version}

# OPTIM="distributed_fused_adam ++model.optim.bucket_cap_mb=125 ++model.optim.overlap_grad_sync=False"
# OPTIM="fused_adam"

# NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=0 \
torchrun --nproc_per_node=${NUM_DEVICES} ${NEMO}/examples/nlp/language_modeling/megatron_gpt_pretraining.py \
--config-path=${NEMO}/debug_llama2_pretrain \
--config-name llama2_7b_hydra.yaml \
++cluster_type=BCP \
trainer.num_nodes=${NUM_NODES} \
trainer.devices=${NUM_DEVICES} \
model.mcore_gpt=True \
model.transformer_engine=True \
model.micro_batch_size=${MICRO_BATCH_SIZE} \
model.sequence_parallel=${SP} \
model.tensor_model_parallel_size=${TP} \
2>&1 | tee llama2_pretrain_${tag}.log
