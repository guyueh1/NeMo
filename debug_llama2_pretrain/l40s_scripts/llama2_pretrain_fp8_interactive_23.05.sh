NEMO=/opt/NeMo
WORKD=/home/guyueh/llama2_l40s_profile/guyueh1_nemo/debug_llama2_pretrain/l40s_scripts
MICRO_BATCH_SIZE=${1:-1}
TP=${2:-1}
PP=${3:-8}
SP=${4:-"False"}
GLOBAL_BATCH_SIZE=${5:-128}
NUM_DEVICES=8
NUM_NODES=1
MODEL="7b"
# MEGATRON_AMP_O2=${6:-True}

version=$(git -C ${NEMO} rev-parse HEAD)

tag=${NUM_NODES}_nodes_${NUM_DEVICES}_devices_TP_${TP}_PP_${PP}_SP_${SP}_MBS_${MICRO_BATCH_SIZE}_GBS_${GLOBAL_BATCH_SIZE}_${MODEL}_O2_v_${version}

# NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=0 \
# CUDA_DEVICE_MAX_CONNECTIONS=1 \
# NCCL_DEBUG=TRACE NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_P2P_DIRECT_DISABLE=1 \
torchrun --nproc_per_node=${NUM_DEVICES} \
${NEMO}/examples/nlp/language_modeling/megatron_gpt_pretraining.py \
--config-path ${WORKD} \
--config-name llama2_${MODEL}_hydra.yaml \
trainer.num_nodes=${NUM_NODES} \
trainer.devices=${NUM_DEVICES} \
++trainer.replace_sampler_ddp=false \
++cluster_type=BCP \
model.global_batch_size=${GLOBAL_BATCH_SIZE} \
model.micro_batch_size=${MICRO_BATCH_SIZE} \
model.sequence_parallel=${SP} \
model.pipeline_model_parallel_size=${PP} \
model.tensor_model_parallel_size=${TP} \
model.fp8=True \
model.transformer_engine=True \
model.fp8_e4m3=True \
2>&1 | tee llama2_pretrain_fp8_${tag}.log
