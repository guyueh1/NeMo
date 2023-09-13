NEMO=/home/guyueh/llama2_l40s_profile/guyueh1_nemo
# MLM=/home/guyueh/llama2_l40s_profile/mlm-github
# TE=/home/guyueh/llama2_l40s_profile/TransformerEngine
# export PYTHONPATH=${NEMO}:${MLM}:${TE}:$PYTHONPATH
export PYTHONPATH=${NEMO}:$PYTHONPATH
MICRO_BATCH_SIZE=${1:-1}
TP=${2:-1}
PP=${3:-8}
SP=${4:-"False"}
GLOBAL_BATCH_SIZE=${5:-4}
NUM_DEVICES=8
NUM_NODES=1
MODEL="7b"

version=$(git -C ${NEMO} rev-parse HEAD)

tag=GQA_group_8_${NUM_NODES}_nodes_${NUM_DEVICES}_devices_TP_${TP}_PP_${PP}_SP_${SP}_MBS_${MICRO_BATCH_SIZE}_GBS_${GLOBAL_BATCH_SIZE}_${MODEL}_v_${version}

# OPTIM="distributed_fused_adam ++model.optim.bucket_cap_mb=125 ++model.optim.overlap_grad_sync=False"
# OPTIM="fused_adam"

# CUDA_DEVICE_MAX_CONNECTIONS=1 \
# NCCL_DEBUG=TRACE NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_P2P_DIRECT_DISABLE=1 \
# NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=0 \
nsys \
profile -s none -o ./llama2_pretrain_fp8_${tag} \
-t cuda,nvtx --force-overwrite true \
--capture-range=cudaProfilerApi --capture-range-end=stop \
torchrun --nproc_per_node=${NUM_DEVICES} \
${NEMO}/examples/nlp/language_modeling/megatron_gpt_pretraining.py \
--config-path ${NEMO}/debug_llama2_pretrain/l40s_scripts \
--config-name llama2_${MODEL}_hydra_gqa.yaml \
trainer.num_nodes=${NUM_NODES} \
trainer.devices=${NUM_DEVICES} \
++trainer.use_distributed_sampler=false \
++cluster_type=BCP \
model.transformer_engine=true \
model.mcore_gpt=True \
model.global_batch_size=${GLOBAL_BATCH_SIZE} \
model.micro_batch_size=${MICRO_BATCH_SIZE} \
model.sequence_parallel=${SP} \
model.pipeline_model_parallel_size=${PP} \
model.tensor_model_parallel_size=${TP} \
model.fp8=True \
model.fp8_e4m3=True \
model.data.num_workers=2 \
model.nsys_profile.enabled=True \
model.nsys_profile.gen_shape=True \
model.nsys_profile.start_step=2 \
model.nsys_profile.end_step=2 \
2>&1 | tee nsys_llama2_pretrain_fp8_${tag}.log
