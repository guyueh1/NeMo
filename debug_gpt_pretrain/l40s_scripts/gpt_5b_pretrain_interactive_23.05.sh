NEMO=/opt/NeMo
WORKD=/home/guyueh/llama2_l40s_profile/guyueh1_nemo/debug_gpt_pretrain/l40s_scripts
MICRO_BATCH_SIZE=${1:-1}
TP=${2:-1}
PP=${3:-8}
SP=${4:-"False"}
GLOBAL_BATCH_SIZE=${5:-128}
NUM_DEVICES=8
NUM_NODES=1
MODEL="5b"

version=$(git -C ${NEMO} rev-parse HEAD)

tag=${NUM_NODES}_nodes_${NUM_DEVICES}_devices_TP_${TP}_PP_${PP}_SP_${SP}_MBS_${MICRO_BATCH_SIZE}_GBS_${GLOBAL_BATCH_SIZE}_${MODEL}_v_${version}

torchrun --nproc_per_node=${NUM_DEVICES} \
${NEMO}/examples/nlp/language_modeling/megatron_gpt_pretraining.py \
--config-path ${WORKD} \
--config-name gpt_${MODEL}_hydra.yaml \
++trainer.replace_sampler_ddp=false \
trainer.num_nodes=${NUM_NODES} \
trainer.devices=${NUM_DEVICES} \
++cluster_type=BCP \
model.transformer_engine=true \
model.global_batch_size=${GLOBAL_BATCH_SIZE} \
model.micro_batch_size=${MICRO_BATCH_SIZE} \
model.sequence_parallel=${SP} \
model.pipeline_model_parallel_size=${PP} \
model.tensor_model_parallel_size=${TP} \
2>&1 | tee gpt_pretrain_${tag}.log

