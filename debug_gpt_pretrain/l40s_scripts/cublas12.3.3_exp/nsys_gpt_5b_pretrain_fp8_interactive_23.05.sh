NEMO=/opt/NeMo
WORKD=/home/guyueh/llama2_l40s_profile/guyueh1_nemo/debug_gpt_pretrain/l40s_scripts
MICRO_BATCH_SIZE=${1:-1}
TP=${2:-1}
PP=${3:-2}
SP=${4:-"False"}
GLOBAL_BATCH_SIZE=${5:-512}
NUM_DEVICES=8
NUM_NODES=1
MODEL="5b"

version=$(git -C ${NEMO} rev-parse HEAD)

tag=${NUM_NODES}_nodes_${NUM_DEVICES}_devices_TP_${TP}_PP_${PP}_SP_${SP}_MBS_${MICRO_BATCH_SIZE}_GBS_${GLOBAL_BATCH_SIZE}_${MODEL}_v_${version}

export CUBLASLT_DEVICE_NAME="NVIDIA L40S"
export LD_LIBRARY_PATH=/home/guyueh/llama2_l40s_profile/cublas_release_override_device_name_l40s_war_08252023/lib64:$LD_LIBRARY_PATH

LD_PRELOAD="/home/guyueh/llama2_l40s_profile/cublas_release_override_device_name_l40s_war_08252023/lib64/libcublas.so /home/guyueh/llama2_l40s_profile/cublas_release_override_device_name_l40s_war_08252023/lib64/libcublasLt.so /home/guyueh/llama2_l40s_profile/cublas_release_override_device_name_l40s_war_08252023/lib64/libnvblas.so" \
nsys \
profile -s none -o ./gpt_pretrain_fp8_${tag} \
-t cuda,nvtx --force-overwrite true \
--capture-range=cudaProfilerApi --capture-range-end=stop \
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
model.fp8=True \
model.fp8_e4m3=True \
model.nsys_profile.enabled=true \
2>&1 | tee nsys_gpt_pretrain_fp8_${tag}.log

