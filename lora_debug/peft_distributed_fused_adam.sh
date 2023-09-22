export PATH=/usr/local/bin:$PATH
NEMO=/lustre/fsw/joc/guyueh/nemo_peft/NeMo
MLM=/lustre/fsw/joc/guyueh/nemo_peft/megatron-lm
# APEX=/lustre/fsw/joc/guyueh/nemo_peft/apex
PL=/lustre/fsw/joc/guyueh/nemo_peft/lightning/src
export PYTHONPATH=${NEMO}:${MLM}:${PL}:$PYTHONPATH

CKPT=/lustre/fsw/joc/big_nlp/nemo_ci_resources/checkpoints/gpt3_5b_bf16_O2_tp2_pp1.nemo
DATAPATH=/lustre/fsw/joc/guyueh/data
optim=${1:-"distributed_fused_adam"}
tp=${2:-1}
sp=${3:-"False"}
micro_batch=${4:-4}
global_batch=128
max_seq=${5:-2048}
nemo_version=$(git -C ${NEMO} rev-parse HEAD)
mlm_version=$(git -C ${MLM} rev-parse HEAD)

if [ $tp == 2 ]
then
CKPT=/lustre/fsw/joc/big_nlp/nemo_ci_resources/checkpoints/gpt3_5b_bf16_O2_tp2_pp1.nemo
else
CKPT=/lustre/fsw/joc/big_nlp/nemo_ci_resources/checkpoints/megatron_gpt_1.3b.nemo
fi
echo $CKPT

if [ "$sp" == "True" ]
then
peft_scheme=ptuning
else
peft_scheme=lora
fi
echo $peft_scheme

tag="optim_${optim}_mbs_${micro_batch}_tp_${tp}_sp_${sp}_version_${nemo_version}_${mlm_version}"
NSYS_CMD="nsys profile -s none -t cuda,nvtx --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop -o ./nsys_nemo_lora_profile_${tag}"
CMD="torchrun --nproc_per_node=8 ${NEMO}/examples/nlp/language_modeling/tuning/megatron_gpt_peft_tuning.py trainer.devices=8 trainer.num_nodes=1 ++cluster_type=BCP"
# CMD="CUDA_VISIBLE_DEVICES=0 python ${NEMO}/examples/nlp/language_modeling/tuning/megatron_gpt_peft_tuning.py"

$CMD \
model.optim.name=${optim} \
model.tensor_model_parallel_size=${tp} \
model.sequence_parallel=${sp} \
trainer.precision=16 \
model.data.train_ds.max_seq_length=${max_seq} \
trainer.max_epochs=1 \
trainer.val_check_interval=1.0 \
model.global_batch_size=${global_batch} \
model.micro_batch_size=${micro_batch} \
model.restore_from_path=${CKPT} \
model.data.train_ds.concat_sampling_probabilities=[1.0] \
model.data.train_ds.file_names=[${DATAPATH}/SQuAD/squad_train.jsonl] \
model.data.validation_ds.file_names=[${DATAPATH}/SQuAD/squad_val.jsonl] \
model.peft.peft_scheme=${peft_scheme} \
model.answer_only_loss=True \
++model.use_flash_attention=True \
++model.nsys_profile.enabled=True \
++model.nsys_profile.start_step=2 \
++model.nsys_profile.end_step=2 \
++model.nsys_profile.gen_shape=True \
2>&1 | tee ${tag}.log