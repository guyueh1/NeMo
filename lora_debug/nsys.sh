export PATH=/usr/local/bin:$PATH
NEMO=/lustre/fsw/joc/guyueh/nemo_peft/NeMo
MLM=/lustre/fsw/joc/guyueh/nemo_peft/megatron-lm
export PYTHONPATH=${NEMO}:${MLM}:$PYTHONPATH

CKPT=/lustre/fsw/joc/big_nlp/nemo_ci_resources/checkpoints/megatron_gpt_1.3b.nemo
DATAPATH=/lustre/fsw/joc/guyueh/data


tp=${1:-1}
sp=${2:-"False"}
micro_batch=${3:-4}
global_batch=128
max_seq=${4:-2048}
logfile=${5:-"dp_gpt_1.3B_nemo_lora"}
version=$(git -C ${MLM} rev-parse HEAD)

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

tag="mbs_${micro_batch}_tp_${tp}_sp_${sp}_version_${version}"
NSYS_CMD="nsys profile -s none -t cuda,nvtx --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop -o ./nsys_nemo_lora_profile_${tag}"
CMD="torchrun --nproc_per_node=8 ${NEMO}/examples/nlp/language_modeling/tuning/megatron_gpt_peft_tuning.py trainer.devices=8 trainer.num_nodes=1 ++cluster_type=BCP"

$NSYS_CMD \
$CMD \
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
2>&1 | tee nsys_${tag}.log


# export PATH=/usr/local/bin:$PATH
# export PYTHONPATH=/home/scratch.guyueh_sw/2023su/NeMo:/home/scratch.guyueh_sw/2023su/apex:/home/scratch.guyueh_sw/2023su/lightning/src:/home/scratch.guyueh_sw/2023su/Megatron-LM-gitlab:$PYTHONPATH

# micro_batch=${1:-1}
# max_seq=${2:-2048}
# freeze_before_training=${3:-"False"}
# torch_compile=${4:-"False"}
# logfile=${5:-"gpt5B_nemo_lora"}

# /home/scratch.svc_compute_arch/release/nsightSystems/x86_64/rel/2023.2.1.122/bin/nsys \
# profile -s none -o ./nsys_nemo_lora_profile_batch_${micro_batch}_seq_${max_seq}_freeze_${freeze_before_training}_compile_${torch_compile} \
# -t cuda,nvtx --force-overwrite true \
# --capture-range=cudaProfilerApi --capture-range-end=stop \
# python /home/scratch.guyueh_sw/2023su/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_peft_tuning.py \
# trainer.precision=bf16 \
# ++model.data.train_ds.pad_to_max_length=True \
# model.data.train_ds.max_seq_length=${max_seq} \
# trainer.max_epochs=1 \
# trainer.val_check_interval=1.0 \
# model.global_batch_size=4 \
# model.micro_batch_size=${micro_batch} \
# model.restore_from_path=/home/scratch.guyueh_sw/2023su/ckpt/nemo_gpt5B \
# model.data.train_ds.concat_sampling_probabilities=[1.0] \
# model.data.train_ds.file_names=[/home/scratch.guyueh_sw/2023su/dataset/SQuAD/squad_train.jsonl] \
# model.data.validation_ds.file_names=[/home/scratch.guyueh_sw/2023su/dataset/SQuAD/squad_val.jsonl] \
# model.peft.peft_scheme=lora \
# model.answer_only_loss=True \
# ++model.use_flash_attention=True \
# ++model.freeze_before_training=${freeze_before_training} \
# ++model.torch_compile=${torch_compile} \
# ++model.nsys_profile.enabled=True \
# ++model.nsys_profile.start_step=2 \
# ++model.nsys_profile.end_step=2 \
# ++model.nsys_profile.gen_shape=True
# 2>&1 | tee nsys_batch_${micro_batch}_seq_${max_seq}_freeze_${freeze_before_training}_compile_${torch_compile}_${logfile}.log