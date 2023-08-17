export PATH=/usr/local/bin:$PATH
NEMO=/lustre/fsw/joc/guyueh/nemo_peft/NeMo
MLM=/lustre/fsw/joc/guyueh/nemo_peft/megatron-lm
export PYTHONPATH=${NEMO}:${MLM}:$PYTHONPATH

CKPT=/lustre/fsw/joc/big_nlp/nemo_ci_resources/checkpoints/gpt3_5b_bf16_O2_tp2_pp1.nemo
DATAPATH=/lustre/fsw/joc/guyueh/data

micro_batch=${1:-4}
global_batch=128
max_seq=${2:-2048}
logfile=${5:-"tp2_sp_gpt_5B_nemo_ptuning"}
version=$(git -C ${MLM} rev-parse HEAD)

CMD="torchrun --nproc_per_node=8 ${NEMO}/examples/nlp/language_modeling/tuning/megatron_gpt_peft_tuning.py trainer.devices=8 trainer.num_nodes=1 ++cluster_type=BCP"
# CMD="CUDA_VISIBLE_DEVICES=0 python ${NEMO}/examples/nlp/language_modeling/tuning/megatron_gpt_peft_tuning.py"
$CMD \
model.tensor_model_parallel_size=2 \
model.sequence_parallel=True \
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
model.peft.peft_scheme=ptuning \
model.answer_only_loss=True \
++model.use_flash_attention=True \
++model.nsys_profile.enabled=True \
++model.nsys_profile.start_step=2 \
++model.nsys_profile.end_step=2 \
++model.nsys_profile.gen_shape=True \
2>&1 | tee batch_${micro_batch}_seq_${max_seq}_${logfile}.${version}.log