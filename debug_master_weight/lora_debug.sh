export PATH=/usr/local/bin:$PATH
NEMO=/home/scratch.guyueh_sw/2023su/nemo_remove_master_weight/NeMo
export PYTHONPATH=${NEMO}:$PYTHONPATH

peft_scheme=${1:-"lora"}
amp_O2=${2:-"True"}
micro_batch=${3:-1}
gbs=4
freeze_before_training=${4:-"True"}
logfile="gpt5B_nemo_${peft_scheme}"

/home/scratch.svc_compute_arch/release/nsightSystems/x86_64/rel/2023.2.1.122/bin/nsys \
profile -s none -o ./nsys_batch_${micro_batch}_freeze_${freeze_before_training}_${logfile} \
-t cuda,nvtx --force-overwrite true \
--capture-range=cudaProfilerApi --capture-range-end=stop \
python ${NEMO}/examples/nlp/language_modeling/tuning/megatron_gpt_peft_tuning.py \
trainer.precision=bf16 \
trainer.max_epochs=1 \
trainer.val_check_interval=1.0 \
model.megatron_amp_O2=${amp_O2} \
model.global_batch_size=${gbs} \
model.micro_batch_size=${micro_batch} \
model.restore_from_path=/home/scratch.guyueh_sw/2023su/ckpt/nemo_gpt5B \
model.data.train_ds.concat_sampling_probabilities=[1.0] \
model.data.train_ds.file_names=[/home/scratch.guyueh_sw/2023su/dataset/SQuAD/squad_train.jsonl] \
model.data.validation_ds.file_names=[/home/scratch.guyueh_sw/2023su/dataset/SQuAD/squad_val.jsonl] \
model.peft.peft_scheme=${peft_scheme} \
model.answer_only_loss=True \
++model.freeze_before_training=${freeze_before_training} \
++model.nsys_profile.enabled=True \
++model.nsys_profile.start_step=2 \
++model.nsys_profile.end_step=2 \
++model.nsys_profile.gen_shape=True \
2>&1 | tee nsys_batch_${micro_batch}_freeze_${freeze_before_training}_${logfile}.log