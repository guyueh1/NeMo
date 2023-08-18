export PATH=/usr/local/bin:$PATH
NEMO=/home/scratch.guyueh_sw/2023su/nemo_remove_master_weight/NeMo
export PYTHONPATH=${NEMO}:$PYTHONPATH

micro_batch=${1:-1}
gbs=4
freeze_before_training=${2:-"True"}
logfile="gpt5B_nemo_sft"

/home/scratch.svc_compute_arch/release/nsightSystems/x86_64/rel/2023.2.1.122/bin/nsys \
profile -s none -o ./nsys_batch_${micro_batch}_freeze_${freeze_before_training}_${logfile} \
-t cuda,nvtx --force-overwrite true \
--capture-range=cudaProfilerApi --capture-range-end=stop \
python ${NEMO}/examples/nlp/language_modeling/tuning/megatron_gpt_sft.py \
trainer.precision=bf16 \
trainer.max_epochs=1 \
trainer.val_check_interval=1.0 \
model.megatron_amp_O2=True \
model.global_batch_size=${gbs} \
model.micro_batch_size=${micro_batch} \
model.restore_from_path=/home/scratch.guyueh_sw/2023su/ckpt/nemo_gpt5B \
model.data.train_ds.concat_sampling_probabilities=[1.0] \
model.data.train_ds.file_names=[/home/scratch.guyueh_sw/2023su/dataset/SQuAD/squad_train.jsonl] \
model.data.validation_ds.file_names=[/home/scratch.guyueh_sw/2023su/dataset/SQuAD/squad_val.jsonl] \
model.data.test_ds.file_names=[/home/scratch.guyueh_sw/2023su/dataset/SQuAD/squad_test.jsonl] \
model.answer_only_loss=True \
++model.freeze_before_training=${freeze_before_training} \
++model.nsys_profile.enabled=True \
++model.nsys_profile.start_step=2 \
++model.nsys_profile.end_step=2 \
++model.nsys_profile.gen_shape=True
2>&1 | tee nsys_batch_${micro_batch}_freeze_${freeze_before_training}_${logfile}.log