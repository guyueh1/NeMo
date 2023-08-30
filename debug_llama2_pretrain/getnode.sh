#!/bin/bash
#SBATCH

ACCOUNT=coreai_dlalgo_llm
job=${ACCOUNT}-llama2:pretrain_debug
partition=interactive
NODE=1
# TASK_PER_NODE=8

srun \
-t 02:00:00 \
-p $partition \
-A ${ACCOUNT} \
-N $NODE \
--container-image /lustre/fsw/joc/guyueh/nemo_llama2.sqsh \
--container-mounts "\
/lustre/fsw/joc/guyueh/llama2_a100_perf:/lustre/fsw/joc/guyueh/llama2_a100_perf,\
/lustre/fsw/joc/guyueh/data:/lustre/fsw/joc/guyueh/data,\
/lustre/fsw/joc/big_nlp/nemo_gpt3:/lustre/fsw/joc/big_nlp/nemo_gpt3,\
/lustre/fsw/joc/big_nlp/nemo_ci_resources:/lustre/fsw/joc/big_nlp/nemo_ci_resources" \
--job-name ${job} \
--pty bash