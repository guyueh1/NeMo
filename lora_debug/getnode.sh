#!/bin/bash
#SBATCH

ACCOUNT=joc
job=${ACCOUNT}-nemo_peft:tuning_debug
partition=interactive
NODE=1
# TASK_PER_NODE=8

srun \
-t 02:00:00 \
-p $partition \
-A ${ACCOUNT} \
-N $NODE \
--container-image nvcr.io/ea-bignlp/nemofw-training:23.07-py3 \
--container-mounts "\
/lustre/fsw/joc/guyueh/nemo_peft:/lustre/fsw/joc/guyueh/nemo_peft,\
/lustre/fsw/joc/guyueh/data:/lustre/fsw/joc/guyueh/data,\
/lustre/fsw/joc/big_nlp/nemo_ci_resources:/lustre/fsw/joc/big_nlp/nemo_ci_resources" \
--job-name ${job} \
--pty bash