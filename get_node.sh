#!/bin/bash
#SBATCH
PARTITION=interactive
ACCOUNT=joc
jobname=joc-nemo_peft:skip_wgrad_debug

srun \
-t 02:00:00 \
-p ${PARTITION} \
-A ${ACCOUNT} \
-N 1 \
--ntasks-per-node 8 \
--container-image nvcr.io/ea-bignlp/nemofw-training:23.07-py3 \
--container-mounts "\
/lustre/fsw/joc/guyueh/:/lustre/fsw/joc/guyueh,\
/lustre/fsw/joc/big_nlp/nemo_ci_resources:/lustre/fsw/joc/big_nlp/nemo_ci_resources" \
--job-name ${jobname} \
--pty bash \

# /lustre/fsw/joc/yuya/stable_diffusion/NeMo-Megatron-Launcher/launcher_scripts:/lustre/fsw/joc/yuya/stable_diffusion/NeMo-Megatron-Launcher/launcher_scripts,\
# /lustre/fsw/joc/yuya/stable_diffusion/NeMo-Megatron-Launcher/launcher_scripts/data:/lustre/fsw/joc/yuya/stable_diffusion/NeMo-Megatron-Launcher/launcher_scripts/data,\
# /lustre/fsw/joc/yuya/stable_diffusion/NeMo-Megatron-Launcher/launcher_scripts/results:/lustre/fsw/joc/yuya/stable_diffusion/NeMo-Megatron-Launcher/launcher_scripts/results,\
# /lustre/fsw/joc/mingyuanm/huggingface-v1.2:/lustre/fsw/joc/mingyuanm/huggingface-v1.2,\
# /lustre/fsw/joc/yuya/stable_diffusion:/lustre/fsw/joc/yuya/stable_diffusion,\
# /lustre/fsw/joc/multimodal/datasets/:/lustre/fsw/joc/multimodal/datasets,\
# /lustre/fsw/adlr/adlr-nlp/vkorthikanti/ImageNet_s480_q95:/lustre/fsw/adlr/adlr-nlp/vkorthikanti/ImageNet_s480_q95,\
# /lustre/fsw/joc/chcui/NeMo-Megatron-Launcher/launcher_scripts/data:/lustre/fsw/joc/chcui/NeMo-Megatron-Launcher/launcher_scripts/data,\
# /lustre/fsw/joc/big_nlp/gpt3/prepare_dataset/the_pile/train:/lustre/fsw/joc/big_nlp/gpt3/prepare_dataset/the_pile/train,\
# /lustre/fsw/joc/yuya/nemo_clip:/lustre/fsw/joc/yuya/nemo_clip,\
# /lustre/fsw/adlr/adlr-nlp/mpatwary/data/multilingual/multi-1.1t-gtc:/lustre/fsw/adlr/adlr-nlp/mpatwary/data/multilingual/multi-1.1t-gtc \