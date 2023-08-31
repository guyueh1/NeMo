#!/bin/bash

./reinstall.sh

export NVTE_FRAMEWORK=pytorch
pip install -e /home/guyueh/llama2_l40s_profile/mlm-github/
pip install -e /home/guyueh/llama2_l40s_profile/TransformerEngine
pip install -U transformers

