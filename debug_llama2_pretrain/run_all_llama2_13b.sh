# sbatch  llama2_pretrain_13b.sh  1 2 1 True 512
# sbatch  llama2_pretrain_13b.sh  2 2 2 True 512
# sbatch  llama2_pretrain_13b.sh  2 2 4 True 512
# sbatch  llama2_pretrain_13b.sh  1 1 2 False 512
# sbatch  llama2_pretrain_13b.sh  1 1 4 False 512
sbatch  llama2_pretrain_13b.sh  2 1 2 False 512
