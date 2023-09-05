# sbatch  llama2_pretrain_13b.sh  1 2 1 True 128
# sbatch  llama2_pretrain_13b.sh  2 2 2 True 128
# sbatch  llama2_pretrain_13b.sh  2 2 4 True 128
sbatch  llama2_pretrain_13b.sh  1 1 1 False 128
sbatch  llama2_pretrain_13b.sh  1 1 2 False 128
sbatch  llama2_pretrain_13b.sh  2 1 2 False 128
# sbatch  llama2_pretrain_13b.sh  1 1 4 False 128
# sbatch  llama2_pretrain_13b.sh  2 1 2 False 128
