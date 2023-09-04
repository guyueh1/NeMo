# 7b
sbatch llama2_pretrain_7b_nightly.sh 1 1 4 
sbatch llama2_pretrain_7b_nightly.sh 1 1 8

# 13b
sbatch llama2_pretrain_13b_nightly.sh 1 1 8

# 33b
sbatch llama2_pretrain_33b_nightly.sh 1 2 8 True
sbatch llama2_pretrain_33b_nightly.sh 1 4 4 True