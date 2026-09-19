#!/bin/bash
#SBATCH  -J  test   # 名称
#SBATCH -p qdhcnormal  #队列 
#SBATCH -N 1 #节点 
#SBATCH -n 32 #总核心cpu    
#SBATCH --exclusive  # 独占队列  按照整个节点计费
#SBATCH --comment={glibcVersion:2.31}

export HF_ENDPOINT=https://hf-mirror.com
export http_proxy='http://lpkkokjo:066642e3@10.1.4.17:3128'
export https_proxy='http://lpkkokjo:066642e3@10.1.4.17:3128'
export ftp_proxy='http://lpkkokjo:066642e3@10.1.4.17:3128' 
source ~/miniconda3/bin/activate tabfm_env
module load    compiler/gcc/9.3.0    
module load   compiler/cmake/3.20.4
#执行命令
python classification_example.py
