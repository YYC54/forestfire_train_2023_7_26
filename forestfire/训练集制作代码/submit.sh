#!/bin/bash
#SBATCH --job-name=interp
#SBATCH --output=./job-%j.log
#SBATCH --error=./job-%j.err
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128

cd /public/home/lihf_hx/yyc/森林火险模型/forestfire_train_2023_7_26/forestfire/训练集制作代码

ulimit -s unlimited
/public/home/lihf_hx/anaconda3/envs/daily/bin/python main.py
