#!/bin/bash
# 
if [ $# -eq 0 ]; then
    echo "未提供时间参数，将使用当前时间作为默认值"
    current_time=$(date +"%Y%m%d")
    time_to_use=$current_time
    echo $time_to_use
else
    time_param="$1"
    time_to_use=$time_param
    echo $time_to_use
fi
cd /public/home/lihf_hx/yyc/森林火险模型/forestfire_train_2023_7_26/forestfire
pwd
/public/home/lihf_hx/anaconda3/envs/ai-lab/bin/python main.py $time_to_use > /public/home/lihf_hx/yyc/森林火险模型/forestfire_train_2023_7_26/forestfire/output.log
