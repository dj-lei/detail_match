#!/bin/sh
LANG=en_US.UTF-8

# 杀掉所有python进程
# kill -9 $(ps -e|grep python |awk '{print $1}')

# 进入工程目录
project_path=/home/ml/evaluation-detail-match
cd ${project_path}

# 创建日志
if [ ! -d ./log/ ];
then
    mkdir -p log;
    touch ./log/cron.log
fi

# 定时下载模型
nohup scp -r root@train_ml:/home/ml/evaluation-detail-match/match/predict/map ./match/predict/ > ./log/cron.log 2>&1
nohup scp -r root@train_ml:/home/ml/evaluation-detail-match/match/predict/model ./match/predict/ > ./log/cron.log 2>&1