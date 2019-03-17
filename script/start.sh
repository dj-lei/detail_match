#!/bin/bash
LANG=en_US.UTF-8

# 杀掉所有python进程
# kill -9 $(ps -e|grep python |awk '{print $1}')

# 进入工程目录
project_path=/home/ml/detail-match
cd ${project_path}

# 创建日志
if [ ! -d ./log/ ];
then
    mkdir -p log;
    touch ./log/cron.log
fi

# 执行定时任务
source /etc/profile
nohup /usr/bin/python3 -u start.py > ./log/cron.log 2>&1