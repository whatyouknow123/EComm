#!/bin/bash

AWS="/usr/bin/aws"
PYTHON="/usr/bin/python"
SPARK="/usr/bin/spark-submit"



# main

$SPARK \
--name research_merge_swing_daily_online \
--master yarn  --deploy-mode cluster \
--executor-memory 10g \
--executor-cores 8 \
--num-executors 30 \
--driver-memory 10g \
--py-files util.py,conf.py \
--conf spark.executor.memoryOverhead=10g \
--conf spark.dynamicAllocation.enabled=false \
--conf spark.yarn.maxAppAttempts=1 \
swing/merge_swing_daily_online.py $1 $2