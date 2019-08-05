#!/bin/bash

AWS="/usr/bin/aws"
PYTHON="/usr/bin/python"
SPARK="/usr/bin/spark-submit"

# main

$SPARK \
--name research_get_item_pair_swing_online \
--master yarn  --deploy-mode cluster \
--executor-memory 25g \
--executor-cores 10 \
--num-executors 100 \
--driver-memory 8g \
--conf spark.dynamicAllocation.enabled=false \
--conf spark.yarn.maxAppAttempts=1 \
swing_daily_online.py $1


