#!/bin/bash -l

source ~/venv/ai-economist/bin/activate

param1=$1
param2=$2

echo ${param1}
echo ${param2}

destnode=`uname -n`
echo "destnode is = [$destnode]"

ray start --address="${param1}" --redis-password="${param2}" #--num-cpus 4 --num-gpus 1