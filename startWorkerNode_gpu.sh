#!/bin/bash -l

source ~/venv/ai-economist/bin/activate

cd $PBS_O_WORKDIR

param1=$1
param2=$2

destnode=`uname -n`
echo "destnode is = [$destnode]"

ray start --address="${param1}" --redis-password="${param2}" #--num-cpus 4 --num-gpus 1