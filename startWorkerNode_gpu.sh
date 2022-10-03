#!/bin/bash -l

source ../venv/ai-economist/bin/activate

ray start --block --address=$1 \
--redis-password=$2 --num-cpus 4 --num-gpus 1

ray stop
conda deactivate