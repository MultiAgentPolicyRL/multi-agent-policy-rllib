#!/bin/bash
### $1 is path to experiment config

#PBS -l select=5:ncpus=4:mem=20gb
#PBS -l walltime=00:10:00
#PBS -q short_cpuQ

source ~/venv/ai-economist/bin/activate

ip_prefix=`hostname -i`
suffix=':6379'
ip_head=$ip_prefix$suffix
redis_password=$(uuidgen)

echo parameters: $ip_head $redis_password

ray start --head --port=6379 \
--redis-password=$redis_password \
--num-cpus 4 --num-gpus 0
sleep 10

for (( n=4; n<$PBS_NCPUS; n+=4 ))
do
  pbsdsh -n $n -v startWorkerNode.sh \
  $ip_head $redis_password &
  sleep 10
done

cd ~/ai-economist-ppo-decision-tree/ai-economist/tutorials/rllib/ || exit
python3 training_2_algos.py --run-dir ./experiments/check/phase1/ --pw $redis_password --ip_address $ip_head

ray stop
deactivate