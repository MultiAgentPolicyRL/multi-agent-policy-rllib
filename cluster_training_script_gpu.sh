#!/bin/bash
### $1 is path to experiment config

#PBS -l select=5:ncpus=4:mem=10gb
#PBS -l walltime=00:05:00
#PBS -q short_gpuQ
#PBS -j oe  

source ~/venv/ai-economist/bin/activate

## Change to working directory
cd $PBS_O_WORKDIR

# Get node list from PBS and format for job:
JOB_NODES=`uniq -c ${PBS_NODEFILE} | awk -F. '{ print $1 }' | awk '{print $2 ":" $1}' | paste -s -d ':'`  

echo "pbs_o_workdir is: $PBS_O_WORKDIR"
thishost=`uname -n`
thishostip=`hostname -i`
suffix=':6379'
ip_head=$thishostip$suffix
redis_password=$(uuidgen)

echo "thishost=[$thishost]"
echo "thishostip=[$thishostip]"

ray start --head --redis-port=6379 \
--redis-password=$redis_password \
--num-cpus 4 --num-gpus 1
sleep 5


# c=$((PBS_NCPUS*PBS_NUM_NODES))
n=4
c=20

while [ $n -lt $c ]
do
  n=$((n+=4))
  pbsdsh -n $n -v ~/ai-economist-ppo-decision-tree/startWorkerNode_gpu.sh \
  "${ip_head}" "${redis_password}" &
done

sleep 20

python3 ~/ai-economist-ppo-decision-tree/ai-economist/tutorials/rllib/training_2_algos.py --run-dir ~/ai-economist-ppo-decision-tree/ai-economist/tutorials/rllib/experiments/check/phase1_gpu/ --pw $redis_password --ip_address $ip_head

ray stop
deactivate
#