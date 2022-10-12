#!/bin/bash
### $1 is path to experiment config

#PBS -l select=5:ncpus=4:mem=10gb
#PBS -l walltime=00:10:00
#PBS -q short_gpuQ
#PBS -j oe  

source ~/venv/ai-economist/bin/activate

## Change to working directory
ln -s $PWD $PBS_O_WORKDIR/$PBS_JOBID
cd $PBS_O_WORKDIR

# Get node list from PBS and format for job:
jobnodes=`uniq -c ${PBS_NODEFILE} | awk -F. '{print $1 }' | awk '{print $2}' | paste -s -d " "`

thishost=`uname -n | awk -F. '{print $1.}'`
thishostip=`hostname -i`
rayport=6379

thishostNport="${thishostip}:${rayport}"
redis_password=$(uuidgen)

echo "Allocate Nodes = <$jobnodes>"
for n in `echo ${jobnodes}`
do
        if [[ ${n} == "${thishost}" ]]
        then
                echo "first allocate node - use as headnode ..."
                ray start --head --redis-port=6379 \
                --redis-password=$redis_password \
                --num-cpus 4 --num-gpus 1
                sleep 5
        else
                ssh ${n} $PBS_O_WORKDIR/startWorkerNode_gpu.sh ${thishostNport} "${redis_password}" 
                sleep 10
        fi
done 

python3 ~/ai-economist-ppo-decision-tree/ai-economist/tutorials/rllib/training_2_algos.py --run-dir ~/ai-economist-ppo-decision-tree/ai-economist/tutorials/rllib/experiments/check/phase1_gpu/ --pw $redis_password --ip_address $thishostNport

ray stop
deactivate
rm $PBS_O_WORKDIR/$PBS_JOBID
#