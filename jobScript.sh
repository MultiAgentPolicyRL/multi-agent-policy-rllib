#!/bin/bash

#PBS -N Exp_2_Phase1_2AlgosTab
#PBS -l select=2:ncpus=15:mem=10gb:ngpus=1:mpiprocs=1
#PBS -l walltime=3:00:00
#PBS -j oe 

# Queues | common_gpuQ | short_gpuQ
#PBS -q short_gpuQ

# Email references
#PBS -m abe
#PBS -M ettore.saggiorato@studenti.unitn.it
#PBS -M ettoreitaut4.1@gmail.com


# then start limiting ram usage
# IDK
# #PBS -P CSCIxxxx

ln -s $PWD $PBS_O_WORKDIR/$PBS_JOBID

cd $PBS_O_WORKDIR

jobnodes=`uniq -c ${PBS_NODEFILE} | awk -F. '{print $1 }' | awk '{print $2}' | paste -s -d " "`

thishost=`uname -n | awk -F. '{print $1.}'`
thishostip=`hostname -i`
rayport=3679

thishostNport="${thishostip}:${rayport}"
redis_password=$(uuidgen)

# dashboard_port=3752
# echo "Dashboard will use port: " $dashboard_port
export PORT=dashboard_port
export HEAD_NODE=thishost

echo "HEAD NODE: " $HEAD_NODE
echo "Allocate Nodes = <$jobnodes>"
# export thishostNport
 
echo "set up ray cluster..." 
echo 
echo 
J=0
for n in `echo ${jobnodes}`
do
        echo Working with node $n
        if [[ ${n} == "${thishost}" ]]
        then
                echo "first allocate node - use as headnode ..."
                source ~/venv/ai-economist/bin/activate
                # https://docs.ray.io/en/latest/cluster/vms/user-guides/large-cluster-best-practices.html#configuring-the-head-node
                ray start --head --redis-port=$rayport --redis-password=$redis_password --num-gpus 1 # --webui-host 127.0.0.1 # --resources {"CPU": 0} --num-gpus 1  --memory 10000000000 --object-store-memory 10000000000 --num-cpus 4 --num-gpus 1
                sleep 5
                echo 
        else
                echo "then allocate other nodes: " $J
                # Run pbsdsh on the J'th node, and do it in the background.
                pbsdsh -n $J -s $PBS_O_WORKDIR/startWorkerNode.sh ${thishostNport} ${redis_password} &
                # c'era il -v
                sleep 10
                echo 
        fi
J=$((J+1))
done 

echo "done, now launching python program"

source ~/venv/ai-economist/bin/activate

python3 -u ~/ai-economist-ppo-decision-tree/trainer/training_2_algos.py --run-dir ~/ai-economist-ppo-decision-tree/trainer/experiments/check/phase1/Exp_2_Phase1_2AlgosTab --pw $redis_password --ip_address $thishostNport --cluster True



#cd ~/ai-economist-ppo-decision-tree/trainer/
# python3 -u ~/ai-economist-ppo-decision-tree/ai-economist/tutorials/rllib/training_2_algos.py --run-dir ~/ai-economist-ppo-decision-tree/ai-economist/tutorials/rllib/experiments/check/phase1/ --pw $redis_password --ip_address $thishostNport
# python3 -u training_2_algos.py --run-dir experiments/check/phase1/Exp_2_Phase1_2Algos/ --pw $redis_password --ip_address $thishostNport --cluster True
# python3 -u training_2_algos.py --run-dir experiments/check/phase1 --pw $redis_password --ip_address $thishostNport --cluster True

# python3 -u training_2_algos.py --run-dir experiments/check/phase1/Exp_2_Phase1_2AlgosTab/ --pw $redis_password --ip_address $thishostNport --cluster True
# python3 -u training_script.py --run-dir experiments/check/phase1/Exp_2_Phase1_Original --pw $redis_password --ip_address $thishostNport --cluster True

ray stop
deactivate
rm $PBS_O_WORKDIR/$PBS_JOBID