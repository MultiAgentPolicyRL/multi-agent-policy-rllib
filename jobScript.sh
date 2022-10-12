#!/bin/bash
#PBS -N Exp_1_TEST2_Phase1
#PBS -l select=2:ncpus=15:mem=10gb:ngpus=1:mpiprocs=1

#PBS -m abe
#PBS -M ettore.saggiorato@studenti.unitn.it
#PBS -M ettoreitaut4.1@gmail.com

# #PBS -N Experiment_5_5_nodes_15_cpu_num_workers_5 ### 11 done in 10 mins
# #PBS -N Experiment_5_20_nodes_5_cpu_num_workers_20 ### 3 done in 10 mins
# #PBS -N Experiment_5_20_nodes_15_cpu_num_workers_20 ### 11 done in 10 mins

# #PBS -N Experiment_6_10_nodes_15_cpu_num_workers_10 ### 4 done in 10 mins
# #PBS -N Experiment_6_4_nodes_15_cpu_num_workers_4_gpu_1_gpus_per_worker_0_25_gpus_1 ### 7 done in 10 mins
# #PBS -N Experiment_6_4_nodes_15_cpu_num_workers_4_gpu_1_gpus_per_worker_0_25 ### 7 done in 10 mins
# #PBS -N Experiment_6_4_nodes_15_cpu_num_workers_4_gpu_1_gpus_per_worker_0 ### 12 done in 10 mins
# #PBS -N Experiment_6_4_nodes_15_cpu_num_workers_4_gpu_1_gpus_per_worker_0 ### 39 done in 30 mins
# #PBS -N Experiment_6_4_nodes_5_cpu_num_workers_4_gpu_1_gpus_per_worker_0 ### 7 done in 10 mins
# #PBS -N Experiment_6_2_nodes_15_cpu_num_workers_2_gpu_1_gpus_per_worker_0 ### doesn't work

# #PBS -N Experiment_7_2_nodes_15_cpu_num_workers_2_gpu_1_gpus_per_worker_0 ### 12 done in 10 mins
# #PBS -N Experiment_7_2_nodes_15_cpu_num_workers_1_gpu_1_gpus_per_worker_0 ### 8 done in 10 mins

# #PBS -N Experiment_8_4_nodes_15_cpu_num_workers_3_gpu_1_gpus_per_worker_0 ### 8 done in 10 mins
# #PBS -N Experiment_8_2_nodes_15_cpu_num_workers_2_gpu_1_gpus_per_worker_0


# then start limiting ram usage

# #PBS -q common_gpuQ
#PBS -q short_gpuQ

#PBS -l walltime=00:10:00
#PBS -j oe  

# #PBS -l select=2:ncpus=24:mpiprocs=1
# #PBS -P CSCIxxxx
# #PBS -m abe
# #PBS -M xxxxx@gmail.com

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
# python3 -u sample_code_for_ray.py --ip_address ${thishostNport} --pw ${redis_password}
cd ~/ai-economist-ppo-decision-tree/trainer
# python3 -u ~/ai-economist-ppo-decision-tree/ai-economist/tutorials/rllib/training_2_algos.py --run-dir ~/ai-economist-ppo-decision-tree/ai-economist/tutorials/rllib/experiments/check/phase2/ --pw $redis_password --ip_address $thishostNport
python3 -u training_2_algos.py --run-dir experiments/check/phase2/ --pw $redis_password --ip_address $thishostNport

ray stop
deactivate
rm $PBS_O_WORKDIR/$PBS_JOBID