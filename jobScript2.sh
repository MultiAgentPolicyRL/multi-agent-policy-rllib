#!/bin/bash

#PBS -N Exp_2_Phase1_Original
#PBS -l select=2:ncpus=15:mem=10gb:ngpus=1:mpiprocs=1
#PBS -l walltime=05:00:00
#PBS -j oe 

# Queues | common_gpuQ | short_gpuQ
#PBS -q short_gpuQ

# Email references
#PBS -m abe
#PBS -M ettore.saggiorato@studenti.unitn.it
#PBS -M ettoreitaut4.1@gmail.com


source ~/venv/ai-economist/bin/activate
# python3 -u sample_code_for_ray.py --ip_address ${thishostNport} --pw ${redis_password}
# cd ~/ai-economist-ppo-decision-tree/trainer
# python3 -u ~/ai-economist-ppo-decision-tree/ai-economist/tutorials/rllib/training_2_algos.py --run-dir ~/ai-economist-ppo-decision-tree/ai-economist/tutorials/rllib/experiments/check/phase2/ --pw $redis_password --ip_address $thishostNport
# python3 -u training_2_algos.py --run-dir experiments/check/phase1/Exp_2_Phase1_2Algos --pw $redis_password --ip_address $thishostNport --cluster True
# python3 -u training_2_algos.py --run-dir experiments/check/phase1/Exp_2_Phase1_2AlgosTab --pw $redis_password --ip_address $thishostNport --cluster True

python3 -u ~/ai-economist-ppo-decision-tree/trainer/training_script.py --run-dir ~/ai-economist-ppo-decision-tree/trainer/experiments/check/phase1/Exp_2_Phase1_Original


# python3 -u training_script.py --run-dir experiments/check/phase1/Exp_2_Phase1_Original