#!/bin/bash -l

source $HOME/.bashrc
cd $PBS_O_WORKDIR

param1=$1
param2=$2 # redis_pwd

thishost=`uname -n`
thishostip=`hostname -i`

echo node NAME: ${thishost}
echo node IP: ${thishostip}
echo dest IP: ${param1}

source ~/venv/ai-economist/bin/activate

ray start --address=${param1} --redis-password=${param2} # --num-gpus=1

# Here, sleep for the duration of the job, so ray does not stop

# WALLTIME=$(qstat -f $PBS_JOBID | sed -rn 's/.*Resource_List.walltime = (.*)/\1/p')
# SECONDS=`echo $WALLTIME | awk -F: '{ print ($1 * 3600) + ($2 * 60) + $3 }'`

# echo "SLEEPING FOR $SECONDS s"
# sleep $SECONDS

echo exiting ${thishost}
#