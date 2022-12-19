#!/bin/sh
### Set the job name (for your reference)
#PBS -N roberta_large_train
### Set the project name, your department code by default
#PBS -P col774
### Request email when job begins and ends
#PBS -m bea
### Specify email address to use for notification.
#PBS -M $USER@iitd.ac.in
####
#PBS -l select=1:ncpus=2:ngpus=2:mem=16G
### Specify "wallclock time" required for this job, hhh:mm:ss
#PBS -l walltime=02:00:00

#PBS -l software=PYTORCH
# After job starts, must goto working directory. 
# $PBS_O_WORKDIR is the directory from where the job is fired. 
echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="
cd $PBS_O_WORKDIR
# proxy activate
cd ../proxy/
python iitdlogin.py ani.cred&
proxy_pid=$!
cd $PBS_O_WORKDIR
#job 
export http_proxy=http://10.10.78.22:3128
export https_proxy=http://10.10.78.22:3128

module load apps/anaconda/3
conda activate dl_35
module unload apps/anaconda/3

python3 roberta.py
kill $proxy_pid
#time -p mpirun -n {n*m} executable
#NOTE
# The job line is an example : users need to change it to suit their applications
# The PBS select statement picks n nodes each having m free processors
# OpenMPI needs more options such as $PBS_NODEFILE
