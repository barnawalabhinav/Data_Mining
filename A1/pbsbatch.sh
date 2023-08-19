#!/bin/sh
### Set the job name (for your reference)
#PBS -N Data_Mining
### Set the project name, your department code by default
#PBS -P cse
### Request email when job begins and ends, don't change anything on the below line 
#PBS -m bea
### Specify email address to use for notification, don't change anything on the below line
#PBS -M $USER@iitd.ac.in
#### Request your resources, just change the numbers
#PBS -l select=1:ncpus=1:ngpus=0:mem=16G
### Specify "wallclock time" required for this job, hhh:mm:ss
#PBS -l walltime=05:00:00
#PBS -l software=C++

# After job starts, must goto working directory. 
# $PBS_O_WORKDIR is the directory from where the job is fired. 
echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="
cd $PBS_O_WORKDIR
echo $PBS_O_WORKDIR

### module () {
###         eval `/usr/share/Modules/$MODULE_VERSION/bin/modulecmd bash $*`
### }

### module load apps/anaconda/3
### source activate ~/myenv
### module unload apps/anaconda/3

module purge
module load compiler/gcc/9.1.0
### module load compiler/cuda/9.2/compilervars
### module load suite/nvidia-hpc-sdk/20.11/cuda11.0

### module load compiler/python/3.6.0/ucs4/gnu/447
### module load pythonpackages/3.6.0/ucs4/gnu/447/pip/9.0.1/gnu
### module load pythonpackages/3.6.0/ucs4/gnu/447/setuptools/34.3.2/gnu
### module load pythonpackages/3.6.0/ucs4/gnu/447/wheel/0.30.0a0/gnu
### module load pythonpackages/3.6.0/numpy/1.16.1/gnu
### module load pythonpackages/3.6.0/pandas/0.23.4/gnu
### module load compiler/cuda/9.2/compilervars
### module load compiler/gcc/9.1.0
### module load apps/pythonpackages/3.6.0/tensorflow/1.9.0/gpu
### module load pythonpackages/3.6.0/tensorflow_tensorboard/1.10.0/gnu
### module load apps/pythonpackages/3.6.0/keras/2.2.2/gpu
### module load pythonpackages/3.6.0/tqdm/4.25.0/gnu
### module load compiler/gcc/6.5/openmpi/4.0.2
### module load compiler/gcc/9.1/openmpi/4.1.2

### module load apps/gromacs/2019.4/intel
### export OMP_NUM_THREADS=4
### mpirun -np 64 ./a2 --taskid=1 --inputpath=test14/test-input-14.gra --headerpath=test14/test-header-14.dat --outputpath=output-14.txt --verbose=1 --startk=2 --endk=2 --p=20

time -v sh run.sh A1_datasets/D_large.dat
