. /home/missa18/spack/share/spack/setup-env.sh

spack load nvshmem@2.7.0-6
spack load /4dc2b3r

export  UCX_WARN_UNUSED_ENV_VARS=n
export NVSHMEM_HOME=$SPACK_ROOT/opt/spack/linux-ubuntu20.04-zen2/gcc-11.1.0/nvshmem-2.7.0-6-tcyd3yuwcp7y2brryqgy7cwd6f3ifbvm/
export MPI_HOME=$SPACK_ROOT/opt/spack/linux-ubuntu20.04-zen2/gcc-11.1.0/openmpi-4.1.4-bfuh6wxkio4tl6mucawvxfuvgxogwbiw/
export CUDA_HOME=$SPACK_ROOT/opt/spack/linux-ubuntu20.04-zen2/gcc-11.1.0/cuda-11.8.0-4dc2b3rpmq4qntukk7tvcb3t5uceeax5/
export UCX_HOME=$SPACK_ROOT/opt/spack/linux-ubuntu20.04-zen2/gcc-11.1.0/ucx-1.13.1-tui3lsbmbzf4dcjo2jbdfz3uieoq3ai4/
export LD_LIBRARY_PATH="$NVSHMEM_HOME/lib:$UCX_HOME/lib:$CUDA_HOME/lib64:$MPI_HOME/lib"
echo $SPACK_ROOT
