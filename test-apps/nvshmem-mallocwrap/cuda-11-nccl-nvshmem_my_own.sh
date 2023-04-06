. ~/.spack/Spack/share/spack/setup-env.sh

spack load nvshmem@2.7.0-6
spack load /4dc2b3r

export  UCX_WARN_UNUSED_ENV_VARS=n
export NVSHMEM_HOME=$SPACK_ROOT/opt/spack/linux-ubuntu20.04-zen2/gcc-11.1.0/nvshmem-2.7.0-6-gwjkk2hhqwszpubr62ywj4bkaeka4qub/
export MPI_HOME=$SPACK_ROOT/opt/spack/linux-ubuntu20.04-zen2/gcc-11.1.0/openmpi-4.1.5-jtjtnnckak2g7z2itpcqostiqnthg4wk/
export CUDA_HOME=$SPACK_ROOT/opt/spack/linux-ubuntu20.04-zen2/gcc-11.1.0/cuda-11.8.0-l6jcnhuigsdga6urrrhahmysv4klqa47/
export UCX_HOME=$SPACK_ROOT/opt/spack/linux-ubuntu20.04-zen2/gcc-11.1.0/ucx-1.14.0-dgytw2yvlffwxyvvcllzh5i7y4iiix37/
export LD_LIBRARY_PATH="$NVSHMEM_HOME/lib:$UCX_HOME/lib:$CUDA_HOME/lib64:$MPI_HOME/lib"
echo $SPACK_ROOT
