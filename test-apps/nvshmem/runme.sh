#!/usr/bin/env bash

. ./cuda-11-nccl-nvshmem.sh

export NVSHMEM_NVTX=common



# mpirun -x LD_PRELOAD="/home/missa18/proj/nvbit/nvbit_release/tools/cudaops/cudaops.so" -np 4 ./main
# mpirun -x LD_PRELOAD="/home/missa18/proj/nvbit/nvbit_release/tools/mem_trace/mem_trace.so" -np 2 ./main
mpirun -x KERNEL_NAME="simple_shift" -x LD_PRELOAD="../../tools/mem_multigpu/mem_multigpu.so" -np 2 ./main
