#!/usr/bin/env bash

. ~/envs/cuda-11-nccl-nvshmem.sh

export NVSHMEM_NVTX=common
# export NCCL_DEBUG=info


# TOOL_VERBOSE=1 KERNEL_NAME="ncclKernel_SendRecv_RING_SIMPLE_Sum_int8_t" LD_PRELOAD="$HOME/proj/nvbit/nvbit_release/tools/noop/noop.so" ./main2
# TOOL_VERBOSE=1 KERNEL_NAME="ncclKernel_SendRecv_RING_SIMPLE_Sum_int8_t" LD_PRELOAD="$HOME/proj/nvbit/nvbit_release/tools/mem_multigpu/mem_multigpu.so" ./main2


# mpirun -x LD_PRELOAD="/home/missa18/proj/nvbit/nvbit_release/tools/cudaops/cudaops.so" -np 4 ./main
mpirun -x KERNEL_NAME="void nvshmemi_transfer_rma_p<int>" -x LD_PRELOAD="$HOME/proj/nvbit/nvbit_release/tools/mem_multigpu/mem_multigpu.so" -np 2 ./main
# mpirun -x LD_PRELOAD="/home/missa18/proj/nvbit/nvbit_release/tools/mem_trace/mem_trace.so" -np 2 ./main
