#!/usr/bin/env bash

. ~/envs/cuda-11-nccl-nvshmem.sh

# export NVSHMEM_NVTX=common
# export NCCL_DEBUG=info


# TOOL_VERBOSE=1 KERNEL_NAME="ncclKernel_SendRecv_RING_SIMPLE_Sum_int8_t" LD_PRELOAD="$HOME/proj/nvbit/nvbit_release/tools/noop/noop.so" ./main2
# TOOL_VERBOSE=1 KERNEL_NAME="ncclKernel_SendRecv_RING_SIMPLE_Sum_int8_t" LD_PRELOAD="$HOME/proj/nvbit/nvbit_release/tools/cudaops/cudaops.so" ./main2
TOOL_VERBOSE=1 KERNEL_NAME="ncclKernel_SendRecv_RING_SIMPLE_Sum_int8_t" LD_PRELOAD="$HOME/proj/nvbit/nvbit_release/tools/mem_multigpu/mem_multigpu.so" ./main2