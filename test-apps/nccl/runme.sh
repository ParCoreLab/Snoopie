#!/usr/bin/env bash

. ~/envs/cuda-11-nccl-nvshmem.sh

# If more verbosity is needed, uncomment the following line:
# export NCCL_DEBUG=info


# To test the overhead of a "noop" instrumentation":
# TOOL_VERBOSE=1 KERNEL_NAME="ncclKernel_SendRecv_RING_SIMPLE_Sum_int8_t" LD_PRELOAD="$HOME/proj/nvbit/nvbit_release/tools/noop/noop.so" ./main

# To list the cuda operations in the binary:
# TOOL_VERBOSE=1 KERNEL_NAME="ncclKernel_SendRecv_RING_SIMPLE_Sum_int8_t" LD_PRELOAD="$HOME/proj/nvbit/nvbit_release/tools/cudaops/cudaops.so" ./main

TOOL_VERBOSE=1 KERNEL_NAME="ncclKernel_SendRecv_RING_SIMPLE_Sum_int8_t" LD_PRELOAD="$HOME/proj/nvbit/nvbit_release/tools/mem_multigpu/mem_multigpu.so" ./main
