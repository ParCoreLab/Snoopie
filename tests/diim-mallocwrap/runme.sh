#!/usr/bin/env bash

. ~/envs/cuda-11-nccl-nvshmem.sh

LD_PRELOAD="$HOME/proj/nvbit/nvbit_release/src/mem_multigpu/libmem_multigpu.so" KERNEL_NAME="simple_kernel" ./diim
