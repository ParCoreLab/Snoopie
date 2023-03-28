#!/usr/bin/env bash

. ~/envs/cuda-11-nccl-nvshmem.sh

LD_PRELOAD="../../tools/mem_multigpu/mem_multigpu.so" KERNEL_NAME="simple_kernel" ./diim
