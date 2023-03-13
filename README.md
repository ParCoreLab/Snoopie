# Snoopie: A Multi-GPU communication profiler

## Features:
* [x] Track Device Initiated Remote Memory Operations
* [X] Track Specific Kernels
* [ ] Track Specific Thread blocks
* [ ] Track Host Initiated Remote Memory Operations
* [ ] Attribute Objects to the tracked Memory Operations
* [-] Track NCCL Remote Memory Operations
* [ ] Track NVSHMEM Remote Memory Operations

## Build

```bash
$ git clone https://github.com/mktip/nvbit-profiler
$ cd tools/mem_multigpu/ && ARCH=80 make # builds mem_multigpu.so
```

## Usage

```
$ LD_PRELOAD="/path/to/mem_multigpu.so" KERNEL_NAME="kernel_name_to_track" ./multigpu-app
```

> Note: You can use "all" for the `KERNEL_NAME` variable to track all kernels


> Note: To find the exact name of the kernel you want to track, you can compile
> and use the `./tools/cudaops/` to profile your application to get the kernel
> names within it
