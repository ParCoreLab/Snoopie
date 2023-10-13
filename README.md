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
$ cd src/mem_multigpu/ && ARCH=80 make # builds mem_multigpu.so

# Or with Cmake
$ cmake .
$ make mem_multigpu
```

## Usage

```
$ LD_PRELOAD="/path/to/mem_multigpu.so" KERNEL_NAME="kernel_name_to_track" ./multigpu-app
```
> To enable source code line attribution, pass CODE_ATTRIBUTION=1 environment variable as follow.

$ LD_PRELOAD="/path/to/mem_multigpu.so" KERNEL_NAME="kernel_name_to_track" CODE_ATTRIBUTION=1 ./multigpu-app

> To run with NVSHMEM version besides 2.8 and 2.9

$ LD_PRELOAD="/path/to/mem_multigpu.so" NVSHMEM_VERSION="2.7" KERNEL_NAME="kernel_name_to_track" CODE_ATTRIBUTION=1 ./multigpu-app

> When running with nvshmem version 2.8 or 2.9, you need to specify the number of GPUs being used as follows

$ LD_PRELOAD="/path/to/mem_multigpu.so" NVSHMEM_NGPUS="4" KERNEL_NAME="kernel_name_to_track" CODE_ATTRIBUTION=1 ./multigpu-app

> To enable sampling-based profiling, pass SAMPLE_SIZE=<sample_size> environment variable as follow.

$ LD_PRELOAD="/path/to/mem_multigpu.so" KERNEL_NAME="kernel_name_to_track" SAMPLE_SIZE=<sample_size> ./multigpu-app

> Note: <sample_size> determines the sample size. If <sample_size> is 100, it means 1/100 of population is sampled.

> Note: You can use "all" for the `KERNEL_NAME` variable to track all kernels


> Note: To find the exact name of the kernel you want to track, you can compile
> and use the `./tools/cudaops/` to profile your application to get the kernel
> names within it

## Visualizer

### Installation and requirements

Language: python 3.7+
> Required streamlit python libraries: streamlit, streamlit_agraph, streamlit-aggrid,
>                                      extra_streamlit_components, streamlit_plotly_events, st-clickable-images
> Other required libraries: seaborn, pandas, plotly, zstandard
```
pip install seaborn pandas plotly streamlit streamlit_agraph streamlit-aggrid extra_streamlit_components streamlit_plotly_events zstandard st-clickable-images
cd visualizer; pip install st-click-detector-0.1.3/
```

### Usage
```
usage: streamlit run /path/to/parse_and_vis.py -- [optional arguments]

Snoopie, a multigpu profiler

positional arguments:
  files                 List of logfiles. Either compressed zst or uncompressed

options:
  -h, --help            show this help message and exit
  --gpu-num GPU_NUM, -n GPU_NUM
                        Number of gpus the code was run on.
  --src-code-folder SRC_CODE_FOLDER, -s SRC_CODE_FOLDER
                        Source code folder for code attribution.
  --sampling-period SAMPLING_PERIOD, -p SAMPLING_PERIOD
                        Sampling period
```

Running without options will display a GUI for uploading files and setting parameters.

By default, the file upload limit is 200MB, to change it set the `STREAMLIT_SERVER_MAX_UPLOAD_SIZE` environment variable.

### Electron Build

See [here](./electron_builder) for electron build instructions.
