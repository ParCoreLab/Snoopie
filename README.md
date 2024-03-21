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
> To enable source code line attribution, pass CODE_ATTRIBUTION=1 environment variable as follow.

$ LD_PRELOAD="/path/to/mem_multigpu.so" KERNEL_NAME="kernel_name_to_track" CODE_ATTRIBUTION=1 ./multigpu-app

> Note: You can use "all" for the `KERNEL_NAME` variable to track all kernels


> Note: To find the exact name of the kernel you want to track, you can compile
> and use the `./tools/cudaops/` to profile your application to get the kernel
> names within it

## Visualizer

### Installation and requirements

Language: python 3.7+
Required streamlit python libraries: streamlit, streamlit_agraph, extra_streamlit_components, streamlit_plotly_events
Other required libraries: seaborn, pandas, plotly
```
pip install seaborn pandas plotly streamlit streamlit_agraph extra_streamlit_components streamlit_plotly_events zstandard
cd visualizer; pip install st-click-detector-0.1.3/
```

### Usage
```
In the directory where the log file is located
streamlit run visualizer/parse_and_vis.py <my_data_file.zst>
```
