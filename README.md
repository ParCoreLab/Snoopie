# Snoopie: A Multi-GPU communication profiler

## Features:
* [x] Track Device Initiated Remote Memory Operations
* [X] Track Specific Kernels
* [X] Track Host Initiated Remote Memory Operations
* [X] Attribute Objects to the tracked Memory Operations
* [X] Track NCCL Remote Memory Operations
* [X] Track NVSHMEM Remote Memory Operations
* [ ] Track Specific Thread blocks

## Build

```bash
$ cmake .
$ make snoop
$ . ./snoopie_path.sh
```

## Usage

```
$ LD_PRELOAD="/path/to/mem_multigpu.so" KERNEL_NAME="kernel_name_to_track" /path/to/multigpu-app
#or
$ ./snoop --kernel-name kernel_name_to_track command -- /path/to/multigpu-app
```
> To enable source code line attribution, pass CODE_ATTRIBUTION=1 environment variable as follow.

```
$ LD_PRELOAD="/path/to/mem_multigpu.so" KERNEL_NAME="kernel_name_to_track" CODE_ATTRIBUTION=1 /path/to/multigpu-app
#or
$ ./snoop --kernel-name kernel_name_to_track command -- /path/to/multigpu-app
```
> To run with NVSHMEM version besides 2.8 and 2.9

```
$ LD_PRELOAD="/path/to/mem_multigpu.so" NVSHMEM_VERSION="2.7" KERNEL_NAME="kernel_name_to_track" CODE_ATTRIBUTION=1 /path/to/multigpu-app
#or
$ ./snoop --nvshmem-version 2.7 --kernel-name kernel_name_to_track command -- /path/to/multigpu-app
```
> When running with nvshmem version 2.8 or 2.9, you need to specify the number of GPUs being used as follows

```
$ LD_PRELOAD="/path/to/mem_multigpu.so" NVSHMEM_NGPUS="4" KERNEL_NAME="kernel_name_to_track" CODE_ATTRIBUTION=1 ./multigpu-app
#or
$ ./snoop --nvshmem-ngpus 4 --kernel-name kernel_name_to_track command -- /path/to/multigpu-app
```
> To enable sampling-based profiling, pass SAMPLE_SIZE=<sample_size> environment variable as follow.

```
$ LD_PRELOAD="/path/to/mem_multigpu.so" KERNEL_NAME="kernel_name_to_track" SAMPLE_SIZE=<sample_size> ./multigpu-app
#or
$ ./snoop --sample-size <sample_size> --kernel-name kernel_name_to_track command -- /path/to/multigpu-app
```
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
> (Optional) For interactive source code folder picking needs tkinter
```
sudo apt install python3-tk -y
```

### Usage
```
usage:
streamlit run /path/to/parse_and_vis.py **files** -- [optional arguments]

for example:
streamlit run visualizer/parse_and_vis.py logs/stencil-p2p_base_run0/* -- --gpu-num 4 --src-code-folder tests/stencil/stencil-p2p_base --sampling-period 1

all arguments are optional (can use GUI to provide log files and options)

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
