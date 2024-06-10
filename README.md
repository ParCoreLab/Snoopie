# Snoopie: A Multi-GPU communication profiler

With data movement becoming one of the most expensive
bottlenecks in computing, the need for profiling tools to
analyze communication becomes crucial for effectively scaling multi-GPU applications. While existing profiling tools
including first-party software by GPU vendors are robust
and excel at capturing compute operations within a single GPU, support for monitoring GPU-GPU data transfers
and calls issued by communication libraries is currently
inadequate. To fill these gaps, we introduce Snoopie, an
instrumentation-based multi-GPU communication profiling
tool built on NVBit, capable of tracking peer-to-peer transfers and GPU-centric communication library calls. To increase programmer productivity, Snoopie can attribute data
movement to the source code line and the data objects involved. It comes with multiple visualization modes at varying
granularities, from a coarse view of the data movement in the
system as a whole to specific instructions and addresses. Our
case studies demonstrate Snoopie’s effectiveness in monitoring data movement, locating performance bugs in applica-
tions, and understanding concrete data transfers abstracted
beneath communication libraries

![Snoopie](https://github.com/ParCoreLab/snoopie/assets/45905717/2c2e73f4-2f8d-47ca-b4a7-f830d7216640)


## Dependencies
* CMake 3.23 or above
* CUDA 11.8 or above
* NVIDIA driver 545.23.08
* OPENMPI 4.1.4 or above
* NVSHMEM 2.7 or above
* NVBit 1.5.5
* Python 3
* Python 3 development library (python3-dev in Ubuntu/Debian)
* Zstandard development library (libzstd-dev in Ubuntu/Debian)
* libunwind development library (libunwind-dev Ubuntu/Debian)
* NumPy

## Build

```bash
$ cmake .
$ make snoop
$ . ./snoopie_path.sh
```

## Easy Docker usage
1. Build the image
```bash
docker build -t snoopie .
```

2. Create a container
```bash
docker run --gpus all -it --rm -p 8000:8000 snoopie:latest
```

3. Run a sample application
```bash
cd /snoopie
snoop.py build/tests/stencil/stencil-p2p_base/stencil-p2p_base -- -niter 1
```

4. Open `localhost:8000` in your browser and upload the generated log files

Alternatively, run your own application
```bash
docker run -it --rm \
	--gpus all \
	--user $(id -u):$(id -g) \
	--volume $(pwd)/src:/src/ \
	--workdir /src \
	snoopie:latest

# Compile your code
snoop.py ./your-app
```

## Usage

```
$ LD_PRELOAD="/path/to/mem_multigpu.so" KERNEL_NAME="kernel_name_to_track" /path/to/multigpu-app
#or
$ ./snoop --kernel-name kernel_name_to_track -- /path/to/multigpu-app
```
> To enable source code line attribution, pass CODE_ATTRIBUTION=1 environment variable as follow.

```
$ LD_PRELOAD="/path/to/mem_multigpu.so" KERNEL_NAME="kernel_name_to_track" CODE_ATTRIBUTION=1 /path/to/multigpu-app
#or
$ ./snoop --kernel-name kernel_name_to_track -- /path/to/multigpu-app
```
> To run with NVSHMEM version besides 2.8 and 2.9

```
$ LD_PRELOAD="/path/to/mem_multigpu.so" NVSHMEM_VERSION="2.7" KERNEL_NAME="kernel_name_to_track" CODE_ATTRIBUTION=1 /path/to/multigpu-app
#or
$ ./snoop --nvshmem-version 2.7 --kernel-name kernel_name_to_track -- /path/to/multigpu-app
```
> When running with nvshmem version 2.8 or 2.9, you need to specify the number of GPUs being used as follows

```
$ LD_PRELOAD="/path/to/mem_multigpu.so" NVSHMEM_NGPUS="4" KERNEL_NAME="kernel_name_to_track" CODE_ATTRIBUTION=1 /path/to/multigpu-app
#or
$ ./snoop --nvshmem-ngpus 4 --kernel-name kernel_name_to_track -- /path/to/multigpu-app
```
> To enable sampling-based profiling, pass SAMPLE_SIZE=<sample_size> environment variable as follow.

```
$ LD_PRELOAD="/path/to/mem_multigpu.so" KERNEL_NAME="kernel_name_to_track" SAMPLE_SIZE=<sample_size> /path/to/multigpu-app
#or
$ ./snoop --sample-size <sample_size> --kernel-name kernel_name_to_track -- /path/to/multigpu-app
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



# Citing


```bibtex
@inproceedings{Snoopie,
  author = {Issa, Mohammad Kefah Taha and Sasongko, Muhammad Aditya and Turimbetov, Ilyas and Baydamirli, Javid and Sa\u{g}bili, Do\u{g}an and Unat, Didem},
  title = {Snoopie: A Multi-GPU Communication Profiler and Visualizer},
  year = {2024},
  url = {https://doi.org/10.1145/3650200.3656597},
  doi = {10.1145/3650200.3656597},
  booktitle = {Proceedings of the 38th International Conference on Supercomputing},
  series = {ICS '24}
}
```

# Acknowledgment
> [!NOTE]
> This project has received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (grant agreement No 949587).

