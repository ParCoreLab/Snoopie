FROM nvcr.io/nvidia/nvhpc:24.3-devel-cuda12.3-ubuntu22.04

# Where snoopie is installed
ENV SNOOPIE_HOME="/snoopie/"

# Path of the .so file for LD_PRELOAD
ENV SNOOPIE_PATH="${SNOOPIE_HOME}/build/src/mem_multigpu/libmem_multigpu.so"

ARG GIT_COMMIT
ENV GIT_COMMIT="${GIT_COMMIT}"

WORKDIR "$SNOOPIE_HOME"
COPY . "$SNOOPIE_HOME"

# Install Snoopie dependencies
RUN apt-get update && apt-get install -y python3-pip python3-dev libzstd-dev libunwind-dev
RUN python3 -m pip install --upgrade pip
RUN pip3 install numpy

# Compile Snoopie
RUN rm -rf build/
RUN mkdir build/
RUN cmake -DCMAKE_CXX_COMPILER=/usr/bin/g++ -B build/
RUN cmake --build build/

# Add cli to path
RUN rm -rf bin/
RUN mkdir bin/
RUN cp src/main.py bin/snoop.py
RUN chmod ugo+ bin/snoop.py
ENV PATH="${PATH}:${SNOOPIE_HOME}/bin"

# Install Visualiser dependencies
RUN cd visualizer && python3 -m pip install -r requirements.txt

# Start the visualiser and a shell
#CMD sh -c "streamlit run --server.port 8000 ./visualizer/streamlit_app.py & sleep 4 && bash"
CMD sh -c "streamlit run --server.port 8000 ./visualizer/parse_and_vis.py -- --sampling-period 1 & sleep 4 && bash"
