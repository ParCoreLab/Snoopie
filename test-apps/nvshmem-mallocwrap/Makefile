NP ?= 2
NVCC ?= nvcc
MPIRUN ?= mpirun
MPICCX = mpic++
CXX ?= g++

GENCODE_SM80    := -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80
GENCODE_FLAGS	:= $(GENCODE_SM80)

NVCC_NV_FLAGS = -dc -Xcompiler -fopenmp $(GENCODE_FLAGS) -std=c++17 -ccbin=$(MPICCX) -I$(NVSHMEM_HOME)/include -I$(MPI_HOME)/include -I./include_nvshmem 
NVCC_NV_LDFLAGS = -ccbin=$(MPICCX) -lgomp -L$(CUDA_HOME)/lib64 -lcuda -lcudart -lnvidia-ml -L$(NVSHMEM_HOME)/lib -lnvshmem_host -lnvshmem_device -L$(MPI_HOME)/lib -lmpi -L$(UCX_HOME)/lib -lucp -lucs -luct -lucm -lmlx5 -L. -lwrapper -lineinfo
#NVCC_NV_LDFLAGS = -ccbin=$(MPICCX) -lgomp -L$(CUDA_HOME)/lib64 -lcuda -lcudart -lnvidia-ml -L$(NVSHMEM_HOME)/lib -lnvshmem -L$(MPI_HOME)/lib -lmpi -L$(UCX_HOME)/lib -lucp -lucs -luct -lucm -lmlx5 -L. -lwrapper -lineinfo

MAKEFLAGS += -j

all: main main.ptx 

main: libwrapper.so main.o 
	$(NVCC) -g $(GENCODE_FLAGS) -o $@ $^ $(NVCC_NV_LDFLAGS)
	

main.o: main.cu
	nvcc -cubin -lineinfo main.cu
	nvdisasm --print-line-info main.cubin > testfile.txt
	$(NVCC) $(NVCC_NV_FLAGS) -o $@ $<

main.ptx: main.cu
	$(NVCC) --ptx $(NVCC_NV_FLAGS) -o $@ $< $(NVCC_NV_LDFLAGS)

libwrapper.so: wrapper.o
	$(NVCC) -g -shared --generate-line-info -o $@ $< -lineinfo

wrapper.o: wrapper.cpp
	$(CXX) -g -c -Wall -Werror -fPIC $<  -L/usr/local/cuda/lib64 -lcudart -I/usr/local/cuda/include -o $@ -lineinfo

run_main: main
	mpirun -x KERNEL_NAME="void nvshmemi_transfer_rma_p<int>" -x LD_LIBRARY_PATH=/usr/local/cuda/lib64:.:$(NVSHMEM_HOME)/lib:$$LD_LIBRARY_PATH -x LD_PRELOAD="../../tools/mem_multigpu/mem_multigpu.so" CODE_ATTRIBUTION=1 -np 2 ./main

clean:
	rm -f *.so *.o main main.ptx *txt *cubin
