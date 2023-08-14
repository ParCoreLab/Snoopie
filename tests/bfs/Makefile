NP ?= 4
NVCC ?= nvcc
MPIRUN ?= mpirun
MPICCX = mpic++
CXX ?= g++
NITER ?= 200
SIZE ?= 1024

GENCODE_SM80    := -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80
GENCODE_FLAGS	:= $(GENCODE_SM80)

NVCC_NV_FLAGS = -dc -Xcompiler -fopenmp $(GENCODE_FLAGS) -std=c++17 -ccbin=$(MPICCX) -I$(NVSHMEM_HOME)/include -I$(MPI_HOME)/include -I./include_nvshmem 
NVCC_NV_LDFLAGS = -ccbin=$(MPICCX) -lgomp -L$(CUDA_HOME)/lib64 -lcuda -lcudart -lnvidia-ml -L$(NVSHMEM_HOME)/lib -lnvshmem_host -lnvshmem_device -L$(MPI_HOME)/lib -lmpi -L$(UCX_HOME)/lib -lucp -lucs -luct -lucm -lmlx5 -L. -lwrapper -lineinfo
#NVCC_NV_LDFLAGS = -ccbin=$(MPICCX) -lgomp -L$(CUDA_HOME)/lib64 -lcuda -lcudart -lnvidia-ml -L$(NVSHMEM_HOME)/lib -lnvshmem -L$(MPI_HOME)/lib -lmpi -L$(UCX_HOME)/lib -lucp -lucs -luct -lucm -lmlx5 -L. -lwrapper -lineinfo

MAKEFLAGS += -j

bfs: libwrapper.so bfs.o 
	$(NVCC) -g $(GENCODE_FLAGS) -o $@ $^ $(NVCC_NV_LDFLAGS)
	

bfs.o: bfs_basic.cu
	#nvcc -cubin -lineinfo main.cu
	#nvdisasm --print-line-info main.cubin > testfile.txt
	$(NVCC) $(NVCC_NV_FLAGS) -o $@ $< -lineinfo
	cuobjdump $@ -xelf all
	nvdisasm --print-line-info bfs*.sm_80.cubin > testfile.txt

libwrapper.so: wrapper.o
	$(NVCC) -g -shared --generate-line-info -o $@ $< -lineinfo

wrapper.o: wrapper.cpp
	$(CXX) -g -c -Wall -Werror -fPIC $<  -L/usr/local/cuda/lib64 -lcudart -I/usr/local/cuda/include -o $@ -lineinfo

clean:
	rm -f *.so *.o jacobi *txt *cubin *zst

all:	bfs bfs_default_run 

#bfs_compile:
#	#nvcc -o test test.cu -l./wrapper.a -cudart shared
#	nvcc -Xcompiler -fPIC -c wrapper.cpp -gencode arch=compute_80,code=sm_80 -I/usr/local/cuda/include -o wrapper.o -lineinfo
#	nvcc -g -L. --generate-line-info -o bfs bfs_basic.cu -lwrapper -lineinfo
#	nvcc -ptx bfs_basic.cu
#	nvcc -cubin -lineinfo bfs_basic.cu
#	nvdisasm --print-line-info bfs_basic.cubin > testfile.txt

bfs_run:
	LD_LIBRARY_PATH=.:$$LD_LIBRARY_PATH LD_PRELOAD=../../tools/mem_multigpu/mem_multigpu.so KERNEL_NAME="all" CODE_ATTRIBUTION=1 SAMPLE_SIZE=10 ./bfs -f gupta2/gupta2.mtx -n 4 -b 10 -t 512 -v 0 -m 0 -r 1 -o 1
	#LD_LIBRARY_PATH=/usr/local/cuda/lib64:.:$$LD_LIBRARY_PATH LD_PRELOAD=../../tools/mem_multigpu/mem_multigpu.so ./device-initiated_implicit_memcpy

bfs_default_run:
	LD_LIBRARY_PATH=.:$$LD_LIBRARY_PATH ./bfs -f gupta2/gupta2.mtx -n 4 -b 10 -t 512 -v 0 -m 0 -r 1 -o 1

clean:
	rm -f *.so *.o bfs *txt *cubin *zst
