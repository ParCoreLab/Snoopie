NP ?= 2
NVCC ?= nvcc
MPIRUN ?= mpirun
MPICCX = mpic++
CXX ?= g++

GENCODE_SM80    := -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80
GENCODE_FLAGS	:= $(GENCODE_SM80)

NVCC_NV_FLAGS = -dc -Xcompiler -fopenmp $(GENCODE_FLAGS) -std=c++17 -ccbin=$(MPICCX) -I$(NVSHMEM_HOME)/include -I$(MPI_HOME)/include -I./include_nvshmem
#NVCC_NV_LDFLAGS = -ccbin=$(MPICCX) -lgomp -L$(CUDA_HOME)/lib64 -lcuda -lcudart -lnvidia-ml -L$(NVSHMEM_HOME)/lib -lnvshmem -L$(MPI_HOME)/lib -lmpi -L$(UCX_HOME)/lib -lucp -lucs -luct -lucm -lmlx5
NVCC_NV_LDFLAGS = -ccbin=$(MPICCX) -lgomp -L$(CUDA_HOME)/lib64 -lcuda -lcudart -lnvidia-ml -L$(NVSHMEM_HOME)/lib -lnvshmem_host -lnvshmem_device -L$(MPI_HOME)/lib -lmpi -L$(UCX_HOME)/lib -lucp -lucs -luct -lucm -lmlx5 -lineinfo

MAKEFLAGS += -j

all: main main.ptx 

main: main.o 
	$(NVCC) -g $(GENCODE_FLAGS) -o $@ $^ $(NVCC_NV_LDFLAGS)
	

main.o: main.cu
	#$(NVCC) -I$(NVSHMEM_HOME)/include -cubin -lineinfo main.cu
	#nvdisasm --print-line-info main.cubin > testfile.txt
	$(NVCC) $(NVCC_NV_FLAGS) -o $@ $< -lineinfo
	cuobjdump $@ -xelf all
	nvdisasm --print-line-info main.sm_80.cubin > testfile.txt
	#nvdisasm --print-line-info $@ > testfile.txt

main.ptx: main.cu
	$(NVCC) --ptx $(NVCC_NV_FLAGS) -o $@ $< $(NVCC_NV_LDFLAGS)


run_main: main
	$(MPIRUN) -np $(NP) ./main

clean:
	rm -f main.o main main.o.ptx main.ptx *cubin *txt
