NP ?= 4
NVCC ?= nvcc
MPIRUN ?= mpirun
MPICCX = mpic++

SIZE ?= 32

GENCODE_SM80  := -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80
GENCODE_FLAGS	:= $(GENCODE_SM80)

NVCC_NV_FLAGS = -dc -Xcompiler -fopenmp $(GENCODE_FLAGS) -std=c++17 -ccbin=$(MPICCX) -I$(NVSHMEM_HOME)/include -I$(MPI_HOME)/include -I./include_nvshmem
NVCC_NV_LDFLAGS = -ccbin=$(MPICCX) -lgomp -L$(CUDA_HOME)/lib64 -lcuda -lcudart -lnvidia-ml -L$(NVSHMEM_HOME)/lib -lnvshmem -L$(MPI_HOME)/lib -lmpi -L$(UCX_HOME)/lib -lucp -lucs -luct -lucm -lmlx5

MAKEFLAGS += -j

all: main 

main: main.o 
	$(NVCC) -g $(GENCODE_FLAGS) -o $@ $^ $(NVCC_NV_LDFLAGS)
	

main.o: main.cu
	$(NVCC) $(NVCC_NV_FLAGS) -o $@ $<

ptx: 
	$(NVCC) --ptx $(NVCC_NV_FLAGS) -o $@ $< $(NVCC_NV_LDFLAGS)


run: main
	@rm -f *.zst
	@rm -f *.txt
	@$(MPIRUN) --oversubscribe -x NVSHMEM_NGPUS=4 -x NOBANNER=1 -x LD_PRELOAD="../../tools/mem_multigpu/mem_multigpu.so" -x KERNEL_NAME="simple_shift" -np $(NP) ./main -n $(SIZE)

clean:
	rm -f main.o main main.o.ptx main.ptx
