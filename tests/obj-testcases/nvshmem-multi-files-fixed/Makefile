NP ?= 2
NVCC ?= nvcc
MPIRUN ?= mpirun
MPICCX = mpic++

SIZE ?= 32

GENCODE_SM80  := -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80
GENCODE_FLAGS	:= $(GENCODE_SM80)

NVCC_NV_FLAGS = -dc -Xcompiler -fopenmp $(GENCODE_FLAGS) -std=c++17 -ccbin=$(MPICCX) -I$(NVSHMEM_HOME)/include -I$(MPI_HOME)/include -I./include_nvshmem
NVCC_NV_LDFLAGS = -ccbin=$(MPICCX) -lgomp -L$(CUDA_HOME)/lib64 -lcuda -lcudart -lnvidia-ml -L$(NVSHMEM_HOME)/lib -lnvshmem_host -lnvshmem_device -L$(MPI_HOME)/lib -lmpi -L$(UCX_HOME)/lib -lucp -lucs -luct -lucm -lmlx5 -L. -lwrapper

MAKEFLAGS += -j

all: device-initiated-direct-access 

#device-initiated-direct-access: device-initiated-direct-access.o 
#	$(NVCC) -g $(GENCODE_FLAGS) -o $@ $^ $(NVCC_NV_LDFLAGS)
	
#main: main.o
#        $(NVCC) -g $(GENCODE_FLAGS) -o $@ $^ $(NVCC_NV_LDFLAGS)
        

#main.o: main.cu
#        $(NVCC) $(NVCC_NV_FLAGS) -o $@ $< -lineinfo
#        cuobjdump $@ -xelf all
#        nvdisasm --print-line-info main*.sm_80.cubin > testfile.txt


device-initiated-direct-access: libwrapper.so device-initiated-direct-access.cu simple1.cu simple2.cu
	$(NVCC) $(NVCC_NV_FLAGS) -g -rdc=true simple1.cu -c -o simple1.o -lineinfo
	$(NVCC) $(NVCC_NV_FLAGS) -g -rdc=true simple2.cu -c -o simple2.o -lineinfo
	$(NVCC) $(NVCC_NV_FLAGS) -g -rdc=true device-initiated-direct-access.cu -c -o device-initiated-direct-access.o -lineinfo
	$(NVCC) -g $(GENCODE_FLAGS) -o $@ libwrapper.so device-initiated-direct-access.o simple1.o simple2.o $(NVCC_NV_LDFLAGS)
	$(NVCC) -cubin -rdc=true -lineinfo device-initiated-direct-access.cu
	nvdisasm --print-line-info device-initiated-direct-access.cubin > testfile.txt
	$(NVCC) -cubin -rdc=true -lineinfo simple1.cu
	nvdisasm --print-line-info simple1.cubin >> testfile.txt
	$(NVCC) -cubin -rdc=true -lineinfo simple2.cu
	nvdisasm --print-line-info simple2.cubin >> testfile.txt
	#cuobjdump $@ -xelf all
	#nvdisasm --print-line-info device-initiated-direct-access*.sm_80.cubin > testfile.txt	

libwrapper.so: wrapper.o
	$(NVCC) -g -shared --generate-line-info -o $@ $< -lineinfo

wrapper.o: wrapper.cpp
	$(CXX) -g -c -Wall -Werror -fPIC $<  -L/usr/local/cuda/lib64 -lcudart -I/usr/local/cuda/include -o $@ -lineinfo

ptx: 
	$(NVCC) --ptx $(NVCC_NV_FLAGS) -o $@ $< $(NVCC_NV_LDFLAGS)


run: device-initiated-direct-access
	#time -p $(MPIRUN) --oversubscribe -x NVSHMEM_NGPUS=4 -x NOBANNER=1 -x LD_PRELOAD="../../../tools/mem_multigpu/mem_multigpu.so" -x KERNEL_NAME="void simple_shift" -np $(NP) ./main -n $(SIZE)
	time -p $(MPIRUN) --oversubscribe -x NVSHMEM_NGPUS=2 -x NOBANNER=1 -x LD_PRELOAD="../../../tools/mem_multigpu/mem_multigpu.so" -x LD_LIBRARY_PATH=/usr/local/cuda/lib64:.:$(NVSHMEM_HOME)/lib:$$LD_LIBRARY_PATH -x KERNEL_NAME=all -x CODE_ATTRIBUTION=1 -np $(NP) ./device-initiated-direct-access -n $(SIZE)

clean:
	rm -f device-initiated-direct-access.o device-initiated-direct-access device-initiated-direct-access.o.ptx device-initiated-direct-access.ptx *.zst *.txt *.cubin *.o *.so
