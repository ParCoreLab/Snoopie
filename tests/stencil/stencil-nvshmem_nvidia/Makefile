NP ?= 4
NVCC ?= nvcc
MPIRUN ?= mpirun
MPICCX = mpic++
CXX ?= g++
NITER ?= 5
SIZE ?= 1024

GENCODE_SM80    := -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80
GENCODE_FLAGS	:= $(GENCODE_SM80)

NVCC_NV_FLAGS = -dc -Xcompiler -fopenmp $(GENCODE_FLAGS) -std=c++17 -ccbin=$(MPICCX) -I$(NVSHMEM_HOME)/include -I$(MPI_HOME)/include -I./include_nvshmem 
NVCC_NV_LDFLAGS = -ccbin=$(MPICCX) -lgomp -L$(CUDA_HOME)/lib64 -lcuda -lcudart -lnvidia-ml -L$(NVSHMEM_HOME)/lib -lnvshmem_host -lnvshmem_device -L$(MPI_HOME)/lib -lmpi -L$(UCX_HOME)/lib -lucp -lucs -luct -lucm -lmlx5 -L. -lwrapper -lineinfo
#NVCC_NV_LDFLAGS = -ccbin=$(MPICCX) -lgomp -L$(CUDA_HOME)/lib64 -lcuda -lcudart -lnvidia-ml -L$(NVSHMEM_HOME)/lib -lnvshmem -L$(MPI_HOME)/lib -lmpi -L$(UCX_HOME)/lib -lucp -lucs -luct -lucm -lmlx5 -L. -lwrapper -lineinfo

MAKEFLAGS += -j

jacobi: libwrapper.so jacobi.o 
	$(NVCC) -g $(GENCODE_FLAGS) -o $@ $^ $(NVCC_NV_LDFLAGS)
	

jacobi.o: jacobi.cu
	#nvcc -cubin -lineinfo main.cu
	#nvdisasm --print-line-info main.cubin > testfile.txt
	$(NVCC) -g $(NVCC_NV_FLAGS) -o $@ $< -lineinfo
	cuobjdump $@ -xelf all
	nvdisasm --print-line-info *.cubin > memop_to_line.txt

libwrapper.so: wrapper.o
	$(NVCC) -g -shared --generate-line-info -o $@ $< -lineinfo

wrapper.o: wrapper.cpp
	$(CXX) -g -c -Wall -Werror -fPIC $<  -L/usr/local/cuda/lib64 -lcudart -I/usr/local/cuda/include -o $@ -lineinfo

run: jacobi
	#$(MPIRUN) -x LD_LIBRARY_PATH=/usr/local/cuda/lib64:.:$(NVSHMEM_HOME)/lib:$$LD_LIBRARY_PATH -np $(NP) ./jacobi -nx $(SIZE) -ny $(SIZE) -niter $(NITER)
	$(MPIRUN) --oversubscribe -x LD_LIBRARY_PATH=/usr/local/cuda/lib64:.:$(NVSHMEM_HOME)/lib:$$LD_LIBRARY_PATH -np $(NP) ./jacobi -nx $(SIZE) -ny $(SIZE) -niter $(NITER)

run_main: jacobi
	$(MPIRUN) --oversubscribe -x NVSHMEM_NGPUS=4 -x KERNEL_NAME="void jacobi_kernel<32, 32>" -x LD_LIBRARY_PATH=/usr/local/cuda/lib64:.:$(NVSHMEM_HOME)/lib:$(CPPTRACE_HOME)/lib64:$$LD_LIBRARY_PATH -x LD_PRELOAD="../../../src/mem_multigpu/mem_multigpu.so" -x CODE_ATTRIBUTION=1 -np $(NP) ./jacobi -nx $(SIZE) -ny $(SIZE) -niter $(NITER)

clean:
	rm -f *.so *.o jacobi *txt *cubin *zst
