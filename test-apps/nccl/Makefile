all: main 

NVCC=nvcc
GENCODE_SM80    := -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80
GENCODE_FLAGS	:= $(GENCODE_SM80)
ifdef DISABLE_CUB
        NVCC_FLAGS = -Xptxas --optimize-float-atomics
else
        NVCC_FLAGS = -DHAVE_CUB
endif

NVCC_FLAGS += -g -G $(GENCODE_FLAGS) -std=c++17
LD_FLAGS = -L$(CUDA_HOME)/lib64 -lcudart -lnvToolsExt -lnccl
main: Makefile main.cu 
	$(NVCC) $(NVCC_FLAGS) main.cu $(LD_FLAGS) -o main

.PHONY.: clean
clean:
	rm -f main