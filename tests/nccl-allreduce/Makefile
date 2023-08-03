SIZE ?= 32
NGPUS ?= 4

all: all_reduce 

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
all_reduce: Makefile all_reduce.cu 
	$(NVCC) -cubin -lineinfo all_reduce.cu -o all_reduce.cubin
	nvdisasm --print-line-info all_reduce.cubin > testfile.txt
	$(NVCC) $(NVCC_FLAGS) all_reduce.cu $(LD_FLAGS) -o all_reduce

.PHONY.: clean
clean:
	rm -f all_reduce *cubin *txt

run:
	./all_reduce -n $(SIZE) -g $(NGPUS)

profile:
	@rm -f *.zst
	@NOBANNER=1 KERNEL_NAME="nccl" LD_PRELOAD="../../tools/mem_multigpu/mem_multigpu.so" ./all_reduce -n $(SIZE) -g $(NGPUS)
