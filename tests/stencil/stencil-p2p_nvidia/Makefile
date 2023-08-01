# Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
NITER ?= 32768
SIZE ?= 1024

NVCC=nvcc
GENCODE_SM80    := -gencode arch=compute_80,code=compute_80
GENCODE_FLAGS	:=  $(GENCODE_SM80)
ifdef DISABLE_CUB
        NVCC_FLAGS = -Xptxas --optimize-float-atomics
else
        NVCC_FLAGS = -DHAVE_CUB
endif
NVCC_FLAGS += -Xcompiler -fopenmp -lineinfo -DUSE_NVTX -lnvToolsExt  -std=c++14
jacobi: Makefile jacobi.cu
	#$(NVCC) $(NVCC_FLAGS) $(GENCODE_FLAGS) jacobi.cu -o jacobi 
	g++ -g -c -Wall -Werror -fPIC wrapper.cpp  -L/usr/local/cuda/lib64 -lcudart -I/usr/local/cuda/include -o wrapper.o -lineinfo
	nvcc -g -shared --generate-line-info -o libwrapper.so wrapper.o -lineinfo
	$(NVCC) -cubin $(NVCC_FLAGS) jacobi.cu
	nvdisasm --print-line-info jacobi.cubin > testfile.txt
	$(NVCC) $(NVCC_FLAGS) jacobi.cu -o jacobi -L. -lwrapper -lineinfo

.PHONY.: clean
clean:
	rm -f jacobi jacobi.qdrep jacobi.cubin libwrapper.so testfile.txt wrapper.o *zst *txt

sanitize: jacobi
	compute-sanitizer ./jacobi -niter 10

run: jacobi
	./jacobi -nx $(SIZE) -ny $(SIZE) -niter $(NITER)
	@ # LD_PRELOAD="../../tools/cudaops/cudaops.so" ./jacobi -nx $(SIZE) -ny $(SIZE) -niter $(NITER)

profile: jacobi
	#NOBANNER=1 KERNEL_NAME="void jacobi_kernel<32, 32>" CODE_ATTRIBUTION=1 SAMPLE_SIZE=10 LD_PRELOAD="../../../tools/mem_multigpu/mem_multigpu.so" ./jacobi -nx $(SIZE) -ny $(SIZE) -niter $(NITER)
	NOBANNER=1 KERNEL_NAME=all CODE_ATTRIBUTION=1 SAMPLE_SIZE=10 LD_PRELOAD="../../../tools/mem_multigpu/mem_multigpu.so" ./jacobi -nx $(SIZE) -ny $(SIZE) -niter $(NITER)
	# NOBANNER=1 SILENT=1 LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:. SILENT=0 KERNEL_NAME="void jacobi_kernel<32, 32>" LD_PRELOAD="../../tools/mem_multigpu/mem_multigpu.so" ./jacobi -nx $(SIZE) -ny $(SIZE) -niter $(NITER)
	# NOBANNER=1 SILENT=0 LD_PRELOAD="../../tools/mem_trace/mem_trace.so" ./jacobi -nx $(SIZE) -ny $(SIZE) -niter $(NITER)
	@# nsys profile --trace=cuda,nvtx -o jacobi ./jacobi -niter 10
