SIZE ?= 48

all:
	nvcc -ccbin=$(CXX) -g -arch=sm_80 -rdc=true simple1.cu -c -o simple1.o -lineinfo
	nvcc -ccbin=$(CXX) -g -arch=sm_80 -rdc=true simple2.cu -c -o simple2.o -lineinfo
	nvcc -ccbin=$(CXX) -g -arch=sm_80 -rdc=true device-initiated-direct-access.cu -c -o device-initiated-direct-access.o -lineinfo
	nvcc -ccbin=$(CXX) -arch=sm_80 -rdc=true simple1.o simple2.o device-initiated-direct-access.o -o diim
	nvcc -cubin -rdc=true -lineinfo device-initiated-direct-access.cu
	nvdisasm --print-line-info device-initiated-direct-access.cubin > memop_to_line.txt
	nvcc -cubin -rdc=true -lineinfo simple1.cu
	nvdisasm --print-line-info simple1.cubin >> memop_to_line.txt
	nvcc -cubin -rdc=true -lineinfo simple2.cu
	nvdisasm --print-line-info simple2.cubin >> memop_to_line.txt

ptx:
	nvcc -ccbin=$(CXX) --ptx -O3 -arch=sm_80 device-initiated-direct-access.cu -o diim.ptx

clean:
	rm -f diim diim.ptx *.zst *txt *cubin *.o

run:
	@NOBANNER=1 KERNEL_NAME=all LD_PRELOAD="../../../src/mem_multigpu/libmem_multigpu.so" CODE_ATTRIBUTION=1 ./diim -n $(SIZE)
