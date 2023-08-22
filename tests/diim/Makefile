SIZE ?= 32

all:
	nvcc -ccbin=$(CXX) -g -arch=sm_80 device-initiated-direct-access.cu -o diim -lineinfo

ptx:
	nvcc -ccbin=$(CXX) --ptx -O3 -arch=sm_80 device-initiated-direct-access.cu -o diim.ptx

clean:
	rm -f diim diim.ptx *.zst *txt

run:
	@NOBANNER=1 KERNEL_NAME="simple_kernel" LD_PRELOAD="../../tools/mem_multigpu/mem_multigpu.so" ./diim -n $(SIZE)
