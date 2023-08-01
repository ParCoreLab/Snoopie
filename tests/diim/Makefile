SIZE ?= 32

all:
	nvcc -ccbin=$(CXX) -O3 -arch=sm_80 device-initiated-direct-access.cu -o diim

ptx:
	nvcc -ccbin=$(CXX) --ptx -O3 -arch=sm_80 device-initiated-direct-access.cu -o diim.ptx

clean:
	rm -f diim diim.ptx

run:
	@rm -f *.zst
	./diim -n $(SIZE)

profile:
	@NOBANNER=1 KERNEL_NAME="simple_kernel" LD_PRELOAD="../../tools/mem_multigpu/mem_multigpu.so" ./diim -n $(SIZE)
