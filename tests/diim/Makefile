SIZE ?= 32

all:
	nvcc -ccbin=$(CXX) -g -arch=sm_80 device-initiated-direct-access.cu -o diim -lineinfo
	nvcc -cubin -lineinfo device-initiated-direct-access.cu
	nvdisasm --print-line-info device-initiated-direct-access.cubin > memop_to_line.txt

ptx:
	nvcc -ccbin=$(CXX) --ptx -O3 -arch=sm_80 device-initiated-direct-access.cu -o diim.ptx

clean:
	rm -f diim diim.ptx *.zst *txt *cubin

run:
	@rm -f *.zst
	@NOBANNER=1 KERNEL_NAME="simple_kernel" LD_PRELOAD=${SNOOPIE_PATH} CODE_ATTRIBUTION=1 ./diim -n $(SIZE)
