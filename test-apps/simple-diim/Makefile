all:    device-initiated_implicit_memcpy_compile device-initiated_implicit_memcpy_run

device-initiated_implicit_memcpy_compile:
	nvcc -ccbin=$(CXX) -O3 -arch=sm_80 device-initiated_implicit_memcpy.cu -o simple-diim -lineinfo
	nvcc -ccbin=$(CXX) --ptx -O3 -arch=sm_80 device-initiated_implicit_memcpy.cu -o simple-diim.ptx -lineinfo
	nvcc -cubin -lineinfo device-initiated_implicit_memcpy.cu
	nvdisasm --print-line-info device-initiated_implicit_memcpy.cubin > testfile.txt

device-initiated_implicit_memcpy_run:
	LD_LIBRARY_PATH=/usr/local/cuda/lib64:.:$$LD_LIBRARY_PATH LD_PRELOAD=../../tools/mem_multigpu/mem_multigpu.so KERNEL_NAME=simple_kernel ./simple-diim

clean:
	rm -f simple-diim simple-diim.ptx *cubin *txt
