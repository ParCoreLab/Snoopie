all:	device-initiated_implicit_memcpy_compile device-initiated_implicit_memcpy_run

device-initiated_implicit_memcpy_compile:
	g++ -c -Wall -Werror -fPIC wrapper.cpp  -L/usr/local/cuda/lib64 -lcudart -I/usr/local/cuda/include -O3 -o wrapper.o -lineinfo
	nvcc -shared -O3 -o libwrapper.so wrapper.o -lineinfo
	nvcc -ccbin=$(CXX) -O3 -L. -o device-initiated_implicit_memcpy device-initiated_implicit_memcpy.cu -lwrapper -lineinfo
	nvcc -ccbin=$(CXX) --ptx -O3 device-initiated_implicit_memcpy.cu -o device-initiated_implicit_memcpy.ptx -lineinfo
	nvcc -cubin -lineinfo device-initiated_implicit_memcpy.cu
	nvdisasm --print-line-info device-initiated_implicit_memcpy.cubin > memop_to_line.txt

device-initiated_implicit_memcpy_run:
	LD_LIBRARY_PATH=/usr/local/cuda/lib64:.:$$LD_LIBRARY_PATH LD_PRELOAD=../../src/mem_multigpu/libmem_multigpu.so KERNEL_NAME=all CODE_ATTRIBUTION=1 ./device-initiated_implicit_memcpy
	#LD_LIBRARY_PATH=/usr/local/cuda/lib64:.:$$LD_LIBRARY_PATH LD_PRELOAD=../../src/mem_multigpu/libmem_multigpu.so ./device-initiated_implicit_memcpy

clean:
	rm -f *.so *.o device-initiated_implicit_memcpy *ptx *cubin *txt
