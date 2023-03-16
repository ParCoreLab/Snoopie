all:	device-initiated_implicit_memcpy_compile device-initiated_implicit_memcpy_run 

device-initiated_implicit_memcpy_compile:
	#nvcc -o test test.cu -l./wrapper.a -cudart shared
	g++ -c -Wall -Werror -fPIC wrapper.cpp  -L/usr/local/cuda/lib64 -lcudart -I/usr/local/cuda/include -o wrapper.o
	nvcc -shared -o libwrapper.so wrapper.o
	nvcc -L. -o device-initiated_implicit_memcpy device-initiated_implicit_memcpy.cu -lwrapper

device-initiated_implicit_memcpy_run:
	LD_LIBRARY_PATH=/usr/local/cuda/lib64:.:$$LD_LIBRARY_PATH LD_PRELOAD=../../tools/mem_multigpu/mem_multigpu.so KERNEL_NAME=simple_kernel ./device-initiated_implicit_memcpy
	#LD_LIBRARY_PATH=/usr/local/cuda/lib64:.:$$LD_LIBRARY_PATH LD_PRELOAD=../../tools/mem_multigpu/mem_multigpu.so ./device-initiated_implicit_memcpy

clean:
	rm -f *.so *.o device-initiated_implicit_memcpy
