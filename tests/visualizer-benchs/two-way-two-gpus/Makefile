all:	two-gpus_compile two-gpus_run

two-gpus_compile:
	g++ -c -Wall -Werror -fPIC wrapper.cpp  -L/usr/local/cuda/lib64 -lcudart -I/usr/local/cuda/include -O3 -o wrapper.o -lineinfo
	nvcc -shared -O3 -o libwrapper.so wrapper.o -lineinfo
	nvcc -ccbin=$(CXX) -O3 -L. -o two-gpus two-gpus.cu -lwrapper -lineinfo
	nvcc -ccbin=$(CXX) --ptx -O3 two-gpus.cu -o two-gpus.ptx -lineinfo
	nvcc -cubin -lineinfo two-gpus.cu
	nvdisasm --print-line-info two-gpus.cubin > memop_to_line.txt

two-gpus_run:
	LD_LIBRARY_PATH=/usr/local/cuda/lib64:.:$$LD_LIBRARY_PATH LD_PRELOAD=../../../src/mem_multigpu/libmem_multigpu.so KERNEL_NAME=all CODE_ATTRIBUTION=1 ./two-gpus
	#LD_LIBRARY_PATH=/usr/local/cuda/lib64:.:$$LD_LIBRARY_PATH LD_PRELOAD=../../src/mem_multigpu/libmem_multigpu.so ./two-gpus

clean:
	rm -f *.so *.o two-gpus *ptx *cubin *txt
