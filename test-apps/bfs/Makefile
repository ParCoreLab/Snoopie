all:	bfs_compile bfs_run 

bfs_compile:
	#nvcc -o test test.cu -l./wrapper.a -cudart shared
	g++ -g -c -Wall -Werror -fPIC wrapper.cpp  -L/usr/local/cuda/lib64 -lcudart -I/usr/local/cuda/include -o wrapper.o -lineinfo
	nvcc -g -shared --generate-line-info -o libwrapper.so wrapper.o -lineinfo
	nvcc -g -L. --generate-line-info -o bfs bfs_basic.cu -lwrapper -lineinfo
	nvcc -ptx bfs_basic.cu
	nvcc -cubin -lineinfo bfs_basic.cu
	nvdisasm --print-line-info bfs_basic.cubin > testfile.txt

bfs_run:
	LD_LIBRARY_PATH=/usr/local/cuda/lib64:.:$$LD_LIBRARY_PATH LD_PRELOAD=../../tools/mem_multigpu/mem_multigpu.so KERNEL_NAME=all CODE_ATTRIBUTION=1 ./bfs -f gupta2/gupta2.mtx -n 4 -b 10 -t 512 -v 0 -m 0 -r 1 -o 1
	#LD_LIBRARY_PATH=/usr/local/cuda/lib64:.:$$LD_LIBRARY_PATH LD_PRELOAD=../../tools/mem_multigpu/mem_multigpu.so KERNEL_NAME="kernel_workq" CODE_ATTRIBUTION=1 ./bfs -f gupta2/gupta2.mtx -n 4 -b 10 -t 512 -v 0 -m 0 -r 1 -o 1
	#LD_LIBRARY_PATH=/usr/local/cuda/lib64:.:$$LD_LIBRARY_PATH LD_PRELOAD=../../tools/mem_multigpu/mem_multigpu.so KERNEL_NAME="kernel_workq" CODE_ATTRIBUTION=1 ./bfs -f Tina_AskCog/Tina_AskCog.mtx -n 4 -b 10 -t 512 -v 0 -m 0 -r 2 -o 1
	#LD_LIBRARY_PATH=/usr/local/cuda/lib64:.:$$LD_LIBRARY_PATH LD_PRELOAD=../../tools/mem_multigpu/mem_multigpu.so KERNEL_NAME="kernel_workq" CODE_ATTRIBUTION=1 ./bfs -f Tina_AskCog/Tina_AskCog.mtx -n 4 -b 10 -t 512 -v 0 -m 0 -r 2 -o 1
	#LD_LIBRARY_PATH=/usr/local/cuda/lib64:.:$$LD_LIBRARY_PATH LD_PRELOAD=../../tools/mem_multigpu/mem_multigpu.so ./device-initiated_implicit_memcpy

clean:
	rm -f *.so *.o bfs *txt *cubin *zst
