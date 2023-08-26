all:
	nvcc -cubin -lineinfo pmcs.cu
	nvdisasm --print-line-info pmcs.cubin > testfile.txt
	nvcc  -lcuda -lcudart -Xcompiler=-fopenmp -O3 -arch=sm_75 pmcs.cu -o pmcs 

clean:
	rm -f pmcs *cubin *txt
