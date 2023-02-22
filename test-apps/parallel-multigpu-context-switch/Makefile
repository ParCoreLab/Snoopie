all:
	nvcc  -lcuda -lcudart -Xcompiler=-fopenmp -O3 -arch=sm_75 pmcs.cu -o pmcs 

clean:
	rm -f pmcs
