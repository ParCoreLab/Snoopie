GENCODE_SM80    := -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80
GENCODE_FLAGS	:= $(GENCODE_SM80)

NVCC_FLAGS = -dc $(GENCODE_FLAGS) 
NVCC_LDFLAGS = $(GENCODE_FLAGS) -lcuda -lcudart -lcudadevrt

all:
	nvcc --ptx $(NVCC_FLAGS)  -o main.o main.cu -lineinfo
	nvcc --ptx $(NVCC_LDFLAGS) -o main.ptx   main.o   -lineinfo
	nvcc -cubin -lineinfo main.cu
	nvdisasm --print-line-info main.cubin > testfile.txt

clean:
	rm -f main *ptx *cubin *txt *.o
