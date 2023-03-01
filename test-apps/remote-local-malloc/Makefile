GENCODE_SM80    := -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80
GENCODE_FLAGS	:= $(GENCODE_SM80)

NVCC_FLAGS = -dc $(GENCODE_FLAGS) 
NVCC_LDFLAGS = $(GENCODE_FLAGS) -lcuda -lcudart -lcudadevrt

all:
	nvcc --ptx $(NVCC_FLAGS)  -o main.o main.cu
	nvcc --ptx $(NVCC_LDFLAGS) -o main.ptx   main.o  
clean:
	rm -f main
