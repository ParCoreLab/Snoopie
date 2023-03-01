all:
	nvcc -O3 -Xcicc -O3 -Xptxas -O3 -arch=sm_80 clocking-writes.cu -o main 
	nvcc --ptx -O3 -Xcicc -O3 -Xptxas -O3 -arch=sm_80 clocking-writes.cu -o main.ptx

clean:
	rm -f main
