all:
	nvcc -ccbin=$(CXX) -O3 -arch=sm_80 device-initiated_implicit_memcpy.cu -o simple-diim -lineinfo
	nvcc -ccbin=$(CXX) --ptx -O3 -arch=sm_80 device-initiated_implicit_memcpy.cu -o simple-diim.ptx -lineinfo
	nvcc -cubin -lineinfo device-initiated_implicit_memcpy.cu
	nvdisasm --print-line-info device-initiated_implicit_memcpy.cubin > testfile.txt

clean:
	rm -f simple-diim simple-diim.ptx *cubin *txt
