all:
	nvcc -ccbin=$(CXX) -O3 -arch=sm_80 device-initiated_implicit_memcpy.cu -o simple-diim
	nvcc -ccbin=$(CXX) --ptx -O3 -arch=sm_80 device-initiated_implicit_memcpy.cu -o simple-diim.ptx

clean:
	rm -f simple-diim simple-diim.ptx
