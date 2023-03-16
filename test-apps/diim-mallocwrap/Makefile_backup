all:
	nvcc -ccbin=$(CXX) -O3 -arch=sm_80 device-initiated_implicit_memcpy.cu -o diim
	nvcc -ccbin=$(CXX) --ptx -O3 -arch=sm_80 device-initiated_implicit_memcpy.cu -o diim.ptx

clean:
	rm -f diim diim.ptx
