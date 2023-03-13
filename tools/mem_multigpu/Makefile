NVCC=nvcc -ccbin=$(CXX) -D_FORCE_INLINES -lcudadevrt

NVCC_VER_REQ=10.1
NVCC_VER=$(shell $(NVCC) --version | grep release | cut -f2 -d, | cut -f3 -d' ')
NVCC_VER_CHECK=$(shell echo "${NVCC_VER} >= $(NVCC_VER_REQ)" | bc)

ifeq ($(NVCC_VER_CHECK),0)
$(error ERROR: nvcc version >= $(NVCC_VER_REQ) required to compile an nvbit tool! Instrumented applications can still use lower versions of nvcc.)
endif

NVBIT_PATH=../../core
INCLUDES=-I$(NVBIT_PATH)

LIBS=-L$(NVBIT_PATH) -lnvbit
NVCC_PATH=-L $(subst bin/nvcc,lib64,$(shell which nvcc | tr -s /))

SOURCES=mem_multigpu.cu inject_funcs.cu

OBJECTS=$(SOURCES:.cu=.o)
ARCH?=35

mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
current_dir := $(notdir $(patsubst %/,%,$(dir $(mkfile_path))))

NVBIT_TOOL=$(current_dir).so

all: $(NVBIT_TOOL)

$(NVBIT_TOOL): $(OBJECTS) $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=sm_$(ARCH) -O3 $(OBJECTS) $(LIBS) $(NVCC_PATH) -lcuda -lcudart_static -shared -o $@

mem_multigpu.o: mem_multigpu.cu
	$(NVCC) -dc -c -std=c++11 $(INCLUDES) -Xptxas -cloning=no -Xcompiler -Wall -allow-unsupported-compiler -arch=sm_$(ARCH) -O3 -Xcompiler -fPIC $< -o $@

inject_funcs.o: inject_funcs.cu
	$(NVCC) $(INCLUDES) -maxrregcount=24 -Xptxas -astoolspatch --keep-device-functions -arch=sm_$(ARCH) -Xcompiler -Wall -Xcompiler -fPIC -allow-unsupported-compiler -c $< -o $@

test_code:
	#nvcc -o test test.cu -l./wrapper.a -cudart shared
	g++ -c -Wall -Werror -fPIC wrapper.cpp  -L/usr/local/cuda/lib64 -lcudart -I/usr/local/cuda/include
	nvcc -shared -o libwrapper.so wrapper.o
	nvcc -L. -o test_mult test_mult.cu -lwrapper

test_run:
	LD_LIBRARY_PATH=/usr/local/cuda/lib64:.:$$LD_LIBRARY_PATH LD_PRELOAD=./mem_multigpu.so ./test_mult

clean:
	rm -f *.so *.o test_mult
