NVCC=nvcc -ccbin=$(CXX) -D_FORCE_INLINES -lcudadevrt

NVCC_VER_REQ=10.1
NVCC_VER=$(shell $(NVCC) --version | grep release | cut -f2 -d, | cut -f3 -d' ')
NVCC_VER_CHECK=$(shell echo "${NVCC_VER} >= $(NVCC_VER_REQ)" | bc)

ifeq ($(NVCC_VER_CHECK),0)
$(error ERROR: nvcc version >= $(NVCC_VER_REQ) required to compile an nvbit tool! Instrumented applications can still use lower versions of nvcc.)
endif

NVBIT_PATH=../../core
INCLUDES=-I$(NVBIT_PATH) -I. -I../../../libunwind-install/include

LIBS=-L$(NVBIT_PATH) -lnvbit -L../../../libunwind-install/lib
NVCC_PATH=-L $(subst bin/nvcc,lib64,$(shell which nvcc | tr -s /))

SOURCES=mem_multigpu.cu inject_funcs.cu obj_database.cu

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

obj_database.o: obj_database.cu
	$(NVCC) $(INCLUDES) -Xptxas -astoolspatch --keep-device-functions -arch=sm_$(ARCH) -Xcompiler -Wall -Xcompiler -fPIC -allow-unsupported-compiler -c $< -o $@

allocation_only_test_compile:
	#nvcc -o test test.cu -l./wrapper.a -cudart shared
	g++ -c -Wall -Werror -fPIC allocation-only-test/wrapper.cpp  -L/usr/local/cuda/lib64 -lcudart -I/usr/local/cuda/include -o allocation-only-test/wrapper.o
	nvcc -shared -o allocation-only-test/libwrapper.so allocation-only-test/wrapper.o
	nvcc -L./allocation-only-test -o allocation-only-test/test_mult allocation-only-test/test_mult.cu -lwrapper

allocation_only_test_run:
	LD_LIBRARY_PATH=/usr/local/cuda/lib64:./allocation-only-test:$$LD_LIBRARY_PATH LD_PRELOAD=./mem_multigpu.so allocation-only-test/test_mult

clean:
	rm -f *.so *.o test_mult
