/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <assert.h>
#include <dlfcn.h>
#include <iostream>
#include <map>
#include <sstream>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>

#include <adm_common.h>
#include <adm_config.h>
#include <adm_database.h>
#include <adm_memory.h>
#include <adm_splay.h>
#include <cpptrace/cpptrace.hpp>

#include "Python.h"
//#include "ndarrayobject.h"
#include <numpy/ndarrayobject.h>
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/pytypes.h>

#include <iostream>

/* every tool needs to include this once */
//#include "nvbit_tool.h"

/* nvbit interface file */
//#include "nvbit.h"

/* for channel */
//#include "utils/channel.hpp"

/* contains definition of the mem_access_t structure */
//#include "common.h"
//#include "util.h"

//#include "mpi.h"
//#include "nvshmem.h"
//#include "nvshmemx.h"

#include <sys/syscall.h>
#define gettid() syscall(SYS_gettid)

#define HEX(x)                                                                 \
	"0x" << std::setfill('0') << std::setw(16) << std::hex << (uint64_t)x        \
<< std::dec

#define CHANNEL_SIZE (1l << 30)

#define JSON 0
#define EQUAL_STRS 0

#define FILE_NAME_SIZE 256
#define PATH_NAME_SIZE 5000

#include "adm.h"

namespace py = pybind11;
using namespace cpptrace;
using namespace adamant;
using namespace std;

#define CHILD 1
#define SIBLING 2

PyObject* orig_cudadevicendarray_func;
PyObject* orig_cudadevicerecord_func;
PyObject* orig_cudapinnedarray_func;
PyObject* orig_torch_cuda_func;
//PyObject* orig_torchtensor_func;
//PyObject* orig_torchto_func;
extern std::vector<adm_range_t *> range_nodes;

pthread_mutex_t mutex_pytorch;

extern int code_context;
extern int data_object_attribution;

inline py::object extract_python_callpath()
{
	py::object traceback = py::module::import("traceback");
	py::object extract_summary = traceback.attr("StackSummary").attr("extract");
	py::object walk_stack = traceback.attr("walk_stack");
	return extract_summary(walk_stack(py::none()));
}

void update_allocation_site_tree(py::object& summary, allocation_site_t **allocation_site, allocation_site_t **parent);

//#if 0
void update_exec_site_tree(py::object& summary, execution_site_t **execution_site, execution_site_t **parent);

void record_exec_context(execution_site_t *parent); 

void record_object_allocation_context(allocation_site_t *parent); 
//#endif



PYBIND11_MODULE(libmem_multigpu, m) {
	pthread_mutexattr_t attr;
	pthread_mutexattr_init(&attr);
	pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
	pthread_mutex_init(&mutex_pytorch, &attr);
	py::object traceback = py::module::import("traceback");
	//py::object nb = py::module::import("numba");
	py::object torch = py::module::import("torch");
	//PyObject* name = PyUnicode_FromString("numpy");
	//PyObject* np = PyImport_Import(name);
	//import_arr();

	auto my_injection = [](py::object obj, std::string func_name) 
	{
		if(func_name == "cuda.cudadrv.devicearray.DeviceNDArray") {
			std::cerr << "cuda.cudadrv.devicearray.DeviceNDArray is injected\n";
			PyObject* mod = obj.ptr();
			PyObject* cuda_obj = PyObject_GetAttrString(mod, "cuda");
			PyObject* cudadrv_obj = PyObject_GetAttrString(cuda_obj, "cudadrv");
			PyObject* devicearray_obj = PyObject_GetAttrString(cudadrv_obj, "devicearray");
			orig_cudadevicendarray_func = PyObject_GetAttrString(devicearray_obj, "DeviceNDArray");		

			obj.attr("cuda").attr("cudadrv").attr("devicearray").attr("DeviceNDArray") = py::cpp_function([](const py::args &args, const py::kwargs &kwargs) {
					//std::cout << msg.cast<std::string>();
					std::cerr << "cuda.cudadrv.devicearray.DeviceNDArray is intercepted\n";

					PyObject* result = PyObject_Call(orig_cudadevicendarray_func, (PyObject *) args.ptr(), (PyObject *) kwargs.ptr());
					
					py::object summary = extract_python_callpath();
                                        std::vector<py::handle> stack_vec;

                                        allocation_site_t *allocation_site = NULL;
                                        allocation_site_t *parent = NULL;

                                        update_allocation_site_tree(summary, &allocation_site, &parent);

                                        std::string filename;

                                        record_object_allocation_context(parent);

					PyObject* gpu_data_obj = PyObject_GetAttrString(result, "gpu_data");
					PyObject* ptr_obj = PyObject_GetAttrString(gpu_data_obj, "device_ctypes_pointer");
					PyObject* ptr_val_obj = PyObject_GetAttrString(ptr_obj, "value");
					unsigned long long offset_val = PyLong_AsUnsignedLongLongMask(ptr_val_obj);
					PyObject* alloc_size_obj = PyObject_GetAttrString(result, "alloc_size");
					unsigned long long alloc_size_val = PyLong_AsUnsignedLongLongMask(alloc_size_obj);

					if (parent) {
						int deviceID = -1;
						cudaGetDevice(&deviceID);

						adm_range_insert(offset_val, alloc_size_val, parent->get_pc(),
							deviceID, "", ADM_STATE_ALLOC);
						range_nodes.push_back(new adm_range_t(
							offset_val, alloc_size_val, parent->get_object_id(), deviceID));
					}	

					fprintf(stderr, "offset value: %lx and allocation size: %ld\n", offset_val, alloc_size_val);
					return py::reinterpret_borrow<py::object>(result);//result;//orig_empty_like_func(args/*, kwargs*/);
			});
		} else if(func_name == "cuda.cudadrv.devicearray.DeviceRecord") {
			std::cerr << "cuda.cudadrv.devicearray.DeviceRecord is injected\n";
			PyObject* mod = obj.ptr();
			PyObject* cuda_obj = PyObject_GetAttrString(mod, "cuda");
			PyObject* cudadrv_obj = PyObject_GetAttrString(cuda_obj, "cudadrv");
			PyObject* devicearray_obj = PyObject_GetAttrString(cudadrv_obj, "devicearray");
			orig_cudadevicerecord_func = PyObject_GetAttrString(devicearray_obj, "DeviceRecord");		

			obj.attr("cuda").attr("cudadrv").attr("devicearray").attr("DeviceRecord") = py::cpp_function([](const py::args &args, const py::kwargs &kwargs) {
					//std::cout << msg.cast<std::string>();
					std::cerr << "cuda.cudadrv.devicearray.DeviceRecord is intercepted\n";

					PyObject* result = PyObject_Call(orig_cudadevicerecord_func, (PyObject *) args.ptr(), (PyObject *) kwargs.ptr());
					py::object summary = extract_python_callpath();
					PyObject* gpu_data_obj = PyObject_GetAttrString(result, "gpu_data");
					PyObject* ptr_obj = PyObject_GetAttrString(gpu_data_obj, "device_ctypes_pointer");
					PyObject* ptr_val_obj = PyObject_GetAttrString(ptr_obj, "value");
					unsigned long long offset_val = PyLong_AsUnsignedLongLongMask(ptr_val_obj);
					PyObject* alloc_size_obj = PyObject_GetAttrString(result, "alloc_size");
					unsigned long long alloc_size_val = PyLong_AsUnsignedLongLongMask(alloc_size_obj);
					fprintf(stderr, "offset value: %lx and allocation size: %ld\n", offset_val, alloc_size_val);
					return py::reinterpret_borrow<py::object>(result);//result;//orig_empty_like_func(args/*, kwargs*/);
			});
		} else if(func_name == "cuda.pinned_array") {
			std::cerr << "cuda.pinned_array is injected\n";
			PyObject* mod = obj.ptr();
			PyObject* cuda_obj = PyObject_GetAttrString(mod, "cuda");
			orig_cudapinnedarray_func = PyObject_GetAttrString(cuda_obj, "pinned_array");		

			obj.attr("cuda").attr("pinned_array") = py::cpp_function([](const py::args &args, const py::kwargs &kwargs) {
					//std::cout << msg.cast<std::string>();
					std::cerr << "cuda.pinned_array is intercepted\n";
					py::object summary = extract_python_callpath();
					PyObject* result = PyObject_Call(orig_cudapinnedarray_func, (PyObject *) args.ptr(), (PyObject *) kwargs.ptr());
					npy_intp size;
					long *dptr;  /* could make this any variable type */
					PyArrayObject * obj_arr = (PyArrayObject *)result;
					//#if 0
					//dptr = (long *) PyArray_DATA(allocated_mem_ptr);
					dptr = (long *) PyArray_DATA(result);
					//long *base_ptr = (long *) PyArray_BASE(allocated_mem_ptr);
					//int typ=PyArray_TYPE(allocated_mem_ptr);
					int typ=PyArray_TYPE(result);
					long element_size = 0;
					switch(typ) {
						case NPY_BYTE:
						case NPY_BOOL:
							//case NPY_INT8:
						case NPY_UBYTE:
							//case NPY_UINT8:
							element_size = 1;
							break;
						case NPY_SHORT:
							//case NPY_INT16:
						case NPY_USHORT:
							//case NPY_UINT16:
							element_size = 2;
							break;
						case NPY_INT:
						case NPY_FLOAT:
							//case NPY_INT32:
						case NPY_UINT:
							//case NPY_UINT32:
							element_size = 4;
							break;
						case NPY_LONG:
						case NPY_LONGLONG:
						case NPY_DOUBLE:
							//case NPY_INT64:

							element_size = 8;
							break;
						default:
							std::cerr << "unknown type " << typ << "\n";
					}
					long element_count = 1;
					for(int i = 0; i < obj_arr->nd; i++) {
						element_count *= obj_arr->dimensions[i];
					}	
					long memory_size = element_count * element_size;
					fprintf(stderr, "offset of pinned_array: %lx, size of object: %ld\n", dptr, memory_size);		
					return py::reinterpret_borrow<py::object>(result);//result;//orig_empty_like_func(args/*, kwargs*/);
			});
		} else if(func_name == "tensor") {
			std::cout << "torch.tensor is injected\n";
			PyObject* mod = obj.ptr();
			PyObject* orig_torchtensor_func = PyObject_GetAttrString(mod, "tensor");		

			obj.attr("tensor") = py::cpp_function([orig_torchtensor_func](const py::args &args, const py::kwargs &kwargs) {
					//std::cout << msg.cast<std::string>();
					std::cout << "torch.tensor is intercepted\n";
					//py::object allocated_mem = orig_empty_like_func(args/*, kwargs*/);

					//PyObject * allocated_mem_ptr = allocated_mem.ptr();//orig_array_func(args/*, kwargs*/);
					PyObject* result = PyObject_Call(orig_torchtensor_func, (PyObject *) args.ptr(), (PyObject *) kwargs.ptr());

					PyObject* is_cuda_obj = PyObject_GetAttrString(result, "is_cuda");
                                        if (PyBool_Check(is_cuda_obj)) {
                                                if(is_cuda_obj == Py_True) {
							//std::cout << "torch.tensor is intercepted 1\n";
							PyObject* ptr_obj = PyObject_GetAttrString(result, "data_ptr");
							PyObject *empty_tuple = PyTuple_Pack(0);
							//PyObject* ptr_val_obj = PyObject_CallNoArgs(ptr_obj, empty_tuple, NULL);
							PyObject* ptr_val_obj = PyObject_Call(ptr_obj, empty_tuple, NULL);

							//std::cout << "torch.tensor is intercepted 2\n";
							unsigned long long offset_val = PyLong_AsUnsignedLongLongMask(ptr_val_obj);
							//PyObject* size_obj = PyObject_GetAttrString(result, "__len__");
							PyObject* size_obj = PyObject_GetAttrString(result, "dim");
							//PyObject* size_val_obj = PyObject_CallNoArgs(size_obj);
							PyObject* size_val_obj = PyObject_Call(size_obj, empty_tuple, NULL);
							//std::cout << "torch.tensor is intercepted 3\n";
							unsigned long long element_count = PyLong_AsUnsignedLongLongMask(size_val_obj);
							unsigned long long alloc_size_val = 1;

							for(long i = 0; i < element_count; i++) {
                                                                PyObject* torchsize_obj = PyObject_GetAttrString(result, "size");
                                                                PyObject *dim_obj = PyLong_FromLong(i);
                                                                PyObject *dim_tuple = PyTuple_Pack(1, dim_obj);
                                                                PyObject* size_dim_obj = PyObject_CallObject(torchsize_obj, dim_tuple);
                                                                unsigned long long dim_size = PyLong_AsUnsignedLongLongMask(size_dim_obj);
                                                                alloc_size_val *= dim_size;
                                                        }
							//std::cout << "torch.tensor is intercepted 3 1\n";
							PyObject* elem_size_obj = PyObject_GetAttrString(result, "element_size");

							if(elem_size_obj != NULL) {
                                                        	PyObject* elem_size_val_obj = PyObject_Call(elem_size_obj, empty_tuple, NULL);
                                                        	//std::cerr << "torch.tensor is intercepted 3 2 1\n";
                                                        	if(elem_size_val_obj != NULL) {
                                                        		//std::cout << "torch.tensor is intercepted 3 3\n";
                                                        		unsigned long long element_size = PyLong_AsUnsignedLongLongMask(elem_size_val_obj);
                                                       			alloc_size_val *= element_size;
									py::object summary = extract_python_callpath();
									//#if 0
									std::vector<py::handle> stack_vec;
									allocation_site_t *allocation_site = NULL;
									allocation_site_t *parent = NULL;

									update_allocation_site_tree(summary, &allocation_site, &parent);

									if (parent) {
										record_object_allocation_context(parent);
										int deviceID = -1;
										cudaGetDevice(&deviceID);

										fprintf(stderr, "torch.tensor func call captured for GPU Object, offset value: %lx, allocation size: %ld recorded\n", offset_val, alloc_size_val);
										adm_range_insert(offset_val, alloc_size_val, parent->get_pc(),
											deviceID, "", ADM_STATE_ALLOC);
										range_nodes.push_back(new adm_range_t(
											offset_val, alloc_size_val, parent->get_object_id(), deviceID));
									}
								} 
							}
							//fprintf(stderr, "offset value: %lx, allocation size: %ld\n", offset_val, alloc_size_val);
						}
					}

					//fprintf(stderr, "offset value: %lx, allocation size: %ld\n", offset_val, alloc_size_val);	
					//std::cout << "torch.tensor is intercepted 5\n";
					py::object result_obj = py::reinterpret_borrow<py::object>(result);
					PyObject* orig_torchto_func = PyObject_GetAttrString(result, "to"); 
					result_obj.attr("to") = py::cpp_function([orig_torchto_func](const py::args &args, const py::kwargs &kwargs) {

							std::cerr << "torch.tensor.to is intercepted\n";
							//py::object allocated_mem = orig_empty_like_func(args/*, kwargs*/);

							//PyObject * allocated_mem_ptr = allocated_mem.ptr();//orig_array_func(args/*, kwargs*/);
							PyObject* result1 = PyObject_Call(orig_torchto_func, (PyObject *) args.ptr(), (PyObject *) kwargs.ptr());
							PyObject* is_cuda_obj = PyObject_GetAttrString(result1, "is_cuda");
                                                        if (PyBool_Check(is_cuda_obj)) {
                                                                if(is_cuda_obj == Py_True) {
									//#if 0
									PyObject* ptr_obj = PyObject_GetAttrString(result1, "data_ptr");
									std::cerr << "here 4\n";
									//PyObject* ptr_val_obj = PyObject_CallNoArgs(ptr_obj);
									PyObject *empty_tuple = PyTuple_Pack(0);
									PyObject* ptr_val_obj = PyObject_Call(ptr_obj, empty_tuple, NULL);
									unsigned long long offset_val1 = PyLong_AsUnsignedLongLongMask(ptr_val_obj);
									//fprintf(stderr, "offset value: %lx\n", offset_val);
									//PyObject* size_obj = PyObject_GetAttrString(result1, "__len__");
									PyObject* size_obj = PyObject_GetAttrString(result1, "dim");
									//PyObject* size_val_obj = PyObject_CallNoArgs(size_obj);
									PyObject* size_val_obj = PyObject_Call(size_obj, empty_tuple, NULL);
									unsigned long long element_count = PyLong_AsUnsignedLongLongMask(size_val_obj);
									unsigned long long alloc_size_val = 1;

									for(long i = 0; i < element_count; i++) {
                                                                		PyObject* torchsize_obj = PyObject_GetAttrString(result1, "size");
                                                                		PyObject *dim_obj = PyLong_FromLong(i);
                                                                		PyObject *dim_tuple = PyTuple_Pack(1, dim_obj);
                                                                		PyObject* size_dim_obj = PyObject_CallObject(torchsize_obj, dim_tuple);
                                                                		unsigned long long dim_size = PyLong_AsUnsignedLongLongMask(size_dim_obj);
                                                                		alloc_size_val *= dim_size;
                                                        		}

									PyObject* elem_size_obj = PyObject_GetAttrString(result1, "element_size");

									if(elem_size_obj != NULL) {
                                                                		//std::cout << "torch.tensor is intercepted 3 2\n";
                                                                        	//PyObject* elem_size_val_obj = PyObject_CallNoArgs(elem_size_obj);
                                                                        	PyObject* elem_size_val_obj = PyObject_Call(elem_size_obj, empty_tuple, NULL);
                                                                        	//std::cerr << "torch.tensor is intercepted 3 2 1\n";
                                                                        	if(elem_size_val_obj != NULL) {
                                                                                	//std::cout << "torch.tensor is intercepted 3 3\n";
                                                                                	unsigned long long element_size = PyLong_AsUnsignedLongLongMask(elem_size_val_obj);
                                                                                	alloc_size_val *= element_size;
                                                                        	}
                                                        		}


									py::object summary = extract_python_callpath(); 
									std::vector<py::handle> stack_vec;
									allocation_site_t *allocation_site = NULL;
									allocation_site_t *parent = NULL;

									update_allocation_site_tree(summary, &allocation_site, &parent);

									if (parent) {
										record_object_allocation_context(parent);
										int deviceID = -1;
										cudaGetDevice(&deviceID);

										fprintf(stderr, "to func call captured for GPU Object, offset value: %lx, allocation size: %ld recorded\n", offset_val1, alloc_size_val);
										adm_range_insert(offset_val1, alloc_size_val, parent->get_pc(),
												deviceID, "", ADM_STATE_ALLOC);
										range_nodes.push_back(new adm_range_t(
													offset_val1, alloc_size_val, parent->get_object_id(), deviceID));
									}	

								} 
							}
							//#endif
							return py::reinterpret_borrow<py::object>(result1);
					});
					PyObject* orig_torchcuda_func = PyObject_GetAttrString(result, "cuda");
					result_obj.attr("cuda") = py::cpp_function([orig_torchcuda_func](const py::args &args, const py::kwargs &kwargs) {
							std::cerr << "tensor.cuda is intercepted\n";
							//PyObject * allocated_mem_ptr = allocated_mem.ptr();//orig_array_func(args/*, kwargs*/);
							PyObject* result1 = PyObject_Call(orig_torchcuda_func, (PyObject *) args.ptr(), (PyObject *) kwargs.ptr());
							//#if 0
							PyObject* ptr_obj = PyObject_GetAttrString(result1, "data_ptr");
							//PyObject* ptr_val_obj = PyObject_CallNoArgs(ptr_obj);
							PyObject *empty_tuple = PyTuple_Pack(0);
							PyObject* ptr_val_obj = PyObject_Call(ptr_obj, empty_tuple, NULL);
							unsigned long long offset_val1 = PyLong_AsUnsignedLongLongMask(ptr_val_obj);
							//fprintf(stderr, "offset value: %lx\n", offset_val);
							//#if 0
							//PyObject* size_obj = PyObject_GetAttrString(result1, "__len__");
							PyObject* size_obj = PyObject_GetAttrString(result1, "dim");
							//PyObject* size_val_obj = PyObject_CallNoArgs(size_obj);
							PyObject* size_val_obj = PyObject_Call(size_obj, empty_tuple, NULL);
							unsigned long long element_count = PyLong_AsUnsignedLongLongMask(size_val_obj);

							unsigned long long alloc_size_val = 1;
//#if 0
							for(long i = 0; i < element_count; i++) {
								PyObject* torchsize_obj = PyObject_GetAttrString(result1, "size");
								PyObject *dim_obj = PyLong_FromLong(i);
								PyObject *dim_tuple = PyTuple_Pack(1, dim_obj);
								PyObject* size_dim_obj = PyObject_CallObject(torchsize_obj, dim_tuple);
								unsigned long long dim_size = PyLong_AsUnsignedLongLongMask(size_dim_obj);
								alloc_size_val *= dim_size;
							}
//#endif
							//#if 0
							std::cerr << "tensor.cuda is intercepted before element_size\n";
							PyObject* elem_size_obj = PyObject_GetAttrString(result1, "element_size");
                                        		if(elem_size_obj != NULL) {
                                        			//std::cout << "torch.tensor is intercepted 3 2\n";
                                        				//PyObject* elem_size_val_obj = PyObject_CallNoArgs(elem_size_obj);
									PyObject* elem_size_val_obj = PyObject_Call(elem_size_obj, empty_tuple, NULL);
                                        				//std::cerr << "torch.tensor is intercepted 3 2 1\n";
                                        				if(elem_size_val_obj != NULL) {
                                        					//std::cout << "torch.tensor is intercepted 3 3\n";
                                        					unsigned long long element_size = PyLong_AsUnsignedLongLongMask(elem_size_val_obj);
										alloc_size_val *= element_size;
									}
							}
							//PyObject* elem_size_obj = PyObject_GetAttrString(result1, "element_size");
							std::cerr << "tensor.cuda is intercepted after element_size\n";
							//#if 0
							//if(elem_size_obj != NULL) {
							std::cerr << "tensor.cuda is intercepted after if 1\n";
							fprintf(stderr, "cuda func call captured, offset value: %lx, allocation size: %ld\n", offset_val1, alloc_size_val);	
							py::object summary = extract_python_callpath();
							std::vector<py::handle> stack_vec;
							allocation_site_t *allocation_site = NULL;
							allocation_site_t *parent = NULL;

							update_allocation_site_tree(summary, &allocation_site, &parent);

							if (parent) {
								record_object_allocation_context(parent);
								int deviceID = -1;
								cudaGetDevice(&deviceID);

								adm_range_insert(offset_val1, alloc_size_val, parent->get_pc(),
										deviceID, "", ADM_STATE_ALLOC);
								range_nodes.push_back(new adm_range_t(
											offset_val1, alloc_size_val, parent->get_object_id(), deviceID));
							}
							//}
							//#endif
							std::cerr << "tensor.cuda is intercepted after if 2\n";	
							//#endif
							return py::reinterpret_borrow<py::object>(result1);
					});	

					return result_obj;//result;//orig_empty_like_func(args/*, kwargs*/);
			});
		} else if(func_name == "randn") {
			std::cerr << "torch.randn is injected\n";
			PyObject* mod = obj.ptr();
			PyObject* orig_torchrandn_func = PyObject_GetAttrString(mod, "randn");		

			obj.attr("randn") = py::cpp_function([orig_torchrandn_func](const py::args &args, const py::kwargs &kwargs) {
					//std::cout << msg.cast<std::string>();
					std::cout << "torch.randn is intercepted\n";
					PyObject* result = PyObject_Call(orig_torchrandn_func, (PyObject *) args.ptr(), (PyObject *) kwargs.ptr());

					PyObject* is_cuda_obj = PyObject_GetAttrString(result, "is_cuda");
                                        if (PyBool_Check(is_cuda_obj)) {
                                                if(is_cuda_obj == Py_True) {
							//std::cout << "torch.tensor is intercepted 1\n";
							PyObject* ptr_obj = PyObject_GetAttrString(result, "data_ptr");
							PyObject *empty_tuple = PyTuple_Pack(0);
							//PyObject* ptr_val_obj = PyObject_CallNoArgs(ptr_obj, empty_tuple, NULL);
							PyObject* ptr_val_obj = PyObject_Call(ptr_obj, empty_tuple, NULL);

							//std::cout << "torch.tensor is intercepted 2\n";
							unsigned long long offset_val = PyLong_AsUnsignedLongLongMask(ptr_val_obj);
							//PyObject* size_obj = PyObject_GetAttrString(result, "__len__");
							PyObject* size_obj = PyObject_GetAttrString(result, "dim");
							//PyObject* size_val_obj = PyObject_CallNoArgs(size_obj);
							PyObject* size_val_obj = PyObject_Call(size_obj, empty_tuple, NULL);
							//std::cout << "torch.tensor is intercepted 3\n";
							unsigned long long element_count = PyLong_AsUnsignedLongLongMask(size_val_obj);
							unsigned long long alloc_size_val = 1;

							for(long i = 0; i < element_count; i++) {
                                                                PyObject* torchsize_obj = PyObject_GetAttrString(result, "size");
                                                                PyObject *dim_obj = PyLong_FromLong(i);
                                                                PyObject *dim_tuple = PyTuple_Pack(1, dim_obj);
                                                                PyObject* size_dim_obj = PyObject_CallObject(torchsize_obj, dim_tuple);
                                                                unsigned long long dim_size = PyLong_AsUnsignedLongLongMask(size_dim_obj);
                                                                alloc_size_val *= dim_size;
                                                        }
							//std::cout << "torch.tensor is intercepted 3 1\n";
							PyObject* elem_size_obj = PyObject_GetAttrString(result, "element_size");

							if(elem_size_obj != NULL) {
                                                        	PyObject* elem_size_val_obj = PyObject_Call(elem_size_obj, empty_tuple, NULL);
                                                        	//std::cerr << "torch.tensor is intercepted 3 2 1\n";
                                                        	if(elem_size_val_obj != NULL) {
                                                        		//std::cout << "torch.tensor is intercepted 3 3\n";
                                                        		unsigned long long element_size = PyLong_AsUnsignedLongLongMask(elem_size_val_obj);
                                                       			alloc_size_val *= element_size;
									py::object summary = extract_python_callpath();
									//#if 0
									std::vector<py::handle> stack_vec;
									allocation_site_t *allocation_site = NULL;
									allocation_site_t *parent = NULL;

									update_allocation_site_tree(summary, &allocation_site, &parent);

									if (parent) {
										record_object_allocation_context(parent);
										int deviceID = -1;
										cudaGetDevice(&deviceID);

										fprintf(stderr, "torch.tensor func call captured for GPU Object, offset value: %lx, allocation size: %ld recorded\n", offset_val, alloc_size_val);
										adm_range_insert(offset_val, alloc_size_val, parent->get_pc(),
											deviceID, "", ADM_STATE_ALLOC);
										range_nodes.push_back(new adm_range_t(
											offset_val, alloc_size_val, parent->get_object_id(), deviceID));
									}
								} 
							}
							//fprintf(stderr, "offset value: %lx, allocation size: %ld\n", offset_val, alloc_size_val);
						}
					}	

					py::object result_obj = py::reinterpret_borrow<py::object>(result);
					PyObject* orig_torchto_func = PyObject_GetAttrString(result, "to"); 
					result_obj.attr("to") = py::cpp_function([orig_torchto_func](const py::args &args, const py::kwargs &kwargs) {

							std::cerr << "torch.tensor.to is intercepted\n";
							//py::object allocated_mem = orig_empty_like_func(args/*, kwargs*/);

							//PyObject * allocated_mem_ptr = allocated_mem.ptr();//orig_array_func(args/*, kwargs*/);
							PyObject* result1 = PyObject_Call(orig_torchto_func, (PyObject *) args.ptr(), (PyObject *) kwargs.ptr());
							PyObject* is_cuda_obj = PyObject_GetAttrString(result1, "is_cuda");
                                                        if (PyBool_Check(is_cuda_obj)) {
                                                                if(is_cuda_obj == Py_True) {
									//#if 0
									PyObject* ptr_obj = PyObject_GetAttrString(result1, "data_ptr");
									std::cerr << "here 4\n";
									//PyObject* ptr_val_obj = PyObject_CallNoArgs(ptr_obj);
									PyObject *empty_tuple = PyTuple_Pack(0);
									PyObject* ptr_val_obj = PyObject_Call(ptr_obj, empty_tuple, NULL);
									unsigned long long offset_val1 = PyLong_AsUnsignedLongLongMask(ptr_val_obj);
									//fprintf(stderr, "offset value: %lx\n", offset_val);
									//PyObject* size_obj = PyObject_GetAttrString(result1, "__len__");
									PyObject* size_obj = PyObject_GetAttrString(result1, "dim");
									//PyObject* size_val_obj = PyObject_CallNoArgs(size_obj);
									PyObject* size_val_obj = PyObject_Call(size_obj, empty_tuple, NULL);
									unsigned long long element_count = PyLong_AsUnsignedLongLongMask(size_val_obj);
									unsigned long long alloc_size_val = 1;

									for(long i = 0; i < element_count; i++) {
                                                                		PyObject* torchsize_obj = PyObject_GetAttrString(result1, "size");
                                                                		PyObject *dim_obj = PyLong_FromLong(i);
                                                                		PyObject *dim_tuple = PyTuple_Pack(1, dim_obj);
                                                                		PyObject* size_dim_obj = PyObject_CallObject(torchsize_obj, dim_tuple);
                                                                		unsigned long long dim_size = PyLong_AsUnsignedLongLongMask(size_dim_obj);
                                                                		alloc_size_val *= dim_size;
                                                        		}

									PyObject* elem_size_obj = PyObject_GetAttrString(result1, "element_size");

									if(elem_size_obj != NULL) {
                                                                		//std::cout << "torch.tensor is intercepted 3 2\n";
                                                                        	//PyObject* elem_size_val_obj = PyObject_CallNoArgs(elem_size_obj);
                                                                        	PyObject* elem_size_val_obj = PyObject_Call(elem_size_obj, empty_tuple, NULL);
                                                                        	//std::cerr << "torch.tensor is intercepted 3 2 1\n";
                                                                        	if(elem_size_val_obj != NULL) {
                                                                                	//std::cout << "torch.tensor is intercepted 3 3\n";
                                                                                	unsigned long long element_size = PyLong_AsUnsignedLongLongMask(elem_size_val_obj);
                                                                                	alloc_size_val *= element_size;
                                                                        	}
                                                        		}

									py::object summary = extract_python_callpath(); 
									std::vector<py::handle> stack_vec;
									allocation_site_t *allocation_site = NULL;
									allocation_site_t *parent = NULL;

									update_allocation_site_tree(summary, &allocation_site, &parent);

									if (parent) {
										record_object_allocation_context(parent);
										int deviceID = -1;
										cudaGetDevice(&deviceID);

										fprintf(stderr, "to func call captured for GPU Object, offset value: %lx, allocation size: %ld recorded\n", offset_val1, alloc_size_val);
										adm_range_insert(offset_val1, alloc_size_val, parent->get_pc(),
												deviceID, "", ADM_STATE_ALLOC);
										range_nodes.push_back(new adm_range_t(
													offset_val1, alloc_size_val, parent->get_object_id(), deviceID));
									}	

								} 
							}
							//#endif
							return py::reinterpret_borrow<py::object>(result1);
					});
										
					PyObject* orig_torchcuda_func = PyObject_GetAttrString(result, "cuda");
					result_obj.attr("cuda") = py::cpp_function([orig_torchcuda_func](const py::args &args, const py::kwargs &kwargs) {
							std::cerr << "tensor.cuda is intercepted\n";
							//PyObject * allocated_mem_ptr = allocated_mem.ptr();//orig_array_func(args/*, kwargs*/);
							PyObject* result1 = PyObject_Call(orig_torchcuda_func, (PyObject *) args.ptr(), (PyObject *) kwargs.ptr());
							//#if 0
							PyObject* ptr_obj = PyObject_GetAttrString(result1, "data_ptr");
							//PyObject* ptr_val_obj = PyObject_CallNoArgs(ptr_obj);
							PyObject *empty_tuple = PyTuple_Pack(0);
							PyObject* ptr_val_obj = PyObject_Call(ptr_obj, empty_tuple, NULL);
							unsigned long long offset_val1 = PyLong_AsUnsignedLongLongMask(ptr_val_obj);
							//fprintf(stderr, "offset value: %lx\n", offset_val);
							//#if 0
							//PyObject* size_obj = PyObject_GetAttrString(result1, "__len__");
							PyObject* size_obj = PyObject_GetAttrString(result1, "dim");
							//PyObject* size_val_obj = PyObject_CallNoArgs(size_obj);
							PyObject* size_val_obj = PyObject_Call(size_obj, empty_tuple, NULL);
							unsigned long long element_count = PyLong_AsUnsignedLongLongMask(size_val_obj);

							unsigned long long alloc_size_val = 1;
//#if 0
							for(long i = 0; i < element_count; i++) {
								PyObject* torchsize_obj = PyObject_GetAttrString(result1, "size");
								PyObject *dim_obj = PyLong_FromLong(i);
								PyObject *dim_tuple = PyTuple_Pack(1, dim_obj);
								PyObject* size_dim_obj = PyObject_CallObject(torchsize_obj, dim_tuple);
								unsigned long long dim_size = PyLong_AsUnsignedLongLongMask(size_dim_obj);
								alloc_size_val *= dim_size;
							}
//#endif
							//#if 0
							std::cerr << "tensor.cuda is intercepted before element_size\n";
							PyObject* elem_size_obj = PyObject_GetAttrString(result1, "element_size");
                                        		if(elem_size_obj != NULL) {
                                        			//std::cout << "torch.tensor is intercepted 3 2\n";
                                        				//PyObject* elem_size_val_obj = PyObject_CallNoArgs(elem_size_obj);
									PyObject* elem_size_val_obj = PyObject_Call(elem_size_obj, empty_tuple, NULL);
                                        				//std::cerr << "torch.tensor is intercepted 3 2 1\n";
                                        				if(elem_size_val_obj != NULL) {
                                        					//std::cout << "torch.tensor is intercepted 3 3\n";
                                        					unsigned long long element_size = PyLong_AsUnsignedLongLongMask(elem_size_val_obj);
										alloc_size_val *= element_size;
									}
							}
							//PyObject* elem_size_obj = PyObject_GetAttrString(result1, "element_size");
							std::cerr << "tensor.cuda is intercepted after element_size\n";
							//#if 0
							//if(elem_size_obj != NULL) {
							std::cerr << "tensor.cuda is intercepted after if 1\n";
							fprintf(stderr, "cuda func call captured, offset value: %lx, allocation size: %ld\n", offset_val1, alloc_size_val);	
							py::object summary = extract_python_callpath();
							std::vector<py::handle> stack_vec;
							allocation_site_t *allocation_site = NULL;
							allocation_site_t *parent = NULL;

							update_allocation_site_tree(summary, &allocation_site, &parent);

							if (parent) {
								record_object_allocation_context(parent);
								int deviceID = -1;
								cudaGetDevice(&deviceID);

								adm_range_insert(offset_val1, alloc_size_val, parent->get_pc(),
										deviceID, "", ADM_STATE_ALLOC);
								range_nodes.push_back(new adm_range_t(
											offset_val1, alloc_size_val, parent->get_object_id(), deviceID));
							}
							//}
							//#endif
							std::cerr << "tensor.cuda is intercepted after if 2\n";	
							//#endif
							return py::reinterpret_borrow<py::object>(result1);
					});	

					return result_obj;//result;//orig_empty_like_func(args/*, kwargs*/);
			});
		} else if(func_name == "full") {
			std::cerr << "torch.full is injected\n";
			PyObject* mod = obj.ptr();
			PyObject* orig_torchfull_func = PyObject_GetAttrString(mod, "full");		

			obj.attr("full") = py::cpp_function([orig_torchfull_func](const py::args &args, const py::kwargs &kwargs) {
					//std::cout << msg.cast<std::string>();
					std::cout << "torch.full is intercepted\n";
					PyObject* result = PyObject_Call(orig_torchfull_func, (PyObject *) args.ptr(), (PyObject *) kwargs.ptr());

					PyObject* is_cuda_obj = PyObject_GetAttrString(result, "is_cuda");
                                        if (PyBool_Check(is_cuda_obj)) {
                                                if(is_cuda_obj == Py_True) {
							//std::cout << "torch.tensor is intercepted 1\n";
							PyObject* ptr_obj = PyObject_GetAttrString(result, "data_ptr");
							PyObject *empty_tuple = PyTuple_Pack(0);
							//PyObject* ptr_val_obj = PyObject_CallNoArgs(ptr_obj, empty_tuple, NULL);
							PyObject* ptr_val_obj = PyObject_Call(ptr_obj, empty_tuple, NULL);

							//std::cout << "torch.tensor is intercepted 2\n";
							unsigned long long offset_val = PyLong_AsUnsignedLongLongMask(ptr_val_obj);
							//PyObject* size_obj = PyObject_GetAttrString(result, "__len__");
							PyObject* size_obj = PyObject_GetAttrString(result, "dim");
							//PyObject* size_val_obj = PyObject_CallNoArgs(size_obj);
							PyObject* size_val_obj = PyObject_Call(size_obj, empty_tuple, NULL);
							//std::cout << "torch.tensor is intercepted 3\n";
							unsigned long long element_count = PyLong_AsUnsignedLongLongMask(size_val_obj);
							unsigned long long alloc_size_val = 1;

							for(long i = 0; i < element_count; i++) {
                                                                PyObject* torchsize_obj = PyObject_GetAttrString(result, "size");
                                                                PyObject *dim_obj = PyLong_FromLong(i);
                                                                PyObject *dim_tuple = PyTuple_Pack(1, dim_obj);
                                                                PyObject* size_dim_obj = PyObject_CallObject(torchsize_obj, dim_tuple);
                                                                unsigned long long dim_size = PyLong_AsUnsignedLongLongMask(size_dim_obj);
                                                                alloc_size_val *= dim_size;
                                                        }
							//std::cout << "torch.tensor is intercepted 3 1\n";
							PyObject* elem_size_obj = PyObject_GetAttrString(result, "element_size");

							if(elem_size_obj != NULL) {
                                                        	PyObject* elem_size_val_obj = PyObject_Call(elem_size_obj, empty_tuple, NULL);
                                                        	//std::cerr << "torch.tensor is intercepted 3 2 1\n";
                                                        	if(elem_size_val_obj != NULL) {
                                                        		//std::cout << "torch.tensor is intercepted 3 3\n";
                                                        		unsigned long long element_size = PyLong_AsUnsignedLongLongMask(elem_size_val_obj);
                                                       			alloc_size_val *= element_size;
									py::object summary = extract_python_callpath();
									//#if 0
									std::vector<py::handle> stack_vec;
									allocation_site_t *allocation_site = NULL;
									allocation_site_t *parent = NULL;

									update_allocation_site_tree(summary, &allocation_site, &parent);

									if (parent) {
										record_object_allocation_context(parent);
										int deviceID = -1;
										cudaGetDevice(&deviceID);

										fprintf(stderr, "torch.tensor func call captured for GPU Object, offset value: %lx, allocation size: %ld recorded\n", offset_val, alloc_size_val);
										adm_range_insert(offset_val, alloc_size_val, parent->get_pc(),
											deviceID, "", ADM_STATE_ALLOC);
										range_nodes.push_back(new adm_range_t(
											offset_val, alloc_size_val, parent->get_object_id(), deviceID));
									}
								} 
							}
							//fprintf(stderr, "offset value: %lx, allocation size: %ld\n", offset_val, alloc_size_val);
						}
					}	

					py::object result_obj = py::reinterpret_borrow<py::object>(result);
					PyObject* orig_torchto_func = PyObject_GetAttrString(result, "to"); 
					result_obj.attr("to") = py::cpp_function([orig_torchto_func](const py::args &args, const py::kwargs &kwargs) {

							std::cerr << "torch.tensor.to is intercepted\n";
							//py::object allocated_mem = orig_empty_like_func(args/*, kwargs*/);

							//PyObject * allocated_mem_ptr = allocated_mem.ptr();//orig_array_func(args/*, kwargs*/);
							PyObject* result1 = PyObject_Call(orig_torchto_func, (PyObject *) args.ptr(), (PyObject *) kwargs.ptr());
							PyObject* is_cuda_obj = PyObject_GetAttrString(result1, "is_cuda");
                                                        if (PyBool_Check(is_cuda_obj)) {
                                                                if(is_cuda_obj == Py_True) {
									//#if 0
									PyObject* ptr_obj = PyObject_GetAttrString(result1, "data_ptr");
									std::cerr << "here 4\n";
									//PyObject* ptr_val_obj = PyObject_CallNoArgs(ptr_obj);
									PyObject *empty_tuple = PyTuple_Pack(0);
									PyObject* ptr_val_obj = PyObject_Call(ptr_obj, empty_tuple, NULL);
									unsigned long long offset_val1 = PyLong_AsUnsignedLongLongMask(ptr_val_obj);
									//fprintf(stderr, "offset value: %lx\n", offset_val);
									//PyObject* size_obj = PyObject_GetAttrString(result1, "__len__");
									PyObject* size_obj = PyObject_GetAttrString(result1, "dim");
									//PyObject* size_val_obj = PyObject_CallNoArgs(size_obj);
									PyObject* size_val_obj = PyObject_Call(size_obj, empty_tuple, NULL);
									unsigned long long element_count = PyLong_AsUnsignedLongLongMask(size_val_obj);
									unsigned long long alloc_size_val = 1;

									for(long i = 0; i < element_count; i++) {
                                                                		PyObject* torchsize_obj = PyObject_GetAttrString(result1, "size");
                                                                		PyObject *dim_obj = PyLong_FromLong(i);
                                                                		PyObject *dim_tuple = PyTuple_Pack(1, dim_obj);
                                                                		PyObject* size_dim_obj = PyObject_CallObject(torchsize_obj, dim_tuple);
                                                                		unsigned long long dim_size = PyLong_AsUnsignedLongLongMask(size_dim_obj);
                                                                		alloc_size_val *= dim_size;
                                                        		}

									PyObject* elem_size_obj = PyObject_GetAttrString(result1, "element_size");

									if(elem_size_obj != NULL) {
                                                                		//std::cout << "torch.tensor is intercepted 3 2\n";
                                                                        	//PyObject* elem_size_val_obj = PyObject_CallNoArgs(elem_size_obj);
                                                                        	PyObject* elem_size_val_obj = PyObject_Call(elem_size_obj, empty_tuple, NULL);
                                                                        	//std::cerr << "torch.tensor is intercepted 3 2 1\n";
                                                                        	if(elem_size_val_obj != NULL) {
                                                                                	//std::cout << "torch.tensor is intercepted 3 3\n";
                                                                                	unsigned long long element_size = PyLong_AsUnsignedLongLongMask(elem_size_val_obj);
                                                                                	alloc_size_val *= element_size;
                                                                        	}
                                                        		}

									py::object summary = extract_python_callpath(); 
									std::vector<py::handle> stack_vec;
									allocation_site_t *allocation_site = NULL;
									allocation_site_t *parent = NULL;

									update_allocation_site_tree(summary, &allocation_site, &parent);

									if (parent) {
										record_object_allocation_context(parent);
										int deviceID = -1;
										cudaGetDevice(&deviceID);

										fprintf(stderr, "to func call captured for GPU Object, offset value: %lx, allocation size: %ld recorded\n", offset_val1, alloc_size_val);
										adm_range_insert(offset_val1, alloc_size_val, parent->get_pc(),
												deviceID, "", ADM_STATE_ALLOC);
										range_nodes.push_back(new adm_range_t(
													offset_val1, alloc_size_val, parent->get_object_id(), deviceID));
									}	

								} 
							}
							//#endif
							return py::reinterpret_borrow<py::object>(result1);
					});
										
					PyObject* orig_torchcuda_func = PyObject_GetAttrString(result, "cuda");
					result_obj.attr("cuda") = py::cpp_function([orig_torchcuda_func](const py::args &args, const py::kwargs &kwargs) {
							std::cerr << "tensor.cuda is intercepted\n";
							//PyObject * allocated_mem_ptr = allocated_mem.ptr();//orig_array_func(args/*, kwargs*/);
							PyObject* result1 = PyObject_Call(orig_torchcuda_func, (PyObject *) args.ptr(), (PyObject *) kwargs.ptr());
							//#if 0
							PyObject* ptr_obj = PyObject_GetAttrString(result1, "data_ptr");
							//PyObject* ptr_val_obj = PyObject_CallNoArgs(ptr_obj);
							PyObject *empty_tuple = PyTuple_Pack(0);
							PyObject* ptr_val_obj = PyObject_Call(ptr_obj, empty_tuple, NULL);
							unsigned long long offset_val1 = PyLong_AsUnsignedLongLongMask(ptr_val_obj);
							//fprintf(stderr, "offset value: %lx\n", offset_val);
							//#if 0
							//PyObject* size_obj = PyObject_GetAttrString(result1, "__len__");
							PyObject* size_obj = PyObject_GetAttrString(result1, "dim");
							//PyObject* size_val_obj = PyObject_CallNoArgs(size_obj);
							PyObject* size_val_obj = PyObject_Call(size_obj, empty_tuple, NULL);
							unsigned long long element_count = PyLong_AsUnsignedLongLongMask(size_val_obj);

							unsigned long long alloc_size_val = 1;
//#if 0
							for(long i = 0; i < element_count; i++) {
								PyObject* torchsize_obj = PyObject_GetAttrString(result1, "size");
								PyObject *dim_obj = PyLong_FromLong(i);
								PyObject *dim_tuple = PyTuple_Pack(1, dim_obj);
								PyObject* size_dim_obj = PyObject_CallObject(torchsize_obj, dim_tuple);
								unsigned long long dim_size = PyLong_AsUnsignedLongLongMask(size_dim_obj);
								alloc_size_val *= dim_size;
							}
//#endif
							//#if 0
							std::cerr << "tensor.cuda is intercepted before element_size\n";
							PyObject* elem_size_obj = PyObject_GetAttrString(result1, "element_size");
                                        		if(elem_size_obj != NULL) {
                                        			//std::cout << "torch.tensor is intercepted 3 2\n";
                                        				//PyObject* elem_size_val_obj = PyObject_CallNoArgs(elem_size_obj);
									PyObject* elem_size_val_obj = PyObject_Call(elem_size_obj, empty_tuple, NULL);
                                        				//std::cerr << "torch.tensor is intercepted 3 2 1\n";
                                        				if(elem_size_val_obj != NULL) {
                                        					//std::cout << "torch.tensor is intercepted 3 3\n";
                                        					unsigned long long element_size = PyLong_AsUnsignedLongLongMask(elem_size_val_obj);
										alloc_size_val *= element_size;
									}
							}
							//PyObject* elem_size_obj = PyObject_GetAttrString(result1, "element_size");
							std::cerr << "tensor.cuda is intercepted after element_size\n";
							//#if 0
							//if(elem_size_obj != NULL) {
							std::cerr << "tensor.cuda is intercepted after if 1\n";
							fprintf(stderr, "cuda func call captured, offset value: %lx, allocation size: %ld\n", offset_val1, alloc_size_val);	
							py::object summary = extract_python_callpath();
							std::vector<py::handle> stack_vec;
							allocation_site_t *allocation_site = NULL;
							allocation_site_t *parent = NULL;

							update_allocation_site_tree(summary, &allocation_site, &parent);

							if (parent) {
								record_object_allocation_context(parent);
								int deviceID = -1;
								cudaGetDevice(&deviceID);

								adm_range_insert(offset_val1, alloc_size_val, parent->get_pc(),
										deviceID, "", ADM_STATE_ALLOC);
								range_nodes.push_back(new adm_range_t(
											offset_val1, alloc_size_val, parent->get_object_id(), deviceID));
							}
							//}
							//#endif
							std::cerr << "tensor.cuda is intercepted after if 2\n";	
							//#endif
							return py::reinterpret_borrow<py::object>(result1);
					});	

					return result_obj;//result;//orig_empty_like_func(args/*, kwargs*/);
			});
		} else if(func_name == "from_numpy") {
			std::cerr << "torch.from_numpy is injected\n";
			PyObject* mod = obj.ptr();
			PyObject* orig_torchfromnumpy_func = PyObject_GetAttrString(mod, "from_numpy");		

			obj.attr("from_numpy") = py::cpp_function([orig_torchfromnumpy_func](const py::args &args, const py::kwargs &kwargs) {
					//std::cout << msg.cast<std::string>();
					std::cout << "torch.from_numpy is intercepted\n";
					PyObject* result = PyObject_Call(orig_torchfromnumpy_func, (PyObject *) args.ptr(), (PyObject *) kwargs.ptr());

					PyObject* is_cuda_obj = PyObject_GetAttrString(result, "is_cuda");
                                        if (PyBool_Check(is_cuda_obj)) {
                                                if(is_cuda_obj == Py_True) {
							//std::cout << "torch.tensor is intercepted 1\n";
							PyObject* ptr_obj = PyObject_GetAttrString(result, "data_ptr");
							PyObject *empty_tuple = PyTuple_Pack(0);
							//PyObject* ptr_val_obj = PyObject_CallNoArgs(ptr_obj, empty_tuple, NULL);
							PyObject* ptr_val_obj = PyObject_Call(ptr_obj, empty_tuple, NULL);

							//std::cout << "torch.tensor is intercepted 2\n";
							unsigned long long offset_val = PyLong_AsUnsignedLongLongMask(ptr_val_obj);
							//PyObject* size_obj = PyObject_GetAttrString(result, "__len__");
							PyObject* size_obj = PyObject_GetAttrString(result, "dim");
							//PyObject* size_val_obj = PyObject_CallNoArgs(size_obj);
							PyObject* size_val_obj = PyObject_Call(size_obj, empty_tuple, NULL);
							//std::cout << "torch.tensor is intercepted 3\n";
							unsigned long long element_count = PyLong_AsUnsignedLongLongMask(size_val_obj);
							unsigned long long alloc_size_val = 1;

							for(long i = 0; i < element_count; i++) {
                                                                PyObject* torchsize_obj = PyObject_GetAttrString(result, "size");
                                                                PyObject *dim_obj = PyLong_FromLong(i);
                                                                PyObject *dim_tuple = PyTuple_Pack(1, dim_obj);
                                                                PyObject* size_dim_obj = PyObject_CallObject(torchsize_obj, dim_tuple);
                                                                unsigned long long dim_size = PyLong_AsUnsignedLongLongMask(size_dim_obj);
                                                                alloc_size_val *= dim_size;
                                                        }
							//std::cout << "torch.tensor is intercepted 3 1\n";
							PyObject* elem_size_obj = PyObject_GetAttrString(result, "element_size");

							if(elem_size_obj != NULL) {
                                                        	PyObject* elem_size_val_obj = PyObject_Call(elem_size_obj, empty_tuple, NULL);
                                                        	//std::cerr << "torch.tensor is intercepted 3 2 1\n";
                                                        	if(elem_size_val_obj != NULL) {
                                                        		//std::cout << "torch.tensor is intercepted 3 3\n";
                                                        		unsigned long long element_size = PyLong_AsUnsignedLongLongMask(elem_size_val_obj);
                                                       			alloc_size_val *= element_size;
									py::object summary = extract_python_callpath();
									//#if 0
									std::vector<py::handle> stack_vec;
									allocation_site_t *allocation_site = NULL;
									allocation_site_t *parent = NULL;

									update_allocation_site_tree(summary, &allocation_site, &parent);

									if (parent) {
										record_object_allocation_context(parent);
										int deviceID = -1;
										cudaGetDevice(&deviceID);

										fprintf(stderr, "torch.tensor func call captured for GPU Object, offset value: %lx, allocation size: %ld recorded\n", offset_val, alloc_size_val);
										adm_range_insert(offset_val, alloc_size_val, parent->get_pc(),
											deviceID, "", ADM_STATE_ALLOC);
										range_nodes.push_back(new adm_range_t(
											offset_val, alloc_size_val, parent->get_object_id(), deviceID));
									}
								} 
							}
							//fprintf(stderr, "offset value: %lx, allocation size: %ld\n", offset_val, alloc_size_val);
						}
					}	

					py::object result_obj = py::reinterpret_borrow<py::object>(result);
					PyObject* orig_torchto_func = PyObject_GetAttrString(result, "to"); 
					result_obj.attr("to") = py::cpp_function([orig_torchto_func](const py::args &args, const py::kwargs &kwargs) {

							std::cerr << "torch.tensor.to is intercepted\n";
							//py::object allocated_mem = orig_empty_like_func(args/*, kwargs*/);

							//PyObject * allocated_mem_ptr = allocated_mem.ptr();//orig_array_func(args/*, kwargs*/);
							PyObject* result1 = PyObject_Call(orig_torchto_func, (PyObject *) args.ptr(), (PyObject *) kwargs.ptr());
							PyObject* is_cuda_obj = PyObject_GetAttrString(result1, "is_cuda");
                                                        if (PyBool_Check(is_cuda_obj)) {
                                                                if(is_cuda_obj == Py_True) {
									//#if 0
									PyObject* ptr_obj = PyObject_GetAttrString(result1, "data_ptr");
									std::cerr << "here 4\n";
									//PyObject* ptr_val_obj = PyObject_CallNoArgs(ptr_obj);
									PyObject *empty_tuple = PyTuple_Pack(0);
									PyObject* ptr_val_obj = PyObject_Call(ptr_obj, empty_tuple, NULL);
									unsigned long long offset_val1 = PyLong_AsUnsignedLongLongMask(ptr_val_obj);
									//fprintf(stderr, "offset value: %lx\n", offset_val);
									//PyObject* size_obj = PyObject_GetAttrString(result1, "__len__");
									PyObject* size_obj = PyObject_GetAttrString(result1, "dim");
									//PyObject* size_val_obj = PyObject_CallNoArgs(size_obj);
									PyObject* size_val_obj = PyObject_Call(size_obj, empty_tuple, NULL);
									unsigned long long element_count = PyLong_AsUnsignedLongLongMask(size_val_obj);
									unsigned long long alloc_size_val = 1;

									for(long i = 0; i < element_count; i++) {
                                                                		PyObject* torchsize_obj = PyObject_GetAttrString(result1, "size");
                                                                		PyObject *dim_obj = PyLong_FromLong(i);
                                                                		PyObject *dim_tuple = PyTuple_Pack(1, dim_obj);
                                                                		PyObject* size_dim_obj = PyObject_CallObject(torchsize_obj, dim_tuple);
                                                                		unsigned long long dim_size = PyLong_AsUnsignedLongLongMask(size_dim_obj);
                                                                		alloc_size_val *= dim_size;
                                                        		}

									PyObject* elem_size_obj = PyObject_GetAttrString(result1, "element_size");

									if(elem_size_obj != NULL) {
                                                                		//std::cout << "torch.tensor is intercepted 3 2\n";
                                                                        	//PyObject* elem_size_val_obj = PyObject_CallNoArgs(elem_size_obj);
                                                                        	PyObject* elem_size_val_obj = PyObject_Call(elem_size_obj, empty_tuple, NULL);
                                                                        	//std::cerr << "torch.tensor is intercepted 3 2 1\n";
                                                                        	if(elem_size_val_obj != NULL) {
                                                                                	//std::cout << "torch.tensor is intercepted 3 3\n";
                                                                                	unsigned long long element_size = PyLong_AsUnsignedLongLongMask(elem_size_val_obj);
                                                                                	alloc_size_val *= element_size;
                                                                        	}
                                                        		}

									py::object summary = extract_python_callpath(); 
									std::vector<py::handle> stack_vec;
									allocation_site_t *allocation_site = NULL;
									allocation_site_t *parent = NULL;

									update_allocation_site_tree(summary, &allocation_site, &parent);

									if (parent) {
										record_object_allocation_context(parent);
										int deviceID = -1;
										cudaGetDevice(&deviceID);

										fprintf(stderr, "to func call captured for GPU Object, offset value: %lx, allocation size: %ld recorded\n", offset_val1, alloc_size_val);
										adm_range_insert(offset_val1, alloc_size_val, parent->get_pc(),
												deviceID, "", ADM_STATE_ALLOC);
										range_nodes.push_back(new adm_range_t(
													offset_val1, alloc_size_val, parent->get_object_id(), deviceID));
									}	

								} 
							}
							//#endif
							return py::reinterpret_borrow<py::object>(result1);
					});
										
					PyObject* orig_torchcuda_func = PyObject_GetAttrString(result, "cuda");
					result_obj.attr("cuda") = py::cpp_function([orig_torchcuda_func](const py::args &args, const py::kwargs &kwargs) {
							std::cerr << "tensor.cuda is intercepted\n";
							//PyObject * allocated_mem_ptr = allocated_mem.ptr();//orig_array_func(args/*, kwargs*/);
							PyObject* result1 = PyObject_Call(orig_torchcuda_func, (PyObject *) args.ptr(), (PyObject *) kwargs.ptr());
							//#if 0
							PyObject* ptr_obj = PyObject_GetAttrString(result1, "data_ptr");
							//PyObject* ptr_val_obj = PyObject_CallNoArgs(ptr_obj);
							PyObject *empty_tuple = PyTuple_Pack(0);
							PyObject* ptr_val_obj = PyObject_Call(ptr_obj, empty_tuple, NULL);
							unsigned long long offset_val1 = PyLong_AsUnsignedLongLongMask(ptr_val_obj);
							//fprintf(stderr, "offset value: %lx\n", offset_val);
							//#if 0
							//PyObject* size_obj = PyObject_GetAttrString(result1, "__len__");
							PyObject* size_obj = PyObject_GetAttrString(result1, "dim");
							//PyObject* size_val_obj = PyObject_CallNoArgs(size_obj);
							PyObject* size_val_obj = PyObject_Call(size_obj, empty_tuple, NULL);
							unsigned long long element_count = PyLong_AsUnsignedLongLongMask(size_val_obj);

							unsigned long long alloc_size_val = 1;
//#if 0
							for(long i = 0; i < element_count; i++) {
								PyObject* torchsize_obj = PyObject_GetAttrString(result1, "size");
								PyObject *dim_obj = PyLong_FromLong(i);
								PyObject *dim_tuple = PyTuple_Pack(1, dim_obj);
								PyObject* size_dim_obj = PyObject_CallObject(torchsize_obj, dim_tuple);
								unsigned long long dim_size = PyLong_AsUnsignedLongLongMask(size_dim_obj);
								alloc_size_val *= dim_size;
							}
//#endif
							//#if 0
							std::cerr << "tensor.cuda is intercepted before element_size\n";
							PyObject* elem_size_obj = PyObject_GetAttrString(result1, "element_size");
                                        		if(elem_size_obj != NULL) {
                                        			//std::cout << "torch.tensor is intercepted 3 2\n";
                                        				//PyObject* elem_size_val_obj = PyObject_CallNoArgs(elem_size_obj);
									PyObject* elem_size_val_obj = PyObject_Call(elem_size_obj, empty_tuple, NULL);
                                        				//std::cerr << "torch.tensor is intercepted 3 2 1\n";
                                        				if(elem_size_val_obj != NULL) {
                                        					//std::cout << "torch.tensor is intercepted 3 3\n";
                                        					unsigned long long element_size = PyLong_AsUnsignedLongLongMask(elem_size_val_obj);
										alloc_size_val *= element_size;
									}
							}
							//PyObject* elem_size_obj = PyObject_GetAttrString(result1, "element_size");
							std::cerr << "tensor.cuda is intercepted after element_size\n";
							//#if 0
							//if(elem_size_obj != NULL) {
							std::cerr << "tensor.cuda is intercepted after if 1\n";
							fprintf(stderr, "cuda func call captured, offset value: %lx, allocation size: %ld\n", offset_val1, alloc_size_val);	
							py::object summary = extract_python_callpath();
							std::vector<py::handle> stack_vec;
							allocation_site_t *allocation_site = NULL;
							allocation_site_t *parent = NULL;

							update_allocation_site_tree(summary, &allocation_site, &parent);

							if (parent) {
								record_object_allocation_context(parent);
								int deviceID = -1;
								cudaGetDevice(&deviceID);

								adm_range_insert(offset_val1, alloc_size_val, parent->get_pc(),
										deviceID, "", ADM_STATE_ALLOC);
								range_nodes.push_back(new adm_range_t(
											offset_val1, alloc_size_val, parent->get_object_id(), deviceID));
							}
							//}
							//#endif
							std::cerr << "tensor.cuda is intercepted after if 2\n";	
							//#endif
							return py::reinterpret_borrow<py::object>(result1);
					});	

					return result_obj;//result;//orig_empty_like_func(args/*, kwargs*/);
			}); 
		} else if(func_name == "empty_like") {
			std::cerr << "torch.empty_like is injected\n";
			PyObject* mod = obj.ptr();
			PyObject* orig_torchemptylike_func = PyObject_GetAttrString(mod, "empty_like");
			obj.attr("empty_like") = py::cpp_function([orig_torchemptylike_func](const py::args &args, const py::kwargs &kwargs) {
				std::cerr << "torch.empty_like is intercepted\n";
				PyObject* result = PyObject_Call(orig_torchemptylike_func, (PyObject *) args.ptr(), (PyObject *) kwargs.ptr());
				py::object result_obj = py::reinterpret_borrow<py::object>(result);
				return result_obj;
			});
		} else if(func_name == "nn.parallel.comm") {
                        std::cerr << "nn.parallel.comm is injected\n";
                        PyObject* mod = obj.ptr();
                        PyObject* nn_obj = PyObject_GetAttrString(mod, "nn");
                        PyObject* parallel_obj = PyObject_GetAttrString(nn_obj, "parallel");
                        PyObject* comm_obj = PyObject_GetAttrString(parallel_obj, "comm");
                        PyObject*  orig_broadcastcoalesced_func = PyObject_GetAttrString(comm_obj, "broadcast_coalesced");

                        obj.attr("nn").attr("parallel").attr("comm").attr("broadcast_coalesced") = py::cpp_function([orig_broadcastcoalesced_func](const py::args &args, const py::kwargs &kwargs) {
                                        //std::cout << msg.cast<std::string>();
                                        std::cerr << "nn.parallel.comm.broadcast_coalesced is intercepted\n";
					if(code_context) {
                                        	py::object summary = extract_python_callpath();//extract_summary(walk_stack(py::none()));
						execution_site_t *execution_site = NULL;
						execution_site_t *parent = NULL;	
						update_exec_site_tree(summary, &execution_site, &parent);
						record_exec_context(parent);
					}
                                        PyObject* result = PyObject_Call(orig_broadcastcoalesced_func, (PyObject *) args.ptr(), (PyObject *) kwargs.ptr());
                                        return py::reinterpret_borrow<py::object>(result);//result;//orig_empty_like_func(args/*, kwargs*/);
                        });
			
			PyObject*  orig_broadcast_func = PyObject_GetAttrString(comm_obj, "broadcast");

			obj.attr("nn").attr("parallel").attr("comm").attr("broadcast") = py::cpp_function([orig_broadcast_func](const py::args &args, const py::kwargs &kwargs) {
                                        //std::cout << msg.cast<std::string>();
                                        std::cerr << "nn.parallel.comm.broadcast is intercepted\n";
					if(code_context) {
                                        	py::object summary = extract_python_callpath();//extract_summary(walk_stack(py::none()));
                                        	execution_site_t *execution_site = NULL;
                                        	execution_site_t *parent = NULL;
                                        	update_exec_site_tree(summary, &execution_site, &parent);
                                        	record_exec_context(parent); 
 					}
                                        PyObject* result = PyObject_Call(orig_broadcast_func, (PyObject *) args.ptr(), (PyObject *) kwargs.ptr());
                                        return py::reinterpret_borrow<py::object>(result);//result;//orig_empty_like_func(args/*, kwargs*/);
                        });

			PyObject*  orig_reduceadd_func = PyObject_GetAttrString(comm_obj, "reduce_add");

                        obj.attr("nn").attr("parallel").attr("comm").attr("reduce_add") = py::cpp_function([orig_reduceadd_func](const py::args &args, const py::kwargs &kwargs) {
                                        //std::cout << msg.cast<std::string>();
                                        std::cerr << "nn.parallel.comm.reduce_add is intercepted\n";
					if(code_context) {
                                       		py::object summary = extract_python_callpath();//extract_summary(walk_stack(py::none()));
                                        	execution_site_t *execution_site = NULL;
                                        	execution_site_t *parent = NULL;
                                        	update_exec_site_tree(summary, &execution_site, &parent);
                                        	record_exec_context(parent); 
					}
                                        PyObject* result = PyObject_Call(orig_reduceadd_func, (PyObject *) args.ptr(), (PyObject *) kwargs.ptr());
                                        return py::reinterpret_borrow<py::object>(result);//result;//orig_empty_like_func(args/*, kwargs*/);
                        });

			PyObject*  orig_scatter_func = PyObject_GetAttrString(comm_obj, "scatter");

                        obj.attr("nn").attr("parallel").attr("comm").attr("scatter") = py::cpp_function([orig_scatter_func](const py::args &args, const py::kwargs &kwargs) {
                                        //std::cout << msg.cast<std::string>();
                                        std::cerr << "nn.parallel.comm.scatter is intercepted\n";
					if(code_context) {
                                        	py::object summary = extract_python_callpath();//extract_summary(walk_stack(py::none()));
                                        	execution_site_t *execution_site = NULL;
                                        	execution_site_t *parent = NULL;
                                        	update_exec_site_tree(summary, &execution_site, &parent);
                                        	record_exec_context(parent);
					}
                                        PyObject* result = PyObject_Call(orig_scatter_func, (PyObject *) args.ptr(), (PyObject *) kwargs.ptr());
                                        return py::reinterpret_borrow<py::object>(result);//result;//orig_empty_like_func(args/*, kwargs*/);
                        });

			PyObject*  orig_gather_func = PyObject_GetAttrString(comm_obj, "gather");

                        obj.attr("nn").attr("parallel").attr("comm").attr("gather") = py::cpp_function([orig_gather_func](const py::args &args, const py::kwargs &kwargs) {
                                        //std::cout << msg.cast<std::string>();
                                        std::cerr << "nn.parallel.comm.gather is intercepted\n";
					if(code_context) {
                                        	py::object summary = extract_python_callpath();//extract_summary(walk_stack(py::none()));
                                        	execution_site_t *execution_site = NULL;
                                        	execution_site_t *parent = NULL;
                                        	update_exec_site_tree(summary, &execution_site, &parent);
                                        	record_exec_context(parent); 
					}
                                        PyObject* result = PyObject_Call(orig_gather_func, (PyObject *) args.ptr(), (PyObject *) kwargs.ptr());
                                        return py::reinterpret_borrow<py::object>(result);//result;//orig_empty_like_func(args/*, kwargs*/);
                        });

                }	
	};	    
	if(data_object_attribution) {
		my_injection(torch, "tensor");
		my_injection(torch, "randn");
		my_injection(torch, "full");
		my_injection(torch, "from_numpy");
		//my_injection(torch, "to");
		my_injection(torch, "empty_like");
	}
	my_injection(torch, "nn.parallel.comm");
	std::cerr << "until here\n";
}
