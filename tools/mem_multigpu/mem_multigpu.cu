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
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <map>
#include <sstream>
#include <string>
#include <unordered_set>
#include <unordered_map>

#include <adm_config.h>
#include <adm_common.h>
#include <adm_splay.h>
#include <adm_memory.h>
#include <adm_database.h>

/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

/* for channel */
#include "utils/channel.hpp"

/* contains definition of the mem_access_t structure */
#include "common.h"
#include "util.h"

#include "mpi.h"
#include "nvshmem.h"
#include "nvshmemx.h"

#define HEX(x)                                                          \
  "0x" << std::setfill('0') << std::setw(16) << std::hex << (uint64_t)x \
  << std::dec

#define CHANNEL_SIZE (1l << 30)

#define JSON 0
#define EQUAL_STRS 0

#define FILE_NAME_SIZE 256
#define PATH_NAME_SIZE 5000

#include "adm.h"

using namespace adamant;

//static adm_splay_tree_t* tree = nullptr;
static bool object_attribution = false;
pool_t<adm_splay_tree_t, ADM_DB_OBJ_BLOCKSIZE>* nodes = nullptr;
pool_t<adm_range_t, ADM_DB_OBJ_BLOCKSIZE>* ranges = nullptr;
static int global_index = 0;
//static std::pair<std::vector<int>, std::vector<int>> line_tracking;
std::map<std::string, std::tuple<std::string, std::vector<int>, std::vector<int>>> line_tracking;

struct CTXstate
{
  /* context id */
  int id;

  /* Channel used to communicate from GPU to CPU receiving thread */
  ChannelDev *channel_dev;
  ChannelHost channel_host;
};

struct MemoryAllocation
{
  int deviceID;
  uint64_t pointer;
  uint64_t bytesize;
};

void initialize_object_table(int size);

void initialize_line_table(int size);

bool line_exists(int index);

std::string get_line_file_name(int index);

std::string get_line_dir_name(int index); 

std::string get_line_sass(int index);

uint32_t get_line_line_num(int index); 

short get_line_estimated_status(int index);

std::string get_object_var_name(uint64_t pc);

std::string get_object_file_name(uint64_t pc);

std::string get_object_func_name(uint64_t pc);

uint32_t get_object_line_num(uint64_t pc);

int get_object_device_id(uint64_t pc);

void set_object_device_id(uint64_t pc, int dev_id);

uint32_t get_object_data_type_size(uint64_t pc);

void set_object_data_type_size(uint64_t pc, const uint32_t type_size);

bool object_exists(uint64_t pc);
/* lock */
pthread_mutex_t mutex1;

/* map to store context state */
std::unordered_map<CUcontext, CTXstate *> ctx_state_map;

/* skip flag used to avoid re-entry on the nvbit_callback when issuing
 * flush_channel kernel call */
bool skip_callback_flag = false;

/* global control variables for this tool */
uint32_t instr_begin_interval = 0;
uint32_t instr_end_interval = UINT32_MAX;
std::string kernel_name;
int verbose = 0;
int silent = 0;
int code_attribution = 0;

/* opcode to id map and reverse map  */
std::map<std::string, int> opcode_to_id_map;
std::map<int, std::string> id_to_opcode_map;
std::vector<MemoryAllocation> mem_allocs;

int64_t find_nvshmem_dev_of_ptr(int mype, uint64_t mem_addr) {

  int size = 10; 

  int region = -1; 

  // 0x000012020000000 is nvshmem's first address for a remote peer
  uint64_t start =  0x000012020000000;

  // 0x000010020000000 is nvshmem's address for the peer itself
  uint64_t incrmnt = (uint64_t) 0x000012020000000 - (uint64_t) 0x000010020000000;

  for (int i = 1; i <= size; i++) {
    uint64_t bottom =  (uint64_t) start + (i - 1) * incrmnt;
    uint64_t top =  (uint64_t) start + i * incrmnt;
    if ((uint64_t) bottom <= (uint64_t) mem_addr && (uint64_t) mem_addr < (uint64_t) top) {
      region = i - 1; break;
    }   
  }

  if (region == -1) {
    return -1; 
  }

  if (mype == region) {
    return (mype + 1) % size;
  } 


  for (int i = 0; i < size; i++) {
    if (mype == i) continue;

    if (region == 0) {
      return i;
    }   

    region--;
  }   

  return -1;
}

int64_t find_dev_of_ptr(uint64_t ptr)
{

  for (MemoryAllocation ma : mem_allocs)
  {
    if (ma.pointer <= ptr && ptr < ma.pointer + ma.bytesize)
    {
      return ma.deviceID;
    }
  }

  return -1;
}

/* grid launch id, incremented at every launch */
uint64_t grid_launch_id = 0;

const char* whitespace = " ,\"\t\n\r\f\v";

// trim from end of string (right)
inline std::string& rtrim(std::string& s, const char* t = whitespace)
{
    s.erase(s.find_last_not_of(t) + 1);
    return s;
}

// trim from beginning of string (left)
inline std::string& ltrim(std::string& s, const char* t = whitespace)
{
    s.erase(0, s.find_first_not_of(t));
    return s;
}

// trim from both ends of string (right then left)
inline std::string& trim(std::string& s, const char* t = whitespace)
{
    return ltrim(rtrim(s, t), t);
}

void memop_to_line () {
   // open a file in read mode.
   ifstream infile;
   infile.open("testfile.txt");

   if(!infile){
        cerr << "Please generate a cubin file using nvcc -cubin "
		"-lineinfo command and run nvdisasm --print-line-info "
		"on the generated cubin file with the output directed to testfile.txt" << endl;
        exit(1);
    }
   //std::pair<std::vector<int>, std::vector<int>> line_tracking;
   int curr_line;
   std::string full_path;
   std::string kern_name;
   //std::cerr << "in memop_to_line before\n";
   for (std::string line; std::getline(infile, line); )
   {
        std::istringstream input1(line);
	std::string prev_word;
        for (std::string word; std::getline(input1, word, ' '); ) {
		if(word.substr(0,6) == ".text.") {
			rtrim(word, ":");
                        kern_name = word;
			//std::cerr << "identified kern_name: " << kern_name << "\n";
                }
                if(word == "line") {
			full_path = trim(prev_word);
                        std::getline(input1, word, ' ');
                        curr_line = std::stoi(word);
			get<0>(line_tracking[kern_name]) = full_path;
			//std::cerr << "a line is found in path: " << full_path << "\n";
                }
                if(word.substr(0,3) == "LDG" || word.substr(0,3) == "LD.") {
                       //std::cout << word.substr(0,3) << " found in line " << curr_line << "\n";
		       get<1>(line_tracking[kern_name]).push_back(curr_line);
                }
                else if(word.substr(0,3) == "STG" || word.substr(0,3) == "ST.") {
                        //std::cout << word.substr(0,3) << " found in line " << curr_line << "\n";
			get<2>(line_tracking[kern_name]).push_back(curr_line);
                }
		prev_word = word;
        }
   }
   //std::cerr << "in memop_to_line after\n";
#if 0
   std::cout << "LDG is detected in the following lines\n";
   for(auto a : line_tracking.first)
           std::cout << a << "\n";
   std::cout << "STG is detected in the following lines\n";
   for(auto a : line_tracking.second)
           std::cout << a << "\n";
#endif
   infile.close();
}

std::string find_recorded_kernel(const std::string& curr_kernel) 
{
	std::string chosen_key;
	size_t shortest_len = 1000;
	//for(const auto& [key, value] : line_tracking) {
	for(auto& x: line_tracking) {
		std::string key_str = x.first;
		//std::cerr << "key_str: " << key_str << ", curr_kernel: " << curr_kernel << "\n";
		std::istringstream tokenized_kern_name(curr_kernel);
		std::string name;
		size_t old_pos = 0;
		size_t pos = 0;
		int token_count = 0;
		int match_count = 0;
		while (std::getline(tokenized_kern_name, name, ':')) {
			if(name.length() == 0)
				continue;
			//std::cerr << "name: " << name << "\n";
			pos = key_str.find(name);
			if (pos != std::string::npos) {
                        	//std::cerr << "found key_str: " << key_str << ", curr_kernel: " << curr_kernel << "\n";
				if (pos >= old_pos) {
					match_count++;
					old_pos = pos;
				}
                	}
			token_count++;
		}
		
		if(token_count != 0 && token_count == match_count && shortest_len > key_str.size()) {
			chosen_key = key_str;
			shortest_len = key_str.size();
		}	
	}
	//std::cerr << "chosen_key: " << chosen_key << "\n"; 
	return chosen_key;
}

void nvbit_at_init()
{
  setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
  GET_VAR_INT(
      instr_begin_interval, "INSTR_BEGIN", 0,
      "Beginning of the instruction interval where to apply instrumentation");
  GET_VAR_INT(
      instr_end_interval, "INSTR_END", UINT32_MAX,
      "End of the instruction interval where to apply instrumentation");
  GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");
  GET_VAR_INT(silent,  "SILENT",       0, "Silence long output of the tool");

  GET_VAR_STR(kernel_name, "KERNEL_NAME", "Specify the name of the kernel to track");
  GET_VAR_INT(code_attribution, "CODE_ATTRIBUTION", 0, "Enable source code line attribution");

  std::string pad(100, '-');
  if (verbose)
  {
    std::cout << pad << std::endl;
  }
  // read the file with line info here
  initialize_object_table(100);
  initialize_line_table(100);
  //std::cerr << "code_attribution: " << code_attribution << std::endl;
  if(code_attribution) {
  	memop_to_line();
  }
  adm_db_init();
  /* set mutex as recursive */
  pthread_mutexattr_t attr;
  pthread_mutexattr_init(&attr);
  pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
  pthread_mutex_init(&mutex1, &attr);


}

/* Set used to avoid re-instrumenting the same functions multiple times */
std::unordered_set<CUfunction> already_instrumented;
std::unordered_map<int, std::string> instrumented_functions;

void instrument_function_if_needed(CUcontext ctx, CUfunction func)
{
  assert(ctx_state_map.find(ctx) != ctx_state_map.end());
  CTXstate *ctx_state = ctx_state_map[ctx];

  // std::cout << "about to instrument" << std::endl;
  if (already_instrumented.count(func))
  {
    return;
  }

  /* Get related functions of the kernel (device function that can be
   * called by the kernel) */
  std::vector<CUfunction> related_functions = nvbit_get_related_functions(ctx, func);

  /* add kernel itself to the related function vector */
  related_functions.push_back(func);

  /* iterate on function */
  for (auto f : related_functions)
  {
    /* "recording" function was instrumented, if set insertion failed
     * we have already encountered this function */
    if (!already_instrumented.insert(f).second)
    {
      continue;
    }

    int func_id = instrumented_functions.size();
    instrumented_functions[func_id] = nvbit_get_func_name(ctx, f);



    /* get vector of instructions of function "f" */
    const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);

    if (verbose)
    {
      std::cout << "instrumenting: " << nvbit_get_func_name(ctx, f) << std::endl;
      printf(
          "MEMTRACE: CTX %p, Inspecting CUfunction %p name %s at address "
          "0x%lx\n",
          ctx, f, nvbit_get_func_name(ctx, f), nvbit_get_func_addr(f));
    }

    std::string curr_kernel_name = nvbit_get_func_name(ctx, f);

    std::size_t parenthes_pos = curr_kernel_name.find_first_of('(');
    //std::cerr << "before encoded_kernel_name\n";
    if(parenthes_pos != std::string::npos)
    	curr_kernel_name.erase(parenthes_pos);
    std::string encoded_kernel_name;

    std::string file;
    std::string path;

    if(code_attribution) {
	//std::cerr << "test 1\n";
	curr_kernel_name = nvbit_get_func_name(ctx, f);
	//std::cerr << "curr_kernel_name: " << curr_kernel_name << "\n";
	parenthes_pos = curr_kernel_name.find_first_of('<');
	if(parenthes_pos != std::string::npos) {
		curr_kernel_name.erase(parenthes_pos);
	//std::cerr << "curr_kernel_name: " << curr_kernel_name << " 2\n";
	}
	else {
		//std::cerr << "curr_kernel_name 2: " << curr_kernel_name << "\n";
		parenthes_pos = curr_kernel_name.find_first_of('(');
		if(parenthes_pos != std::string::npos)
			curr_kernel_name.erase(parenthes_pos);
	}
	//std::cerr << "curr_kernel_name 3: " << curr_kernel_name << "\n";
	std::istringstream tokenized_kern_name(curr_kernel_name);
	std::string name;
	while (std::getline(tokenized_kern_name, name, ' '));
	//name.erase(0, name.find_last_of(':'));
	//trim(name, ":");
	//std::cerr << "test 1.1 " << name << "\n";
	encoded_kernel_name = find_recorded_kernel(name);
	//std::cerr << "test 1.2 encoded_kernel_name: " << encoded_kernel_name << "\n";
	path = get<0>(line_tracking[encoded_kernel_name]);
	if(path.size() > 0) {
		std::istringstream tokenized_path(path);
    		while (std::getline(tokenized_path, file, '/'));
    		//path = get<0>(line_tracking[encoded_kernel_name]);
		//std::cerr << "path: " << path << " file: " << file << "\n";
    		path.erase(path.size()-file.size()-1, file.size()+1);
		//std::cerr << "test 2\n";
	}
    }

    std::string prev_valid_file_name;
    std::string prev_valid_dir_name;
    uint32_t prev_valid_line_num = 0;  
    uint32_t cnt = 0;
    int ldg_count = 0;
    int stg_count = 0; 
    /* iterate on all the static instructions in the function */
    for (auto instr : instrs)
    {
      uint32_t instr_offset = instr->getOffset();
      char *file_name = (char*)malloc(sizeof(char)*FILE_NAME_SIZE);
      file_name[0] = '\0';
      char *dir_name = (char*)malloc(sizeof(char)*PATH_NAME_SIZE);
      dir_name[0] = '\0';
      uint32_t line_num = 0;
      bool ret_line_info = nvbit_get_line_info(ctx, f, instr_offset, &file_name, &dir_name, &line_num);
      std::string filename = file_name;
      std::string dirname = dir_name;
      std::string sass = instr->getSass();
      //std::cout << "sass is detected: " << sass << "\n";
 
// before
      if(code_attribution && path.size() > 0) {
      	std::istringstream input1(sass);
      	for (std::string word; std::getline(input1, word, ' '); ) {
	      if(word.substr(0,3) == "LDG" || word.substr(0,3) == "LD.") {
		      //std::cout << word.substr(0,3) << " found in line " << curr_line << "\n";
		      //line_tracking.first.push_back(curr_line);
		      if(!ret_line_info) {
			//std::cout << "ldg_count: " << ldg_count << " " << line_tracking.first[ldg_count] << "\n";
		      	line_num = get<1>(line_tracking[encoded_kernel_name])[ldg_count]; //line_tracking.first[ldg_count];
			dirname = path;
			filename = file;
		      }
		      ldg_count++;
	      } 
	      else if(word.substr(0,3) == "STG" || word.substr(0,3) == "ST.") {
		      //std::cout << word.substr(0,3) << " found in line " << curr_line << "\n";
		      if(!ret_line_info) {
			//std::cout << "stg_count: " << stg_count << " " << line_tracking.second[stg_count] << "\n";
                        line_num = get<2>(line_tracking[encoded_kernel_name])[stg_count]; //line_tracking.second[stg_count];
			dirname = path;
                        filename = file;
		      }
		      stg_count++;
	      }
      	}
      }
// after      
      //free(file_name);
      //free(dir_name);
      short estimated_status = 2; // it is estimated
      if(line_num != 0) {

        estimated_status = 1; // it is original
        adm_line_location_insert(global_index, filename, dirname, sass, line_num, estimated_status);
        prev_valid_file_name = filename;
        prev_valid_dir_name = dirname;
        prev_valid_line_num = line_num;
      } else {
        adm_line_location_insert(global_index, prev_valid_file_name, prev_valid_dir_name, sass, prev_valid_line_num, estimated_status);
      }
      global_index++;
      if (cnt < instr_begin_interval || cnt >= instr_end_interval ||
          instr->getMemorySpace() == InstrType::MemorySpace::NONE ||
          instr->getMemorySpace() == InstrType::MemorySpace::CONSTANT)
      {
        cnt++;
        continue;
      }
      if (verbose)
      {
        instr->printDecoded();
      }

      if (opcode_to_id_map.find(instr->getOpcode()) ==
          opcode_to_id_map.end())
      {
        int opcode_id = opcode_to_id_map.size();
        opcode_to_id_map[instr->getOpcode()] = opcode_id;
        id_to_opcode_map[opcode_id] = std::string(instr->getOpcode());
      }

      int opcode_id = opcode_to_id_map[instr->getOpcode()];
      int mref_idx = 0;
      /* iterate on the operands */
      for (int i = 0; i < instr->getNumOperands(); i++)
      {
        /* get the operand "i" */
        const InstrType::operand_t *op = instr->getOperand(i);

        if (op->type == InstrType::OperandType::MREF)
        {
          /* insert call to the instrumentation function with its
           * arguments */
          nvbit_insert_call(instr, "instrument_mem", IPOINT_BEFORE);
          /* predicate value */
          nvbit_add_call_arg_guard_pred_val(instr);
          /* opcode id */
          nvbit_add_call_arg_const_val32(instr, opcode_id);
          /* device id */
          int dev_id = -1;
          cudaGetDevice(&dev_id);
          nvbit_add_call_arg_const_val32(instr, dev_id);
          /* memory reference 64 bit address */
          nvbit_add_call_arg_mref_addr64(instr, mref_idx);
          /* add "space" for kernel function pointer that will be set
           * at launch time (64 bit value at offset 0 of the dynamic
           * arguments)*/
          nvbit_add_call_arg_launch_val64(instr, 0);
          /* add pointer to channel_dev*/
          nvbit_add_call_arg_const_val64(
              instr, (uint64_t)ctx_state->channel_dev);
          nvbit_add_call_arg_const_val32(instr, global_index-1);
          nvbit_add_call_arg_const_val32(instr, func_id);
          mref_idx++;
        }
      }
      cnt++;
    }
  }
}

__global__ void flush_channel(ChannelDev *ch_dev)
{
  /* set a CTA id = -1 to indicate communication thread that this is the
   * termination flag */
  mem_access_t ma;
  ma.lane_id = -1;
  ch_dev->push(&ma, sizeof(mem_access_t));
  /* flush channel */
  ch_dev->flush();
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
    const char *name, void *params, CUresult *pStatus)
{
  pthread_mutex_lock(&mutex1);

  /* we prevent re-entry on this callback when issuing CUDA functions inside
   * this function */
  if (skip_callback_flag)
  {
    pthread_mutex_unlock(&mutex1);
    return;
  }
  skip_callback_flag = true;

  assert(ctx_state_map.find(ctx) != ctx_state_map.end());
  CTXstate *ctx_state = ctx_state_map[ctx];

  if (!is_exit && cbid == API_CUDA_cuLaunchKernel_ptsz ||
      cbid == API_CUDA_cuLaunchKernel)
  {
    cuLaunchKernel_params *p = (cuLaunchKernel_params *)params;

    /* Make sure GPU is idle */
    // cudaDeviceSynchronize();
    // assert(cudaGetLastError() == cudaSuccess);

    /* get function name and pc */

    // gets the kernel signature
    std::string func_name(nvbit_get_func_name(ctx, p->f));
    uint64_t pc = nvbit_get_func_addr(p->f);

    std::vector<CUfunction> related_functions = nvbit_get_related_functions(ctx, p->f);
    related_functions.push_back(p->f);

    for (auto f : related_functions)
    {
      // only instrument kernel's with the kernel name supplied by the user,
      // the substr and find are to extract the func name from the func
      // signature
      std::string func_name(nvbit_get_func_name(ctx, f));
      if (kernel_name == "all" || kernel_name == func_name.substr(0, func_name.find("(")))
      {
        instrument_function_if_needed(ctx, f);
      } else if (kernel_name == "nccl" && func_name.substr(0, std::string("ncclKernel").length()).compare(std::string("ncclKernel")) == 0) {
        instrument_function_if_needed(ctx, f);
      }

      int nregs = 0;
      CUDA_SAFECALL(
          cuFuncGetAttribute(&nregs, CU_FUNC_ATTRIBUTE_NUM_REGS, f));

      int shmem_static_nbytes = 0;
      CUDA_SAFECALL(
          cuFuncGetAttribute(&shmem_static_nbytes,
            CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, f));

      /* set grid launch id at launch time */
      nvbit_set_at_launch(ctx, f, &grid_launch_id, sizeof(uint64_t));
      /* increment grid launch id for next launch */
      grid_launch_id++;

      /* enable instrumented code to run */
      nvbit_enable_instrumented(ctx, f, true);

      if (verbose)
      {
        printf(
            "MEMTRACE: CTX 0x%016lx - LAUNCH - Kernel pc 0x%016lx - Kernel "
            "name %s - grid launch id %ld\n",
            (uint64_t)ctx, pc, func_name.c_str(), grid_launch_id);
      }
    }
  }
  else if (!is_exit && cbid == API_CUDA_cuLaunchCooperativeKernel ||
      cbid == API_CUDA_cuLaunchCooperativeKernel_ptsz)
  {
    cuLaunchCooperativeKernel_params *p = (cuLaunchCooperativeKernel_params *)params;

    /* get function name and pc */
    // gets the kernel signature
    uint64_t pc = nvbit_get_func_addr(p->f);

    std::vector<CUfunction> related_functions = nvbit_get_related_functions(ctx, p->f);
    related_functions.push_back(p->f);

    // only instrument kernel's with the kernel name supplied by the user,
    // the substr and find are to extract the func name from the func
    // signature
    for (auto f : related_functions)
    {
      std::string func_name(nvbit_get_func_name(ctx, f));
      if (kernel_name == "all" || kernel_name == func_name.substr(0, func_name.find("(")))
      {
        /* instrument */
        instrument_function_if_needed(ctx, p->f);
      } else if (kernel_name == "nccl" && func_name.substr(0, std::string("ncclKernel").length()).compare(std::string("ncclKernel")) == 0) {
        instrument_function_if_needed(ctx, f);
      }


      /* set grid launch id at launch time */
      nvbit_set_at_launch(ctx, f, &grid_launch_id, sizeof(uint64_t));
      /* increment grid launch id for next launch */
      grid_launch_id++;

      /* enable instrumented code to run */
      nvbit_enable_instrumented(ctx, f, true);

      if (verbose)
      {
        printf(
            "MEMTRACE: CTX 0x%016lx - LAUNCH - Kernel pc 0x%016lx - Kernel "
            "name %s - grid launch id %ld\n",
            (uint64_t)ctx, pc, func_name.c_str(), grid_launch_id);
      }
    }
  }
  else if (is_exit && cbid == API_CUDA_cuMemAlloc_v2)
  {
    cuMemAlloc_v2_params *p = (cuMemAlloc_v2_params *)params;
    std::stringstream ss;
    ss << HEX(*p->dptr);
    std::stringstream ss2;
    ss2 << HEX(*p->dptr + p->bytesize);
    int deviceID = -1;
    uint64_t pointer = *p->dptr;
    uint64_t bytesize = p->bytesize;

    cudaGetDevice(&deviceID);
    assert(cudaGetLastError() == cudaSuccess);

    MemoryAllocation ma = {deviceID, pointer, bytesize};
    mem_allocs.push_back(ma);

    if (JSON) {
      std::cout << "{\"op\": \"mem_alloc\", " << "\"dev_id\": " << deviceID << ", " << "\"bytesize\": " << p->bytesize << ", \"start\": \"" << ss.str() << "\", \"end\": \"" << ss2.str() << "\"}" << std::endl;
    }
  }

  skip_callback_flag = false;
  pthread_mutex_unlock(&mutex1);
}

cudaError_t cudaMallocHostWrap ( void** devPtr, size_t size, const char *var_name, const uint32_t element_size, const char *fname, const char *fxname, int lineno) {
  cudaError_t errorOutput = cudaMallocHost( devPtr, size );
  if(*devPtr /*&& adm_set_tracing(0)*/) {
    if(!object_attribution) {
      object_attribution = true;
    }
    uint64_t allocation_pc = (uint64_t) __builtin_extract_return_addr (__builtin_return_address (0));
    std::string vname = var_name;
    adm_range_t* range = adm_range_insert(reinterpret_cast<uint64_t>(*devPtr), size, allocation_pc, -1, vname, ADM_STATE_ALLOC);

    if(range) {
      adm_object_t* obj = adm_object_insert(allocation_pc, var_name, element_size, fname, fxname, lineno, ADM_STATE_ALLOC);
      if(obj) {
        range->set_index_in_object(obj->get_range_count());
        obj->inc_range_count();
      }
    }
  }

  return errorOutput;	
}

cudaError_t cudaMallocWrap ( void** devPtr, size_t size, const char *var_name, const uint32_t element_size, const char *fname, const char *fxname, int lineno/*, const std::experimental::source_location& location = std::experimental::source_location::current()*/) {
  cudaError_t errorOutput = cudaMalloc( devPtr, size );
  if(*devPtr /*&& adm_set_tracing(0)*/) {
    if(!object_attribution) {
      object_attribution = true;
    }
    uint64_t allocation_pc = (uint64_t) __builtin_extract_return_addr (__builtin_return_address (0));
    std::string vname = var_name;
    int dev_id = -1;
    cudaGetDevice(&dev_id);
    adm_range_t* range = adm_range_insert(reinterpret_cast<uint64_t>(*devPtr), size, allocation_pc, dev_id, vname, ADM_STATE_ALLOC);

    if(range) {
      adm_object_t* obj = adm_object_insert(allocation_pc, var_name, element_size, fname, fxname, lineno, ADM_STATE_ALLOC);	
      if(obj) {
        range->set_index_in_object(obj->get_range_count());
        obj->inc_range_count();
      }
    }
  }

  return errorOutput;
}

//#if 0
void * nvshmem_mallocWrap ( size_t size, const char *var_name, const uint32_t element_size, const char *fname, const char *fxname, int lineno/*, const std::experimental::source_location& location = std::experimental::source_location::current()*/) {
  void * allocated_memory = nvshmem_malloc( size );
  //fprintf(stderr, "nvshmem_malloc is intercepted\n");
  if(allocated_memory /*&& adm_set_tracing(0)*/) {
    if(!object_attribution) {
      object_attribution = true;
    }
    uint64_t allocation_pc = (uint64_t) __builtin_extract_return_addr (__builtin_return_address (0));
    std::string vname = var_name;
    int dev_id = -1;
    cudaGetDevice(&dev_id);
    adm_range_t* range = adm_range_insert(reinterpret_cast<uint64_t>(allocated_memory), size, allocation_pc, dev_id, vname, ADM_STATE_ALLOC);

    if(range) {
      adm_object_t* obj = adm_object_insert(allocation_pc, var_name, element_size, fname, fxname, lineno, ADM_STATE_ALLOC);
      if(obj) {
        range->set_index_in_object(obj->get_range_count());
        obj->inc_range_count();
      }
    }
  }

  return allocated_memory;
}

void * nvshmem_alignWrap ( size_t alignment, size_t size, const char *var_name, const uint32_t element_size, const char *fname, const char *fxname, int lineno/*, const std::experimental::source_location& location = std::experimental::source_location::current()*/) {
  void * allocated_memory = nvshmem_align( alignment, size );
  //fprintf(stderr, "nvshmem_align is intercepted\n");
  if(allocated_memory /*&& adm_set_tracing(0)*/) {
    if(!object_attribution) {
      object_attribution = true;
    }
    uint64_t allocation_pc = (uint64_t) __builtin_extract_return_addr (__builtin_return_address (0));
    std::string vname = var_name;
    int dev_id = -1;
    cudaGetDevice(&dev_id);
    adm_range_t* range = adm_range_insert(reinterpret_cast<uint64_t>(allocated_memory), size, allocation_pc, dev_id, vname, ADM_STATE_ALLOC);

    if(range) {
      adm_object_t* obj = adm_object_insert(allocation_pc, var_name, element_size, fname, fxname, lineno, ADM_STATE_ALLOC);
      if(obj) {
        range->set_index_in_object(obj->get_range_count());
        obj->inc_range_count();
      }
    }
  }

  return allocated_memory;
}
//#endif

void *recv_thread_fun(void *args)
{


  CUcontext ctx = (CUcontext)args;

  pthread_mutex_lock(&mutex1);
  /* get context state from map */
  assert(ctx_state_map.find(ctx) != ctx_state_map.end());
  CTXstate *ctx_state = ctx_state_map[ctx];

  ChannelHost *ch_host = &ctx_state->channel_host;
  pthread_mutex_unlock(&mutex1);
  char *recv_buffer = (char *)malloc(CHANNEL_SIZE);

  bool done = false;
  while (!done)
  {
    /* receive buffer from channel */
    uint32_t num_recv_bytes = ch_host->recv(recv_buffer, CHANNEL_SIZE);
    if (num_recv_bytes > 0)
    {
      uint32_t num_processed_bytes = 0;
      while (num_processed_bytes < num_recv_bytes)
      {
        mem_access_t *ma =
          (mem_access_t *)&recv_buffer[num_processed_bytes];

        /* when we receive a CTA_id_x it means all the kernels
         * completed, this is the special token we receive from the
         * flush channel kernel that is issues at the end of the
         * context */

        if (ma->lane_id == -1)
        {
          done = true;
          break;
        }
        adm_range_t* range = nullptr; //adm_range_find(ma.addrs[0]);
        uint64_t allocation_pc = 0; //obj->get_allocation_pc();	
        std::string varname;
        std::string filename;
        std::string funcname;
        uint32_t linenum;
        uint32_t data_type_size = 1;
        int dev_id = -1;
        int line_index = ma->global_index;	
        std::string line_filename = get_line_file_name(line_index);
        std::string line_dirname = get_line_dir_name(line_index);
        std::string line_sass = get_line_sass(line_index);
        uint32_t line_linenum = get_line_line_num(line_index);
        short line_estimated_status = get_line_estimated_status(line_index);
        uint64_t offset_address_range = 0;

        for (int i = 0; i < 32; i++)
        {

          if (ma->addrs[i] == 0x0)
            continue;

          int mem_device_id = find_dev_of_ptr(ma->addrs[i]);

          // nvshmem heap_base = 0x10020000000
          // ignore operations on memory locations not allocated by cudaMalloc on the host
          if (mem_device_id == -1 && (ma->addrs[i] >= 0x0000010020000000))
            mem_device_id = find_nvshmem_dev_of_ptr(ma->dev_id, ma->addrs[i]);

          // ignore operations on the same device
          if (mem_device_id == ma->dev_id)
            continue;

          if (mem_device_id == -1)
            continue;

          uint32_t index_in_object = 0;
          uint32_t index_in_malloc = 0;

          if (object_attribution) {
            range = adm_range_find(ma->addrs[i]);
            allocation_pc = range->get_allocation_pc();
            if(object_exists(allocation_pc)) {
              varname = get_object_var_name(allocation_pc);
              filename = get_object_file_name(allocation_pc);
              funcname = get_object_func_name(allocation_pc);
              linenum = get_object_line_num(allocation_pc);
              dev_id = range->get_device_id();
              data_type_size = get_object_data_type_size(allocation_pc);
              index_in_object = range->get_index_in_object();
            }
            index_in_malloc = (ma->addrs[i] - range->get_address())/data_type_size;
            offset_address_range = range->get_address();
          }

          if (silent) continue;

          std::stringstream ss;
          if (JSON) {
            ss << "{\"op\": \"" << id_to_opcode_map[ma->opcode_id]  << "\", "
              << "\"kernel_name\": \"" << instrumented_functions[ma->func_id] << "\", "
              << "\"addr\": \"" << HEX(ma->addrs[i]) << "\","
              << "\"object_allocation_pc\": \"" << HEX(allocation_pc) << "\", "
              << "\"object_variable_name\": \"" << varname << "\", "
              << "\"malloc_index_in_object\": " << index_in_object << ", "
              << "\"element_index_in_malloc\": " << index_in_malloc << ", "
              << "\"object_allocation_file_name\": \"" << filename << "\", "
              << "\"object_allocation_func_name\": \"" << funcname << "\", "
              << "\"object_allocation_line_num\": " << linenum << ", "
              << "\"object_allocation_device_id\": " << dev_id << ", "
              << "\"thread_index\": " << ma->thread_index << ", "
              << "\"lane_id\": " << ma->lane_id << ", "
              << "\"running_device_id\": " << ma->dev_id << ", "
              << "\"mem_device_id\": " << mem_device_id << ", "
              << "\"code_line_index\": \"" << line_index << "\", "
              << "\"code_line_filename\": \"" << line_filename << "\", "
              << "\"code_line_dirname\": \"" << line_dirname << "\", "
              << "\"code_line_linenum\": " << line_linenum << ", "
              << "\"code_line_estimated_status\": " << line_estimated_status
              << "}" << std::endl;
          } else {
            ss << id_to_opcode_map[ma->opcode_id] << "," 
              << HEX(ma->addrs[i]) << ","
              << ma->thread_index  << ","
              << ma->dev_id        << ","
              << mem_device_id     << ","
              << line_linenum      << ","
              << line_index        << ","
              << line_estimated_status << ","
              << HEX(offset_address_range)
              << std::endl;
          }
          std::cout << ss.str() << std::flush;
        }
        num_processed_bytes += sizeof(mem_access_t);
      }
    }
  }
  free(recv_buffer);
  return NULL;
}

void nvbit_at_ctx_init(CUcontext ctx)
{
  pthread_mutex_lock(&mutex1);
  if (verbose)
  {
    printf("MEMTRACE: STARTING CONTEXT %p\n", ctx);
  }
  CTXstate *ctx_state = new CTXstate;
  assert(ctx_state_map.find(ctx) == ctx_state_map.end());
  ctx_state_map[ctx] = ctx_state;
  cudaMallocManaged(&ctx_state->channel_dev, sizeof(ChannelDev));
  ctx_state->channel_host.init((int)ctx_state_map.size() - 1, CHANNEL_SIZE,
      ctx_state->channel_dev, recv_thread_fun, ctx);
  nvbit_set_tool_pthread(ctx_state->channel_host.get_thread());
  pthread_mutex_unlock(&mutex1);
  if (!silent && ((int)ctx_state_map.size() - 1 == 0)) {
    std::cout << "op_code, addr, thread_indx, running_dev_id, mem_dev_id, code_linenum, code_line_index, code_line_estimated_status, obj_offset" << std::endl;
  }
}

void nvbit_at_ctx_term(CUcontext ctx)
{
  pthread_mutex_lock(&mutex1);
  skip_callback_flag = true;
  if (verbose)
  {
    printf("MEMTRACE: TERMINATING CONTEXT %p\n", ctx);
  }
  /* get context state from map */
  assert(ctx_state_map.find(ctx) != ctx_state_map.end());
  CTXstate *ctx_state = ctx_state_map[ctx];

  /* flush channel */
  flush_channel<<<1, 1>>>(ctx_state->channel_dev);
  /* Make sure flush of channel is complete */
  cudaDeviceSynchronize();
  assert(cudaGetLastError() == cudaSuccess);

  ctx_state->channel_host.destroy(false);
  cudaFree(ctx_state->channel_dev);
  skip_callback_flag = false;
  delete ctx_state;
  pthread_mutex_unlock(&mutex1);
}

void nvbit_at_term()
{
  if (!silent) {
    adm_ranges_print();
  }

  // TODO: Print the below agian at some point
  // adm_line_table_print();
  adm_db_fini();
}
