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

#define _GNU_SOURCE
#include <dlfcn.h>
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <map>
#include <sstream>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <iostream>

#include <adm_config.h>
#include <adm_common.h>
#include <adm_splay.h>
#include <adm_memory.h>
#include <adm_database.h>
#include <cpptrace/cpptrace.hpp>
#include <iostream>

/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

/* for channel */
#include "utils/channel.hpp"

/* contains definition of the mem_access_t structure */
#include "common.h"
#include "util.h"

#define HEX(x)                                                          \
  "0x" << std::setfill('0') << std::setw(16) << std::hex << (uint64_t)x \
  << std::dec

#define CHANNEL_SIZE (1l << 30)

#define JSON 0
#define EQUAL_STRS 0

#define FILE_NAME_SIZE 256
#define PATH_NAME_SIZE 5000

#include "adm.h"


using namespace cpptrace;
using namespace adamant;
using namespace std;

hash<string> hash_func;

#define CHILD 1
#define SIBLING 2

int object_counter = 0;

static bool nvshmem_malloc_handled = false;
static bool object_attribution = false;
pool_t<adm_splay_tree_t, ADM_DB_OBJ_BLOCKSIZE>* nodes = nullptr;
pool_t<adm_range_t, ADM_DB_OBJ_BLOCKSIZE>* ranges = nullptr;
static int global_index = 0;

static allocation_site_t* root = NULL;

static allocation_line_hash_table_t* allocation_line_table;

std::vector<adm_range_t*> range_nodes;

std::vector<adm_object_t*> object_nodes;

Logger logger("snoopie_log_" + std::to_string(getpid()) + ".zst");

std::map<std::string, std::tuple<std::string, std::vector<int>, std::vector<int>>> line_tracking;

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
std::string profiled_nccl_file = "";
int on_dev_filtering = 1;
int time_log = 0;
int verbose = 0;
std::string nvshmem_version = "2.8";
int nvshmem_ngpus = 10;
int silent = 0;
int code_attribution = 0;
int sample_size;

/* opcode to id map and reverse map  */
std::map<std::string, int> opcode_to_id_map;
std::map<int, std::string> id_to_opcode_map;
std::vector<MemoryAllocation> mem_allocs;

void log_time(string msg) {
  if (!time_log) return;

  std::cout << msg << ": " << std::chrono::time_point_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now())
              .time_since_epoch()
              .count()
       << std::endl;
}

int64_t find_nvshmem_dev_of_ptr(int mype,uint64_t mem_addr, int nvshmem_ngpus,
    std::string version) {

  int size = 15;

  int region = -1;

  // 0x000012020000000 is nvshmem's first address for a remote peer
  uint64_t start = 0x000012020000000;

  // 0x000010020000000 is nvshmem's address for the peer itself
  uint64_t incrmnt = (uint64_t)0x000012020000000 - (uint64_t)0x000010020000000;

  for (int i = 1; i <= size; i++) {
    uint64_t bottom = (uint64_t)start + (i - 1) * incrmnt;
    uint64_t top = (uint64_t)start + i * incrmnt;
    if ((uint64_t)bottom <= (uint64_t)mem_addr &&
        (uint64_t)mem_addr < (uint64_t)top) {
      region = i - 1;
      break;
    }
  }

  if (region == -1) {
    return -1;
  }

  if (version == "2.9" || version == "2.8") {
    region += mype;
  }

  if (mype == region) {
    return (mype + 1) % nvshmem_ngpus;
  }

  for (int i = 0; i < size; i++) {
    if (mype == i)
      continue;

    if (region == 0) {
      return i % nvshmem_ngpus;
    }

    region--;
  }

  return -1;
}

allocation_site_t* search_at_level(allocation_site_t* allocation_site, uint64_t pc)
{
    if (allocation_site == NULL || allocation_site->get_pc() == pc)
        return allocation_site;

    return search_at_level(allocation_site->get_next_sibling(), pc);
}

uint64_t normalise_nvshmem_ptr(uint64_t mem_addr) {
  return mem_addr & 0x0000F0FFFFFFFFF;
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

   int curr_line;
   std::string full_path;
   std::string kern_name;

   for (std::string line; std::getline(infile, line); )
   {
        std::istringstream input1(line);
	std::string prev_word;
        for (std::string word; std::getline(input1, word, ' '); ) {
		if(word.substr(0,6) == ".text.") {
			rtrim(word, ":");
                        kern_name = word;

                }
                if(word == "line" && prev_word.find(".cu") != std::string::npos) {
			full_path = trim(prev_word);
                        std::getline(input1, word, ' ');
                        curr_line = std::stoi(word);
			get<0>(line_tracking[kern_name]) = full_path;

                }
                if(word.substr(0,3) == "LDG" || word.substr(0,3) == "LD.") {

		       get<1>(line_tracking[kern_name]).push_back(curr_line);
                }
                else if(word.substr(0,3) == "STG" || word.substr(0,3) == "ST.") {

			get<2>(line_tracking[kern_name]).push_back(curr_line);
                }
		prev_word = word;
        }
   }

   infile.close();
}

std::string find_recorded_kernel(const std::string& curr_kernel)
{
	std::string chosen_key;
	size_t shortest_len = 1000;

	for(auto& x: line_tracking) {
		std::string key_str = x.first;

		std::istringstream tokenized_kern_name(curr_kernel);
		std::string name;
		size_t old_pos = 0;
		size_t pos = 0;
		int token_count = 0;
		int match_count = 0;
		while (std::getline(tokenized_kern_name, name, ':')) {
			if(name.length() == 0)
				continue;

			pos = key_str.find(name);
			if (pos != std::string::npos) {

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

	return chosen_key;
}

// Function to print the
// N-ary tree graphically
void printNTree(allocation_site_t* x,
    vector<bool> flag,
    int depth = 0, bool isLast = false)
{

    // Condition when allocation_site is None
    if (x == NULL)
        return;

    // Loop to print the depths of the
    // current allocation_site
    for (int i = 1; i < depth; ++i) {

        // Condition when the depth
        // is exploring
        if (flag[i] == true) {
            cout << "| "
                << " "
                << " "
                << " ";
        }

        // Otherwise print
        // the blank spaces
        else {
            cout << " "
                << " "
                << " "
                << " ";
        }
    }

    // Condition when the current
    // allocation_site is the root allocation_site
    uint64_t pc = x->get_pc();
    int obj_id = x->get_object_id();
    if (depth == 0) {
        cout << pc << endl;

    // Condition when the allocation_site is
    // the last allocation_site of
    // the exploring depth
    } else if (isLast) {
        cout << "+--- " << pc;

	if (obj_id > 0)
		cout << " " << obj_id;

	cout << endl;
        // No more childrens turn it
        // to the non-exploring depth

        flag[depth] = false;

    }
    else {

        cout << "+--- " << pc;

	if (obj_id > 0)
		cout << " " << obj_id;
	cout << endl;


    }

    int it = 0;
    x = x->get_first_child();
        // Recursive call for the
        // children allocation_sites

    while(x != NULL) {
        printNTree(x, flag, depth + 1,
            x->get_next_sibling() == NULL);
	x = x->get_next_sibling();
    }
    flag[depth] = true;
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
  GET_VAR_INT(time_log, "TIME_LOG", 0, "Enable time logging inside the tool");
  GET_VAR_INT(on_dev_filtering, "ON_DEVICE_FILTERING", 1, "Enables on device filtering instead of on host fitering instead ");
  GET_VAR_INT(silent,  "SILENT",       0, "Silence long output of the tool");

  GET_VAR_STR(nvshmem_version, "NVSHMEM_VERSION", "Specify the nvshmem version to use the correct memory mapping");
  GET_VAR_INT(nvshmem_ngpus, "NVSHMEM_NGPUS", 10, "Setting the number of GPUS nvshmem will use");

  GET_VAR_STR(kernel_name, "KERNEL_NAME", "Specify the name of the kernel to track");
  GET_VAR_STR(profiled_nccl_file, "PROFILED_NCCL_FILE", "Specify the name of the file that has the NCCL function calls");
  GET_VAR_INT(code_attribution, "CODE_ATTRIBUTION", 0, "Enable source code line attribution");
  GET_VAR_INT(sample_size, "SAMPLE_SIZE", 1, "Setting the sample size, if 100, it means 1/100 of population is sampled");

  std::string pad(100, '-');
  if (verbose)
  {
    std::cout << pad << std::endl;
  }
  // read the file with line info here
  initialize_object_table(100);
  allocation_line_table = new allocation_line_hash_table_t(100);
  initialize_line_table(100);

  if(code_attribution) {
  	memop_to_line();
  }
  adm_db_init();
  /* set mutex as recursive */
  string txt_str(".txt");
  pthread_mutexattr_t attr;
  pthread_mutexattr_init(&attr);
  pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
  pthread_mutex_init(&mutex1, &attr);

  log_time("Bgn Snoopie");

}

/* Set used to avoid re-instrumenting the same functions multiple times */
std::unordered_set<CUfunction> already_instrumented;
std::unordered_map<int, std::string> instrumented_functions;

void instrument_function_if_needed(CUcontext ctx, CUfunction func)
{
 std::string main_func_name(nvbit_get_func_name(ctx, func));

  log_time("Bgn Instrumentation of func: " + main_func_name);
  assert(ctx_state_map.find(ctx) != ctx_state_map.end());
  CTXstate *ctx_state = ctx_state_map[ctx];

  if (already_instrumented.count(func))
  {
    log_time("End Instrumentation of func: " + main_func_name);
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

    if(parenthes_pos != std::string::npos)
    	curr_kernel_name.erase(parenthes_pos);
    std::string encoded_kernel_name;

    std::string file;
    std::string path;

    if(code_attribution) {

	curr_kernel_name = nvbit_get_func_name(ctx, f);
	parenthes_pos = curr_kernel_name.find_first_of('<');
	if(parenthes_pos != std::string::npos) {
		curr_kernel_name.erase(parenthes_pos);
	}
	else {
		parenthes_pos = curr_kernel_name.find_first_of('(');
		if(parenthes_pos != std::string::npos)
			curr_kernel_name.erase(parenthes_pos);
	}
	std::istringstream tokenized_kern_name(curr_kernel_name);
	std::string name;
	while (std::getline(tokenized_kern_name, name, ' '));
	encoded_kernel_name = find_recorded_kernel(name);
	path = get<0>(line_tracking[encoded_kernel_name]);
	if(path.size() > 0) {
		std::istringstream tokenized_path(path);
    		while (std::getline(tokenized_path, file, '/'));
    		path.erase(path.size()-file.size()-1, file.size()+1);
	}
    }

    // change here
    uint32_t nccl_line_num = 0;
    std::string nccl_filename;
    std::string nccl_dirname;
    if(!profiled_nccl_file.empty()) {
	    cout << "profiled_nccl_file is not empty\n";
	    std::vector<stacktrace_frame> trace = generate_trace();
	    allocation_site_t* call_site = root;
	    allocation_site_t* parent = NULL;
	    for (auto itr = trace.rbegin(); itr != trace.rend(); ++itr) {
		allocation_line_t* line = allocation_line_table->find(itr->address);
                if(line == NULL) {
                        allocation_line_table->insert(new allocation_line_t(itr->address, itr->symbol, itr->filename, itr->line));
                }
		if(root == NULL) {
                        root = new allocation_site_t (itr->address);
                        call_site = root;
                        parent = call_site;
                        call_site = call_site->get_first_child();
                        continue;
                }
		allocation_site_t* temp = call_site;
                call_site = search_at_level(call_site, itr->address);
		if(call_site == NULL) {
			if(temp != NULL) {
                                while(temp->get_next_sibling() != NULL)
                                        temp = temp->get_next_sibling();
                                temp->set_next_sibling(new allocation_site_t(itr->address));
                                call_site = temp->get_next_sibling();
				call_site->set_parent(temp->get_parent());
                        } else {
                                parent->set_first_child(new allocation_site_t(itr->address));
                                call_site = parent->get_first_child();
                                call_site->set_parent(parent);
                        }
		}
		parent = call_site;
		call_site = call_site->get_first_child();
	    }
	    string file_name;
	    if(parent) {
                file_name = allocation_line_table->find(parent->get_pc())->get_file_name();
                while(file_name.find(/*str1*/profiled_nccl_file) == string::npos) {
                        parent = parent->get_parent();
                        if(parent)
                                file_name = allocation_line_table->find(parent->get_pc())->get_file_name();
                        else
                                break;
                }
            }
	    if(parent) {
		allocation_line_t* node = allocation_line_table->find(parent->get_pc());
		path = node->get_file_name();
        	if(path.size() > 0) {
                	std::istringstream tokenized_path(path);
                	while (std::getline(tokenized_path, file, '/'));
                	path.erase(path.size()-file.size()-1, file.size()+1);
			nccl_line_num = node->get_line_num();
    			nccl_filename = file;
    			nccl_dirname = path;
        	}
	    }
    } else {
	    cout << "profiled_nccl_file is empty\n";
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
      bool ret_line_info;
      std::string filename;
      std::string dirname;
      std::string sass;

      if(profiled_nccl_file.empty()) {
	      ret_line_info = nvbit_get_line_info(ctx, f, instr_offset, &file_name, &dir_name, &line_num);
	      filename = file_name;
	      dirname = dir_name;
	      sass = instr->getSass();
      if(code_attribution && path.size() > 0) {
      	std::istringstream input1(sass);
      	for (std::string word; std::getline(input1, word, ' '); ) {
	      if(word.substr(0,3) == "LDG" || word.substr(0,3) == "LD.") {
		      if(!ret_line_info) {
		      	line_num = get<1>(line_tracking[encoded_kernel_name])[ldg_count]; //line_tracking.first[ldg_count];
			dirname = path;
			filename = file;
		      }
		      ldg_count++;
	      }
	      else if(word.substr(0,3) == "STG" || word.substr(0,3) == "ST.") {
		      if(!ret_line_info) {
                        line_num = get<2>(line_tracking[encoded_kernel_name])[stg_count]; //line_tracking.second[stg_count];
			dirname = path;
                        filename = file;
		      }
		      stg_count++;
	      }
      	}
      } 
      } else {
	      filename = nccl_filename;
	      dirname = nccl_dirname;
	      line_num = nccl_line_num;
      }

      short estimated_status = 2; // it is estimated
      if(line_num != 0) {

        estimated_status = 1; // it is original
	std::string hashed_string = dirname + filename + ":" + std::to_string(line_num);
	size_t global_index = hash_func(hashed_string);
	cout << "global_index: "  << global_index << endl;
        adm_line_location_insert(global_index, filename, dirname, sass, line_num, estimated_status);
        prev_valid_file_name = filename;
        prev_valid_dir_name = dirname;
        prev_valid_line_num = line_num;
      } else {
	std::string hashed_string = prev_valid_dir_name + prev_valid_file_name + ":" + std::to_string(prev_valid_line_num);
	size_t global_index = hash_func(hashed_string);
	cout << "global_index: "  << global_index << endl;
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
          //  nvbit_add_call_arg_const_val32(instr, ctx_state->id);
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
	  nvbit_add_call_arg_const_val32(instr, sample_size);
          mref_idx++;

        }
      }
      cnt++;
    }
  }

  log_time("End Instrumentation of func: " + main_func_name);
}

__global__ void flush_channel(ChannelDev *ch_dev)
{
  /* set a CTA id = -1 to indicate communication thread that this is the
   * termination flag */
  mem_access_t ma;
  ma.lane_id = -1;
  ch_dev->push(&ma, sizeof(mem_access_t), 0);
  /* flush channel */
  ch_dev->flush();
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
    const char *name, void *params, CUresult *pStatus)
{
  pthread_mutex_lock(&mutex1);
  log_time(std::string("Bgn Cuda Event ") + (is_exit ? "Exit" : "Enter") +  find_cbid_name(cbid));

  /* we prevent re-entry on this callback when issuing CUDA functions inside
   * this function */
  if (skip_callback_flag || nvshmem_malloc_handled)
  {
    log_time(std::string("End Cuda Event ") + (is_exit ? "Exit" : "Enter") +  find_cbid_name(cbid));
    pthread_mutex_unlock(&mutex1);
    return;
  }
  skip_callback_flag = true;

  assert(ctx_state_map.find(ctx) != ctx_state_map.end());
  CTXstate *ctx_state = ctx_state_map[ctx];

  MemoryAllocation ma;
  if (!is_exit && (cbid == API_CUDA_cuLaunchKernel_ptsz ||
      cbid == API_CUDA_cuLaunchKernel))
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

      // NOTE: Needs to verify if cuda_sm_20_div_s64 contains any addrs writes
      // or not. Avoid instrumentting this (possibly a whole family of
      // functions similar to this should be avoided to speed up NCCL
      // profiling)
      if (strcmp(nvbit_get_func_name(ctx, f), "__cuda_sm20_div_s64") == 0) {
        continue;
      }

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
  else if (!is_exit && (cbid == API_CUDA_cuLaunchCooperativeKernel ||
      cbid == API_CUDA_cuLaunchCooperativeKernel_ptsz))
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


    ma.deviceID = deviceID;
    ma.pointer = pointer;
    ma.bytesize = bytesize;
    mem_allocs.push_back(ma);

    for (const auto & ctx_map_pair : ctx_state_map) {
      ctx_map_pair.second->channel_dev->add_malloc(ma);
    }

    if (JSON) {
      std::cout << "{\"op\": \"mem_alloc\", " << "\"dev_id\": " << deviceID << ", " << "\"bytesize\": " << p->bytesize << ", \"start\": \"" << ss.str() << "\", \"end\": \"" << ss2.str() << "\"}" << std::endl;
    }
  }
  else if(is_exit && cbid == API_CUDA_cuMemAlloc)
  {
	cuMemAlloc_params *p = (cuMemAlloc_params *)params;
    std::stringstream ss;
    ss << HEX(*p->dptr);
    std::stringstream ss2;
    ss2 << HEX(*p->dptr + p->bytesize);
    int deviceID = -1;
    uint64_t pointer = *p->dptr;
    uint64_t bytesize = p->bytesize;

    cudaGetDevice(&deviceID);
    assert(cudaGetLastError() == cudaSuccess);


    ma.deviceID = deviceID;
    ma.pointer = pointer;
    ma.bytesize = bytesize;
    mem_allocs.push_back(ma);

    for (const auto & ctx_map_pair : ctx_state_map) {
      ctx_map_pair.second->channel_dev->add_malloc(ma);
    }

    if (JSON) {
      std::cout << "{\"op\": \"mem_alloc\", " << "\"dev_id\": " << deviceID << ", " << "\"bytesize\": " << p->bytesize << ", \"start\": \"" << ss.str() << "\", \"end\": \"" << ss2.str() << "\"}" << std::endl;
    }
  }
  else if (cbid == API_CUDA_cuMemAllocHost)
  {
    cuMemAllocHost_params *p = (cuMemAllocHost_params *)params;
    std::stringstream ss;
    ss << HEX(*p->pp);
    std::stringstream ss2;
    ss2 << HEX(*p->pp + p->bytesize);
    int deviceID = -1;
    uint64_t pointer = (uint64_t) *p->pp;
    uint64_t bytesize = p->bytesize;
    assert(cudaGetLastError() == cudaSuccess);


    ma.deviceID = deviceID;
    ma.pointer = pointer;
    ma.bytesize = bytesize;
    mem_allocs.push_back(ma);

    for (const auto & ctx_map_pair : ctx_state_map) {
      ctx_map_pair.second->channel_dev->add_malloc(ma);
    }
  }
  else if (cbid == API_CUDA_cuMemAllocHost_v2)
  {
    print_trace();
    std::cerr << "API_CUDA_cuMemAllocHost_v2 is detected\n";
    cuMemAllocHost_v2_params *p = (cuMemAllocHost_v2_params *)params;
    std::stringstream ss;
    ss << HEX(*p->pp);
    std::stringstream ss2;
    ss2 << HEX(*p->pp + p->bytesize);
    int deviceID = -1;
    uint64_t pointer = (uint64_t) *p->pp;
    uint64_t bytesize = p->bytesize;
    assert(cudaGetLastError() == cudaSuccess);


    ma.deviceID = deviceID;
    ma.pointer = pointer;
    ma.bytesize = bytesize;
    mem_allocs.push_back(ma);

    for (const auto & ctx_map_pair : ctx_state_map) {
      ctx_map_pair.second->channel_dev->add_malloc(ma);
    }
  }
  else if (cbid == API_CUDA_cuMemHostAlloc)
  {
    cuMemHostAlloc_params *p = (cuMemHostAlloc_params *)params;
    std::stringstream ss;
    ss << HEX(*p->pp);
    std::stringstream ss2;
    ss2 << HEX(*p->pp + p->bytesize);
    int deviceID = -1;
    uint64_t pointer = (uint64_t) *p->pp;
    uint64_t bytesize = p->bytesize;
    assert(cudaGetLastError() == cudaSuccess);

    //MemoryAllocation ma = {deviceID, pointer, bytesize};
    ma.deviceID = deviceID;
    ma.pointer = pointer;
    ma.bytesize = bytesize;
    mem_allocs.push_back(ma);

    for (const auto & ctx_map_pair : ctx_state_map) {
      ctx_map_pair.second->channel_dev->add_malloc(ma);
    }
  }
  else if (is_exit && cbid == API_CUDA_cuMemcpyDtoDAsync_v2)
  {
    cuMemcpyDtoDAsync_v2_params *p = (cuMemcpyDtoDAsync_v2_params *) params;

    CUdevice srcDeviceID;
    CUdevice dstDeviceID;


    cuPointerGetAttribute(&srcDeviceID, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, p->srcDevice);
    cuPointerGetAttribute(&dstDeviceID, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, p->dstDevice);


    adm_range_t* range = nullptr;
    uint64_t offset_address_range = 0;

    if (object_attribution) {
        range = adm_range_find(p->dstDevice);
        offset_address_range = range->get_address();
    }

    // Log this operation

    uint64_t addr1;
          if (p->dstDevice >= 0x0000010020000000) {
             addr1 = normalise_nvshmem_ptr(p->dstDevice);
          } else {
             addr1 = p->dstDevice;
          }
    std::stringstream ss;
    ss << find_cbid_name(cbid) << ","
      << 	HEX(addr1) << ","
      << -1  << ","
      << srcDeviceID       << ","
      << dstDeviceID       << ","
      << -1 << ","
      << -1 << ","
      << -1 << ","
      << HEX(offset_address_range) << ","
      << p->ByteCount
      << std::endl;
    logger.log(ss.str());

  } else if (is_exit && cbid == API_CUDA_cuMemcpyDtoD_v2) {


    // Check if copy operation was successful from the result field

    cuMemcpyDtoD_v2_params *p = (cuMemcpyDtoD_v2_params *)params;
    CUdevice srcDeviceID;
    CUdevice dstDeviceID;


    cuPointerGetAttribute(&srcDeviceID, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, p->srcDevice);
    cuPointerGetAttribute(&dstDeviceID, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, p->dstDevice);


    adm_range_t* range = nullptr;
    uint64_t offset_address_range = 0;


    if (object_attribution) {
        range = adm_range_find(p->dstDevice);
        offset_address_range = range->get_address();
    }

    // Log this operation

    std::stringstream ss;
    ss << find_cbid_name(cbid) << ","
      << HEX(p->dstDevice) << ","
      << -1  << ","
      << srcDeviceID       << ","
      << dstDeviceID       << ","
      << -1 << ","
      << -1 << ","
      << -1 << ","
      << HEX(offset_address_range) << ","
      << p->ByteCount
      << std::endl;

    logger.log(ss.str());
  }


  if(is_exit && (cbid == API_CUDA_cuMemAlloc ||  cbid == API_CUDA_cuMemAlloc_v2 || cbid == API_CUDA_cuMemAllocHost || cbid == API_CUDA_cuMemAllocHost_v2 || cbid == API_CUDA_cuMemHostAlloc)) {

        std::vector<stacktrace_frame> trace = generate_trace();


	allocation_site_t* allocation_site = root;
	allocation_site_t* parent = NULL;

	for (auto itr = trace.rbegin(); itr != trace.rend(); ++itr) {

		allocation_line_t* line = allocation_line_table->find(itr->address);
        	if(line == NULL) {
                	allocation_line_table->insert(new allocation_line_t(itr->address, itr->symbol, itr->filename, itr->line));
        	}
		if(root == NULL) {
			root = new allocation_site_t (itr->address);
			allocation_site = root;

			parent = allocation_site;
                	allocation_site = allocation_site->get_first_child();
			continue;
		}
		allocation_site_t* temp = allocation_site;
		allocation_site = search_at_level(allocation_site, itr->address);
		if(allocation_site == NULL) {
			if(temp != NULL) {

				while(temp->get_next_sibling() != NULL)
					temp = temp->get_next_sibling();
				temp->set_next_sibling(new allocation_site_t(itr->address));

				allocation_site = temp->get_next_sibling();
				allocation_site->set_parent(temp->get_parent());
			} else {

				parent->set_first_child(new allocation_site_t(itr->address));

				allocation_site = parent->get_first_child();
				allocation_site->set_parent(parent);
			}
		}
		parent = allocation_site;
		allocation_site = allocation_site->get_first_child();
        }

	string func_name;
	if(parent) {
		func_name = allocation_line_table->find(parent->get_pc())->get_func_name();
        	while(func_name.find(/*str1*/"cudaMalloc") == string::npos && func_name.find(/*str1*/"nvshmem_malloc") == string::npos && func_name.find(/*str1*/"nvshmem_align") == string::npos) {
			parent = parent->get_parent();
			if(parent)
				func_name = allocation_line_table->find(parent->get_pc())->get_func_name();
			else
				break;
		}
	}
	if(parent && func_name.find(/*str1*/"nvshmem_malloc") != string::npos)
	if(parent) {
		while(func_name.find(/*str1*/"cudaMalloc") != string::npos || func_name.find(/*str1*/"nvshmem_malloc") != string::npos || func_name.find(/*str1*/"nvshmem_align") != string::npos) {
			parent = parent->get_parent();
			if (parent)
				func_name = allocation_line_table->find(parent->get_pc())->get_func_name();
			else
				break;
		}
	}
	if(parent && parent->get_object_id() == 0) {
		parent->set_object_id(++object_counter);
		object_nodes.push_back(new adm_object_t(parent->get_object_id(), parent, 8));
	}
	adm_range_t* range;
	if(parent) {
        	range = adm_range_insert(ma.pointer, ma.bytesize, parent->get_pc(), ma.deviceID, "", ADM_STATE_ALLOC);
		range_nodes.push_back(new adm_range_t(ma.pointer, ma.bytesize, parent->get_object_id(), ma.deviceID));
	}
  } 
  skip_callback_flag = false;
  log_time(std::string("End Cuda Event ") + (is_exit ? "Exit" : "Enter") +  find_cbid_name(cbid));
  pthread_mutex_unlock(&mutex1);
}

cudaError_t cudaMallocHostWrap ( void** devPtr, size_t size, const char *var_name, const uint32_t element_size, const char *fname, const char *fxname, int lineno) {
  cudaError_t errorOutput = cudaMallocHost( devPtr, size );
  if(*devPtr) {
    if(!object_attribution) {
      object_attribution = true;
    }
    uint64_t allocation_pc = (uint64_t) __builtin_extract_return_addr (__builtin_return_address (0));
    std::string vname = var_name;


    adm_range_t* range = adm_range_find(reinterpret_cast<uint64_t>(*devPtr));
    range->set_var_name(vname);
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
  if(*devPtr) {
    if(!object_attribution) {
      object_attribution = true;
    }
    uint64_t allocation_pc = (uint64_t) __builtin_extract_return_addr (__builtin_return_address (0));
    std::string vname = var_name;
    int dev_id = -1;
    cudaGetDevice(&dev_id);

    adm_range_t* range = adm_range_find(reinterpret_cast<uint64_t>(*devPtr));
    range->set_var_name(vname);

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

void * nvshmem_malloc ( size_t size) {
  void *(*ori_nvshmem_malloc)(size_t) = (void *(*)(size_t)) dlsym(RTLD_NEXT, "nvshmem_malloc");
  nvshmem_malloc_handled = true;
  void * allocated_memory = ori_nvshmem_malloc( size );
  nvshmem_malloc_handled = false;

  int deviceID = -1;
  cudaGetDevice(&deviceID);
        std::vector<stacktrace_frame> trace = generate_trace();
        allocation_site_t* allocation_site = root;
        allocation_site_t* parent = NULL;
        for (auto itr = trace.rbegin(); itr != trace.rend(); ++itr) {
                allocation_line_t* line = allocation_line_table->find(itr->address);
                if(line == NULL) {
                        allocation_line_table->insert(new allocation_line_t(itr->address, itr->symbol, itr->filename, itr->line));
                }
                if(root == NULL) {
                        root = new allocation_site_t (itr->address);
                        allocation_site = root;
                        parent = allocation_site;
                        allocation_site = allocation_site->get_first_child();
                        continue;
                }
                allocation_site_t* temp = allocation_site;
                allocation_site = search_at_level(allocation_site, itr->address);
                if(allocation_site == NULL) {
                        if(temp != NULL) {
                                while(temp->get_next_sibling() != NULL)
                                        temp = temp->get_next_sibling();
                                temp->set_next_sibling(new allocation_site_t(itr->address));
                                allocation_site = temp->get_next_sibling();
                                allocation_site->set_parent(temp->get_parent());
                        } else {
                                parent->set_first_child(new allocation_site_t(itr->address));
                                allocation_site = parent->get_first_child();
                                allocation_site->set_parent(parent);
                        }
                }
                parent = allocation_site;
                allocation_site = allocation_site->get_first_child();
        }

        string func_name;
        if(parent) {
                func_name = allocation_line_table->find(parent->get_pc())->get_func_name();
                while(func_name.find(/*str1*/"nvshmem_malloc") == string::npos) {
                        parent = parent->get_parent();
                        if(parent)
                                func_name = allocation_line_table->find(parent->get_pc())->get_func_name();
                        else
                                break;
                }
        }

        if(parent) {
                while(func_name.find(/*str1*/"nvshmem_malloc") != string::npos) {
                        parent = parent->get_parent();
                        if (parent)
                                func_name = allocation_line_table->find(parent->get_pc())->get_func_name();
                        else
                                break;
                }
        }

        if(parent && parent->get_object_id() == 0) {
                parent->set_object_id(++object_counter);
                object_nodes.push_back(new adm_object_t(parent->get_object_id(), parent, 8));
        }

        adm_range_t* range;
	MemoryAllocation ma;
        if(parent) {
                range = adm_range_insert((uint64_t) allocated_memory, size, parent->get_pc(), deviceID, "", ADM_STATE_ALLOC);
                range_nodes.push_back(new adm_range_t((uint64_t) allocated_memory, size, parent->get_object_id(), deviceID));
		ma.deviceID = deviceID;
		ma.pointer = (uint64_t) allocated_memory;
		ma.bytesize = size;
		mem_allocs.push_back(ma);
        }
  return allocated_memory;
}

void * nvshmem_alignWrap ( size_t alignment, size_t size, const char *var_name, const uint32_t element_size, const char *fname, const char *fxname, int lineno/*, const std::experimental::source_location& location = std::experimental::source_location::current()*/) {
  void *(*ori_nvshmem_align)(size_t, size_t) = (void *(*)(size_t, size_t)) dlsym(RTLD_NEXT, "nvshmem_malloc");
  void * allocated_memory = ori_nvshmem_align( alignment, size );
  if(allocated_memory /*&& adm_set_tracing(0)*/) {
    if(!object_attribution) {
      object_attribution = true;
    }
    uint64_t allocation_pc = (uint64_t) __builtin_extract_return_addr (__builtin_return_address (0));
    std::string vname = var_name;
    int dev_id = -1;
    cudaGetDevice(&dev_id);

    adm_range_t* range = adm_range_find(reinterpret_cast<uint64_t>(allocated_memory));
    range->set_var_name(vname);

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

  int dev_id = -1;
  cudaGetDevice(&dev_id);

  log_time(std::string("Bgn Recv Thread " + to_string(dev_id)));

  ChannelHost *ch_host = &ctx_state->channel_host;

  pthread_mutex_unlock(&mutex1);
  char *recv_buffer = (char *)malloc(CHANNEL_SIZE);



  if (!silent && ((int)ctx_state_map.size() - 1 == 0)) {
    std::stringstream ss;
    ss << "op_code, addr, thread_indx, running_dev_id, mem_dev_id, code_linenum, code_line_index, code_line_estimated_status, obj_offset, mem_range" << std::endl;
    logger.log(ss.str());
  }

  bool done = false;
  bool waiting = false;
  while (!done)
  {

    if (!waiting) {
      log_time(std::string("Bgn Waiting Recv Thread " + to_string(dev_id)));
      waiting = true;
    }

    /* receive buffer from channel */
    uint32_t num_recv_bytes = ch_host->recv(recv_buffer, CHANNEL_SIZE);

    if (num_recv_bytes > 0)
    {
      log_time(std::string("End Waiting Recv Thread " + to_string(dev_id)));
      waiting = false;
      log_time(std::string("Bgn Processing Recv Thread " + to_string(dev_id)));
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
          bool nvshmem_flag = false;
          if (mem_device_id == -1 && (ma->addrs[i] >= 0x0000010020000000)) {
	    nvshmem_flag = true;
            mem_device_id = find_nvshmem_dev_of_ptr(ma->dev_id, ma->addrs[i], nvshmem_ngpus, nvshmem_version);
          }

          // ignore operations on the same device
          if (mem_device_id == ma->dev_id)
            continue;

          if (mem_device_id == -1)
            continue;

          uint32_t index_in_object = 0;
          uint32_t index_in_malloc = 0;

          if (silent) continue;

          std::stringstream ss;
	  uint64_t addr1;
	  if (nvshmem_flag) {
             addr1 = normalise_nvshmem_ptr(ma->addrs[i]);
          } else {
	     addr1 = ma->addrs[i];
	  }

	range = adm_range_find(addr1);
            if(range != nullptr) {
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

          if (JSON) {
            ss << "{\"op\": \"" << id_to_opcode_map[ma->opcode_id]  << "\", "
              << "\"kernel_name\": \"" << instrumented_functions[ma->func_id] << "\", "
              << "\"addr\": \"" << HEX(addr1) << "\","
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
              << HEX(addr1) << ","
              << ma->thread_index  << ","
              << ma->dev_id        << ","
              << mem_device_id     << ","
              << line_linenum      << ","
              << line_index        << ","
              << line_estimated_status << ","
              << HEX(offset_address_range) << ","
              << 4
              << std::endl;
          }
          logger.log(ss.str());
          // memop_outfile << ss.str() << std::flush;
        }
        num_processed_bytes += sizeof(mem_access_t);
      }

      log_time(std::string("End Processing Recv Thread " + to_string(dev_id)));
    }
  }

  log_time(std::string("End Recv Thread " + to_string(dev_id)));
  return NULL;
}

void nvbit_at_ctx_init(CUcontext ctx)
{
  pthread_mutex_lock(&mutex1);
  int dev_id = -1;
  cudaGetDevice(&dev_id);

  log_time("Bgn Context" + to_string(dev_id));
  if (verbose)
  {
    printf("MEMTRACE: STARTING CONTEXT %p\n", ctx);
  }
  CTXstate *ctx_state = new CTXstate;
  assert(ctx_state_map.find(ctx) == ctx_state_map.end());
  ctx_state_map[ctx] = ctx_state;
  cudaMallocManaged(&ctx_state->channel_dev, sizeof(ChannelDev));

  ctx_state->channel_host.init((int)ctx_state_map.size() - 1, CHANNEL_SIZE,
      ctx_state->channel_dev, recv_thread_fun, on_dev_filtering, ctx);
  nvbit_set_tool_pthread(ctx_state->channel_host.get_thread());
  pthread_mutex_unlock(&mutex1);
}

void nvbit_at_ctx_term(CUcontext ctx)
{
  pthread_mutex_lock(&mutex1);
  int dev_id = -1;
  cudaGetDevice(&dev_id);
  log_time("End Context" + to_string(dev_id));

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
    if(object_attribution)
        adm_ranges_print();
    adm_line_table_print();
  }
  vector<bool> flag(100, true);
  cout << "Tree of data objects\n";
  printNTree(root, flag);

  ofstream object_outfile;
  string object_str("mem_alloc_site_log_");
  string txt_str(".txt");
  string object_log_str = object_str + to_string(getpid()) + txt_str;
  object_outfile.open(object_log_str);
  object_outfile << "pc,func_name,file_name,line_no\n";
  allocation_line_table->print(object_outfile);
  object_outfile.close();

  ofstream object_outfile1;
  string object_str1("address_range_log_");
  string object_log_str1 = object_str1 + to_string(getpid()) + txt_str;
  object_outfile1.open(object_log_str1);
  object_outfile1 << "offset,size,obj_id,dev_id\n";
  for(auto i : range_nodes)
        i->print(object_outfile1);
  object_outfile1.close();

  ofstream object_outfile2;
  string object_str2("data_object_log_");
  string object_log_str2 = object_str2 + to_string(getpid()) + txt_str;
  object_outfile2.open(object_log_str2);
  object_outfile2 << "obj_id,var_name,call_stack\n";
  for(auto i : object_nodes)
        i->print(object_outfile2);
  object_outfile2.close();
  delete allocation_line_table;
  delete root;
  log_time("End Snoopie");
  adm_db_fini();
}
