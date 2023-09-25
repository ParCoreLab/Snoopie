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
#include <map>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <unordered_set>

/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

/* provide some __device__ functions */
#include "utils/utils.h"

/* kernel id counter, maintained in system memory */
uint32_t kernel_id = 0;

/* total instruction counter, maintained in system memory, incremented by
 * "counter" every time a kernel completes  */
uint64_t tot_app_instrs = 0;

/* kernel instruction counter, updated by the GPU threads */
#define MAX_OPCODES (16 * 1024)
__managed__ uint64_t histogram[MAX_OPCODES];

/* global control variables for this tool */
uint32_t instr_begin_interval = 0;
uint32_t instr_end_interval = UINT32_MAX;
uint32_t ker_begin_interval = 0;
uint32_t ker_end_interval = UINT32_MAX;
int verbose = 0;
int count_warp_level = 1;
int exclude_pred_off = 0;

/* instruction to opcode map, used for final print of the opcodes */
std::map<std::string, int> instr_opcode_to_num_map;

/* a pthread mutex, used to prevent multiple kernels to run concurrently and
 * therefore to "corrupt" the counter variable */
pthread_mutex_t mutex;

/* nvbit_at_init() is executed as soon as the nvbit tool is loaded. We typically
 * do initializations in this call. In this case for instance we get some
 * environment variables values which we use as input arguments to the tool */
void nvbit_at_init() {
  /* just make sure all managed variables are allocated on GPU */
  setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);

  /* we get some environment variables that are going to be use to selectively
   * instrument (within a interval of kernel indexes and instructions). By
   * default we instrument everything. */
  GET_VAR_INT(
      instr_begin_interval, "INSTR_BEGIN", 0,
      "Beginning of the instruction interval where to apply instrumentation");
  GET_VAR_INT(instr_end_interval, "INSTR_END", UINT32_MAX,
              "End of the instruction interval where to apply instrumentation");
  GET_VAR_INT(ker_begin_interval, "KERNEL_BEGIN", 0,
              "Beginning of the kernel launch interval where to apply "
              "instrumentation");
  GET_VAR_INT(
      ker_end_interval, "KERNEL_END", UINT32_MAX,
      "End of the kernel launch interval where to apply instrumentation");
  GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");
  GET_VAR_INT(count_warp_level, "COUNT_WARP_LEVEL", 1,
              "Count warp level or thread level instructions");
  GET_VAR_INT(exclude_pred_off, "EXCLUDE_PRED_OFF", 0,
              "Exclude predicated off instruction from count");

  std::string pad(100, '-');
  printf("%s\n", pad.c_str());
}

/* Set used to avoid re-instrumenting the same functions multiple times */
std::unordered_set<CUfunction> already_instrumented;

void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
  /* Get related functions of the kernel (device function that can be
   * called by the kernel) */
  std::vector<CUfunction> related_functions =
      nvbit_get_related_functions(ctx, func);

  /* add kernel itself to the related function vector */
  related_functions.push_back(func);

  /* iterate on function */
  for (auto f : related_functions) {
    /* "recording" function was instrumented, if set insertion failed
     * we have already encountered this function */
    if (!already_instrumented.insert(f).second) {
      continue;
    }

    /* Get the vector of instruction composing the loaded CUFunction "func"
     */
    const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, func);

    /* Get kernel name */
    std::string kernel_name = nvbit_get_func_name(ctx, func);

    /* If verbose we print function name and number of" static" instructions
     */
    if (verbose) {
      printf("inspecting %s - num instrs %ld\n", kernel_name.c_str(),
             instrs.size());
    }

    /* We iterate on the vector of instruction */
    for (auto i : instrs) {
      /* Check if the instruction falls in the interval where we want to
       * instrument */
      if (i->getIdx() < instr_begin_interval ||
          i->getIdx() >= instr_end_interval) {
        continue;
      }
      /* If verbose we print which instruction we are instrumenting */
      if (verbose) {
        i->print();
      }

      std::string opcode = i->getOpcode();
      if (instr_opcode_to_num_map.find(opcode) ==
          instr_opcode_to_num_map.end()) {
        size_t size = instr_opcode_to_num_map.size();
        instr_opcode_to_num_map[opcode] = size;
      }
      int instr_type = instr_opcode_to_num_map[opcode];

      /* Insert a call to "count_instrs" before the instruction "i" */
      nvbit_insert_call(i, "count_instrs", IPOINT_BEFORE);
      /* Add argument to the instrumentation function */
      if (exclude_pred_off) {
        /* pass predicate value */
        nvbit_add_call_arg_guard_pred_val(i);
      } else {
        /* pass always true */
        nvbit_add_call_arg_const_val32(i, 1);
      }

      /* add instruction type id */
      nvbit_add_call_arg_const_val32(i, instr_type);
      /* add count warps option */
      nvbit_add_call_arg_const_val32(i, count_warp_level);
      /* add pointer to counter location */
      nvbit_add_call_arg_const_val64(i, (uint64_t)histogram);
    }
  }
}

/* This call-back is triggered every time a CUDA event is encountered.
 * Here, we identify CUDA kernel launch events and reset the "counter" before
 * th kernel is launched, and print the counter after the kernel has completed
 * (we make sure it has completed by using cudaDeviceSynchronize()). To
 * selectively run either the original or instrumented kernel we used
 * nvbit_enable_instrumented() before launching the kernel. */
void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus) {
  /* Identify all the possible CUDA launch events */
  if (cbid == API_CUDA_cuLaunch || cbid == API_CUDA_cuLaunchKernel_ptsz ||
      cbid == API_CUDA_cuLaunchGrid || cbid == API_CUDA_cuLaunchGridAsync ||
      cbid == API_CUDA_cuLaunchKernel) {
    /* cast params to cuLaunch_params since if we are here we know these are
     * the right parameters type */
    cuLaunch_params *p = (cuLaunch_params *)params;

    if (!is_exit) {
      /* if we are entering in a kernel launch:
       * 1. Lock the mutex to prevent multiple kernels to run concurrently
       * (overriding the counter) in case the user application does that
       * 2. Instrument the function if needed
       * 3. Select if we want to run the instrumented or original
       * version of the kernel
       * 4. Reset the kernel instruction counter */

      pthread_mutex_lock(&mutex);
      instrument_function_if_needed(ctx, p->f);

      if (kernel_id >= ker_begin_interval && kernel_id < ker_end_interval) {
        nvbit_enable_instrumented(ctx, p->f, true);
      } else {
        nvbit_enable_instrumented(ctx, p->f, false);
      }
      memset(histogram, 0, sizeof(uint64_t) * MAX_OPCODES);
    } else {
      /* if we are exiting a kernel launch:
       * 1. Wait until the kernel is completed using
       * cudaDeviceSynchronize()
       * 2. Get number of thread blocks in the kernel
       * 3. Print the thread instruction counters
       * 4. Release the lock*/
      CUDA_SAFECALL(cudaDeviceSynchronize());
      int num_ctas = 0;
      if (cbid == API_CUDA_cuLaunchKernel_ptsz ||
          cbid == API_CUDA_cuLaunchKernel) {
        cuLaunchKernel_params *p2 = (cuLaunchKernel_params *)params;
        num_ctas = p2->gridDimX * p2->gridDimY * p2->gridDimZ;
      }
      uint64_t counter = 0;
      for (auto a : instr_opcode_to_num_map) {
        if (histogram[a.second] != 0) {
          counter += histogram[a.second];
        }
      }
      tot_app_instrs += counter;
      printf("kernel %d - %s - #thread-blocks %d,  kernel "
             "instructions %ld, total instructions %ld\n",
             kernel_id++, nvbit_get_func_name(ctx, p->f), num_ctas, counter,
             tot_app_instrs);

      for (auto a : instr_opcode_to_num_map) {
        if (histogram[a.second] != 0) {
          printf("  %s = %ld\n", a.first.c_str(), histogram[a.second]);
        }
      }
      pthread_mutex_unlock(&mutex);
    }
  }
}
