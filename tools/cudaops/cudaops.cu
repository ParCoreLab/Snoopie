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
#include <sstream>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>

/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

/* for channel */
#include "utils/channel.hpp"

/* contains definition of the mem_access_t structure */
#include "common.h"
#include "util.h"

#define HEX(x)                                                                 \
  "0x" << std::setfill('0') << std::setw(16) << std::hex << (uint64_t)x        \
       << std::dec

#define CHANNEL_SIZE (1l << 30)

#define EQUAL_STRS 0

struct CTXstate {
  /* context id */
  int id;

  /* Channel used to communicate from GPU to CPU receiving thread */
  ChannelDev *channel_dev;
  ChannelHost channel_host;
};

struct MemoryAllocation {
  int deviceID;
  uint64_t pointer;
  uint64_t bytesize;
};

/* lock */
pthread_mutex_t mutex;

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

/* grid launch id, incremented at every launch */
uint64_t grid_launch_id = 0;

void nvbit_at_init() {
  setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
  GET_VAR_INT(
      instr_begin_interval, "INSTR_BEGIN", 0,
      "Beginning of the instruction interval where to apply instrumentation");
  GET_VAR_INT(instr_end_interval, "INSTR_END", UINT32_MAX,
              "End of the instruction interval where to apply instrumentation");
  GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");

  GET_VAR_STR(kernel_name, "KERNEL_NAME",
              "Specify the name of the kernel to track");

  std::string pad(100, '-');
  if (verbose) {
    std::cout << pad << std::endl;
  }

  /* set mutex as recursive */
  pthread_mutexattr_t attr;
  pthread_mutexattr_init(&attr);
  pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
  pthread_mutex_init(&mutex, &attr);
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus) {
  pthread_mutex_lock(&mutex);
  std::string cbid_name(find_cbid_name(cbid));

  bool is_func = false;
  std::string func_name;
  CUfunction func;

  if (cbid == API_CUDA_cuLaunchKernel || cbid == API_CUDA_cuLaunchKernel_ptsz) {
    cuLaunchKernel_params *p = (cuLaunchKernel_params *)params;
    func_name = (nvbit_get_func_name(ctx, p->f));
    func = p->f;
    is_func = true;
  } else if (cbid == API_CUDA_cuLaunchCooperativeKernel ||
             cbid == API_CUDA_cuLaunchCooperativeKernel_ptsz) {
    cuLaunchCooperativeKernel_params *p =
        (cuLaunchCooperativeKernel_params *)params;
    func_name = (nvbit_get_func_name(ctx, p->f));
    func = p->f;
    is_func = true;
  }

  std::cout << "###"
            << " " << (is_exit ? "BGN" : "END") << ":\t" << cbid_name
            << (is_func ? func_name : "") << std::endl;

  if (is_func) {
    std::vector<CUfunction> related_functions =
        nvbit_get_related_functions(ctx, func);

    /* iterate on function */
    for (auto f : related_functions) {
      std::string related_func_name(nvbit_get_func_name(ctx, f));
      std::cout << "Related func to: " << func_name
                << " is: " << related_func_name << std::endl;
    }
  }

  pthread_mutex_unlock(&mutex);
}
