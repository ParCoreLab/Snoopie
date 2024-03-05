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

#include <stdint.h>
#include <stdio.h>
//#include <curand.h>
//#include <curand_kernel.h>

#include "utils/utils.h"

/* for channel */
#include "utils/channel.hpp"

/* contains definition of the mem_access_t structure */
#include "common.h"

#if 0
#include "adm.h"
#include "adm_common.h"
#include "adm_database.h"

using namespace adamant;
#endif

__device__ int last_valid_line_index = -1;

extern "C" __device__ __noinline__ void instrument_mem(int pred, int opcode_id, int dev_id,
                                                       uint64_t addr,
                                                       uint64_t grid_launch_id,
                                                       uint64_t pchannel_dev,
                                                       int global_index,
                                                       int func_id,
						       int sample_size) {
    /* if thread is predicated off, return */
    if (!pred) {
        return;
    }

    int active_mask = __ballot_sync(__activemask(), 1);
    const int laneid = get_laneid();
    const int first_laneid = __ffs(active_mask) - 1;

    mem_access_t ma;

    /* collect memory address information from other threads */
    for (int i = 0; i < 32; i++) {
        ma.addrs[i] = __shfl_sync(active_mask, addr, i);
    }

    //adm_range_t* obj = adm_range_find(ma.addrs[0]);
    //ma.allocation_pc = obj->get_allocation_pc();

    // int4 cta = get_ctaid();
    // ma.grid_launch_id = grid_launch_id;
    // ma.cta_id_x = cta.x;
    // ma.cta_id_y = cta.y;
    // ma.cta_id_z = cta.z;
    ma.dev_id = dev_id;
    // ma.warp_id = get_warpid();
    ma.lane_id = get_laneid();
    ma.opcode_id = opcode_id;
    ma.global_index = global_index;
    ma.func_id = func_id;

    uint64_t blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    ma.thread_index = blockId * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

    /* first active lane pushes information on the channel */
//#if 0
    if (first_laneid == laneid) {
        ChannelDev* channel_dev = (ChannelDev*)pchannel_dev;
	//curandState state;
	long rand_num = clock64() % 100000;
        //curand_init(clock64(), ma.thread_index, 0, &state);

        //float randf = curand_uniform(&state);
	if (rand_num < 100000/sample_size)
        	channel_dev->push(&ma, sizeof(mem_access_t), dev_id);
    }
//#endif
}

__global__ void trickary() { instrument_mem(0, 0, 0, 0, 0, 0, 0, 0, 0); }
