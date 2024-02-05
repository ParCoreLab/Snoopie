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

#pragma once

#include "utils.h"
#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define ULL unsigned long long int
#define USE_ASYNC_STREAM

#ifndef MEMTINGS2
#define MEMTINGS2
typedef struct
{
  int dev_id;

  int lane_id;
  int func_id;
  int opcode_id;
  uint64_t addrs[32];

  int global_index;
  uint64_t thread_index;
} mem_access_t;
#endif

#ifndef MEMTINGS
#define MEMTINGS
struct MemoryAllocation
{
  int deviceID;
  uint64_t pointer;
  uint64_t bytesize;
};
#endif

class ChannelDev
{
private:
  int id;
  volatile int *doorbellA;
  volatile int *doorbellB;

  uint8_t *buffA;
  uint8_t *buffB;

  // True for A, false for B;
  bool *currentBuff;

  uint8_t *buff;
  uint8_t *buff_end;

  /* head/tail pointers */
  uint8_t *volatile buff_write_head_ptr;
  uint8_t *volatile buff_write_tail_ptr;

  MemoryAllocation *mallocs_record;
  bool on_dev_filtering = true;

  uint64_t no_mallocs = 0;

public:
  ChannelDev() {}

  __device__ __forceinline__ void
  push(void *packet, uint32_t nbytes, int this_device)
  {
    assert(nbytes != 0);

    uint8_t *curr_ptr = NULL;

    mem_access_t *mc = (mem_access_t *)packet;

    // TODO: CHECK all packets, if any of them is a remote address, allow to
    // push, otherwise skip

    bool found_remote = false;

    if(mc->lane_id != -1 && on_dev_filtering)
      {
        for(int i = 0; i < 32; i++)
          {
            uint64_t ptr = mc->addrs[i];

            if(found_remote)
              break;

            for(int j = 0; j < no_mallocs; j++)
              {
                MemoryAllocation ma = mallocs_record[j];
                if(ma.pointer <= ptr && ptr < ma.pointer + ma.bytesize)
                  {
                    if(ma.deviceID != this_device)
                      {
                        found_remote = true;
                        break;
                      }
                  }

                // check if ptr falls within nvshmem's peer memory address space
                if(0x0000012020000000 <= ptr && ptr <= 0x0000020020000000)
                  {
                    found_remote = true;
                  }
              }
          }

        if(!found_remote)
          {
            return;
          }
      }

    while(curr_ptr == NULL)
      {
        curr_ptr
          = (uint8_t *)atomicAdd((ULL *)&buff_write_head_ptr, (ULL)nbytes);

        /* if the current position plus nbytes is after buff_end, the
         * buffer is full.
         * Many warps could find condition true, but only the first warp
         * will find true the condition after. */
        if(curr_ptr + nbytes > buff_end)
          {
            /* I am the first warp that found the buffer full and
             * I am the one responsible for flushing the buffer out */
            if(curr_ptr <= buff_end)
              {
                /* wait until everyone completed to write */
                while(buff_write_tail_ptr != curr_ptr) {}

                /* flush buffer */
                flush();
              }
            else
              {
                /* waiting for buffer to flush */
                while(buff_write_head_ptr > buff_end) {}
              }
            curr_ptr = NULL;
          }
      }

    memcpy(curr_ptr, packet, nbytes);
    atomicAdd((ULL *)&buff_write_tail_ptr, (ULL)nbytes);

    // if (nbytes != 0) {
    //   return;
    // }
  }

  __device__ __forceinline__ void flush()
  {
    uint32_t nbytes = (uint32_t)(buff_write_tail_ptr - buff);
    // printf("FLUSH CHANNEL#%d: buffer bytes %d, currentBuff: %d\n", id,
    // nbytes, *currentBuff);
    if(nbytes == 0)
      {
        return;
      }

    /* make sure everything is visible in memory */
    __threadfence_system();

    if(*currentBuff)
      {
        assert(*doorbellA == 0);
      }
    else
      {
        assert(*doorbellB == 0);
      }

    /* notify current buffer has something*/
    if(*currentBuff)
      {
        *doorbellA = nbytes;
      }
    else
      {
        *doorbellB = nbytes;
      }

    *currentBuff = !*currentBuff;
    /* wait for host to release the doorbell */

    if(*currentBuff)
      {
        while(*doorbellA != 0)
          ;
      }
    else
      {
        while(*doorbellB != 0)
          ;
      }

    /* reset head/tail */
    if(*currentBuff)
      {
        buff_write_tail_ptr = buffA;
        __threadfence();
        buff_write_head_ptr = buffA;
      }
    else
      {
        buff_write_tail_ptr = buffB;
        __threadfence();
        buff_write_head_ptr = buffB;
      }

    // printf("FLUSH CHANNEL#%d: DONE\n", id);
  }

  void add_malloc(MemoryAllocation ma) { mallocs_record[no_mallocs++] = ma; }

private:
  /* called by the ChannelHost init */
  void init(int id, int *h_doorbellA, int *h_doorbellB, int buff_size,
            bool on_dev_filtering)
  {
    this->on_dev_filtering = on_dev_filtering;
    CUDA_SAFECALL(
      cudaHostGetDevicePointer((void **)&doorbellA, (void *)h_doorbellA, 0));
    CUDA_SAFECALL(
      cudaHostGetDevicePointer((void **)&doorbellB, (void *)h_doorbellB, 0));

/* allocate large buffer */
#ifdef USE_ASYNC_STREAM
    CUDA_SAFECALL(cudaMalloc((void **)&buffA, buff_size));
    CUDA_SAFECALL(cudaMalloc((void **)&buffB, buff_size));
    CUDA_SAFECALL(cudaMalloc((void **)&currentBuff, sizeof(bool)));
    CUDA_SAFECALL(cudaMemset((void **)&currentBuff, 1, sizeof(bool)));
#else
    CUDA_SAFECALL(cudaMallocManaged((void **)&buffA, buff_size));
    CUDA_SAFECALL(cudaMallocManaged((void **)&buffB, buff_size));
#endif
    CUDA_SAFECALL(cudaMallocManaged((void **)&mallocs_record, buff_size));
    buff = buffB;
    buff_write_head_ptr = buff;
    buff_write_tail_ptr = buff;
    buff_end = buff + buff_size;
    this->id = id;
  }

  friend class ChannelHost;
};

class ChannelHost
{
private:
  volatile int *doorbellA;
  volatile int *doorbellB;

  volatile int *h_doorbellA;
  volatile int *h_doorbellB;

  volatile int *h_currentBuff;

  cudaStream_t stream;
  ChannelDev *ch_dev;

  /* pointers to device buffer */
  uint8_t *dev_buff_read_head;
  uint8_t *dev_buffA;
  uint8_t *dev_buffB;

  uint8_t *hdev_buff;

  /* receiving thread */
  pthread_t thread;
  volatile bool thread_started;

public:
  int id;
  int buff_size;

public:
  ChannelHost() {}

  void
  init(int id, int buff_size, ChannelDev *ch_dev, void *(*thread_fun)(void *),
       bool on_dev_filtering, void *args = NULL)
  {
    this->buff_size = buff_size;
    this->id = id;
    /* get device properties */
    cudaDeviceProp prop;
    int device = 0;
    cudaGetDeviceProperties(&prop, device);
    if(prop.canMapHostMemory == 0)
      {
        CUDA_SAFECALL(cudaSetDeviceFlags(cudaDeviceMapHost));
      }

#ifdef USE_ASYNC_STREAM
    /* create stream that will read memory with highest possible priority */
    int priority_high, priority_low;
    CUDA_SAFECALL(
      cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high));
    CUDA_SAFECALL(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking,
                                               priority_high));
#endif

    /* create doorbell */
    CUDA_SAFECALL(
      cudaHostAlloc((void **)&doorbellA, sizeof(int), cudaHostAllocMapped));
    CUDA_SAFECALL(
      cudaHostAlloc((void **)&doorbellB, sizeof(int), cudaHostAllocMapped));
    CUDA_SAFECALL(
      cudaHostAlloc((void **)&h_doorbellA, sizeof(int), cudaHostAllocMapped));
    CUDA_SAFECALL(
      cudaHostAlloc((void **)&h_doorbellB, sizeof(int), cudaHostAllocMapped));
    CUDA_SAFECALL(cudaHostAlloc((void **)&h_currentBuff, sizeof(int),
                                cudaHostAllocMapped));
    CUDA_SAFECALL(
      cudaHostAlloc((void **)&hdev_buff, buff_size, cudaHostAllocMapped));
    /* set doorbell to zero */
    *doorbellA = 0;
    *doorbellB = 0;
    *h_doorbellA = 0;
    *h_doorbellB = 0;
    *h_currentBuff = 1;

    /* initialize device channel */
    this->ch_dev = ch_dev;
    ch_dev->init(id, (int *)doorbellA, (int *)doorbellB, buff_size,
                 on_dev_filtering);

    dev_buffA = ch_dev->buffA;
    dev_buffB = ch_dev->buffB;

    dev_buff_read_head = hdev_buff;
    if(thread_fun != NULL)
      {
        thread_started = true;
        pthread_create(&thread, NULL, (void *(*)(void *))thread_fun, args);
      }
    else
      {
        thread_started = false;
      }
  }

  /* when used in nvbit we don't want to dealloc because
   * when modules are unloaded the driver automatically
   * deallocates CUDA malloc, so further deallocs done
   * here will result in errors */
  void destroy(bool dealloc)
  {
    if(thread_started)
      {
        thread_started = false;
        pthread_join(thread, NULL);
      }
    if(dealloc)
      {
#ifdef USE_ASYNC_STREAM
        CUDA_SAFECALL(cudaStreamDestroy(stream));
#endif
        CUDA_SAFECALL(cudaFree((int *)doorbellA));
        CUDA_SAFECALL(cudaFree((int *)h_doorbellA));
        CUDA_SAFECALL(cudaFree(ch_dev->buff));
      }
  }

  bool is_active() { return thread_started; }

  void load_dev_buff()
  {
    *h_currentBuff = !*h_currentBuff;
    // wait until signaled from the device

    if(*h_currentBuff)
      {
        while(*doorbellA == 0)
          ;
      }
    else
      {
        while(*doorbellB == 0)
          ;
      }

    if(*h_currentBuff)
      {
        *h_doorbellA = *doorbellA;
      }
    else
      {
        *h_doorbellB = *doorbellB;
      }

    if(*h_currentBuff)
      {
#ifdef USE_ASYNC_STREAM
        CUDA_SAFECALL(cudaMemcpyAsync(hdev_buff, dev_buffA, buff_size,
                                      cudaMemcpyDeviceToHost, stream));
        CUDA_SAFECALL(cudaStreamSynchronize(stream));
#else
        memcpy(hdev_buff, dev_buffA, buff_size);
#endif
      }
    else
      {
#ifdef USE_ASYNC_STREAM
        CUDA_SAFECALL(cudaMemcpyAsync(hdev_buff, dev_buffB, buff_size,
                                      cudaMemcpyDeviceToHost, stream));
        CUDA_SAFECALL(cudaStreamSynchronize(stream));
#else
        memcpy(hdev_buff, dev_buffB, buff_size);
#endif
      }

    if(*h_currentBuff)
      {
        *doorbellA = 0;
      }
    else
      {
        *doorbellB = 0;
      }
  }

  uint32_t recv(void *buff, uint32_t max_buff_size)
  {
    assert(max_buff_size > 0);
    assert(h_doorbellA != NULL);
    uint32_t buff_nbytes;
    if(*h_currentBuff)
      {
        buff_nbytes = *h_doorbellA;
      }
    else
      {
        buff_nbytes = *h_doorbellB;
      }

    if(buff_nbytes == 0)
      {
        // only attempt to load device buffer when host buffer is empty
        load_dev_buff();
        return 0;
      }

    int nbytes = buff_nbytes;

    // printf("HOST TO RECEIVE nbytes %d - bytes left %d\n", nbytes, 0);
    if(buff_nbytes > max_buff_size)
      {
        nbytes = max_buff_size;
      }
    memcpy(buff, dev_buff_read_head, nbytes);
    // #endif
    int bytes_left = buff_nbytes - nbytes;
    assert(bytes_left >= 0);
    if(bytes_left > 0)
      {
        dev_buff_read_head += nbytes;
      }
    else
      {
        dev_buff_read_head = hdev_buff;
      }

    if(*h_currentBuff)
      {
        *h_doorbellA = bytes_left;
      }
    else
      {
        *h_doorbellB = bytes_left;
      }
    // printf("HOST RECEIVED nbytes %d - bytes left %d\n", nbytes, bytes_left);
    return nbytes;
  }

  pthread_t get_thread() { return thread; }

  friend class MultiChannelHost;
};
