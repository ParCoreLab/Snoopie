#include <cstdint>
#include <string>
#include <vector>
#include "util.h"

#ifndef PTR_UTIL_H
#define PTR_UTIL_H

int64_t find_nvshmem_dev_of_ptr(int mype, uint64_t mem_addr, int nvshmem_ngpus,
                                std::string version)
{

  int size = 15;

  int region = -1;

  // 0x000012020000000 is nvshmem's first address for a remote peer
  uint64_t start = 0x000012020000000;

  // 0x000010020000000 is nvshmem's address for the peer itself
  uint64_t incrmnt = (uint64_t)0x000012020000000 - (uint64_t)0x000010020000000;

  for (int i = 1; i <= size; i++)
  {
    uint64_t bottom = (uint64_t)start + (i - 1) * incrmnt;
    uint64_t top = (uint64_t)start + i * incrmnt;
    if ((uint64_t)bottom <= (uint64_t)mem_addr &&
        (uint64_t)mem_addr < (uint64_t)top)
    {
      region = i - 1;
      break;
    }
  }

  if (region == -1)
  {
    return -1;
  }

  if (version == "2.9" || version == "2.8")
  {
    region += mype;
  }

  if (mype == region)
  {
    return (mype + 1) % nvshmem_ngpus;
  }

  for (int i = 0; i < size; i++)
  {
    if (mype == i)
      continue;

    if (region == 0)
    {
      return i % nvshmem_ngpus;
    }

    region--;
  }

  return -1;
}

uint64_t normalise_nvshmem_ptr(uint64_t mem_addr)
{
  return mem_addr & 0x0000F0FFFFFFFFF;
}

int64_t find_dev_of_ptr(uint64_t ptr, std::vector<MemoryAllocation> &mem_allocs)
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

#endif // PTR_UTIL_H