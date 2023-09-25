#ifndef __ADAMANT_MEMORY
#define __ADAMANT_MEMORY

#include <stdint.h>

#include <adm_common.h>

namespace adamant {

template <typename T, uint32_t size> class memory_t {
private:
  memory_t<T, size> *const next;
  T memory[size];

public:
  memory_t<T, size>(memory_t<T, size> *n) : next(n) {}

  ~memory_t<T, size>() { delete next; }

  T *get_block(const uint32_t block) noexcept { return memory + block; }

  memory_t<T, size> *get_next() noexcept { return next; }
};

template <typename T, uint32_t size> class pool_t {
private:
  memory_t<T, size> *memory;
  uint32_t block;

public:
  pool_t<T, size>() : memory(nullptr), block(size) {}

  ~pool_t<T, size>() {
    if (memory)
      delete memory;
  }

  T *malloc(const uint32_t n = 1) noexcept {
    if (n < size)
      try {
        if (block + n > size) {
          // unsigned char t = adm_set_tracing(0);
          memory = new memory_t<T, size>(memory);
          // adm_set_tracing(t);
          block = 0;
        }
        uint32_t b = block;
        block += n;
        return memory->get_block(b);
      } catch (const std::bad_alloc &e) {
        adm_warning("Alloocation failed!");
      }
    return nullptr;
  }

  class iterator {
  private:
    pool_t<T, size> &pool;
    memory_t<T, size> *memory;
    uint32_t block;

  public:
    iterator(pool_t<T, size> &p) : pool(p), memory(p.memory), block(0) {}

    T *next() noexcept {
      if (memory && (block < pool.block || memory != pool.memory)) {
        T *n = memory->get_block(block);
        ++block;
        if ((memory == pool.memory && block == pool.block) || block == size) {
          memory = memory->get_next();
          block = 0;
        }
        return n;
      }

      return nullptr;
    }
  };
};

} // namespace adamant

#endif
