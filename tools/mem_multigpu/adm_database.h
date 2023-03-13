#ifndef __ADAMANT_DATABASE
#define __ADAMANT_DATABASE

#include <cstdint>
#include <cstring>
#define UNW_LOCAL_ONLY
#include <libunwind.h>

#include <adm.h>
#include <adm_config.h>
#include <adm_common.h>

namespace adamant
{

enum meta_t {ADM_META_VAR_TYPE=0, ADM_META_OBJ_TYPE=1,
             ADM_META_BIN_TYPE=2, ADM_META_STACK_TYPE=3, ADM_MAX_META_TYPES=4};

enum state_t {ADM_STATE_STATIC=0, ADM_STATE_ALLOC=1, ADM_STATE_FREE=2, ADM_STATE_DONE=3};

enum events_t {ADM_L1_LD=0, ADM_L2_LD=1, ADM_L3_LD=2, ADM_MM_LD=3,
               ADM_L1_ST=4, ADM_L2_ST=5, ADM_L3_ST=6, ADM_MM_ST=7, ADM_EV_COUNT=8, ADM_EVENTS=9};

class adm_meta_t
{
  public:

    uint64_t events[ADM_EVENTS];
    void* meta[ADM_MAX_META_TYPES];

    adm_meta_t() { memset(this, 0, sizeof(adm_meta_t)); }

    bool has_events() const noexcept;

    void process(const adm_event_t& event) noexcept;

    void print() const noexcept;
};

class stack_t
{
  public:

    stack_t() { memset(function, 0, sizeof(function)); memset(ip, 0, sizeof(ip)); }

    char function[ADM_META_STACK_NAMES];
    unw_word_t ip[ADM_META_STACK_DEPTH];
};

class adm_object_t
{
    uint64_t size;
    uint64_t address;

  public:

    adm_meta_t meta;

    adm_object_t(): size(0), address(0) {}

    uint64_t get_address() const noexcept { return address; };

    void set_address(const uint64_t a) noexcept { address=a; };

    uint64_t get_size() const noexcept { return size&0x0FFFFFFFFFFFFFFF; };

    void set_size(const uint64_t s) {size = (size&0xF000000000000000)|s; };

    state_t get_state() const noexcept { return static_cast<state_t>(size>>60); };

    void set_state(const state_t state) {size = (size&0x0FFFFFFFFFFFFFFF)|(static_cast<uint64_t>(state)<<60); };

    void add_state(const state_t state) {size |= (static_cast<uint64_t>(state)<<60); };

    bool has_events() const noexcept { return meta.has_events(); }

    void process(const adm_event_t& event) noexcept { meta.process(event); }

    void print() const noexcept;
};
 
adm_object_t* adm_db_insert(const uint64_t address, const uint64_t size, const state_t state=ADM_STATE_STATIC) noexcept;

adm_object_t* adm_db_find(const uint64_t address) noexcept;

void adm_db_update_size(const uint64_t address, const uint64_t size) noexcept;

void adm_db_update_state(const uint64_t address, const state_t state) noexcept;

void adm_db_init();

void adm_db_fini();

static inline void adm_meta_init() noexcept {};

static inline void adm_meta_fini() noexcept {};

void adm_db_print() noexcept;

}

#endif
