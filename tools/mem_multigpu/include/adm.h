#ifndef __ADAMANT
#define __ADAMANT

#include <cstdint>
#include <linux/perf_event.h>

typedef struct {
  struct perf_event_header hdr;
  uint64_t ip;
  uint32_t pid, tid;
  uint64_t time;
  uint64_t address;
  uint64_t stream_id;
  uint32_t cpu, res;
  uint64_t weight;
  uint64_t data_src;
} adm_event_t;

extern "C" void adm_event(const adm_event_t *event);

extern "C" void adm_events_v(const adm_event_t *event,
                             const uint32_t events_num);

#endif

/*
source encoding:
mem_op
    Type of opcode, a bitwise combination of:
    PERF_MEM_OP_NA          Not available
    PERF_MEM_OP_LOAD        Load instruction
    PERF_MEM_OP_STORE       Store instruction
    PERF_MEM_OP_PFETCH      Prefetch
    PERF_MEM_OP_EXEC        Executable code
mem_lvl
    Memory hierarchy level hit or  miss,  a  bitwise  combination  of  the
following, shifted left by PERF_MEM_LVL_SHIFT: PERF_MEM_LVL_NA         Not
available PERF_MEM_LVL_HIT        Hit PERF_MEM_LVL_MISS       Miss
    PERF_MEM_LVL_L1         Level 1 cache
    PERF_MEM_LVL_LFB        Line fill buffer
    PERF_MEM_LVL_L2         Level 2 cache
    PERF_MEM_LVL_L3         Level 3 cache
    PERF_MEM_LVL_LOC_RAM    Local DRAM
    PERF_MEM_LVL_REM_RAM1   Remote DRAM 1 hop
    PERF_MEM_LVL_REM_RAM2   Remote DRAM 2 hops
    PERF_MEM_LVL_REM_CCE1   Remote cache 1 hop
    PERF_MEM_LVL_REM_CCE2   Remote cache 2 hops
    PERF_MEM_LVL_IO         I/O memory
    PERF_MEM_LVL_UNC        Uncached memory
mem_snoop
    Snoop   mode,   a   bitwise   combination   of  the  following,  shifted
left  by PERF_MEM_SNOOP_SHIFT: PERF_MEM_SNOOP_NA       Not available
    PERF_MEM_SNOOP_NONE     No snoop
    PERF_MEM_SNOOP_HIT      Snoop hit
    PERF_MEM_SNOOP_MISS     Snoop miss
    PERF_MEM_SNOOP_HITM     Snoop hit modified
mem_lock
    Lock instruction,  a  bitwise  combination  of  the  following,  shifted
left  by PERF_MEM_LOCK_SHIFT: PERF_MEM_LOCK_NA        Not available
    PERF_MEM_LOCK_LOCKED    Locked transaction
mem_dtlb
    TLB  access  hit  or miss, a bitwise combination of the following, shifted
left by PERF_MEM_TLB_SHIFT: PERF_MEM_TLB_NA         Not available
    PERF_MEM_TLB_HIT        Hit
    PERF_MEM_TLB_MISS       Miss
    PERF_MEM_TLB_L1         Level 1 TLB
    PERF_MEM_TLB_L2         Level 2 TLB
    PERF_MEM_TLB_WK         Hardware walker
    PERF_MEM_TLB_OS         OS fault handler
*/
