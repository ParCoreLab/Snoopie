#ifndef __ADAMANT_DATABASE
#define __ADAMANT_DATABASE

#include <cstdint>
#include <cstring>
#define UNW_LOCAL_ONLY
#include <libunwind.h>

#include <adm.h>
#include <adm_config.h>
#include <adm_common.h>
#include <iostream>

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

class adm_range_t
{
    uint64_t size;
    uint64_t address;
    uint64_t allocation_pc;
    uint32_t index_in_object;
    int device_id;
    std::string var_name;

  public:

    adm_meta_t meta;

    adm_range_t(): size(0), address(0), allocation_pc(0), index_in_object(999), device_id(0) {}

    uint64_t get_address() const noexcept { return address; };

    void set_address(const uint64_t a) noexcept { address=a; };

    uint64_t get_size() const noexcept { return size&0x0FFFFFFFFFFFFFFF; };

    void set_size(const uint64_t s) {size = (size&0xF000000000000000)|s; };

    uint64_t get_allocation_pc() const noexcept { return allocation_pc; };

    void set_allocation_pc(const uint64_t a) noexcept { allocation_pc=a; };

    uint32_t get_index_in_object() const noexcept { return index_in_object; };

    void set_index_in_object(const uint32_t a) noexcept { index_in_object=a; };

    int get_device_id() const noexcept { return device_id; };

    void set_device_id(const int a) noexcept { device_id=a; };

    std::string get_var_name() const noexcept { return var_name; };

    void set_var_name(std::string varname) noexcept { var_name=varname; };

    state_t get_state() const noexcept { return static_cast<state_t>(size>>60); };

    void set_state(const state_t state) {size = (size&0x0FFFFFFFFFFFFFFF)|(static_cast<uint64_t>(state)<<60); };

    void add_state(const state_t state) {size |= (static_cast<uint64_t>(state)<<60); };

    bool has_events() const noexcept { return meta.has_events(); }

    void process(const adm_event_t& event) noexcept { meta.process(event); }

    void print(std::ofstream& object_outfile) const noexcept;
};
 
adm_range_t* adm_range_insert(const uint64_t address, const uint64_t size, const uint64_t allocation_pc, const int dev_id, std::string var_name, const state_t state=ADM_STATE_STATIC) noexcept;

adm_range_t* adm_range_find(const uint64_t address) noexcept;

class adm_line_location_t
{
    int global_index;
    std::string file_name;
    std::string dir_name;
    std::string sass;
    uint32_t line_num;
    short estimated;

  public:

    adm_line_location_t(): global_index(-1), line_num(0), estimated(0) {}
    int get_global_index() const noexcept { return global_index; };
    void set_global_index(const int idx) noexcept { global_index=idx; };
    void inc_global_index() noexcept { global_index++; };
    std::string get_file_name() const noexcept { return file_name; };
    void set_file_name(std::string filename) {file_name = filename; };
    std::string get_dir_name() const noexcept { return dir_name; };
    void set_dir_name(std::string dirname) {dir_name = dirname; };
    std::string get_sass() const noexcept { return sass; };
    void set_sass(std::string sass_instr) {sass = sass_instr; };
    uint32_t get_line_num() const noexcept { return line_num; };
    void set_line_num(const uint32_t linenum) noexcept { line_num=linenum; };
    short get_estimated_status() const noexcept { return estimated; };
    void set_estimated_status(const short estimated_status) noexcept { estimated=estimated_status; };
    void print(std::ofstream& codeline_outfile) const noexcept;
};

adm_line_location_t* adm_line_location_insert(const int global_index, std::string file_name, std::string dir_name, std::string sass, const uint32_t line_num, short estimated) noexcept;

adm_line_location_t* adm_line_location_find(const int global_index) noexcept;

class adm_object_t
{
    uint64_t allocation_pc;
    std::string var_name;
    std::string file_name;
    std::string func_name;
    uint32_t line_num;
    uint32_t data_type_size;
    uint32_t range_count;

  public:

    adm_meta_t meta;

    adm_object_t(): allocation_pc(0), line_num(0), data_type_size(0), range_count(0) {}

    uint64_t get_allocation_pc() const noexcept { return allocation_pc; };

    void set_allocation_pc(const uint64_t pc) noexcept { allocation_pc=pc; };

    std::string get_var_name() const noexcept { return var_name; };

    void set_var_name(std::string varname) {var_name = varname; };

    std::string get_file_name() const noexcept { return file_name; };

    void set_file_name(std::string filename) noexcept { file_name=filename; };

    std::string get_func_name() const noexcept { return func_name; };

    void set_func_name(std::string funcname) noexcept { func_name=funcname; };

    uint32_t get_line_num() const noexcept { return line_num; };

    void set_line_num(const uint64_t linenum) noexcept { line_num=linenum; };

    uint32_t get_data_type_size() const noexcept { return data_type_size; };

    void set_data_type_size(const int type_size) noexcept { data_type_size=type_size; }; 

    uint32_t get_range_count() const noexcept { return range_count; };

    void inc_range_count() noexcept { range_count++; }; 

    bool has_events() const noexcept { return meta.has_events(); }

    void process(const adm_event_t& event) noexcept { meta.process(event); }

    void print(std::ofstream& object_outfile) const noexcept;
};	

adm_object_t* adm_object_insert(const uint64_t allocation_pc, std::string varname, const uint32_t element_size, std::string filename, std::string funcname, uint32_t linenum, const state_t state=ADM_STATE_STATIC) noexcept;

adm_object_t* adm_object_find(const uint64_t allocation_pc) noexcept;

void adm_db_update_size(const uint64_t address, const uint64_t size) noexcept;

void adm_db_update_state(const uint64_t address, const state_t state) noexcept;

void adm_db_init();

void adm_db_fini();

static inline void adm_meta_init() noexcept {};

static inline void adm_meta_fini() noexcept {};

void adm_ranges_print() noexcept;

void adm_line_table_print() noexcept;

}

#endif
