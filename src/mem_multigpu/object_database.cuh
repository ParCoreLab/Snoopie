#ifndef OBJECT_DATABASE_CUH
#define OBJECT_DATABASE_CUH

#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <iostream>

#include <adm_common.h>
#include <adm_config.h>
#include <adm_database.h>
#include <adm_memory.h>
#include <adm_splay.h>
#include <unistd.h>

using namespace adamant;

namespace adamant {
static adm_splay_tree_t *range_tree;
static object_hash_table_t *object_table;
static line_hash_table_t *line_table;
static pool_t<adm_splay_tree_t, ADM_DB_OBJ_BLOCKSIZE> *range_nodes;
static pool_t<adm_range_t, ADM_DB_OBJ_BLOCKSIZE> *ranges;
static pool_t<adm_object_t, ADM_DB_OBJ_BLOCKSIZE> *objects;
}

void initialize_line_table(int size);

bool line_exists(int index);

std::string get_line_file_name(int global_index);

std::string get_line_dir_name(int global_index);

std::string get_line_sass(int global_index);

uint32_t get_line_line_num(int global_index);

short get_line_estimated_status(int global_index);

void initialize_object_table(int size);

bool object_exists(uint64_t pc);

std::string get_object_var_name(uint64_t pc);

std::string get_object_file_name(uint64_t pc);

std::string get_object_func_name(uint64_t pc);

uint32_t get_object_line_num(uint64_t pc);

uint32_t get_object_data_type_size(uint64_t pc);

void set_object_data_type_size(uint64_t pc, const uint32_t type_size);

static inline adm_splay_tree_t *
adm_range_find_node(const uint64_t address) noexcept;

ADM_VISIBILITY
adm_range_t *adamant::adm_range_find(const uint64_t address) noexcept;

ADM_VISIBILITY
adm_object_t *adamant::adm_object_insert(const uint64_t allocation_pc,
                                         std::string varname,
                                         const uint32_t element_size,
                                         std::string filename,
                                         std::string funcname, uint32_t linenum,
                                         const state_t state) noexcept;

ADM_VISIBILITY
adm_line_location_t *adamant::adm_line_location_insert(
    const int global_index, std::string file_name, std::string dir_name,
    std::string sass, const uint32_t line_num, short estimated) noexcept;

ADM_VISIBILITY
adm_range_t *adamant::adm_range_insert(const uint64_t address,
                                       const uint64_t size,
                                       const uint64_t allocation_pc,
                                       const int dev_id, std::string var_name,
                                       const state_t state) noexcept;

ADM_VISIBILITY
void adamant::adm_db_update_size(const uint64_t address,
                                 const uint64_t size) noexcept;

ADM_VISIBILITY
void adamant::adm_db_update_state(const uint64_t address,
                                  const state_t state) noexcept;

ADM_VISIBILITY
void adamant::adm_ranges_print() noexcept;

ADM_VISIBILITY
void adamant::adm_line_table_print() noexcept;

// ADM_VISIBILITY
void adamant::adm_db_init();

// ADM_VISIBILITY
void adamant::adm_db_fini();

#endif // OBJECT_DATABASE_CUH
