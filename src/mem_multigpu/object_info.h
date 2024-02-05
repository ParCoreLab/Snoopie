// object_info.h

#ifndef OBJECT_INFO_H
#define OBJECT_INFO_H

#include <string>
#include <cstdint>

std::string get_object_var_name(uint64_t pc);

std::string get_object_file_name(uint64_t pc);

std::string get_object_func_name(uint64_t pc);

uint32_t get_object_line_num(uint64_t pc);

int get_object_device_id(uint64_t pc);

void set_object_device_id(uint64_t pc, int dev_id);

uint32_t get_object_data_type_size(uint64_t pc);

void set_object_data_type_size(uint64_t pc, const uint32_t type_size);

bool object_exists(uint64_t pc);

#endif // OBJECT_INFO_H