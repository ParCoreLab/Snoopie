// line_info.h

#ifndef LINE_INFO_H
#define LINE_INFO_H

#include <string>
#include <cstdint>

void initialize_line_table(int size);

bool line_exists(int index);
short get_line_estimated_status(int index);

std::string get_line_dir_name(int index);
std::string get_line_file_name(int index);
std::string get_line_sass(int index);

uint32_t get_line_line_num(int index);


#endif // LINE_INFO_H