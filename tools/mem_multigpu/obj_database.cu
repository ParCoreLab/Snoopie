#include <cstdint>
#include <cstring>
#include <iostream>

#include <adm_config.h>
#include <adm_common.h>
#include <adm_splay.h>
#include <adm_memory.h>
#include <adm_database.h>
#include <unistd.h>

#define HEX(x)                                                          \
  "0x" << std::setfill('0') << std::setw(16) << std::hex << (uint64_t)x \
  << std::dec

using namespace adamant;

static adm_splay_tree_t* range_tree = nullptr;
static object_hash_table_t* object_table;
static line_hash_table_t* line_table;
static pool_t<adm_splay_tree_t, ADM_DB_OBJ_BLOCKSIZE>* range_nodes = nullptr;
static pool_t<adm_range_t, ADM_DB_OBJ_BLOCKSIZE>* ranges = nullptr;
static pool_t<adm_object_t, ADM_DB_OBJ_BLOCKSIZE>* objects = nullptr;


void initialize_line_table(int size) {
        line_table = new line_hash_table_t(size);
}

bool line_exists(int index) {
        adm_line_location_t* line = line_table->find(index);
        if(line) {
                return true;
        }
        return false;
}

std::string get_line_file_name(int global_index) {
        adm_line_location_t* line = line_table->find(global_index);
        if(line) {
                return line->get_file_name();
        }
        return "";
}

std::string get_line_dir_name(int global_index) {
        adm_line_location_t* line = line_table->find(global_index);
        if(line) {
                return line->get_dir_name();
        }
        return "";
}

std::string get_line_sass(int global_index) {
        adm_line_location_t* line = line_table->find(global_index);
        if(line) {
                return line->get_sass();
        }
        return "";
}

uint32_t get_line_line_num(int global_index) {
        adm_line_location_t* line = line_table->find(global_index);
        if(line) {
                return line->get_line_num();
        }
        return 0;
}

short get_line_estimated_status(int global_index) {
        adm_line_location_t* line = line_table->find(global_index);
        if(line) {
                return line->get_estimated_status();
        }
        return 0;
}

void initialize_object_table(int size) {
	object_table = new object_hash_table_t(size);
}

bool object_exists(uint64_t pc) {
	adm_object_t* obj = object_table->find(pc);
        if(obj) {
                return true;
        }
	return false;	
}

std::string get_object_var_name(uint64_t pc) {
	adm_object_t* obj = object_table->find(pc);
	if(obj) {
        	return obj->get_var_name();
	}
	return "";
}

std::string get_object_file_name(uint64_t pc) {
        adm_object_t* obj = object_table->find(pc);
        if(obj) {
                return obj->get_file_name();
        }
        return "";
}

std::string get_object_func_name(uint64_t pc) {
        adm_object_t* obj = object_table->find(pc);
        if(obj) {
                return obj->get_func_name();
        }
        return "";
}

uint32_t get_object_line_num(uint64_t pc) {
        adm_object_t* obj = object_table->find(pc);
        if(obj) {
                return obj->get_line_num();
        }
        return 0;
}

uint32_t get_object_data_type_size(uint64_t pc) {
        adm_object_t* obj = object_table->find(pc);
        if(obj) {
                return obj->get_data_type_size();
        }
        return 0;
}

void set_object_data_type_size(uint64_t pc, const uint32_t type_size) {
        adm_object_t* obj = object_table->find(pc);
        if(obj) {
                return obj->set_data_type_size(type_size);
        }
}

static inline
adm_splay_tree_t* adm_range_find_node(const uint64_t address) noexcept
{
  if(range_tree)
    return range_tree->find(address);
  return nullptr;
}

ADM_VISIBILITY
adm_range_t* adamant::adm_range_find(const uint64_t address) noexcept
{
  adm_splay_tree_t* node = adm_range_find_node(address);
  if(node) return node->range;
  return nullptr;
}

ADM_VISIBILITY 
adm_object_t* adamant::adm_object_insert(const uint64_t allocation_pc, std::string varname, const uint32_t element_size, std::string filename, std::string funcname, uint32_t linenum, const state_t state) noexcept
{
	adm_object_t* obj = object_table->find(allocation_pc);
	if(obj == nullptr) {
		obj = new adm_object_t();
		obj->set_allocation_pc(allocation_pc);
		obj->set_var_name(varname);
		obj->set_data_type_size(element_size);
		obj->set_file_name(filename);
		obj->set_func_name(funcname);
		obj->set_line_num(linenum);
		object_table->insert(obj);
	}
	if(obj->get_allocation_pc() == allocation_pc)
                return obj;
	return nullptr;	
}

ADM_VISIBILITY
adm_line_location_t* adamant::adm_line_location_insert(const int global_index, std::string file_name, std::string dir_name, std::string sass, const uint32_t line_num, short estimated) noexcept
{
        adm_line_location_t* line = line_table->find(global_index);
        if(line == nullptr) {
                line = new adm_line_location_t();
                line->set_global_index(global_index);
                line->set_file_name(file_name);
                line->set_dir_name(dir_name);
		line->set_sass(sass);
                line->set_line_num(line_num);
                line->set_estimated_status(estimated);
                line_table->insert(line);
        }
        if(line->get_global_index() == global_index)
                return line;
        return nullptr;
}

ADM_VISIBILITY
adm_range_t* adamant::adm_range_insert(const uint64_t address, const uint64_t size, const uint64_t allocation_pc, const int dev_id, std::string var_name, const state_t state) noexcept
{
  adm_splay_tree_t* obj = nullptr;
  adm_splay_tree_t* pos = nullptr;

  //fprintf(stderr, "inside adm_range_insert before range_tree->find_with_parent\n");
  if(range_tree) range_tree->find_with_parent(address, pos, obj);
  //fprintf(stderr, "inside adm_range_insert after range_tree->find_with_parent\n");
  if(obj==nullptr) {
    //fprintf(stderr, "inside adm_range_insert before range_nodes->malloc\n");
    obj = range_nodes->malloc();
    if(obj==nullptr) return nullptr;

    //fprintf(stderr, "inside adm_range_insert before ranges->malloc\n");
    obj->range = ranges->malloc();
    if(obj->range==nullptr) return nullptr;

    obj->start = address;
    obj->range->set_address(address); 
    obj->end = obj->start+size;
    obj->range->set_size(size);
    obj->range->set_allocation_pc(allocation_pc);
    obj->range->set_device_id(dev_id);
    obj->range->set_var_name(var_name);
    obj->range->set_state(state);

// before
#if 0
    adm_object_t* data_obj = object_table->find(allocation_pc);
    if(data_obj) {
	obj->range->set_index_in_object(data_obj->get_range_count());
	data_obj->inc_range_count();
    }
#endif
// after    
    if(pos!=nullptr)
      pos->insert(obj);
    range_tree = obj->splay();
    //fprintf(stderr, "range is inserted to the splay tree\n");
  }
  else {
    if(!(obj->range->get_state()&ADM_STATE_FREE)) {
      if(obj->start==address)
        adm_warning("db_insert: address already in range_tree and not free - ", address);
      else if(obj->start<address && address<obj->end)
        adm_warning("db_insert: address in range of another address in range_tree - ", obj->start, "..", obj->end, " (", address, ")");
      if(obj->end<address+size) {
        obj->end = address+size;
        obj->range->set_size(size);
      }
      range_tree = obj->splay();
    }
    else {
      obj->range = ranges->malloc();
      if(obj->range==nullptr) return nullptr;

      obj->start = address;
      obj->range->set_address(address); 
      obj->end = obj->start+size;
      obj->range->set_size(size);
      obj->range->set_allocation_pc(allocation_pc);
      obj->range->set_device_id(dev_id);
      obj->range->set_var_name(var_name);
      obj->range->set_state(state);
      range_tree = obj->splay();
    }
  }

  return obj->range;
}

ADM_VISIBILITY
void adamant::adm_db_update_size(const uint64_t address, const uint64_t size) noexcept
{
  adm_splay_tree_t* obj = adm_range_find_node(address);
  if(obj) {
    obj->range->set_size(size);
    if(obj->start!=address) {
      adm_warning("db_update_size: address in range of another address in range_tree - ", obj->start, "..", obj->end, "(", address, ")");
      obj->start = address;
      obj->range->set_address(address);
    }
    obj->end = address+size;
    obj->range->set_size(size);
    range_tree = obj->splay();
  }
  else adm_warning("db_update_size: address not in range_tree - ", address);
}

ADM_VISIBILITY
void adamant::adm_db_update_state(const uint64_t address, const state_t state) noexcept
{
  adm_splay_tree_t* obj = adm_range_find_node(address);
  if(obj) {
    obj->range->add_state(state);
    if(obj->start!=address)
      adm_warning("db_update_state: address in range of another address in range_tree - ", obj->start, "..", obj->end, "(", address, ")");
  }
  else adm_warning("db_update_state: address not in range_tree - ", address);
}

ADM_VISIBILITY
void adamant::adm_ranges_print() noexcept
{

  //bool all = adm_conf_string("+all", "1");
  // std::cout << "List of captured address ranges along with their variable names and code locations:\n";

  int first_iter = 1;  
  ofstream object_outfile;

  string object_str("data_object_log_");
  string txt_str(".txt");
  string object_log_str = object_str + to_string(getpid()) + txt_str;
  object_outfile.open(object_log_str);
  pool_t<adm_splay_tree_t, ADM_DB_OBJ_BLOCKSIZE>::iterator n(*range_nodes);
  for(adm_splay_tree_t* obj = n.next(); obj!=nullptr; obj = n.next()) {
    if (first_iter) {
      object_outfile << "offset, size, device_id, var_name, filename, alloc_line_num" << std::endl;
      first_iter = 0;
    }

    obj->range->print(object_outfile);
  }
  object_outfile.close();
}

ADM_VISIBILITY
void adamant::adm_line_table_print() noexcept
{
  //bool all = adm_conf_string("+all", "1");
  ofstream codeline_outfile;

  string codeline_str("codeline_log_");
  string txt_str(".txt");
  string codeline_log_str = codeline_str + to_string(getpid()) + txt_str;
  codeline_outfile.open(codeline_log_str);
  codeline_outfile << "code_line_index, dir_path, file, code_linenum, code_line_estimated_status\n";
  int size = line_table->get_size();
  for(int i = 0; i < size; i++) {
  	adm_line_location_t* line = line_table->find(i);
	if(line == nullptr)
		break;
	line->print(codeline_outfile);
  }
  codeline_outfile.close();
}

//#if 0
//ADM_VISIBILITY
void adamant::adm_db_init()
{
  range_nodes = new pool_t<adm_splay_tree_t, ADM_DB_OBJ_BLOCKSIZE>;
  ranges = new pool_t<adm_range_t, ADM_DB_OBJ_BLOCKSIZE>;
  objects = new pool_t<adm_object_t, ADM_DB_OBJ_BLOCKSIZE>;
}

//ADM_VISIBILITY
void adamant::adm_db_fini()
{
  delete range_nodes;
  delete ranges;
  delete objects;
}
//#endif

ADM_VISIBILITY
void adm_range_t::print(std::ofstream& object_outfile) const noexcept
{
  //std::cout << "in adm_range_t::print\n";
  uint64_t a = get_address();
  // std::cout << "offset: " << a << ", ";
  uint64_t z = get_size();
  // std::cout << "size: " << z << ", ";
  int dev_id = get_device_id();
  int obj_id = get_object_id();
  // std::cout << "device_id: " << dev_id << ", ";
  uint64_t p = get_allocation_pc();

  object_outfile << HEX(a) << ",";
  object_outfile << z << ",";
  object_outfile << obj_id << ",";
  object_outfile << dev_id << ",";

  adm_object_t* obj = object_table->find(p);
  obj->print(object_outfile);
#if 0
  uint64_t p = get_allocation_pc();
  std::cout << "allocation_pc: " << p << std::endl; 
#endif
  //std::string varname = get_var_name();
  //std::cout << varname << std::endl;
}

ADM_VISIBILITY
void adm_object_t::print(std::ofstream& object_outfile) const noexcept
{
  //std::cout << "in adm_object_t::print\n";

  // uint64_t p = get_allocation_pc();
  // std::cout << "allocation_pc: " << p << ", "; 

  int obj_id = get_object_id();
  std::string varname = get_var_name();

  allocation_site_t* temp = get_allocation_site();
  while(temp) {
	  cout << "," << temp->get_pc();
	  temp = temp->get_parent();
  }
  cout << endl;
}

ADM_VISIBILITY
void adm_line_location_t::print(std::ofstream& codeline_outfile) const noexcept
{
	codeline_outfile << global_index << "," << dir_name << "," << file_name
		<< "," << line_num << "," << estimated << "\n";	
}
