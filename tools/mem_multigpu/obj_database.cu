#include <cstdint>
#include <cstring>
#include <iostream>

#include <adm_config.h>
#include <adm_common.h>
#include <adm_splay.h>
#include <adm_memory.h>
#include <adm_database.h>

using namespace adamant;

static adm_splay_tree_t* tree = nullptr;
static pool_t<adm_splay_tree_t, ADM_DB_OBJ_BLOCKSIZE>* nodes = nullptr;
static pool_t<adm_object_t, ADM_DB_OBJ_BLOCKSIZE>* objects = nullptr;

static inline
adm_splay_tree_t* adm_db_find_node(const uint64_t address) noexcept
{
  if(tree)
    return tree->find(address);
  return nullptr;
}

ADM_VISIBILITY
adm_object_t* adamant::adm_db_find(const uint64_t address) noexcept
{
  adm_splay_tree_t* node = adm_db_find_node(address);
  if(node) return node->object;
  return nullptr;
}

ADM_VISIBILITY
adm_object_t* adamant::adm_db_insert(const uint64_t address, const uint64_t size, const uint64_t allocation_pc, const state_t state) noexcept
{
  adm_splay_tree_t* obj = nullptr;
  adm_splay_tree_t* pos = nullptr;

  fprintf(stderr, "inside adm_db_insert before tree->find_with_parent\n");
  if(tree) tree->find_with_parent(address, pos, obj);
  fprintf(stderr, "inside adm_db_insert after tree->find_with_parent\n");
  if(obj==nullptr) {
    fprintf(stderr, "inside adm_db_insert before nodes->malloc\n");
    obj = nodes->malloc();
    if(obj==nullptr) return nullptr;

    fprintf(stderr, "inside adm_db_insert before objects->malloc\n");
    obj->object = objects->malloc();
    if(obj->object==nullptr) return nullptr;

    obj->start = address;
    obj->object->set_address(address); 
    obj->end = obj->start+size;
    obj->object->set_size(size);
    obj->object->set_allocation_pc(allocation_pc);
    obj->object->set_state(state);
    if(pos!=nullptr)
      pos->insert(obj);
    tree = obj->splay();
    fprintf(stderr, "object is inserted to the splay tree\n");
  }
  else {
    if(!(obj->object->get_state()&ADM_STATE_FREE)) {
      if(obj->start==address)
        adm_warning("db_insert: address already in tree and not free - ", address);
      else if(obj->start<address && address<obj->end)
        adm_warning("db_insert: address in range of another address in tree - ", obj->start, "..", obj->end, " (", address, ")");
      if(obj->end<address+size) {
        obj->end = address+size;
        obj->object->set_size(size);
      }
      tree = obj->splay();
    }
    else {
      obj->object = objects->malloc();
      if(obj->object==nullptr) return nullptr;

      obj->start = address;
      obj->object->set_address(address); 
      obj->end = obj->start+size;
      obj->object->set_size(size);
      obj->object->set_allocation_pc(allocation_pc);
      obj->object->set_state(state);
      tree = obj->splay();
    }
  }

  return obj->object;
}

ADM_VISIBILITY
void adamant::adm_db_update_size(const uint64_t address, const uint64_t size) noexcept
{
  adm_splay_tree_t* obj = adm_db_find_node(address);
  if(obj) {
    obj->object->set_size(size);
    if(obj->start!=address) {
      adm_warning("db_update_size: address in range of another address in tree - ", obj->start, "..", obj->end, "(", address, ")");
      obj->start = address;
      obj->object->set_address(address);
    }
    obj->end = address+size;
    obj->object->set_size(size);
    tree = obj->splay();
  }
  else adm_warning("db_update_size: address not in tree - ", address);
}

ADM_VISIBILITY
void adamant::adm_db_update_state(const uint64_t address, const state_t state) noexcept
{
  adm_splay_tree_t* obj = adm_db_find_node(address);
  if(obj) {
    obj->object->add_state(state);
    if(obj->start!=address)
      adm_warning("db_update_state: address in range of another address in tree - ", obj->start, "..", obj->end, "(", address, ")");
  }
  else adm_warning("db_update_state: address not in tree - ", address);
}

ADM_VISIBILITY
void adamant::adm_db_print() noexcept
{
  //bool all = adm_conf_string("+all", "1");
  pool_t<adm_splay_tree_t, ADM_DB_OBJ_BLOCKSIZE>::iterator n(*nodes);
  for(adm_splay_tree_t* obj = n.next(); obj!=nullptr; obj = n.next())
    //if(obj->object->has_events())
    obj->object->print();
}

//#if 0
//ADM_VISIBILITY
void adamant::adm_db_init()
{
  nodes = new pool_t<adm_splay_tree_t, ADM_DB_OBJ_BLOCKSIZE>;
  objects = new pool_t<adm_object_t, ADM_DB_OBJ_BLOCKSIZE>;
}

//ADM_VISIBILITY
void adamant::adm_db_fini()
{
  delete nodes;
  delete objects;
}
//#endif

ADM_VISIBILITY
void adm_object_t::print() const noexcept
{
  fprintf(stderr, "in adm_object_t::print\n");
  uint64_t a = get_address();
  fprintf(stderr, "offset: %lx ", a);
  uint64_t z = get_size();
  fprintf(stderr, "size: %ld ", z);
  uint64_t p = get_allocation_pc();
  fprintf(stderr, "allocation_pc: %lx\n", p); 
}
