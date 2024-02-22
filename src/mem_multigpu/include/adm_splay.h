#ifndef __ADAMANT_SPLAY
#define __ADAMANT_SPLAY

#include <adm_database.h>
#include <bits/stdc++.h>
using namespace std;

namespace adamant {

class adm_splay_tree_t {
private:
  void zigzig(adm_splay_tree_t *granpa) noexcept {
    adm_splay_tree_t *tmp = parent->right;
    granpa->left = tmp;
    if (tmp)
      tmp->parent = granpa;

    parent->left = right;
    if (right)
      right->parent = parent;

    tmp = granpa->parent;

    parent->right = granpa;
    granpa->parent = parent;
    right = parent;
    right->parent = this;

    parent = tmp;
    if (tmp) {
      if (tmp->right == granpa)
        tmp->right = this;
      else
        tmp->left = this;
    }
  }

  void zagzag(adm_splay_tree_t *granpa) noexcept {
    adm_splay_tree_t *tmp = parent->left;
    granpa->right = tmp;
    if (tmp)
      tmp->parent = granpa;

    parent->right = left;
    if (left)
      left->parent = parent;

    tmp = granpa->parent;

    parent->left = granpa;
    granpa->parent = parent;
    left = parent;
    left->parent = this;

    parent = tmp;
    if (tmp) {
      if (tmp->right == granpa)
        tmp->right = this;
      else
        tmp->left = this;
    }
  }

  void zagzig(adm_splay_tree_t *granpa) noexcept {
    parent->right = left;
    if (left)
      left->parent = parent;

    granpa->left = right;
    if (right)
      right->parent = granpa;

    adm_splay_tree_t *root = granpa->parent;

    left = parent;
    left->parent = this;
    right = granpa;
    right->parent = this;

    parent = root;
    if (root) {
      if (root->right == granpa)
        root->right = this;
      else
        root->left = this;
    }
  }

  void zigzag(adm_splay_tree_t *granpa) noexcept {
    granpa->right = left;
    if (left)
      left->parent = granpa;

    parent->left = right;
    if (right)
      right->parent = parent;

    adm_splay_tree_t *root = granpa->parent;

    left = granpa;
    granpa->parent = this;
    right = parent;
    parent->parent = this;

    parent = root;
    if (root) {
      if (root->right == granpa)
        root->right = this;
      else
        root->left = this;
    }
  }

  void zig() noexcept {
    parent->left = right;
    if (right)
      right->parent = parent;
    right = parent;
    right->parent = this;
    parent = nullptr;
  }

  void zag() noexcept {
    parent->right = left;
    if (left)
      left->parent = parent;
    left = parent;
    left->parent = this;
    parent = nullptr;
  }

  adm_splay_tree_t *parent;
  adm_splay_tree_t *left;
  adm_splay_tree_t *right;

public:
  uint64_t start;
  uint64_t end;

  adm_range_t *range;

  adm_object_t *object;

  adm_splay_tree_t()
      : parent(nullptr), left(nullptr), right(nullptr), start(0), end(0),
        range(nullptr) {}

  adm_splay_tree_t *min() noexcept {
    adm_splay_tree_t *m = this;
    while (m->left)
      m = m->left;
    return m;
  }

  adm_splay_tree_t *find(const uint64_t start) noexcept {
    adm_splay_tree_t *f = this;
    while (f) {
      if (f->start <= start && start < f->end)
        break;
      if (start < f->start)
        f = f->left;
      else
        f = f->right;
    }
    return f;
  }

  void find_with_parent(const uint64_t start, adm_splay_tree_t *&p,
                        adm_splay_tree_t *&f) noexcept {
    f = this;
    p = f->parent;
    while (f) {
      if (f->start == start && start < f->end)
        break;
      p = f;
      if (start < f->start)
        f = f->left;
      else
        f = f->right;
    }
  }

  void insert(adm_splay_tree_t *a) noexcept {
    if (start > a->start)
      left = a;
    else
      right = a;
    a->parent = this;
  }

  adm_splay_tree_t *splay() noexcept {
    while (parent) {
      adm_splay_tree_t *granpa = parent->parent;

      if (granpa) {
        if (parent->right == this) {
          if (granpa->left == parent)
            zagzig(granpa);
          else
            zagzag(granpa);
        } else {
          if (granpa->left == parent)
            zigzig(granpa);
          else
            zigzag(granpa);
        }
      } else {
        if (parent->left == this)
          zig();
        else
          zag();
      }
    }
    return this;
  }
};

class object_hash_table_t {
  int hash_table_size; // No. of buckets

  // Pointer to an array containing buckets
  list<adm_object_t *> *table;

public:
  // hash function to map values to key
  int hashFunction(uint64_t x) { return (x % 54121 % hash_table_size); }

  object_hash_table_t(int bucket) {
    hash_table_size = bucket;
    table = new list<adm_object_t *>[hash_table_size];
  } // Constructor

  // inserts a key into hash table
  void insert(adm_object_t *obj) noexcept {
    int index = hashFunction(obj->get_allocation_pc());
    table[index].push_back(obj);
  }

  // deletes a key from hash table
  adm_object_t *find(uint64_t pc) noexcept {
    // get the hash index of key
    int index = hashFunction(pc);

    if (table[index].empty())
      return nullptr;
    // find the key in (index)th list
    list<adm_object_t *>::iterator i;
    for (i = table[index].begin(); i != table[index].end(); i++) {
      if ((*i)->get_allocation_pc() == pc)
        break;
    }

    // if key is found in hash table, remove it
    if (i != table[index].end())
      return *i;
    return nullptr;
  }

#if 0
    void displayHash() noexcept
    {
    }
#endif
};

class line_hash_table_t {
  int hash_table_size; // No. of buckets

  // Pointer to an array containing buckets
  list<adm_line_location_t *> *table;
  uint32_t entry_size;

public:
  // hash function to map values to key
  int hashFunction(uint64_t x) { return (x % 54121 % hash_table_size); }

  int get_size() { return entry_size; }

  line_hash_table_t(int bucket) {
    entry_size = 0;
    hash_table_size = bucket;
    table = new list<adm_line_location_t *>[hash_table_size];
  } // Constructor

  // inserts a key into hash table
  void insert(adm_line_location_t *line) noexcept {
    int index = hashFunction(line->get_global_index());
    table[index].push_back(line);
    entry_size++;
  }

  adm_line_location_t *find(int global_index) noexcept {
    // get the hash index of key
    int index = hashFunction(global_index);

    if (table[index].empty())
      return nullptr;
    // find the key in (index)th list
    list<adm_line_location_t *>::iterator i;
    for (i = table[index].begin(); i != table[index].end(); i++) {
      if ((*i)->get_global_index() == global_index)
        break;
    }

    // if key is found in hash table, remove it
    if (i != table[index].end())
      return *i;
    return nullptr;
  }
};

class allocation_line_hash_table_t {
  int hash_table_size; // No. of buckets

  // Pointer to an array containing buckets
  list<allocation_line_t *> *table;
  uint32_t entry_size;

public:
  // hash function to map values to key
  int hashFunction(uint64_t x) { return (x % 54121 % hash_table_size); }

  int get_size() { return entry_size; }

  allocation_line_hash_table_t(int bucket) {
    entry_size = 0;
    hash_table_size = bucket;
    table = new list<allocation_line_t *>[hash_table_size];
  } // Constructor

  // inserts a key into hash table
  void insert(allocation_line_t *line) noexcept {
    int index = hashFunction(line->get_pc());
    table[index].push_back(line);
    entry_size++;
  }

  allocation_line_t *find(uint64_t pc1) noexcept {
    // get the hash index of key
    int index = hashFunction(pc1);

    if (table[index].empty())
      return nullptr;
    // find the key in (index)th list
    list<allocation_line_t *>::iterator i;
    for (i = table[index].begin(); i != table[index].end(); i++) {
      if ((*i)->get_pc() == pc1)
        break;
    }

    // if key is found in hash table, remove it
    if (i != table[index].end())
      return *i;
    return nullptr;
  }

  void print(std::ofstream &object_outfile) {
    // find the key in (index)th list
    list<allocation_line_t *>::iterator i;
    for (int j = 0; j < hash_table_size; j++) {
      if (table[j].empty())
        continue;
      for (i = table[j].begin(); i != table[j].end(); i++) {
        (*i)->print(object_outfile);
      }
    }
  }
};


class execution_site_hash_table_t {
  int hash_table_size; // No. of buckets

  // Pointer to an array containing buckets
  list<execution_site_t *> *table;
  uint32_t entry_size;

public:
  // hash function to map values to key
  uint64_t hashFunction(uint64_t x) { return (x % 54121 % hash_table_size); }

  int get_size() { return entry_size; }

  execution_site_hash_table_t(int bucket) {
    entry_size = 0;
    hash_table_size = bucket;
    table = new list<execution_site_t *>[hash_table_size];
  } // Constructor

  // inserts a key into hash table
  void insert(execution_site_t *line) noexcept {
    int index = hashFunction(line->get_exec_site_id());
    table[index].push_back(line);
    entry_size++;
  }

  execution_site_t *find(uint64_t exec_site_id) noexcept {
    // get the hash index of key
    uint64_t index = hashFunction(exec_site_id);

    if (table[index].empty())
      return nullptr;
    // find the key in (index)th list
    list<execution_site_t *>::iterator i;
    for (i = table[index].begin(); i != table[index].end(); i++) {
      if ((*i)->get_exec_site_id() == exec_site_id)
        break;
    }

    // if key is found in hash table, remove it
    if (i != table[index].end())
      return *i;
    return nullptr;
  }
  
  void print(std::ofstream &object_outfile) { 
    // find the key in (index)th list
    list<execution_site_t *>::iterator i;
    for (int j = 0; j < hash_table_size; j++) {
      if (table[j].empty())
        continue;
      for (i = table[j].begin(); i != table[j].end(); i++) {
        (*i)->print(object_outfile);
      }
    }
  }
};

} // namespace adamant

#endif
