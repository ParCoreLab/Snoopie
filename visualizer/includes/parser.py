import streamlit as st
from .tables import *
from copy import deepcopy
from .streamlit_globals import setup_globals
import os
import pickle

tables = {}

current_table = None

_counter = 0
_table_keys = []
_devices = set()


def change_table(line: str):
    global current_table, _table_keys
    for key in tables.keys():
        if line.startswith(tables[key]["starts_with"]):
            current_table = key
            _table_keys = line.strip().split(",")
            return


def parse_line(line: str, gbs: tuple):
    global _counter

    _counter += 1
    if _counter % 100000 == 0:
        print("reading data, line:", counter)

    if current_table == None:
        change_table(line)
    else:
        tables[current_table]["parser"](line, gbs)


def isInt_try(v):
    try:
        i = int(v)
    except:
        return False
    return True


def parse_codeline_info(line: str, gbs: tuple):
    _, ops = gbs
    change_table(line)
    if current_table != "codeline_info":
        CodeLineInfoRow.inferred_home_dir = CodeLineInfoRow.infer_home_dir(CodeLineInfoRow.table())
        return
    data = {}
    vals = line.strip().split(",")  # change this later
    for index in range(len(_table_keys)):
        if isInt_try(vals[index]):
            data[_table_keys[index].strip()] = int(vals[index])
        else:
            data[_table_keys[index].strip()] = vals[index]

    CodeLineInfoRow(**data)


def parse_obj_info(line: str, gbs: tuple):
    _, ops = gbs
    change_table(line)
    if current_table != "obj_info":
        return
    data = {}
    vals = line.strip().split(",")  # change this later
    for index in range(len(_table_keys)):
        if isInt_try(vals[index]):
            data[_table_keys[index].strip()] = int(vals[index])
        else:
            data[_table_keys[index].strip()] = vals[index]

    ObjNameRow(**data)


def parse_offset_info(line: str, gbs: tuple):
    _, ops = gbs
    change_table(line)
    if current_table != "offset_info":
        return
    data = {}
    vals = line.strip().split(",")  # change this later
    for index in range(len(_table_keys)):
        if isInt_try(vals[index]):
            data[_table_keys[index].strip()] = int(vals[index])
        else:
            data[_table_keys[index].strip()] = vals[index]

    _devices.add(data["dev_id"])
    ObjIdRow(**data)


def parse_func_info(line: str, gbs: tuple):
    _, ops = gbs
    change_table(line)
    if current_table != "func_info":
        return
    data = {}

    # since function name contains "," we need to be careful
    # assume filepath does not contain ","

    sep = [-1, -1, -1]
    sep[0] = line.find(",", 0)
    while True:
        i1 = line.find(",", max(sep[1], sep[0]) + 1)
        if i1 == -1:
            break
        i2 = line.find(",", i1 + 1)
        if i2 == -1:
            # sep[1] = sep[2]
            # sep[2] = i1
            break
        sep[1] = i1
        sep[2] = i2
    split_data = (
        line[: sep[0]].strip(),
        line[sep[0] + 1: sep[1]].strip(),
        line[sep[1] + 1: sep[2]].strip(),
        line[sep[2] + 1:].strip(),
    )
    FunctionInfoRow(
        int(split_data[0]), split_data[1], split_data[2], int(split_data[3])
    )


def parse_op_info(line: str, gbs: tuple):
    _, ops = gbs
    change_table(line)
    if current_table != "op_info":
        return

    data = {}
    vals = line.strip().split(",")  # change this later
    for index in range(len(_table_keys)):
        if isInt_try(vals[index]):
            data[_table_keys[index].strip()] = int(vals[index])
        else:
            data[_table_keys[index].strip()] = vals[index]

    address = data["addr"]
    obj_offset = data["obj_offset"]
    operation = data["op_code"]
    line_index = data["code_line_index"]
    mem_range = data.get("mem_range", 4)

    if "U8" in operation:
        mem_range = 4
    if "U16" in operation:
        mem_range = 8
    elif "32" in operation:
        mem_range = 4
    elif "64" in operation:
        mem_range = 8
    elif "128" in operation:
        mem_range = 16

    ops.add(operation)
    new_row = OpInfoRow(**data)
    _devices.add(data["running_dev_id"])
    _devices.add(data["mem_dev_id"])

    # device = "GPU" + str(data["running_dev_id"])
    # owner = "GPU" + str(data["mem_dev_id"])
    # pair = device + "-" + owner


tables = {
    "op_info": {"starts_with": "op_code", "parser": parse_op_info},
    "func_info": {"starts_with": "pc", "parser": parse_func_info},
    "offset_info": {"starts_with": "offset", "parser": parse_offset_info},
    "obj_info": {"starts_with": "obj_id", "parser": parse_obj_info},
    "codeline_info": {"starts_with": "code_line_index", "parser": parse_codeline_info},
}


# @st.cache_data
def read_data(file, filename, gbs):
    if file == None or filename == None:
        st.experimental_rerun()  # this shouldn't be here need to fix the problem soon
    devices, ops = gbs

    # prints all files
    graph_name = ""
    pickle_file = None

    pickle_filename = "".join(filename.split(".")[:-1]) + ".pkl"
    if os.path.isfile(pickle_filename):
        with open(pickle_filename, "rb") as f:
            pickle_file = pickle.load(f)
            # if len(pickle_file) > 6:
            #     new_pickle_file = pickle_file[:5]
            #     new_pickle_file.append(pickle_file[6])
            #     pickle_file = new_pickle_file
            for item in pickle_file:
                pass
                # print(item)
                # print()
            print("Data loaded from " + pickle_filename)

    reading_data = 0
    opkeys = []
    objkeys = []
    counter = 0

    if pickle_file is None:
        for line in file:
            parse_line(line, gbs)

        tempdev = deepcopy(_devices)
        if -1 in tempdev:
            tempdev.remove(-1)

        st.session_state.gpu_num = max(max(tempdev) + 1, len(tempdev))
        gpu_num = max(max(tempdev) + 1, len(tempdev))
        print("Reading complete")

        all_data = [
            OpInfoRow._table,
            FunctionInfoRow.by_pc,
            ObjIdRow.by_dev_offset,
            ObjIdRow.by_dev_id,
            ObjNameRow.by_obj_id,
            CodeLineInfoRow.by_cd_index,
            gpu_num,
            ops,
        ]

        with open(pickle_filename, "wb") as pf:
            pickle.dump(all_data, pf)
            print("Data saved to " + pickle_filename)
    else:
        (
            OpInfoRow._table,
            FunctionInfoRow.by_pc,
            ObjIdRow.by_dev_offset,
            ObjIdRow.by_dev_id,
            ObjNameRow.by_obj_id,
            CodeLineInfoRow.by_cd_index,
            gpu_num,
            ops,
        ) = pickle_file

        st.session_state.gpu_num = gpu_num

    return gpu_num, ops
