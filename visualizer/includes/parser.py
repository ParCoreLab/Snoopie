import streamlit as st
from .tables import *
from copy import deepcopy
from .streamlit_globals import setup_globals
import os
import pickle
from io import TextIOWrapper
from typing import List, Dict, Tuple, Literal, Any
import pandas as pd
from functools import partial

tables: Dict[
    Literal["op_info", "func_info", "offset_info", "obj_info", "codeline_info"],
    Dict[Literal["starts_with", "parser"], Any],
] = {}

current_table = None

_counter = 0
_table_keys = []
_devices = set()
_pid = -1


def change_table(line: str):
    global current_table, _table_keys
    next_key = get_next_table(line)
    if next_key is not None and next_key != current_table:
        current_table = next_key
        _table_keys = line.strip().split(",")
        return True
    return False


def get_next_table(df: pd.DataFrame) -> str | None:
    for key in tables.keys():
        if df.keys()[0] == tables[key]["starts_with"]:
            return key
    return None


def parse_line(line: str, gbs: tuple):
    global _counter

    _counter += 1
    if _counter % 100000 == 0:
        print("reading data, line:", _counter)

    if current_table == None:
        change_table(line)
    else:
        tables[current_table]["parser"](line, gbs)


def parse_file(file: TextIOWrapper, filename: str, gbs: tuple):
    print("parsing file", filename)
    df = pd.read_csv(file)
    print("df successfully read")
    key = get_next_table(df)
    print("key:", key)
    if key is None:
        raise Exception(f"Could not match {filename}: {df.keys()} to any logfile type")
    fnc: function = tables[key]["parser"]
    i = 0
    for index, row in df.iterrows():
        if i % 1e10 == 0:
            print(fnc, row)
        i += 1
        fnc(row, gbs=gbs)

def isInt_try(v):
    try:
        i = int(v)
    except:
        return False
    return True


def parse_codeline_info(line: pd.Series, gbs: tuple):
    _, ops = gbs
    data = {**{k.strip(): v for k, v in line.items()}}
    data["pid"] = _pid
    CodeLineInfoRow(**data)


def parse_obj_info(line: pd.Series, gbs: tuple):
    _, ops = gbs
    data = {k.strip(): v for k, v in line.items()}
    data["pid"] = _pid
    ObjNameRow(**data)


def parse_offset_info(line: pd.Series, gbs: tuple):
    _, ops = gbs
    data = {k.strip(): v for k, v in line.items()}
    data["pid"] = _pid
    _devices.add(data["dev_id"])
    ObjIdRow(**data)


def parse_func_info(line: pd.Series, gbs: tuple):
    _, ops = gbs
    data = {k.strip(): v for k, v in line.items()}
    data["pid"] = _pid
    FunctionInfoRow(**data)


def parse_op_info(line: pd.Series, gbs: tuple):
    _, ops = gbs
    data = {k.strip(): v for k, v in line.items()}
    data["pid"] = _pid

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
    data["mem_range"] = mem_range

    ops.add(operation)
    new_row = OpInfoRow(**data)
    _devices.add(data["running_dev_id"])
    _devices.add(data["mem_dev_id"])

    # device = "GPU" + str(data["running_dev_id"])
    # owner = "GPU" + str(data["mem_dev_id"])
    # pair = device + "-" + owner


tables = {
    "op_info": {"starts_with": "op_code", "parser": parse_op_info},  # snoopie_log
    "func_info": {"starts_with": "pc", "parser": parse_func_info},  # mem_alloc_site
    "offset_info": {"starts_with": "offset", "parser": parse_offset_info},  # addr_range
    "obj_info": {"starts_with": "obj_id", "parser": parse_obj_info},  # data_obj_log
    "codeline_info": {"starts_with": "code_line_index", "parser": parse_codeline_info},
}


def get_pid(filename: str) -> int:
    global _pid
    if filename.endswith(".txt"):
        # one of address_range_log_pid.txt, codeline_log_pid.txt, data_object_log_pid.txt, mem_alloc_site_log_pid.txt
        beforepid = filename.rfind("_")
        slice = filename[beforepid + 1 : -4]
        if not slice.isdigit():
            return -1
        return int(slice)
    else:
        # snoopie-log-pid or snoopie-log-pid.zstd
        last = -4 if filename.endswith(".zst") else -1
        beforepid = filename.rfind("-")
        slice = filename[beforepid + 1 : last]
        if not slice.isdigit():
            return -1
        return int(slice)


# @st.cache_data
def read_data(
    file: TextIOWrapper | List[TextIOWrapper], filename: str | List[str], gbs
):
    global _pid
    print(file, filename)
    if file == None or filename == None:
        st.experimental_rerun()  # this shouldn't be here need to fix the problem soon

    all_pids = []

    devices, ops = gbs

    # prints all files
    graph_name = ""
    pickle_file = None
    pickle_filename = ""

    if isinstance(filename, list):
        all_pids = [get_pid(i) for i in filename]
        all_pids.sort()
        pickle_filename = "-".join([str(i) for i in all_pids]) + ".pkl"
    else:
        pickle_filename = "".join(filename.split(".")[:-1]) + ".pkl"

    st.session_state.pickle_filename = pickle_filename

    if os.path.isfile(pickle_filename):
        with open(pickle_filename, "rb") as f:
        # set gpu num and ops here
        # pickle_file = 0  
            pickle_file = pickle.load(f)
        # # if len(pickle_file) > 6:
        # #     new_pickle_file = pickle_file[:5]
        # #     new_pickle_file.append(pickle_file[6])
        # #     pickle_file = new_pickle_file
        # for item in pickle_file:
        #     pass
        #     # print(item)
        #     # print()
        # print("Data loaded from " + pickle_filename)

    reading_data = 0
    opkeys = []
    objkeys = []
    counter = 0

    if pickle_file is None:
        if isinstance(file, list):
            for f, fn in zip(file, filename):
                _pid = get_pid(fn)
                parse_file(f, fn, gbs)
        else:
            _pid = get_pid(filename)
            parse_file(file, filename, gbs)
        if st.session_state.check_source_code:
            CodeLineInfoRow.inferred_home_dir = CodeLineInfoRow.infer_home_dir(
                CodeLineInfoRow.table()
            )
        ts = set()
        for op in list(OpInfoRow._table.keys()):
            id, name = op.get_obj_info()
            u: UniqueObject = op.get_unique_obj()
            if u is None:
                continue
            if u not in SnoopieObject.all_objects:
                tmp = SnoopieObject(name.var_name, id.obj_id, name.call_stack)
                SnoopieObject.all_objects[u] = tmp
            so: SnoopieObject = SnoopieObject.all_objects[u]
            so.add_op(op)
            so.add_addres_range(id)
            val = OpInfoRow._table[op]
            OpInfoRow._table[op] = OpInfoRowCombined(op, OpInfoRowValue(val.count, so))
        print(len(OpInfoRow.table()))
        tempdev = deepcopy(_devices)
        if -1 in tempdev:
            tempdev.remove(-1)
        print(tempdev)
        st.session_state.gpu_num = max(max(tempdev) + 1, len(tempdev))
        gpu_num = max(max(tempdev) + 1, len(tempdev))
        print("Reading complete")

        all_data = [
            OpInfoRow._table,
            SnoopieObject.all_objects,
            FunctionInfoRow.by_pc,
            ObjIdRow.by_dev_offset,
            ObjIdRow.by_pid_offset,
            ObjNameRow.by_obj_id,
            CodeLineInfoRow.by_cd_index,
            CodeLineInfoRow.inferred_home_dir,
            gpu_num,
            ops,
        ]

        with open(pickle_filename, "wb") as pf:
            pickle.dump(all_data, pf)
            print("Data saved to " + pickle_filename)
    else:
        # pass
        (
            OpInfoRow._table,
            SnoopieObject.all_objects,
            FunctionInfoRow.by_pc,
            ObjIdRow.by_dev_offset,
            ObjIdRow.by_pid_offset,
            ObjNameRow.by_obj_id,
            CodeLineInfoRow.by_cd_index,
            CodeLineInfoRow.inferred_home_dir,
            gpu_num,
            ops,
        ) = pickle_file

        # st.session_state.gpu_num = gpu_num

    return gpu_num, ops
