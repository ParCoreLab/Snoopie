import streamlit as st

tables = {
    "op_info": {"starts_with": "op_code", "parser": parse_op_info},
    "func_info": {"starts_with": "pc", "parser": None},
    "offset_info": {"starts_with": "offset", "parser": None},
    "obj_info": {"starts_with": "obj_id", "parser": None},
    "codeline_info": {"starts_with": "code_line_index", "parser": None},
}

current_table = None

_counter = 0
_table_keys = []


def change_table(line: str):
    global current_table, _table_keys
    for key in tables.keys():
        if line.startswith(tables[key]):
            current_table = key
            _table_keys = line.strip().split(",")
            return


def parse_line(line: str, gbs: tuple):
    global _counter

    _counter += 1
    if counter % 100000 == 0:
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

# old
def parse_op_info(line: str, gbs: tuple):
    data_by_address, data_by_device, data_by_obj, data_by_line, gpu_num, ops = gbs
    change_table(line)
    if current_table != "op_info":
        return

    data = {}
    vals = line.strip().split(",")  # change this later
    for index in range(len(_table_keys)):
        if isInt_try(vals[index]):
            data[_table_keys[index]] = int(vals[index])
        else:
            data[_table_keys[index]] = vals[index]

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

    device = "GPU" + str(data["running_dev_id"])
    owner = "GPU" + str(data["mem_dev_id"])
    pair = device + "-" + owner

    data_by_address[address] = data_by_address.get(
        address, {"total": 0, "by_operation": {}, "by_device": {}, "lines": set(), "by_line_index": {}}
    )

    temp_data = data_by_address[address]
    temp_data["total"] = temp_data.get("total", 0) + 1
    temp_data["by_operation"][operation] = (
        temp_data["by_operation"].get(operation, 0) + 1
    )

    temp_data["by_device"][device] = temp_data["by_device"].get(
        device, {"total": 0, "by_operation": {}}
    )
    temp_data["by_device"][device]["by_operation"][operation] = (
        temp_data["by_device"][device]["by_operation"].get(operation, 0) + 1
    )
    temp_data["by_device"][device]["total"] = (
        temp_data["by_device"][device].get("total", 0) + 1
    )

    temp_data["by_line_index"][line_index] = temp_data["by_line_index"].get(line_index, {"total": 0, "by_operation": {}})

    temp_data["by_line_index"][line_index]["total"] = temp_data["by_line_index"][line_index].get("total", 0) + 1

    temp_data["by_line_index"][line_index]["by_operation"][operation] = temp_data["by_line_index"][line_index]["by_operation"].get(operation, 0) + 1


    temp_lines = temp_data.get("lines", set())
    temp_lines.add(line_index)
    temp_data["lines"] = temp_lines

    data_by_device[pair] = data_by_device.get(pair, {"total": 0, "totalbytes": 0, "operations": {}, "operations_bytes": {}, "obj_offset": {}})
    temp_data = data_by_device[pair]

    temp_data["total"] = temp_data.get("total", 0) + 1
    temp_data["totalbytes"] = temp_data.get("totalbytes", 0) + mem_range

    temp_data["operations"][operation] = temp_data["operations"].get(operation, 0) + 1
    temp_data["operations_bytes"][operation] = temp_data["operations_bytes"].get(operation, 0) + mem_range

    temp_data["obj_offset"][obj_name] = temp_data["obj_offset"].get(obj_name, 0) + 1

    data_by_line[line_index] = data_by_line.get(line_index, {})


    #TODO CHANGE BELOW

    temp_data = data_by_line[line_index]

    temp_data["total"] = temp_data.get("total", 0) + 1
    temp_data[pair] = temp_data.get(pair, 0) + 1
    temp_data[operation] = temp_data.get(operation, 0) + 1
    temp_data[address] = temp_data.get(address, 0) + 1
    temp_data[obj_offset] = temp_data.get(obj_offset, 0) + 1
    temp_objects = temp_data.get("objects", set())
    temp_objects.add(obj_offset)
    




@st.cache_data
def read_data(_file, filename, gbs):
    if _file == None or filename == None:
        st.experimental_rerun()  # this shouldn't be here need to fix the problem soon
    data_by_address, data_by_device, data_by_obj, data_by_line, gpu_num, ops = gbs
    file = _file
    if gpu_num <= 0:
        file = detect_gpu_count(file)
    if file == None:
        logfile_base.seek(0)
        file, _ = filepath_handler.file_from_upload_check(logfile_base)
        global logfile
        logfile = file

    addrs = set()

    # prints all files
    graph_name = ""
    pickle_file = None

    pickle_filename = "".join(filename.split(".")[:-1]) + ".pkl"
    if os.path.isfile(pickle_filename):
        with open(pickle_filename, "rb") as f:
            pickle_file = pickle.load(f)
            if len(pickle_file) > 6:
                new_pickle_file = pickle_file[:5]
                new_pickle_file.append(pickle_file[6])
                pickle_file = new_pickle_file
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
            counter += 1
            if counter % 100000 == 0:
                print(counter)
            if reading_data:
                data = {}
                vals = line.strip().split(",")
                # print(vals)
                if len(vals) != len(opkeys):
                    reading_data = False
                    if line.startswith("offset"):
                        objkeys = line.split(", ")
                        reading_data = True
                    break
                for index in range(len(opkeys)):
                    if isInt_try(vals[index]):
                        data[opkeys[index]] = int(vals[index])
                    else:
                        data[opkeys[index]] = vals[index]

                address = data["addr"]
                obj_name = data["obj_offset"]
                operation = data["op_code"]
                linenum = data["code_linenum"]
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

                addrs.add(address)

                ops.add(operation)

                device = "GPU" + str(data["running_dev_id"])
                owner = "GPU" + str(data["mem_dev_id"])
                pair = device + "-" + owner

                # print(operation)
                # print(device)
                # print(owner)
                # print(address)
                # print(obj_name)

                data_by_address[address] = data_by_address.get(address, {})
                temp_data = data_by_address[address]
                temp_data["total"] = temp_data.get("total", 0) + 1
                temp_data[operation] = temp_data.get(operation, 0) + 1

                temp_data[device] = temp_data.get(device, {})
                temp_data[device][operation] = temp_data[device].get(operation, 0) + 1

                temp_data["line_" + str(linenum)] = temp_data.get(
                    "line_" + str(linenum), {}
                )
                temp_data["line_" + str(linenum)][operation] = (
                    temp_data["line_" + str(linenum)].get(operation, 0) + 1
                )

                temp_lines = temp_data.get("lines", set())
                temp_lines.add("line_" + str(linenum))
                temp_data["lines"] = temp_lines

                data_by_device[pair] = data_by_device.get(pair, {})
                temp_data = data_by_device[pair]
                temp_data["total"] = temp_data.get("total", 0) + 1
                temp_data["totalbytes"] = temp_data.get("totalbytes", 0) + mem_range
                temp_data[operation] = temp_data.get(operation, 0) + 1
                temp_data[operation + "bytes"] = (
                    temp_data.get(operation + "bytes", 0) + mem_range
                )
                temp_data[obj_name] = temp_data.get(obj_name, 0) + 1
                temp_data["line_" + str(linenum)] = (
                    temp_data.get("line_" + str(linenum), 0) + 1
                )
                temp_lines = temp_data.get("lines", set())
                temp_lines.add("line_" + str(linenum))
                temp_data["lines"] = temp_lines

                data_by_device[device] = data_by_device.get(device, {})
                temp_data = data_by_device[device]
                temp_data["totalbytes"] = temp_data.get("totalbytes", 0) + mem_range
                temp_data["total"] = temp_data.get("total", 0) + 1
                temp_data[operation] = temp_data.get(operation, 0) + 1
                temp_data[operation + "bytes"] = (
                    temp_data.get(operation + "bytes", 0) + mem_range
                )
                temp_data[owner] = temp_data.get(owner, 0) + 1
                temp_data[owner + "bytes"] = (
                    temp_data.get(owner + "bytes", 0) + mem_range
                )
                temp_data[obj_name] = temp_data.get(obj_name, 0) + 1

                data_by_line[linenum] = data_by_line.get(linenum, {})
                temp_data = data_by_line[linenum]
                temp_data["total"] = temp_data.get("total", 0) + 1
                temp_data[pair] = temp_data.get(pair, 0) + 1
                temp_data[operation] = temp_data.get(operation, 0) + 1
                temp_data[address] = temp_data.get(address, 0) + 1
                temp_data[obj_name] = temp_data.get(obj_name, 0) + 1
                temp_objects = temp_data.get("objects", set())
                temp_objects.add(obj_name)
            else:
                if line.startswith("filename="):
                    print("This is a graph file!")
                    splt_info = line.strip().split(",")
                    graph_name = splt_info[0].split("/")[-1]
                    gpu_num = int(splt_info[1][-1])
                # op_code, addr, thread_indx, running_dev_id, mem_dev_id, code_linenum, code_line_index, code_line_estimated_status, obj_offset
                elif line.startswith("op_code"):
                    opkeys = line.strip().split(", ")
                    reading_data = True
                # elif (line.startswith('offset')):
                #     objkeys = line.split(', ')
                #     reading_data = 2

        for line in file:
            if reading_data:
                data = {}
                vals = line.strip().split(",")
                if len(vals) != len(objkeys):
                    reading_data = False
                    continue
                for index in range(len(objkeys)):
                    if isInt_try(vals[index]):
                        data[objkeys[index]] = int(vals[index])
                    else:
                        data[objkeys[index]] = vals[index]
                data_by_obj[data[objkeys[0]]] = data
                # for i in range(data['size']):
                #     hex_addr = str.format('0x{:016x}', int_off+(i*4))
                #     addr_obj_map[hex_addr] = hex_off
            else:
                # offset, size, device_id, var_name, filename, alloc_line_num
                if line.startswith("offset"):
                    objkeys = line.split(", ")
                    reading_data = True

        print("Reading complete")

        all_data = [
            data_by_address,
            data_by_device,
            data_by_obj,
            data_by_line,
            gpu_num,
            ops,
        ]

        with open(pickle_filename, "wb") as pf:
            pickle.dump(all_data, pf)
            print("Data saved to " + pickle_filename)
    else:
        (
            data_by_address,
            data_by_device,
            data_by_obj,
            data_by_line,
            gpu_num,
            ops,
        ) = pickle_file

    return data_by_address, data_by_device, data_by_obj, data_by_line, gpu_num, ops
