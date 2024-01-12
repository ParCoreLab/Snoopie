from typing import List, Dict, NamedTuple, Tuple, MutableSet
from abc import ABC, abstractmethod
import os


def quick_add_to_dict(d: Dict[any, Dict[any, any]], key1, key2, value):
    if key1 not in d:
        d[key1] = {}
    d[key1][key2] = value


def table_to_list(d: Dict[any, Dict[any, any]]) -> List[any]:
    ret = []
    for i in d.values():
        ret.extend(i.values())
    return ret


class Table(ABC):
    @staticmethod
    @abstractmethod
    def table():
        pass

    def __init__(self, pid: int):
        self.pid = pid


class UniqueObjectKeyable:
    def __init__(self, key: "UniqueObject") -> None:
        self.key: "UniqueObject" = key

    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, type(self)) and self.key == __value.key

    def __hash__(self) -> int:
        return hash(self.key)


def generic_filter(table: Table, criteria: dict):
    filtered = []
    for row in table.table():
        match = True
        for key, value in criteria.items():
            if not hasattr(row, key) or getattr(row, key) != value:
                match = False
                break
        if match:
            filtered.append(row)
    return filtered


class OpInfoRowKey(NamedTuple):
    pid: int
    op_code: str
    addr: str
    thread_indx: int
    running_dev_id: int
    mem_dev_id: int
    code_linenum: int
    code_line_index: int
    code_line_estimated_status: int
    obj_offset: str
    mem_range: int

    def get_codeline_info(self):
        return CodeLineInfoRow.by_cd_index.get(self.pid).get(self.code_line_index)

    def get_obj_info(self):
        obj_id_info: ObjIdRow | None = ObjIdRow.search(self.pid, self.obj_offset)
        if obj_id_info == None:
            print("OBJ ID INFO NONE", self.obj_offset, self.pid)
            return None, None
        obj_name_info: ObjNameRow | None = ObjNameRow.search(
            self.pid, obj_id_info.obj_id
        )
        return obj_id_info, obj_name_info

    def get_unique_obj(self) -> "UniqueObject":
        id, name = self.get_obj_info()
        if id is None or name is None:
            return None
        return UniqueObject(name.var_name, name.call_stack.get_tuple_stack(), id.obj_id)

    def get_line_info(self, obj_id_info=None, obj_name_info=None):
        if obj_id_info == None or obj_name_info == None:
            obj_id_info, obj_name_info = self.get_obj_info()
            if obj_id_info == None or obj_name_info == None:
                return None
        codeline_info = self.get_codeline_info()
        return LineInfo.get(obj_name_info, obj_id_info, codeline_info)


class OpInfoRowValue(NamedTuple):
    count: int
    related_object: "SnoopieObject"


class OpInfoRowCombined(NamedTuple):
    key: OpInfoRowKey
    value: OpInfoRowValue

    def get_codeline_info(self):
        return self.key.get_codeline_info()

    def get_obj_info(self):
        return self.key.get_obj_info()

    def get_unique_obj(self) -> "UniqueObject":
        return self.key.get_unique_obj()

    def get_line_info(self, obj_id_info=None, obj_name_info=None):
        return self.key.get_line_info(
            obj_id_info=obj_id_info, obj_name_info=obj_name_info
        )


class OpInfoRow(Table):
    _table: Dict[OpInfoRowKey, OpInfoRowValue] = {}

    def __init__(
        self,
        pid: int,
        op_code: str,
        addr: str,
        thread_indx: int,
        running_dev_id: int,
        mem_dev_id: int,
        code_linenum: int,
        code_line_index: int,
        code_line_estimated_status: int,
        obj_offset: str,
        mem_range: int,
    ):
        super().__init__(pid)
        key = OpInfoRowKey(
            pid,
            op_code,
            addr,
            thread_indx,
            running_dev_id,
            mem_dev_id,
            code_linenum,
            code_line_index,
            code_line_estimated_status,
            obj_offset,
            mem_range,
        )

        if key in OpInfoRow._table:
            # print("Key in table")
            OpInfoRow._table[key] = OpInfoRowValue(
                OpInfoRow._table[key].count + 1, OpInfoRow._table[key].related_object
            )
        else:
            OpInfoRow._table[key] = OpInfoRowValue(1, None)

    @staticmethod
    def table():
        return OpInfoRow._table

    @staticmethod
    def filter_by_device_and_ops(
        allowed_ops: List[str],
        allowed_running_devs: List[int],
        allowed_mem_devs: List[int],
    ):
        return [
            OpInfoRowCombined(i, v)
            for i, v in OpInfoRow._table.items()
            if (
                i.op_code in allowed_ops
                and i.running_dev_id in allowed_running_devs
                and i.mem_dev_id in allowed_mem_devs
            )
        ]

    @staticmethod
    def get_total_accesses(ops: List[OpInfoRowCombined], memrange: bool) -> int:
        if not memrange:
            return sum([i.value.count for i in ops])
        _sum = 0
        for op in ops:
            mult = op.key.mem_range * op.value.count
            _sum += mult
        return _sum


class FunctionInfoRow(Table):
    by_pc: Dict[int, Dict[int, Table]] = {}

    class Key(NamedTuple):
        func_name: str
        file_name: str
        line_no: int

    def __init__(self, pid: int, pc: int, func_name: str, file_name: str, line_no: int):
        super().__init__(pid)
        self.pc = pc
        self.func_name = func_name
        self.file_name = file_name
        self.line_no = line_no

        quick_add_to_dict(FunctionInfoRow.by_pc, pid, pc, self)

    @staticmethod
    def search_by_pc(pid: int, pc: str):
        pc = int(pc)
        search_result = FunctionInfoRow.by_pc.get(pid).get(pc)
        return search_result

    @staticmethod
    def table():
        return table_to_list(FunctionInfoRow.by_pc)

    def __str__(self) -> str:
        return f"{self.file_name}[{self.func_name}]:{self.line_no}"

    def get_key(self) -> Key:
        return FunctionInfoRow.Key(self.func_name, self.file_name, self.line_no)


class ObjIdRow(Table):
    class Key(NamedTuple):
        offset: str
        size: int
        obj_id: int
        dev_id: int
        pid: int

    by_dev_offset: Dict[int, Dict[str, Table]] = {}
    by_pid_offset: Dict[int, Dict[str, Table]] = {}

    def __init__(self, pid: int, offset: str, size: int, obj_id: int, dev_id: int):
        super().__init__(pid)
        self.offset = offset
        self.size = size
        self.obj_id = obj_id
        self.dev_id = dev_id

        quick_add_to_dict(ObjIdRow.by_dev_offset, dev_id, offset, self)

        quick_add_to_dict(ObjIdRow.by_pid_offset, pid, offset, self)

    def get_key(self):
        return ObjIdRow.Key(self.offset, self.size, self.obj_id, self.dev_id, self.pid)

    @staticmethod
    def table():
        return table_to_list(ObjIdRow.by_pid_offset)

    @staticmethod
    def search(pid: int, offset: str) -> None | Table:
        temp = ObjIdRow.by_pid_offset.get(pid)
        if temp == None:
            return None
        r = temp.get(offset)
        # check other offsets
        if r is None:
            for p in ObjIdRow.by_pid_offset.keys():
                if p != pid:
                    temp = ObjIdRow.by_pid_offset.get(p)
                    if temp is not None:
                        r = temp.get(offset)
                        if r is not None:
                            break
        return r


class CallStack:
    def __init__(self, pid: int, call_stack_string: str):
        self.pid = pid
        self.stack: List[str] = CallStack.parse_call_stack_string(call_stack_string)

    @staticmethod
    def parse_call_stack_string(call_stack_string: str):
        if not isinstance(call_stack_string, str) or call_stack_string == "":
            return []
        split = call_stack_string.split("<")
        return split

    def get_parsed_stack(self) -> List[FunctionInfoRow]:
        return [
            FunctionInfoRow.search_by_pc(self.pid, pc) for pc in reversed(self.stack)
        ]

    def get_tuple_stack(self) -> Tuple[FunctionInfoRow.Key]:
        return tuple(i.get_key() for i in self.get_parsed_stack())


class UniqueObject(NamedTuple):
    obj_name: str
    initialized_call_stack: Tuple[FunctionInfoRow.Key]
    object_id: int

    def __str__(self) -> str:
        return f"{self.obj_name}:{self.object_id}, l:{len(self.initialized_call_stack)}"


class SnoopieObject(UniqueObjectKeyable):
    all_objects: Dict[UniqueObject, "SnoopieObject"] = {}

    def __init__(self, obj_name: str, obj_id: int, call_stack: CallStack):
        k = UniqueObject(obj_name, call_stack.get_tuple_stack(), obj_id)
        super().__init__(k)
        self.addres_ranges: Dict[ObjIdRow.Key, int] = {}
        self.ops: List[OpInfoRowCombined] = []
        # SnoopieObject.all_objects[k] = self

    def add_op(self, op: OpInfoRowCombined):
        self.ops.append(op)

    def add_addres_range(self, ar: ObjIdRow):
        key = ar.get_key()
        if key not in self.addres_ranges:
            self.addres_ranges[key] = 0
        self.addres_ranges[key] += 1

    def __str__(self) -> str:
        return str(self.key)

    @staticmethod
    def get_display_dict() -> List[Dict]:
        return {str(j): i.display_dict() for j, i in SnoopieObject.all_objects.items()}

    # {
    #     "var_name": self.key.obj_name,
    #     "id": self.key.object_id,
    #     "created_stack": [str(i) for i in self.key.initialized_call_stack]
    # }

    def display_dict(self):
        return {
            "key": self.key,
            "address_ranges": [
                {
                    "count": c,
                    "dev_id": i.dev_id,
                    "offset": i.offset,
                    "size": i.size,
                    "pid": i.pid,
                }
                for i, c in self.addres_ranges.items()
            ],
            "ops": [
                {
                    "op_code": i.op_code,
                    "code_line_index": i.code_line_index,
                    "mem_dev": i.mem_dev_id,
                    "run_dev": i.running_dev_id,
                    "offset": i.addr,
                }
                for i in self.ops
            ],
        }


class ObjNameRow(Table):
    by_obj_id: Dict[int, Dict[int, Table]] = {}

    def __init__(self, pid: int, obj_id: int, var_name: str, call_stack: str):
        super().__init__(pid)
        self.obj_id = obj_id
        self.var_name = var_name
        self.call_stack: CallStack = CallStack(self.pid, call_stack)

        quick_add_to_dict(ObjNameRow.by_obj_id, pid, obj_id, self)

    @staticmethod
    def table():
        return table_to_list(ObjNameRow.by_obj_id)

    @staticmethod
    def search(pid: int, id: int):
        temp = ObjNameRow.by_obj_id.get(pid)
        if temp == None:
            return None
        r = temp.get(id)
        # check other offsets
        if r is None:
            for p in ObjNameRow.by_obj_id.keys():
                if p != pid:
                    temp = ObjNameRow.by_obj_id.get(p)
                    if temp is not None:
                        r = temp.get(id)
                        if r is not None:
                            break
        return r


class CodeLineInfoRow(Table):
    class CodeLineInfoTuple(NamedTuple):
        code_line_index: int
        dir_path: str
        file: str
        code_linenum: int
        code_line_estimated_status: int

    by_cd_index: Dict[int, Dict[int, Table]] = {}
    inferred_home_dir: str = None

    def __init__(
        self,
        pid: int,
        code_line_index: int,
        dir_path: str,
        file: str,
        code_linenum: int,
        code_line_estimated_status: int,
    ):
        super().__init__(pid)
        self.code_line_index = code_line_index
        self.dir_path = dir_path
        self.file = file
        self.code_linenum = code_linenum
        self.code_line_estimated_status = code_line_estimated_status

        quick_add_to_dict(CodeLineInfoRow.by_cd_index, pid, code_line_index, self)

    def to_tuple(self) -> "CodeLineInfoRow.CodeLineInfoTuple":
        return CodeLineInfoRow.CodeLineInfoTuple(
            self.code_line_index,
            self.dir_path,
            self.file,
            self.code_linenum,
            self.code_line_estimated_status,
        )

    @staticmethod
    def table():
        return table_to_list(CodeLineInfoRow.by_cd_index)

    def combined_filepath(self) -> str:
        return os.path.join(self.dir_path, self.file)

    def trimmed_path(self) -> str:
        return self.dir_path[len(CodeLineInfoRow.inferred_home_dir) :]

    def relative_file_path(self) -> str:
        trimmed_f_path = self.trimmed_path()
        joined = os.path.join(trimmed_f_path, self.file)
        if len(joined) == 0:
            return joined
        if joined[-1] == "/":
            joined = joined[:-1]
        if joined[0] == "/":
            joined = joined[1:]
        return joined

    @staticmethod
    def infer_home_dir(rows: List[Table]) -> str:
        dirpaths: List[List[str]] = [
            (i.dir_path).split("/")
            for i in rows
            if len(i.file) > 0 or len(i.dir_path) > 0
        ]
        print("dirpaths", dirpaths)
        index = 0
        for i in range(min([len(i) for i in dirpaths])):
            index = i
            to_check = [j[index] for j in dirpaths]
            check = all(j == to_check[0] for j in to_check)
            if (
                not check
            ):  # some are different, the last split should be the home directory
                index -= 1
                break
        home_dir_path = "/".join(dirpaths[0][: index + 1])
        return home_dir_path


class LineInfo(UniqueObjectKeyable):
    class LineInfoKey(NamedTuple):
        Uobject: UniqueObject
        codeline_info: CodeLineInfoRow.CodeLineInfoTuple

    saved_objects = {}

    @staticmethod
    def get(
        obj_name_info: ObjNameRow, obj_id_info: ObjIdRow, codeline_info: CodeLineInfoRow
    ) -> "LineInfo":
        key = LineInfo.get_key_t(obj_name_info, obj_id_info, codeline_info)
        if key in LineInfo.saved_objects:
            return LineInfo.saved_objects[key]
        else:
            ret = LineInfo(obj_name_info, obj_id_info, codeline_info)
            LineInfo.saved_objects[key] = ret
            return ret

    """
    Don't use __init__ unless really necessary
    """

    def __init__(
        self,
        obj_name_info: ObjNameRow,
        obj_id_info: ObjIdRow,
        codeline_info: CodeLineInfoRow,
    ):
        super().__init__(LineInfo.get_key_t(obj_name_info, obj_id_info, codeline_info))
        self.obj_name_info = obj_name_info
        self.obj_id_info = obj_id_info
        self.codeline_info = codeline_info
        self.call_stack: List[
            FunctionInfoRow
        ] = self.obj_name_info.call_stack.get_parsed_stack()

    def __repr__(self) -> str:
        ret = ""
        for i in self.call_stack:
            ret += str(i) + " -> "
        ret += self.obj_name_info.var_name
        return ret

    def __str__(self) -> str:
        ret = ""
        for i in self.call_stack:
            ret += i.func_name + " -> "
        ret += self.obj_name_info.var_name
        return ret

    @staticmethod
    def get_key_t(
        obj_name_info: ObjNameRow,
        obj_id_info: ObjIdRow,
        code_line_info: CodeLineInfoRow,
    ):
        return LineInfo.LineInfoKey(
            UniqueObject(
                obj_name=obj_name_info.var_name,
                object_id=obj_id_info.obj_id,
                # dev_id=obj_id_info.dev_id,
                initialized_call_stack=obj_name_info.call_stack.get_tuple_stack(),
            ),
            code_line_info.to_tuple(),
        )

    def get_key(self) -> UniqueObject:
        return LineInfo.get_key_t(self.obj_name_info, self.obj_id_info)

    def check_correct_line(self, file: str, line: int) -> bool:
        return (
            file == self.codeline_info.combined_filepath()
            and line == self.codeline_info.code_linenum
        )
