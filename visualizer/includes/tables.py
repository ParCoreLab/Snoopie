from typing import List, Dict
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


class OpInfoRow(Table):
    _table: List[Table] = []

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
        self.op_code = op_code
        self.addr = addr
        self.thread_indx = thread_indx
        self.running_dev_id = running_dev_id
        self.mem_dev_id = mem_dev_id
        self.code_linenum = code_linenum
        self.code_line_index = code_line_index
        self.code_line_estimated_status = code_line_estimated_status
        self.obj_offset = obj_offset
        self.mem_range = mem_range

        OpInfoRow._table.append(self)

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
            i
            for i in OpInfoRow.table()
            if (
                i.op_code in allowed_ops
                and i.running_dev_id in allowed_running_devs
                and i.mem_dev_id in allowed_mem_devs
            )
        ]

    @staticmethod
    def get_total_accesses(ops: List[Table], memrange: bool) -> int:
        if not memrange:
            return len(ops)
        _sum = 0
        for op in ops:
            mult = op.mem_range
            _sum += mult
        return _sum

    def get_codeline_info(self):
        return CodeLineInfoRow.by_cd_index.get(self.pid).get(self.code_line_index)

    def get_obj_info(self):
        obj_id_info: ObjIdRow | None = ObjIdRow.search(self.pid, self.obj_offset)
        if obj_id_info == None:
            return None, None
        obj_name_info: ObjNameRow | None = ObjNameRow.by_obj_id.get(self.pid).get(
            obj_id_info.obj_id
        )
        return obj_id_info, obj_name_info

    def get_line_info(self, obj_id_info=None, obj_name_info=None):
        if obj_id_info == None or obj_name_info == None:
            obj_id_info, obj_name_info = self.get_obj_info()
        codeline_info = self.get_codeline_info()
        return LineInfo(self, obj_name_info, obj_id_info, codeline_info)


class FunctionInfoRow(Table):
    by_pc: Dict[int, Dict[int, Table]] = {}

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


class ObjIdRow(Table):
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

    @staticmethod
    def table():
        return table_to_list(ObjIdRow.by_pid_offset)

    @staticmethod
    def search(pid: int, offset: str) -> None | Table:
        temp = ObjIdRow.by_pid_offset.get(pid)
        if temp == None:
            return None
        return temp.get(offset)


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

    def get_parsed_stack(self):
        return [
            FunctionInfoRow.search_by_pc(self.pid, pc) for pc in reversed(self.stack)
        ]


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


class CodeLineInfoRow(Table):
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


class LineInfo:
    def __init__(
        self,
        op_info: OpInfoRow,
        obj_name_info: ObjNameRow,
        obj_id_info: ObjIdRow,
        codeline_info: CodeLineInfoRow,
    ):
        self.op_info = op_info
        self.obj_name_info = obj_name_info
        self.obj_id_info = obj_id_info
        self.codeline_info = codeline_info
        self.call_stack: List[
            FunctionInfoRow
        ] = self.obj_name_info.call_stack.get_parsed_stack()

    def __repr__(self) -> str:
        ret = ""
        for i in self.call_stack:
            ret += i.func_name + " -> "
        ret += self.obj_name_info.var_name
        ret += (
            " : " + self.codeline_info.file + ":" + str(self.codeline_info.code_linenum)
        )
        return ret

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, LineInfo) and hash(self) == hash(__value)

    def __hash__(self) -> int:
        return hash("LineInfo:" + str(self))

    def check_correct_line(self, file: str, line: int) -> bool:
        return (
            file == self.codeline_info.combined_filepath()
            and line == self.codeline_info.code_linenum
        )
