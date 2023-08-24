from typing import List, Dict
from abc import ABC, abstractmethod





class Table(ABC):
    @staticmethod
    @abstractmethod
    def table():
        pass

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
            for i in OpInfoRow._table
            if (i.op_code in allowed_ops
            and i.running_dev_id in allowed_running_devs
            and i.mem_dev_id in allowed_mem_devs)
        ]
    
    @staticmethod
    def get_total_accesses(ops: List[Table], memrange: bool) -> int:
        sum = 0
        for op in ops:
            mult = 1 if not memrange else op.mem_range
            sum += mult
        return sum

    def get_codeline_info(self):
        return CodeLineInfoRow.by_cd_index.get(self.code_line_index)
    
    def get_obj_info(self):
        obj_id_info : ObjIdRow | None = ObjIdRow.by_offset.get(self.obj_offset)
        if obj_id_info == None: return None
        obj_name_info : ObjNameRow | None = ObjNameRow.by_obj_id.get(obj_id_info.obj_id)
        return obj_id_info, obj_name_info

class FunctionInfoRow(Table):
    by_pc: Dict[int, Table] = {}

    def __init__(self, pc: int, func_name: str, file_name: str, line_no: int):
        self.pc = pc
        self.func_name = func_name
        self.file_name = file_name
        self.line_no = line_no

        FunctionInfoRow.by_pc[pc] = self

    @staticmethod
    def search_by_pc(pc: str):
        pc = int(pc)
        search_result = FunctionInfoRow.by_pc.get(pc)
        return search_result

    def table():
        return list(FunctionInfoRow.by_pc.values())


class ObjIdRow(Table):
    by_offset: Dict[str, Table] = {}

    def __init__(self, offset: str, size: int, obj_id: int, dev_id: int):
        self.offset = offset
        self.size = size
        self.obj_id = obj_id
        self.dev_id = dev_id

        ObjIdRow.by_offset[self.offset] = self

    def table():
        return list(ObjIdRow.by_offset.values())


class CallStack:
    def __init__(self, call_stack_string: str):
        self.stack: List[str] = CallStack.parse_call_stack_string(call_stack_string)

    @staticmethod
    def parse_call_stack_string(call_stack_string: str):
        if not isinstance(call_stack_string, str) or call_stack_string == "":
            return []
        split = call_stack_string.split("<")
        return split

    def get_parsed_stack(self):
        return [FunctionInfoRow.search_by_pc(pc) for pc in self.stack]


class ObjNameRow(Table):
    by_obj_id: Dict[int, Table] = {}

    def __init__(self, obj_id: int, var_name: str, call_stack: str):
        self.obj_id = obj_id
        self.var_name = var_name
        self.call_stack: CallStack = CallStack(call_stack)

        ObjNameRow.by_obj_id[self.obj_id] = self

    def table():
        return list(ObjNameRow.by_obj_id.values())


class CodeLineInfoRow(Table):
    by_cd_index: Dict[int, Table] = {}

    def __init__(
        self,
        code_line_index: int,
        dir_path: str,
        file: str,
        code_linenum: int,
        code_line_estimated_status: int,
    ):
        self.code_line_index = code_line_index
        self.dir_path = dir_path
        self.file = file
        self.code_linenum = code_linenum
        self.code_line_estimated_status = code_line_estimated_status

        CodeLineInfoRow.by_cd_index[self.code_line_index] = self

    def table():
        return list(CodeLineInfoRow.by_cd_index.values())
