import streamlit as st
from .tables import (
    OpInfoRow,
    ObjIdRow,
    ObjNameRow,
    ContextRow,
    CodeLineInfoRow,
    FunctionInfoRow,
    SiteInfoRow,
    LineInfo,
    SnoopieObject
)

gpu_num: int
sampling_period: int
current_folder: None | str
src_code_file = None
logfile = None
logfile_name = None
logfile_base = None
home_folder = None
choose_src_code_file_list = None


def _reset():
    global gpu_num, src_code_file, sampling_period, logfile, logfile_name, logfile_base, home_folder, current_folder, choose_src_code_file_list

    #####################################
    gpu_num = -1
    sampling_period = 1

    # these two variables should be open files or None
    src_code_file = None
    logfile = None
    logfile_name = None
    logfile_base = None  # base uploadad file
    home_folder = None
    current_folder = None
    #####################################
    choose_src_code_file_list = {}


_reset()


def setup_globals():
    global gpu_num, src_code_file, sampling_period, logfile, logfile_name, logfile_base, home_folder, current_folder, choose_src_code_file_list

    if "gpu_num" in st.session_state:
        gpu_num = st.session_state.gpu_num

    if "src_code_file" in st.session_state:
        src_code_file = st.session_state.src_code_file

    if "sampling_period" in st.session_state:
        sampling_period = st.session_state.sampling_period

    if "logfile" in st.session_state:
        logfile = st.session_state.logfile

    if "logfile_name" in st.session_state:
        logfile_name = st.session_state.logfile_name

    if "logfile_base" in st.session_state:
        logfile_base = st.session_state.logfile_base

    if "home_folder" in st.session_state:
        home_folder = st.session_state.home_folder.strip()

    if "current_folder" in st.session_state:
        current_folder = st.session_state.current_folder

    if "choose_src_code_file_list" in st.session_state:
        choose_src_code_file_list = st.session_state.choose_src_code_file_list


def reset_globals():
    if "gpu_num" in st.session_state:
        del st.session_state.gpu_num

    if "src_code_file" in st.session_state:
        del st.session_state.src_code_file

    if "sampling_period" in st.session_state:
        del st.session_state.sampling_period

    if "logfile" in st.session_state:
        del st.session_state.logfile

    if "logfile_name" in st.session_state:
        del st.session_state.logfile_name

    if "logfile_base" in st.session_state:
        del st.session_state.logfile_base

    if "home_folder" in st.session_state:
        del st.session_state.home_folder

    if "current_folder" in st.session_state:
        del st.session_state.current_folder
    
    if "choose_src_code_file_list" in st.session_state:
        del st.session_state.choose_src_code_file_list
    
    if "object_table_selected" in st.session_state:
        del st.session_state.object_table_selected
    
    if "pickle_filename" in st.session_state:
        del st.session_state.pickle_filename

    _reset()

    CodeLineInfoRow.inferred_home_dir = None
    SnoopieObject.all_objects = {}
    CodeLineInfoRow.by_cd_index = {}
    LineInfo.saved_objects = {}
    OpInfoRow._table = []
    ObjIdRow.by_dev_offset = {}
    ObjIdRow.by_pid_offset = {}
    ObjNameRow.by_obj_id = {}
    ContextRow.by_obj_id = {}
    SiteInfoRow.by_pc = {}
    FunctionInfoRow.by_pc = {}
    
