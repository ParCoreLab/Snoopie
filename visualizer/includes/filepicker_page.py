import streamlit as st
from .filepath_handler import *
from .streamlit_globals import *
import subprocess
import os

_logfile = None
_src_code_file = None
_sampling_period = None
_gpu_num = None
_home_folder = None


def folder_choose_dialog():
    path = os.path.abspath("tk_folder_chooser")
    p = subprocess.Popen(
        ["python3", "tk_folder_chooser.py"],
        cwd=path,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    result, error = p.communicate()
    p.terminate()
    print("p:", result, error)
    if isinstance(result, bytes):
        result = result.decode("utf-8")
    if isinstance(result, str):
        return result
    else:
        return "-1"


def filepicker_page():
    global _logfile, _src_code_file, _sampling_period, _gpu_num, _home_folder

    if ("logfile" not in st.session_state):
        _logfile = st.file_uploader("Log File", accept_multiple_files=True)
    # _src_code_file = st.file_uploader("Source Code File", accept_multiple_files=False)

    home_folder_choose_cols = st.columns([1, 4])
    
    choose_home_folder_btn = st.button(
        "Choose source code folder",
        help="Do not use this button if the profiler is running on a remote machine. Fill the folder path manually instead.",
    )
    if choose_home_folder_btn:
        res = folder_choose_dialog()
        print("res:", res)
        if res != "-1":
            _home_folder = res
            print(_home_folder)

    _home_folder = st.text_input(
        "Home folder of source code:",
        value=os.getcwd() if _home_folder is None else _home_folder,
    )

    _gpu_num = st.number_input(
        "Number of GPU's (leave -1 for automatic detection)", -1, 16, -1, help="Leave -1 for automatic detection"
    )

    _sampling_period = st.number_input("Sampling Period (extrapolates the data if sampling was used)", 0, 100, 1)
    filepicker_button = st.button("Start")

    if (
        (_logfile != None or "logfile" in st.session_state)
        and not (isinstance(_logfile, list) and len(_logfile) == 0)
        and filepicker_button == 1
    ):
        if ("logfile" not in st.session_state):
            (
                st.session_state.logfile,
                st.session_state.logfile_base,
            ) = multi_file_from_upload_check(_logfile)
            if isinstance(_logfile, list):
                st.session_state.logfile_name = [i.name for i in _logfile]
            else:
                st.session_state.logfile_name = _logfile.name
            # st.session_state.src_code_file, _ = file_from_upload(_src_code_file)

        st.session_state.sampling_period = _sampling_period
        st.session_state.gpu_num = _gpu_num
        st.session_state.home_folder = _home_folder

        st.session_state.show_filepicker = False
        setup_globals()
        st.experimental_rerun()
