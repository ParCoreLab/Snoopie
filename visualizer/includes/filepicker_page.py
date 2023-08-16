import streamlit as st
from .filepath_handler import *
from .streamlit_globals import *


def filepicker_page():
    logfile = st.file_uploader("Log File", accept_multiple_files=False)
    src_code_file = st.file_uploader("Source Code File", accept_multiple_files=False)
    gpu_num = st.number_input(
        "Number of GPU's", -1, 16, -1, help="Leave -1 for automatic detection"
    )
    _sampling_period = st.number_input("Sampling Period", 0, 100, sampling_period)
    accept_btn = st.button("Start")

    if accept_btn:
        print("clicked", logfile, src_code_file)
        if logfile != None and src_code_file != None:
            st.session_state.logfile = file_from_upload_check(logfile)
            st.session_state.logfile_name = logfile.name
            st.session_state.src_code_file = file_from_upload(src_code_file)

            st.session_state.sampling_period = _sampling_period
            st.session_state.gpu_num = gpu_num

            st.session_state.show_filepicker = False
            accept_btn = False
            setup_globals()
            print(st.session_state.show_filepicker)
            st.experimental_rerun()
        else:
            st.write("Please upload log file and source code files")
