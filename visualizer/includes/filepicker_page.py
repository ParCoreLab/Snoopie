import streamlit as st
from .filepath_handler import *
from . import streamlit_globals


def can_continue():
    if (
        "filepicker_button" in st.session_state
        and st.session_state.filepicker_button == True
        and "show_filepicker" in st.session_state
        and st.session_state.show_filepicker == False
    ):
        return True


def filepicker_page():
    global accept_btn, logfile, src_code_file, _sampling_period, gpu_num

    if can_continue():
        return False

    logfile = st.file_uploader("Log File", accept_multiple_files=False)
    src_code_file = st.file_uploader("Source Code File", accept_multiple_files=False)
    gpu_num = st.number_input(
        "Number of GPU's", -1, 16, -1, help="Leave -1 for automatic detection"
    )

    _sampling_period = st.number_input(
        "Sampling Period", 0, 100, streamlit_globals.sampling_period
    )
    st.session_state.filepicker_button = st.button("Start")

    if logfile != None and src_code_file != None:
        st.session_state.logfile = file_from_upload_check(logfile)
        st.session_state.logfile_name = logfile.name
        st.session_state.src_code_file = file_from_upload(src_code_file)

        st.session_state.sampling_period = _sampling_period
        st.session_state.gpu_num = gpu_num

        st.session_state.show_filepicker = False
        streamlit_globals.setup_globals()
