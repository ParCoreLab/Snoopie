import streamlit as st
from .filepath_handler import *
from .streamlit_globals import *



_logfile = None
_src_code_file = None
_sampling_period = None
_gpu_num = None


def filepicker_page():
    global _logfile, _src_code_file, _sampling_period, _gpu_num

    _logfile = st.file_uploader("Log File", accept_multiple_files=False)
    _src_code_file = st.file_uploader("Source Code File", accept_multiple_files=False)
    _gpu_num = st.number_input(
        "Number of GPU's", -1, 16, -1, help="Leave -1 for automatic detection"
    )

    _sampling_period = st.number_input(
        "Sampling Period", 0, 100, 10
    )
    filepicker_button = st.button("Start")

    if _logfile != None and _src_code_file != None and filepicker_button == 1:
        st.session_state.logfile, st.session_state.logfile_base = file_from_upload_check(_logfile)
        st.session_state.logfile_name = _logfile.name
        st.session_state.src_code_file, _ = file_from_upload(_src_code_file)

        st.session_state.sampling_period = _sampling_period
        st.session_state.gpu_num = _gpu_num

        st.session_state.show_filepicker = False
        setup_globals()
        st.experimental_rerun()
        
