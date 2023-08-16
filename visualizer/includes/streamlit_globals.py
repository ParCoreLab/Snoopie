import streamlit as st

#####################################
gpu_num = -1  
sampling_period = 10

# these two variables should be open files or None
src_code_file = None            
logfile = None
logfile_name = None
#####################################

def setup_globals():
    global gpu_num, src_code_file, sampling_period, logfile, logfile_name

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