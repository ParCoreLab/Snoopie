import argparse
import streamlit as st
from .filepath_handler import *
from .streamlit_globals import *
from .electron_checker import is_electron

_description = "Snoopie, a multigpu profiler"


def _parse():
    parser = argparse.ArgumentParser(
        description=_description,
        usage="streamlit run /path/to/parse_and_vis.py -- [optional arguments]",
    )

    # parser.add_argument(
    #     "--logfile",
    #     "-l",
    #     help="Path to the snoopie log file. Either the compressed .zst file or the decompressed file.",
    #     type=str,
    #     required=False,
    #     default="",
    # )
    parser.add_argument('files', nargs='*', help="List of logfiles. Either compressed zst or uncompressed",)
    parser.add_argument(
        "--gpu-num",
        "-n",
        help="Number of gpus the code was run on.",
        required=False,
        default=-1,
        type=int,
    )
    parser.add_argument(
        "--src-code-folder",
        "-s",
        help="Source code folder for code attribution.",
        required=False,
        default="",
        type=str,
    )
    parser.add_argument(
        "--sampling-period",
        "-p",
        help="Sampling period",
        required=False,
        default=10,
        type=int,
    )
    args = parser.parse_args()
    return args


def parse():
    if not is_electron:
        if "first_run" not in st.session_state:
            st.session_state.first_run = False
            st.session_state.show_filepicker = False

            args = _parse()

            st.session_state.gpu_num = args.gpu_num
            st.session_state.sampling_period = args.sampling_period

            src_code_folder = args.src_code_folder
            logfile = args.files
            print("PARSER logfile", logfile)

            if src_code_folder == "":
                st.session_state.show_filepicker = True
            else:
                st.session_state.home_folder = src_code_folder

            if len(logfile) == 0:
                st.session_state.show_filepicker = True
            else:
                st.session_state.logfile, st.session_state.logfile_base = multi_file_from_filepath_check(logfile)
                print (st.session_state.logfile,
                st.session_state.logfile_base
                )
                st.session_state.logfile_name = logfile
    else:  # if run in electron, there is no argument parser
        if "first_run" not in st.session_state:
            st.session_state.first_run = False
            st.session_state.show_filepicker = True
            st.session_state.gpu_num = 8
            st.session_state.sampling_period = 10
            setup_globals()
