import argparse

_description = "Snoopie, a multigpu profiler"


def parse():
    parser = argparse.ArgumentParser(description=_description, usage="streamlit run /path/to/parse_and_vis.py -- [optional arguments]")

    parser.add_argument(
        "--logfile", "-l",
        help="Path to the snoopie log file. Either the compressed .zst file or the decompressed file.",
        type=str,
        required=False,
        default=""
    )
    parser.add_argument(
        "--gpu-num",
        "-n",
        help="Number of gpus the code was run on.",
        required=False,
        default=-1,
        type=int,
    )
    parser.add_argument(
        "--src-code-file",
        "-s",
        help="Source code file for code attribution.",
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
