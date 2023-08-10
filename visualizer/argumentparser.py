import argparse

_description = "Snoopie, a multigpu profiler"


def parse():
    parser = argparse.ArgumentParser(description=_description, usage="streamlit run /path/to/parse_and_vis.py -- [optional arguments] logfile")

    parser.add_argument(
        "logfile",
        help="Path to the snoopie log file. Either the compressed .zst file or the decompressed file.",
        type=str,
    )
    parser.add_argument(
        "--gpu-num",
        "-n",
        help="Number of gpus the code was run on.",
        required=False,
        default=4,
        type=int,
    )
    parser.add_argument(
        "--src-code-file",
        "-s",
        help="Source code file for code attribution.",
        required=False,
        default="jacobi.cu",
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
