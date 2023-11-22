#!/usr/bin/env python3

import shutil
import subprocess
import argparse
import glob
import time
import os


def main():
    parser = argparse.ArgumentParser(
        description="Snoopie Multi-GPU Communication Profiling Tool"
    )

    parser.add_argument(
        "--kernel-name",
        metavar="KERNEL_NAME",
        default="all",
        help="Kernel Name to Instrument",
    )

    parser.add_argument(
        "--nvshmem-version",
        choices=["2.7", "2.8", "2.9"],
        metavar="NVSHMEM_VERSION",
        default="2.8",
        help="Specify the nvshmem version used (if any)",
    )

    parser.add_argument(
        "--nvshmem-ngpus",
        metavar="NVSHMEM_NGPUS",
        default="8",
        help="Specify the number of GPUs nvshmem is using (only required for nvshmem 2.8 and later)",
    )

    parser.add_argument(
        "--filtering-location",
        metavar="FILTERING_LOCATION",
        choices=["device", "host"],
        default="device",
        help="Specify where the addresses will be filtered (does not affect the results, only performance)",
    )

    parser.add_argument(
        "--sample-size",
        metavar="SAMPLE_SIZE",
        default="1",
        help="Enable Sampling by setting the sample size (if 100, it means 1/100 of population is sampled)",
    )

    parser.add_argument(
        "command",
        metavar="COMMAND",
        nargs="+",
        help="Specify the Kernel Name to Instrument (all are instrumented by default)",
    )

    parser.add_argument(
        "--verbose", action="store_true", help="Enable NVBIT Verbose Mode"
    )

    parser.add_argument(
        "--disable-logs", action="store_true", help="Disable storing the logs into a log file (debugging flag)"
    )

    parser.add_argument(
        "--disable-code-attribution",
        action="store_false",
        help='Disable Code Attribution (can be helpful in cases where a ".cubin" file cannot be generated)',
    )

    args = parser.parse_args()

    run_snoopie(args)

def run_snoopie(args):
    env = os.environ.copy()
    env["LD_PRELOAD"] = env["SNOOPIE_PATH"]

    if (args.verbose):
        env["TOOL_VERBOSE"] = "1"

    if (args.disable_logs):
        env["SILENT"] = "1"

    if (args.disable_code_attribution):
        env["CODE_ATTRIBUTION"] = "0"

    if (args.sample_size):
        env["SAMPLE_SIZE"] = args.sample_size

    if (args.filtering_location == "device"):
        env["ON_DEVICE_FILTERING"] = "1"

    if (args.filtering_location == "host"):
        env["ON_DEVICE_FILTERING"] = "0"

    if (args.nvshmem_ngpus):
        env["NVSHMEM_NGPUS"] = args.nvshmem_ngpus

    if (args.nvshmem_version):
        env["NVSHMEM_VERSION"] = args.nvshmem_version

    if (args.kernel_name):
        env["KERNEL_NAME"] = args.kernel_name


    cmd = " ".join(args.command)

    print(f"Running {cmd}")
    snooped_command = subprocess.Popen(args.command, env=dict(env))
    snooped_command.wait()
    print(f"Finished Execution of {cmd}")

    print(f"Creating results file")
    log_files = glob.glob(f"*{snooped_command.pid}*")
    os.mkdir(f"./results-{snooped_command.pid}")

    for f in log_files:
        shutil.move(f, f"./results-{snooped_command.pid}")

    shutil.make_archive(f"./results-{snooped_command.pid}", "zip", f"./results-{snooped_command.pid}")
    shutil.rmtree(f"./results-{snooped_command.pid}")

if __name__ == "__main__":
    main()
