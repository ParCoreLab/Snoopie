import argparse


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
        "--disable-code-attribution",
        action="store_false",
        help='Disable Code Attribution (can be helpful in cases where a ".cubin" file cannot be generated)',
    )

    args = parser.parse_args()

    print(args.command)
    print(args.kernel_name)


if __name__ == "__main__":
    main()
