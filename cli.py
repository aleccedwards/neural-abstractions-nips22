# Neural Abstractions
# Copyright (c) 2022  Alessandro Abate, Alec Edwards, Mirco Giacobbe

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

import configargparse
import config
import yaml


def parse_command(args=None):
    parser = configargparse.ArgParser(
        default_config_files=["config.yaml"],
        config_file_parser_class=configargparse.YAMLConfigFileParser,
    )

    parser.add_argument("-c", "--config", is_config_file=True, help="config file path")

    parser.add_argument(
        "-e",
        "--error",
        help="Maximum error of ||f(x) - nn(x)||**2",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--stopping-criterion",
        help="Criteria for when to stop learning procedure",
        type=yaml.safe_load,
    )
    parser.add_argument(
        "-w",
        "--width",
        help="Width of relu layer in network",
        nargs="+",
        type=int,
        default=15,
    )
    parser.add_argument(
        "-b",
        "--benchmark",
        help="Benchmark to perform abstraction on",
        type=str,
        required=True,
        choices=(
            "lin",
            "exp",
            "steam",
            "jet",
            "nl1",
            "nl2",
            "tank",
        ),
    )

    parser.add_argument(
        "-t",
        "--timeout",
        help="If passed, synthesis will run until no success in time interval.\
             Default 120s",
        action="store_true",
        required=False,
    )

    parser.add_argument("--timeout-duration", help="Duration of timeout")

    parser.add_argument(
        "-s",
        "--save-net",
        help="Save state-dict of synthesised network",
        action="store_true",
        required=False,
    )

    parser.add_argument(
        "--seed",
        help="Shifts set random seed by given value",
        type=int,
        required=False,
        default=0,
    )

    parser.add_argument(
        "--scalar",
        help="Run in scalar mode where one net learns each fi",
        required=False,
        action="store_true",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        help="Supress Stdout",
        required=False,
        action="store_false",
        default=True,
    )

    parser.add_argument(
        "-i",
        "--iterative",
        help="Run iteratively on success to reduce error",
        action="store_true",
        required=False,
    )

    parser.add_argument(
        "--reduction", help="factor to reduce error iteratively by", required=False
    )

    parser.add_argument(
        "--output-type",
        help="Store abstraction in xml file or produce plot",
        required=False,
        nargs="+",
        type=str,
        default="plot",
        choices=("None", "xml", "plot", "csv", "pkl"),
    )

    parser.add_argument(
        "-f",
        "--output-file",
        help="Output file name without extension.\
             If not given, file will be defualt or plot will be shown",
        type=str,
        required=False,
    )

    parser.add_argument(
        "-r",
        "--repeat",
        help="repeat with same settings for consecutive seeds",
        required=False,
        default=1,
    )

    parser.add_argument(
        "--optimizer",
        help="optimizer to use for learning and settings",
        type=yaml.safe_load,
    )

    parser.add_argument(
        "--bounded-time",
        help="When creating xml file, set time bound for safety verification",
        action="store_true",
    )

    parser.add_argument(
        "--initial",
        help="initial states for safety verification",
        type=yaml.safe_load,
        default=[],
    )

    parser.add_argument(
        "--forbidden",
        help="forbidden states for safety verification",
        type=list,
        default=[],
    )

    parser.add_argument("--time-horizon", help="Time horizon for safety verification")
    args = parser.parse_args(args)
    args = get_name(args)
    return args


def get_config():
    """Read CLI arguments and return a config object"""
    args = parse_command()
    c = config.Config(args)
    return c


def get_default_config():
    """Convenience function to return default config for testing purposes"""
    args = parse_command(["-c", "config.yaml"])
    c = config.Config(args)
    return c


def get_name(args):
    args.fname = (
        "(benchmark="
        + str(args.benchmark)
        + ", error="
        + str(args.error)
        + ", width="
        + str(args.width)
        + ", seed="
        + str(args.seed)
        + ")"
    )
    return args




if __name__ == "__main__":
    config = get_default_config()
    print(config)
