import argparse

import pytest

from roa_processor.cli import (
    build_parser,
    parse_block_range,
    parse_non_negative_float,
    parse_non_negative_int,
)


def test_parse_block_range():
    assert parse_block_range("1-50") == (1, 50)
    assert parse_block_range("001-050") == (1, 50)


def test_process_parser_accepts_block_range():
    args = build_parser().parse_args(["process", "sample_info.txt", "--range", "1-50"])

    assert args.block_range == (1, 50)


def test_process_parser_defaults_output_to_processed():
    args = build_parser().parse_args(["process", "sample_info.txt"])

    assert args.output == "processed"


def test_process_parser_accepts_highest_noise_reject_blocks():
    args = build_parser().parse_args(
        ["process", "sample_info.txt", "--roa-highest-noise-reject-blocks", "2"]
    )

    assert args.roa_highest_noise_reject_blocks == 2


def test_process_parser_accepts_manual_raman_component_subtraction():
    args = build_parser().parse_args(
        [
            "process",
            "sample_info.txt",
            "--raman-component-subtraction",
            "manual",
            "--water-scale",
            "1.2",
            "--quartz-scale",
            "0",
            "--air-scale",
            "2.5",
        ]
    )

    assert args.raman_component_subtraction == "manual"
    assert args.water_scale == 1.2
    assert args.quartz_scale == 0
    assert args.air_scale == 2.5


def test_plot_parser_accepts_isolated_raman_kind():
    args = build_parser().parse_args(["plot", "processed", "--kind", "isolated-raman"])

    assert args.kind == "isolated-raman"


def test_parse_block_range_rejects_invalid_values():
    with pytest.raises(argparse.ArgumentTypeError):
        parse_block_range("50-1")

    with pytest.raises(argparse.ArgumentTypeError):
        parse_block_range("1:")


def test_parse_non_negative_int_rejects_negative_values():
    with pytest.raises(argparse.ArgumentTypeError):
        parse_non_negative_int("-1")


def test_parse_non_negative_float_rejects_negative_values():
    with pytest.raises(argparse.ArgumentTypeError):
        parse_non_negative_float("-0.1")
