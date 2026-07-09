import argparse

import pytest

from roa_processor.cli import build_parser, parse_block_range


def test_parse_block_range():
    assert parse_block_range("1-50") == (1, 50)
    assert parse_block_range("001-050") == (1, 50)


def test_process_parser_accepts_block_range():
    args = build_parser().parse_args(["process", "sample_info.txt", "--range", "1-50"])

    assert args.block_range == (1, 50)


def test_plot_parser_accepts_isolated_raman_kind():
    args = build_parser().parse_args(["plot", "processed", "--kind", "isolated-raman"])

    assert args.kind == "isolated-raman"


def test_parse_block_range_rejects_invalid_values():
    with pytest.raises(argparse.ArgumentTypeError):
        parse_block_range("50-1")

    with pytest.raises(argparse.ArgumentTypeError):
        parse_block_range("1:")
