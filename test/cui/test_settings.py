"""Tests for the settings read from command-line options and conf files."""

from __future__ import annotations

import pathlib

import pytest

from phono3py.cui.phono3py_argparse import get_run_parser
from phono3py.cui.settings import Phono3pyConfParser


@pytest.mark.parametrize(
    "options,expected",
    [
        ([], False),
        (["--symmetrize-tetrahedra"], True),
        (["--no-symmetrize-tetrahedra"], False),
    ],
)
def test_symmetrize_tetrahedra_option(options: list[str], expected: bool):
    """--symmetrize-tetrahedra and --no-symmetrize-tetrahedra set the setting."""
    parser, _ = get_run_parser()
    args = parser.parse_args(options)
    assert Phono3pyConfParser(args=args).settings.symmetrize_tetrahedra is expected


@pytest.mark.parametrize(
    "options,expected",
    [([], True), (["--no-symmetrize-tetrahedra"], False)],
)
def test_symmetrize_tetrahedra_tag(
    tmp_path: pathlib.Path, options: list[str], expected: bool
):
    """SYMMETRIZE_TETRAHEDRA is read, and the command-line option overrides it."""
    conf = tmp_path / "phono3py.conf"
    conf.write_text("SYMMETRIZE_TETRAHEDRA = .TRUE.\n")
    parser, _ = get_run_parser()
    args = parser.parse_args(options)
    settings = Phono3pyConfParser(filename=conf, args=args).settings
    assert settings.symmetrize_tetrahedra is expected
