"""Tests for the settings read from command-line options and conf files."""

from __future__ import annotations

import pathlib

import pytest

from phono3py.cui.phono3py_argparse import get_run_parser
from phono3py.cui.settings import Phono3pyConfParser

BOOLEAN_OPTIONS = ["symmetrize_tetrahedra", "exclude_gamma_acoustic"]


@pytest.mark.parametrize("name", BOOLEAN_OPTIONS)
@pytest.mark.parametrize(
    "prefix,expected", [(None, False), ("--", True), ("--no-", False)]
)
def test_boolean_option(name: str, prefix: str | None, expected: bool):
    """--name and --no-name set the setting, which is off without them."""
    options = [] if prefix is None else [prefix + name.replace("_", "-")]
    parser, _ = get_run_parser()
    args = parser.parse_args(options)
    assert getattr(Phono3pyConfParser(args=args).settings, name) is expected


@pytest.mark.parametrize("name", BOOLEAN_OPTIONS)
@pytest.mark.parametrize("override,expected", [(False, True), (True, False)])
def test_boolean_tag(tmp_path: pathlib.Path, name: str, override: bool, expected: bool):
    """The tag in a conf file is read, and --no-name overrides it."""
    conf = tmp_path / "phono3py.conf"
    conf.write_text(f"{name.upper()} = .TRUE.\n")
    options = ["--no-" + name.replace("_", "-")] if override else []
    parser, _ = get_run_parser()
    args = parser.parse_args(options)
    settings = Phono3pyConfParser(filename=conf, args=args).settings
    assert getattr(settings, name) is expected
