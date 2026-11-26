"""Tests for the settings read from command-line options and conf files."""

from __future__ import annotations

import pathlib

import pytest

from phono3py.cui.phono3py_argparse import get_run_parser
from phono3py.cui.settings import Phono3pyConfParser

# Option name, which is also the conf tag, and the attribute of the settings.
BOOLEAN_OPTIONS = {
    "symmetrize_tetrahedra": "symmetrize_tetrahedra",
    "exclude_gamma_acoustic": "exclude_gamma_acoustic",
    "average_degenerate_weights": "average_degenerate_weights",
    "isotope_two_sided_weights": "use_isotope_two_sided_weights",
}
V5_OPTIONS = [
    "exclude_gamma_acoustic",
    "average_degenerate_weights",
    "isotope_two_sided_weights",
]


@pytest.mark.parametrize("name", BOOLEAN_OPTIONS)
@pytest.mark.parametrize(
    "prefix,expected", [(None, False), ("--", True), ("--no-", False)]
)
def test_boolean_option(name: str, prefix: str | None, expected: bool):
    """--name and --no-name set the setting, which is off without them."""
    options = [] if prefix is None else [prefix + name.replace("_", "-")]
    parser, _ = get_run_parser()
    args = parser.parse_args(options)
    settings = Phono3pyConfParser(args=args).settings
    assert getattr(settings, BOOLEAN_OPTIONS[name]) is expected


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
    assert getattr(settings, BOOLEAN_OPTIONS[name]) is expected


@pytest.mark.parametrize("v5", [False, True])
def test_v5_option(v5: bool):
    """--v5 turns on the options that will be on by default in v5.0."""
    parser, _ = get_run_parser()
    args = parser.parse_args(["--v5"] if v5 else [])
    settings = Phono3pyConfParser(args=args).settings
    for name in V5_OPTIONS:
        assert getattr(settings, BOOLEAN_OPTIONS[name]) is v5


@pytest.mark.parametrize("name", V5_OPTIONS)
def test_v5_option_overridden(name: str):
    """--no-name given with --v5 turns off only that option."""
    parser, _ = get_run_parser()
    args = parser.parse_args(["--v5", "--no-" + name.replace("_", "-")])
    settings = Phono3pyConfParser(args=args).settings
    for other in V5_OPTIONS:
        assert getattr(settings, BOOLEAN_OPTIONS[other]) is (other != name)


@pytest.mark.parametrize("name", V5_OPTIONS)
def test_v5_tag(tmp_path: pathlib.Path, name: str):
    """V5 = .TRUE. in a conf file, where a tag given as .FALSE. takes precedence."""
    conf = tmp_path / "phono3py.conf"
    conf.write_text(f"V5 = .TRUE.\n{name.upper()} = .FALSE.\n")
    parser, _ = get_run_parser()
    args = parser.parse_args([])
    settings = Phono3pyConfParser(filename=conf, args=args).settings
    for other in V5_OPTIONS:
        assert getattr(settings, BOOLEAN_OPTIONS[other]) is (other != name)
