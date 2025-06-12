"""Tests of functions in fc_calculator."""

import pytest
from phonopy.structure.atoms import PhonopyAtoms

from phono3py.api_phono3py import Phono3py
from phono3py.interface.fc_calculator import determine_cutoff_pair_distance


def test_determine_cutoff_pair_distance() -> None:
    """Test determine_cutoff_pair_distance."""
    cutoff = determine_cutoff_pair_distance(fc_calculator_options="|cutoff=4")
    assert cutoff == pytest.approx(4.0)

    cutoff = determine_cutoff_pair_distance(cutoff_pair_distance=5.0)
    assert cutoff == pytest.approx(5.0)

    cutoff = determine_cutoff_pair_distance(
        fc_calculator_options="|cutoff=4", cutoff_pair_distance=5.0
    )
    assert cutoff == pytest.approx(4.0)


def test_determine_cutoff_pair_distance_with_memsize(aln_cell: PhonopyAtoms) -> None:
    """Test determine_cutoff_pair_distance estimated by memsize."""
    pytest.importorskip("symfc")

    ph3 = Phono3py(aln_cell, supercell_matrix=[3, 3, 2])
    cutoff = determine_cutoff_pair_distance(
        fc_calculator="symfc",
        fc_calculator_options="|memsize=0.1",
        supercell=ph3.supercell,
        primitive=ph3.primitive,
        symmetry=ph3.symmetry,
        log_level=1,
    )
    assert cutoff == pytest.approx(3.2)

    cutoff = determine_cutoff_pair_distance(
        fc_calculator="symfc",
        symfc_memory_size=0.2,
        supercell=ph3.supercell,
        primitive=ph3.primitive,
        symmetry=ph3.symmetry,
        log_level=1,
    )
    assert cutoff == pytest.approx(3.7)

    cutoff = determine_cutoff_pair_distance(
        fc_calculator="symfc",
        fc_calculator_options="|memsize=0.1",
        symfc_memory_size=0.2,
        supercell=ph3.supercell,
        primitive=ph3.primitive,
        symmetry=ph3.symmetry,
        log_level=1,
    )
    assert cutoff == pytest.approx(3.2)

    with pytest.raises(RuntimeError):
        cutoff = determine_cutoff_pair_distance(
            fc_calculator="alm",
            fc_calculator_options="|memsize=0.1",
            symfc_memory_size=0.2,
            supercell=ph3.supercell,
            primitive=ph3.primitive,
            symmetry=ph3.symmetry,
            log_level=1,
        )
