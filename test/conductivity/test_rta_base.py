"""Unit tests for ConductivityRTABase interaction-strength branches."""

from types import MethodType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from phono3py.conductivity.rta_base import ConductivityRTABase

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class _FakeCollision:
    def __init__(self, g_zero):
        self._g_zero = g_zero
        self.pp_strength_calls = []
        self.ave_pp_calls = []
        self.run_interaction_calls = []
        self.run_calls = []
        self.imag_self_energy = np.array([0.0], dtype="double")
        self._gamma_N = np.array([0.0], dtype="double")
        self._gamma_U = np.array([0.0], dtype="double")

    def get_integration_weights(self):
        return None, self._g_zero

    def set_interaction_strength(self, pp_strength):
        self.pp_strength_calls.append(pp_strength)

    def set_averaged_pp_interaction(self, ave_pp):
        self.ave_pp_calls.append(ave_pp)

    def set_sigma(self, sigma, sigma_cutoff=None):
        self.sigma = sigma
        self.sigma_cutoff = sigma_cutoff

    def run_integration_weights(self):
        pass

    def run_interaction(self, is_full_pp=False):
        self.run_interaction_calls.append(is_full_pp)

    def run(self):
        self.run_calls.append(True)

    def get_imag_self_energy_N_and_U(self):
        return self._gamma_N, self._gamma_U


def _make_dummy(overrides=None):
    """Return a minimal SimpleNamespace that satisfies ConductivityRTABase methods."""
    base = dict(
        _read_pp=False,
        _use_ave_pp=False,
        _use_const_ave_pp=False,
        _is_full_pp=False,
        _store_pp=False,
        _sigma_cutoff=None,
        _sigmas=[None],
        _log_level=0,
        _pp=SimpleNamespace(
            mesh_numbers=np.array([2, 2, 2], dtype="int64"),
            constant_averaged_interaction=None,
            averaged_interaction=np.array([0.1], dtype="double"),
            interaction_strength=np.zeros((1, 3, 3), dtype="double"),
        ),
        _grid_points=np.array([3], dtype="int64"),
        _pp_filename=None,
        _collision=_FakeCollision(g_zero=np.zeros(3, dtype="int64")),
        _temperatures=np.array([100.0, 300.0], dtype="double"),
        _gamma=np.zeros((1, 2, 1, 1), dtype="double"),
        _gamma_N=np.zeros((1, 2, 1, 1), dtype="double"),
        _gamma_U=np.zeros((1, 2, 1, 1), dtype="double"),
        _is_gamma_detail=False,
        _is_N_U=False,
        _averaged_pp_interaction=np.zeros((1, 1), dtype="double"),
        _read_gamma=False,
        _read_gamma_iso=False,
        _is_isotope=False,
    )
    if overrides:
        base.update(overrides)
    dummy = SimpleNamespace(**base)
    _bind_methods(dummy)
    return dummy


def _bind_methods(dummy):
    helper_names = (
        "_run_sigma_at_grid_point",
        "_show_gamma_sigma_log",
        "_set_interaction_strength_at_sigma",
        "_set_interaction_strength_from_file",
        "_allocate_gamma_detail_at_q_if_needed",
        "_run_collisions_at_temperatures",
        "_requires_full_gamma_path",
    )
    for name in helper_names:
        if hasattr(ConductivityRTABase, name):
            setattr(dummy, name, MethodType(getattr(ConductivityRTABase, name), dummy))


# ---------------------------------------------------------------------------
# Existing tests (preserved)
# ---------------------------------------------------------------------------


def test_set_gamma_at_sigmas_read_pp_uses_pp_from_file(monkeypatch):
    """`_set_gamma_at_sigmas` forwards read pp-strength to collision object."""
    pp_from_file = np.array([1.0, 2.0], dtype="double")
    g_zero_runtime = np.array([1, 0, 1], dtype="int64")

    monkeypatch.setattr(
        "phono3py.conductivity.rta_base.read_pp_from_hdf5",
        lambda *args, **kwargs: (pp_from_file, None),
    )

    fake_collision = _FakeCollision(g_zero=g_zero_runtime)
    dummy = SimpleNamespace(
        _read_pp=True,
        _use_ave_pp=False,
        _use_const_ave_pp=False,
        _is_full_pp=False,
        _sigma_cutoff=None,
        _sigmas=[None],
        _log_level=0,
        _pp=SimpleNamespace(mesh_numbers=np.array([2, 2, 2], dtype="int64")),
        _grid_points=np.array([3], dtype="int64"),
        _pp_filename=None,
        _collision=fake_collision,
        _temperatures=np.array([300.0], dtype="double"),
        _gamma=np.zeros((1, 1, 1, 1), dtype="double"),
        _is_gamma_detail=False,
        _is_N_U=False,
    )
    _bind_methods(dummy)
    dummy_as_any: Any = dummy

    ConductivityRTABase._set_gamma_at_sigmas(dummy_as_any, 0)

    assert len(fake_collision.pp_strength_calls) == 1
    np.testing.assert_allclose(fake_collision.pp_strength_calls[0], pp_from_file)


def test_set_gamma_at_sigmas_use_ave_pp_uses_averaged_pp():
    """`_set_gamma_at_sigmas` forwards averaged pp interaction."""
    averaged_pp = np.array([[5.0, 6.0]], dtype="double")
    fake_collision = _FakeCollision(g_zero=np.array([0], dtype="int64"))

    dummy = SimpleNamespace(
        _read_pp=False,
        _use_ave_pp=True,
        _use_const_ave_pp=False,
        _is_full_pp=False,
        _sigma_cutoff=None,
        _sigmas=[0.1],
        _log_level=0,
        _averaged_pp_interaction=averaged_pp,
        _collision=fake_collision,
        _temperatures=np.array([300.0], dtype="double"),
        _gamma=np.zeros((1, 1, 1, 1), dtype="double"),
        _is_gamma_detail=False,
        _is_N_U=False,
    )
    _bind_methods(dummy)
    dummy_as_any: Any = dummy

    ConductivityRTABase._set_gamma_at_sigmas(dummy_as_any, 0)

    assert len(fake_collision.ave_pp_calls) == 1
    np.testing.assert_allclose(fake_collision.ave_pp_calls[0], averaged_pp[0])


# ---------------------------------------------------------------------------
# _requires_full_gamma_path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "flags,expected",
    [
        (dict(_is_full_pp=True), True),
        (dict(_read_pp=True), True),
        (dict(_use_ave_pp=True), True),
        (dict(_is_gamma_detail=True), True),
        ({}, False),  # all False → low-memory path
    ],
)
def test_requires_full_gamma_path(flags, expected):
    """_requires_full_gamma_path returns True when any triggering flag is set."""
    base_flags = dict(
        _is_full_pp=False,
        _read_pp=False,
        _store_pp=False,
        _use_ave_pp=False,
        _use_const_ave_pp=False,
        _is_gamma_detail=False,
    )
    base_flags.update(flags)
    dummy = _make_dummy(base_flags)
    assert dummy._requires_full_gamma_path() is expected


# ---------------------------------------------------------------------------
# _set_interaction_strength_at_sigma
# ---------------------------------------------------------------------------


def test_set_interaction_strength_at_sigma_runs_interaction_at_first_sigma():
    """At i_sigma==0 with no flags, run_interaction is called."""
    dummy = _make_dummy()
    dummy._set_interaction_strength_at_sigma(0, 0, None)
    assert len(dummy._collision.run_interaction_calls) == 1


def test_set_interaction_strength_at_sigma_reuses_at_second_sigma():
    """At i_sigma!=0 with tetrahedron method, run_interaction is NOT called again."""
    dummy = _make_dummy({"_sigmas": [None, None]})
    dummy._set_interaction_strength_at_sigma(0, 1, None)
    assert len(dummy._collision.run_interaction_calls) == 0


def test_set_interaction_strength_at_sigma_const_ave_pp(monkeypatch):
    """With use_const_ave_pp, run_interaction is called and averaged_pp is stored."""
    dummy = _make_dummy({"_use_const_ave_pp": True})
    dummy._pp.constant_averaged_interaction = 1e-5
    dummy._set_interaction_strength_at_sigma(0, 0, None)
    assert len(dummy._collision.run_interaction_calls) == 1
    np.testing.assert_allclose(
        dummy._averaged_pp_interaction[0], dummy._pp.averaged_interaction
    )


def test_set_interaction_strength_at_sigma_full_pp_stores_averaged(monkeypatch):
    """With is_full_pp, averaged_pp_interaction is stored after run_interaction."""
    dummy = _make_dummy({"_is_full_pp": True})
    dummy._set_interaction_strength_at_sigma(0, 0, None)
    assert len(dummy._collision.run_interaction_calls) == 1
    np.testing.assert_allclose(
        dummy._averaged_pp_interaction[0], dummy._pp.averaged_interaction
    )


# ---------------------------------------------------------------------------
# _set_interaction_strength_from_file
# ---------------------------------------------------------------------------


def test_set_interaction_strength_from_file_raises_on_g_zero_mismatch(monkeypatch):
    """ValueError is raised when g_zero from file differs from runtime g_zero."""
    pp_from_file = np.array([1.0], dtype="double")
    g_zero_from_file = np.array([1, 0], dtype="int64")

    monkeypatch.setattr(
        "phono3py.conductivity.rta_base.read_pp_from_hdf5",
        lambda *args, **kwargs: (pp_from_file, g_zero_from_file),
    )

    g_zero_runtime = np.array([0, 0], dtype="int64")  # different from file
    dummy = _make_dummy(
        {
            "_read_pp": True,
            "_collision": _FakeCollision(g_zero=g_zero_runtime),
        }
    )

    with pytest.raises(ValueError, match="g_zero"):
        dummy._set_interaction_strength_from_file(0, None)


def test_set_interaction_strength_from_file_passes_matching_g_zero(monkeypatch):
    """No error raised when g_zero from file matches runtime g_zero."""
    pp_from_file = np.array([1.0, 2.0], dtype="double")
    g_zero_matching = np.array([0, 1], dtype="int64")

    monkeypatch.setattr(
        "phono3py.conductivity.rta_base.read_pp_from_hdf5",
        lambda *args, **kwargs: (pp_from_file, g_zero_matching),
    )

    dummy = _make_dummy(
        {
            "_read_pp": True,
            "_collision": _FakeCollision(g_zero=g_zero_matching),
        }
    )
    dummy._set_interaction_strength_from_file(0, None)

    assert len(dummy._collision.pp_strength_calls) == 1
    np.testing.assert_allclose(dummy._collision.pp_strength_calls[0], pp_from_file)


# ---------------------------------------------------------------------------
# _allocate_gamma_detail_at_q_if_needed
# ---------------------------------------------------------------------------


def test_allocate_gamma_detail_skipped_when_not_requested():
    """_allocate_gamma_detail_at_q_if_needed does nothing when flag is False."""
    dummy = _make_dummy({"_is_gamma_detail": False, "_gamma_detail_at_q": None})
    dummy._allocate_gamma_detail_at_q_if_needed()
    assert dummy._gamma_detail_at_q is None


def test_allocate_gamma_detail_allocates_when_requested():
    """_allocate_gamma_detail_at_q_if_needed allocates array matching pp shape."""
    n_temp = 2
    pp_shape = (5, 3, 3)
    dummy = _make_dummy(
        {
            "_is_gamma_detail": True,
            "_gamma_detail_at_q": None,
            "_temperatures": np.array([100.0, 300.0]),
            "_pp": SimpleNamespace(
                mesh_numbers=np.array([2, 2, 2]),
                interaction_strength=np.zeros(pp_shape),
                constant_averaged_interaction=None,
                averaged_interaction=np.zeros(1),
            ),
        }
    )
    dummy._allocate_gamma_detail_at_q_if_needed()
    assert dummy._gamma_detail_at_q is not None
    assert dummy._gamma_detail_at_q.shape == (n_temp,) + pp_shape
    assert (dummy._gamma_detail_at_q == 0).all()


# ---------------------------------------------------------------------------
# _run_collisions_at_temperatures
# ---------------------------------------------------------------------------


def test_run_collisions_at_temperatures_stores_gamma():
    """_run_collisions_at_temperatures populates gamma for each temperature."""
    fake_collision = _FakeCollision(g_zero=np.zeros(1, dtype="int64"))
    fake_collision.imag_self_energy = np.array([0.5], dtype="double")

    dummy = _make_dummy(
        {
            "_collision": fake_collision,
            "_temperatures": np.array([100.0, 200.0, 300.0]),
            "_gamma": np.zeros((1, 3, 1, 1), dtype="double"),
            "_is_N_U": False,
            "_is_gamma_detail": False,
        }
    )
    dummy._run_collisions_at_temperatures(0, 0)

    assert len(fake_collision.run_calls) == 3
    np.testing.assert_allclose(dummy._gamma[0, :, 0, :], 0.5)


def test_run_collisions_at_temperatures_stores_N_U():
    """_run_collisions_at_temperatures populates gamma_N and gamma_U."""
    fake_collision = _FakeCollision(g_zero=np.zeros(1, dtype="int64"))
    fake_collision.imag_self_energy = np.array([0.3], dtype="double")
    fake_collision._gamma_N = np.array([0.1], dtype="double")
    fake_collision._gamma_U = np.array([0.2], dtype="double")

    dummy = _make_dummy(
        {
            "_collision": fake_collision,
            "_temperatures": np.array([300.0]),
            "_gamma": np.zeros((1, 1, 1, 1), dtype="double"),
            "_gamma_N": np.zeros((1, 1, 1, 1), dtype="double"),
            "_gamma_U": np.zeros((1, 1, 1, 1), dtype="double"),
            "_is_N_U": True,
            "_is_gamma_detail": False,
        }
    )
    dummy._run_collisions_at_temperatures(0, 0)

    np.testing.assert_allclose(dummy._gamma_N[0, 0, 0], 0.1)
    np.testing.assert_allclose(dummy._gamma_U[0, 0, 0], 0.2)
