"""Unit tests for ConductivityRTABase interaction-strength branches."""

from types import MethodType, SimpleNamespace
from typing import Any

import numpy as np

from phono3py.conductivity.rta_base import ConductivityRTABase


class _FakeCollision:
    def __init__(self, g_zero):
        self._g_zero = g_zero
        self.pp_strength_calls = []
        self.ave_pp_calls = []

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

    def run(self):
        self.imag_self_energy = np.array([0.0], dtype="double")


def _bind_extracted_helpers_if_available(dummy):
    helper_names = (
        "_run_sigma_at_grid_point",
        "_show_gamma_sigma_log",
        "_set_interaction_strength_at_sigma",
        "_set_interaction_strength_from_file",
        "_allocate_gamma_detail_at_q_if_needed",
        "_run_collisions_at_temperatures",
    )
    for name in helper_names:
        if hasattr(ConductivityRTABase, name):
            setattr(dummy, name, MethodType(getattr(ConductivityRTABase, name), dummy))


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
    _bind_extracted_helpers_if_available(dummy)
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
    _bind_extracted_helpers_if_available(dummy)
    dummy_as_any: Any = dummy

    ConductivityRTABase._set_gamma_at_sigmas(dummy_as_any, 0)

    assert len(fake_collision.ave_pp_calls) == 1
    np.testing.assert_allclose(fake_collision.ave_pp_calls[0], averaged_pp[0])
