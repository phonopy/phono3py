"""Tests for mathematical functions in phono3py.phonon.func."""

from __future__ import annotations

import numpy as np
import pytest
from phonopy.physical_units import get_physical_units

from phono3py.phonon.func import mode_cv, sigma_squared

# Value given in the comment of sigma_squared, in AMU * Angstrom^2 * THz.
SIGMA_SQUARED_FACTOR = 1.0107576777968994


def test_mode_cv_equals_textbook_expression():
    """Test mode_cv against k_B x^2 exp(x) / (exp(x) - 1)^2 at moderate x."""
    kb = get_physical_units().KB
    temps = np.array([100.0, 300.0, 1000.0])
    freqs = np.array([0.005, 0.02, 0.05])
    x = np.divide.outer(freqs, kb * temps).T
    expected = kb * x**2 * np.exp(x) / (np.exp(x) - 1) ** 2

    cv = mode_cv(temps, freqs)

    assert cv.shape == (3, 3)
    np.testing.assert_allclose(cv, expected, rtol=1e-12)


def test_mode_cv_high_temperature_limit():
    """Test that mode_cv approaches k_B at high temperature."""
    kb = get_physical_units().KB
    cv = mode_cv(np.array([1e6]), np.array([1e-3]))
    np.testing.assert_allclose(cv, kb, rtol=1e-8)


def test_mode_cv_is_zero_where_exp_x_overflows():
    """Test that mode_cv is zero, not nan, for x above about 709."""
    with np.errstate(over="raise"):
        cv = mode_cv(np.array([0.5]), np.array([0.05, 0.5]))
    np.testing.assert_array_equal(cv, 0.0)


def test_mode_cv_is_even_in_frequency():
    """Test that a negative frequency gives the value of its absolute value."""
    temps = np.array([0.5, 300.0])
    freqs = np.array([0.02, 0.05])
    with np.errstate(over="raise"):
        cv_negative = mode_cv(temps, -freqs)
    np.testing.assert_array_equal(cv_negative, mode_cv(temps, freqs))


@pytest.mark.parametrize("temperature", [1.0, 0.0])
def test_sigma_squared_at_low_temperature(temperature: float):
    """Test that sigma_squared goes to the zero point value without an error."""
    freqs = np.array([1.0, 10.0, 20.0])
    a2 = sigma_squared(freqs, temperature)
    np.testing.assert_allclose(a2, 0.5 / freqs * SIGMA_SQUARED_FACTOR, rtol=1e-12)


def test_sigma_squared_keeps_numpy_error_state():
    """Test that the numpy error state is restored when an exception is raised."""
    state = np.geterr()
    with pytest.raises(FloatingPointError):
        sigma_squared(np.array([0.0]), 300.0)
    assert np.geterr() == state
