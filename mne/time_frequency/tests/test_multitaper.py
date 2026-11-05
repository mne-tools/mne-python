# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal

from mne.time_frequency import psd_array_multitaper
from mne.time_frequency.multitaper import _psd_from_mt, dpss_windows
from mne.utils import _record_warnings


def test_dpss_windows():
    """Test computation of DPSS windows."""
    ni = pytest.importorskip("nitime")

    N = 1000
    half_nbw = 4
    Kmax = int(2 * half_nbw)

    dpss, eigs = dpss_windows(N, half_nbw, Kmax, low_bias=False)
    with _record_warnings():  # conversions
        dpss_ni, eigs_ni = ni.algorithms.dpss_windows(N, half_nbw, Kmax)

    assert_array_almost_equal(dpss, dpss_ni)
    assert_array_almost_equal(eigs, eigs_ni)

    dpss, eigs = dpss_windows(N, half_nbw, Kmax, low_bias=False)
    with _record_warnings():  # conversions
        dpss_ni, eigs_ni = ni.algorithms.dpss_windows(N, half_nbw, Kmax)

    assert_array_almost_equal(dpss, dpss_ni)
    assert_array_almost_equal(eigs, eigs_ni)


@pytest.mark.parametrize("n_times", (100, 101))
@pytest.mark.parametrize("adaptive, n_jobs", [(False, 1), (True, 1), (True, 2)])
def test_multitaper_psd(n_times, adaptive, n_jobs):
    """Test multi-taper PSD computation."""
    ni = pytest.importorskip("nitime")

    n_channels = 5
    data = np.random.default_rng(0).random((n_channels, n_times))
    sfreq = 500
    with pytest.raises(ValueError, match="Invalid value for the 'normaliza"):
        psd_array_multitaper(data, sfreq, normalization="foo")
    # compute with MNE
    psd, freqs = psd_array_multitaper(
        data, sfreq, adaptive=adaptive, n_jobs=n_jobs, normalization="full"
    )
    # compute with nitime
    freqs_ni, psd_ni, _ = ni.algorithms.spectral.multi_taper_psd(
        data, sfreq, adaptive=adaptive, jackknife=False
    )
    # compare
    assert_array_almost_equal(psd, psd_ni, decimal=4)
    del freqs, freqs_ni
    # assert_array_equal(freqs, freqs_ni)
    # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
    # this is commented out because nitime's freq calculations differ from ours
    # so there's no point checking (theirs are wrong; sometimes they return a
    # freq component at exactly sfreq/2 when they shouldn't)
    # nitime  →  np.linspace(0, sfreq / 2, n_times // 2 + 1)
    # mne     →  scipy.fft.rfftfreq(n_times, 1. / sfreq)

    # test with bad bandwidth
    with pytest.raises(ValueError, match="use a value of at least"):
        psd_array_multitaper(data, sfreq, bandwidth=4.9)


def test_adaptive_weights_convergence():
    """Test convergence and lack of convergence when setting adaptive=True."""
    data = np.random.default_rng(0).random((5, 100))
    sfreq = 500
    with pytest.warns(
        RuntimeWarning, match="Iterative multi-taper PSD computation did not converge."
    ):
        psd_array_multitaper(data, sfreq, adaptive=True, max_iter=2)
    psd_array_multitaper(data, sfreq, adaptive=True, max_iter=200)


def test_multitaper_complex_full_normalization():
    """Test full normalization of complex multitaper coefficients."""
    sfreq = 100.0
    data = np.random.default_rng(0).standard_normal((1, 1000))
    psd_length, freqs = psd_array_multitaper(
        data, sfreq, normalization="length", output="power", verbose=False
    )
    psd_full, freqs_full = psd_array_multitaper(
        data, sfreq, normalization="full", output="power", verbose=False
    )
    assert_allclose(freqs, freqs_full)
    assert_allclose(psd_full, psd_length / sfreq)

    coefs_length, _, _ = psd_array_multitaper(
        data, sfreq, normalization="length", output="complex", verbose=False
    )
    for normalization, scale in (("length", 1.0), ("full", np.sqrt(sfreq))):
        coefs, _, weights = psd_array_multitaper(
            data, sfreq, normalization=normalization, output="complex", verbose=False
        )
        psd, _ = psd_array_multitaper(
            data, sfreq, normalization=normalization, output="power", verbose=False
        )
        assert_allclose(_psd_from_mt(coefs, weights), psd)
        assert_allclose(np.abs(coefs), np.abs(coefs_length) / scale)
