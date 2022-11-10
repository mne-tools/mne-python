# -*- coding: utf-8 -*-
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from mne.time_frequency import psd_array_multitaper
from mne.time_frequency.multitaper import dpss_windows
from mne.utils import requires_nitime, _record_warnings


@requires_nitime
def test_dpss_windows():
    """Test computation of DPSS windows."""
    import nitime as ni
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

    with pytest.warns(FutureWarning, match='``interp_from`` option is deprec'):
        dpss_windows(N, half_nbw, Kmax, interp_from=200)
    with pytest.warns(FutureWarning, match='``interp_kind`` option is deprec'):
        dpss_windows(N, half_nbw, Kmax, interp_kind='linear')


@requires_nitime
@pytest.mark.parametrize('n_times', (100, 101))
@pytest.mark.parametrize('adaptive, n_jobs',
                         [(False, 1), (True, 1), (True, 2)])
def test_multitaper_psd(n_times, adaptive, n_jobs):
    """Test multi-taper PSD computation."""
    import nitime as ni
    n_channels = 5
    data = np.random.default_rng(0).random((n_channels, n_times))
    sfreq = 500
    with pytest.raises(ValueError, match="Invalid value for the 'normaliza"):
        psd_array_multitaper(data, sfreq, normalization='foo')
    # compute with MNE
    psd, freqs = psd_array_multitaper(
        data, sfreq, adaptive=adaptive, n_jobs=n_jobs, normalization='full')
    # compute with nitime
    freqs_ni, psd_ni, _ = ni.algorithms.spectral.multi_taper_psd(
        data, sfreq, adaptive=adaptive, jackknife=False)
    # compare
    assert_array_almost_equal(psd, psd_ni, decimal=4)
    # assert_array_equal(freqs, freqs_ni)
    # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
    # this is commented out because nitime's freq calculations differ from ours
    # so there's no point checking (theirs are wrong; sometimes they return a
    # freq component at exactly sfreq/2 when they shouldn't)
    # nitime  →  np.linspace(0, sfreq / 2, n_times // 2 + 1)
    # mne     →  scipy.fft.rfftfreq(n_times, 1. / sfreq)

    # test with bad bandwidth
    with pytest.raises(ValueError, match='use a value of at least'):
        psd_array_multitaper(data, sfreq, bandwidth=4.9)


def test_adaptive_weights_convergence():
    """Test convergence and lack of convergence when setting adaptive=True."""
    data = np.random.default_rng(0).random((5, 100))
    sfreq = 500
    with pytest.warns(
        RuntimeWarning,
        match="Iterative multi-taper PSD computation did not converge."
    ):
        psd_array_multitaper(data, sfreq, adaptive=True, max_iter=2)
    psd_array_multitaper(data, sfreq, adaptive=True, max_iter=200)
