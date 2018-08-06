from distutils.version import LooseVersion

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from mne.time_frequency import psd_multitaper
from mne.time_frequency.multitaper import dpss_windows
from mne.utils import requires_nitime
from mne.io import RawArray
from mne import create_info


@requires_nitime
def test_dpss_windows():
    """Test computation of DPSS windows."""
    import nitime as ni
    N = 1000
    half_nbw = 4
    Kmax = int(2 * half_nbw)

    dpss, eigs = dpss_windows(N, half_nbw, Kmax, low_bias=False)
    with pytest.warns(None):  # conversions
        dpss_ni, eigs_ni = ni.algorithms.dpss_windows(N, half_nbw, Kmax)

    assert_array_almost_equal(dpss, dpss_ni)
    assert_array_almost_equal(eigs, eigs_ni)

    dpss, eigs = dpss_windows(N, half_nbw, Kmax, interp_from=200,
                              low_bias=False)
    with pytest.warns(None):  # conversions
        dpss_ni, eigs_ni = ni.algorithms.dpss_windows(N, half_nbw, Kmax,
                                                      interp_from=200)

    assert_array_almost_equal(dpss, dpss_ni)
    assert_array_almost_equal(eigs, eigs_ni)


@requires_nitime
def test_multitaper_psd():
    """Test multi-taper PSD computation."""
    import nitime as ni
    for n_times in (100, 101):
        n_channels = 5
        data = np.random.RandomState(0).randn(n_channels, n_times)
        sfreq = 500
        info = create_info(n_channels, sfreq, 'eeg')
        raw = RawArray(data, info)
        pytest.raises(ValueError, psd_multitaper, raw, sfreq,
                      normalization='foo')
        ni_5 = (LooseVersion(ni.__version__) >= LooseVersion('0.5'))
        norm = 'full' if ni_5 else 'length'
        for adaptive, n_jobs in zip((False, True, True), (1, 1, 2)):
            psd, freqs = psd_multitaper(raw, adaptive=adaptive,
                                        n_jobs=n_jobs,
                                        normalization=norm)
            with pytest.warns(None):  # nitime integers
                freqs_ni, psd_ni, _ = ni.algorithms.spectral.multi_taper_psd(
                    data, sfreq, adaptive=adaptive, jackknife=False)
            assert_array_almost_equal(psd, psd_ni, decimal=4)
            if n_times % 2 == 0:
                # nitime's frequency definitions must be incorrect,
                # they give the same values for 100 and 101 samples
                assert_array_almost_equal(freqs, freqs_ni)
        with pytest.raises(ValueError, match='use a value of at least'):
            psd_multitaper(raw, bandwidth=4.9)
