# Author: Denis A. Engemann <denis.engemann@gmail.com>
#         Victoria Peterson <victoriapeterson09@gmail.com>
# License: BSD (3-clause)

import numpy as np
import pytest
from numpy.testing import (assert_array_almost_equal, assert_array_equal)
from mne import io
from mne.time_frequency import psd_array_welch
from mne.decoding.ssd import SSD
from mne.utils import requires_sklearn
from mne.filter import filter_data
from mne import create_info
from mne.decoding import CSP

freqs_sig = 9, 12
freqs_noise = 8, 13


def simulate_data(freqs_sig=[9, 12], n_trials=100, n_channels=20,
                  n_samples=500, samples_per_second=250,
                  n_components=5, SNR=0.05, random_state=42):
    """Simulate data according to an instantaneous mixin model.

    Data are simulated in the statistical source space, where n=n_components
    sources contain the peak of interest.
    """
    rng = np.random.RandomState(random_state)

    filt_params_signal = dict(l_freq=freqs_sig[0], h_freq=freqs_sig[1],
                              l_trans_bandwidth=1, h_trans_bandwidth=1,
                              fir_design='firwin')

    # generate an orthogonal mixin matrix
    mixing_mat = np.linalg.svd(rng.randn(n_channels, n_channels))[0]
    # define sources
    S_s = rng.randn(n_trials * n_samples, n_components)
    # filter source in the specific freq. band of interest
    S_s = filter_data(S_s.T, samples_per_second, **filt_params_signal).T
    S_n = rng.randn(n_trials * n_samples, n_channels - n_components)
    S = np.hstack((S_s, S_n))
    # mix data
    X_s = np.dot(mixing_mat[:, :n_components], S_s.T).T
    X_n = np.dot(mixing_mat[:, n_components:], S_n.T).T
    # add noise
    X_s = X_s / np.linalg.norm(X_s, 'fro')
    X_n = X_n / np.linalg.norm(X_n, 'fro')
    X = SNR * X_s + (1 - SNR) * X_n
    X = X.T
    S = S.T
    return X, mixing_mat, S


@pytest.mark.slowtest
def test_ssd():
    """Test Common Spatial Patterns algorithm on raw data."""
    X, A, S = simulate_data()
    sf = 250
    n_channels = X.shape[0]
    info = create_info(ch_names=n_channels, sfreq=sf, ch_types='eeg')
    n_components_true = 5

    # Init
    filt_params_signal = dict(l_freq=freqs_sig[0], h_freq=freqs_sig[1],
                              l_trans_bandwidth=1, h_trans_bandwidth=1)
    filt_params_noise = dict(l_freq=freqs_noise[0], h_freq=freqs_noise[1],
                             l_trans_bandwidth=1, h_trans_bandwidth=1)
    ssd = SSD(info, filt_params_signal, filt_params_noise)
    # freq no int
    freq = 'foo'
    filt_params_signal = dict(l_freq=freq, h_freq=freqs_sig[1],
                              l_trans_bandwidth=1, h_trans_bandwidth=1)
    filt_params_noise = dict(l_freq=freqs_noise[0], h_freq=freqs_noise[1],
                             l_trans_bandwidth=1, h_trans_bandwidth=1)
    with pytest.raises(TypeError, match='must be an instance '):
        ssd = SSD(info, filt_params_signal, filt_params_noise)

    # Wrongly specified noise band
    freq = 2
    filt_params_signal = dict(l_freq=freq, h_freq=freqs_sig[1],
                              l_trans_bandwidth=1, h_trans_bandwidth=1)
    filt_params_noise = dict(l_freq=freqs_noise[0], h_freq=freqs_noise[1],
                             l_trans_bandwidth=1, h_trans_bandwidth=1)
    with pytest.raises(ValueError, match='Wrongly specified '):
        ssd = SSD(info, filt_params_signal, filt_params_noise)

    # filt param no dict
    filt_params_signal = freqs_sig
    filt_params_noise = freqs_noise
    with pytest.raises(ValueError, match='must be defined'):
        ssd = SSD(info, filt_params_signal, filt_params_noise)

    # Data type
    filt_params_signal = dict(l_freq=freqs_sig[0], h_freq=freqs_sig[1],
                              l_trans_bandwidth=1, h_trans_bandwidth=1)
    filt_params_noise = dict(l_freq=freqs_noise[0], h_freq=freqs_noise[1],
                             l_trans_bandwidth=1, h_trans_bandwidth=1)
    ssd = SSD(info, filt_params_signal, filt_params_noise)
    raw = io.RawArray(X, info)

    pytest.raises(TypeError, ssd.fit, raw)

    # More than 1 channel type
    ch_types = np.reshape([['mag'] * 10, ['eeg'] * 10], n_channels)
    info_2 = create_info(ch_names=n_channels, sfreq=sf, ch_types=ch_types)

    with pytest.raises(ValueError, match='At this point SSD'):
        ssd = SSD(info_2, filt_params_signal, filt_params_noise)

    # Number of channels
    info_3 = create_info(ch_names=n_channels + 1, sfreq=sf, ch_types='eeg')
    ssd = SSD(info_3, filt_params_signal, filt_params_noise)
    pytest.raises(ValueError, ssd.fit, X)

    # Fit
    n_components = 10
    ssd = SSD(info, filt_params_signal, filt_params_noise,
              n_components=n_components)

    # Call transform before fit
    pytest.raises(AttributeError, ssd.transform, X)

    # Check outputs
    ssd.fit(X)

    assert (ssd.filters_.shape == (n_channels, n_channels))
    assert (ssd.patterns_.shape == (n_channels, n_channels))

    # Transform
    X_ssd = ssd.fit_transform(X)
    assert (X_ssd.shape[0] == n_components)
    # back and forward
    ssd = SSD(info, filt_params_signal, filt_params_noise,
              n_components=None, sort_by_spectral_ratio=False)
    ssd.fit(X)
    X_denoised = ssd.apply(X)
    assert_array_almost_equal(X_denoised, X)

    # Power ratio ordering
    spec_ratio, _ = ssd.get_spectral_ratio(ssd.transform(X))
    # since we now that the number of true components is 5, the relative
    # difference should be low for the first 5 components and then increases
    index_diff = np.argmax(-np.diff(spec_ratio))
    assert index_diff == n_components_true - 1

    # Check detected peaks
    # fit ssd
    n_components = n_components_true
    filt_params_signal = dict(l_freq=freqs_sig[0], h_freq=freqs_sig[1],
                              l_trans_bandwidth=1, h_trans_bandwidth=1)
    filt_params_noise = dict(l_freq=freqs_noise[0], h_freq=freqs_noise[1],
                             l_trans_bandwidth=1, h_trans_bandwidth=1)
    ssd = SSD(info, filt_params_signal, filt_params_noise,
              n_components=n_components, sort_by_spectral_ratio=False)
    ssd.fit(X)

    out = ssd.transform(X)
    psd_out, _ = psd_array_welch(out[0], sfreq=250, n_fft=250)
    psd_S, _ = psd_array_welch(S[0], sfreq=250, n_fft=250)
    corr = np.abs(np.corrcoef((psd_out, psd_S))[0, 1])
    assert np.abs(corr) > 0.95

    # Check pattern estimation
    # Since there is no exact ordering of the recovered patterns
    # a pair-wise greedy search will be done
    error = list()
    for ii in range(n_channels):
        corr = np.abs(np.corrcoef(ssd.patterns_[ii, :].T, A[:, 0])[0, 1])
        error.append(1 - corr)
        min_err = np.min(error)
    assert min_err < 0.3  # threshold taken from SSD original paper


def test_ssd_epoched_data():
    """Test Common Spatial Patterns algorithm on epoched data.

    Compare the outputs when raw data is used.
    """
    X, A, S = simulate_data(n_trials=100, n_channels=20, n_samples=500)
    sf = 250
    n_channels = X.shape[0]
    info = create_info(ch_names=n_channels, sfreq=sf, ch_types='eeg')
    n_components_true = 5

    # Build epochs as sliding windows over the continuous raw file

    # Epoch length is 1 second
    X_e = np.reshape(X, (100, 20, 500))

    # Fit
    filt_params_signal = dict(l_freq=freqs_sig[0], h_freq=freqs_sig[1],
                              l_trans_bandwidth=4, h_trans_bandwidth=4)
    filt_params_noise = dict(l_freq=freqs_noise[0], h_freq=freqs_noise[1],
                             l_trans_bandwidth=4, h_trans_bandwidth=4)

    # ssd on epochs
    ssd_e = SSD(info, filt_params_signal, filt_params_noise)
    ssd_e.fit(X_e)
    # ssd on raw
    ssd = SSD(info, filt_params_signal, filt_params_noise)
    ssd.fit(X)

    # Check if the 5 first 5 components are the same for both
    _, sorter_spec_e = ssd_e.get_spectral_ratio(ssd_e.transform(X_e))
    _, sorter_spec = ssd.get_spectral_ratio(ssd.transform(X))
    assert_array_equal(sorter_spec_e[:n_components_true],
                       sorter_spec[:n_components_true])


@requires_sklearn
def test_ssd_pipeline():
    """Test if SSD works in a pipeline."""
    from sklearn.pipeline import Pipeline
    sf = 250
    X, A, S = simulate_data(n_trials=100, n_channels=20, n_samples=500)
    X_e = np.reshape(X, (100, 20, 500))
    # define bynary random output
    y = np.random.randint(2, size=100)

    info = create_info(ch_names=20, sfreq=sf, ch_types='eeg')

    filt_params_signal = dict(l_freq=freqs_sig[0], h_freq=freqs_sig[1],
                              l_trans_bandwidth=4, h_trans_bandwidth=4)
    filt_params_noise = dict(l_freq=freqs_noise[0], h_freq=freqs_noise[1],
                             l_trans_bandwidth=4, h_trans_bandwidth=4)
    ssd = SSD(info, filt_params_signal, filt_params_noise)
    csp = CSP()
    pipe = Pipeline([('SSD', ssd), ('CSP', csp)])
    pipe.set_params(SSD__n_components=5)
    pipe.set_params(CSP__n_components=2)
    out = pipe.fit_transform(X_e, y)
    assert (out.shape == (100, 2))
    assert (pipe.get_params()['SSD__n_components'] == 5)
