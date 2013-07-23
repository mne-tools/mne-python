import numpy as np
from nose.tools import assert_raises, assert_equal
from numpy.testing import assert_array_equal

import mne

from mne.fiff import Raw
from mne.datasets import sample
from mne.time_frequency import compute_csd, induced_power

data_path = sample.data_path()
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_raw-eve.fif'

# Read raw data
raw = Raw(raw_fname)
raw.info['bads'] = ['MEG 2443', 'EEG 053']  # 2 bads channels

# Set picks
picks = mne.fiff.pick_types(raw.info, meg=True, eeg=False, eog=False,
                            stim=False, exclude='bads')

# Read several epochs
event_id, tmin, tmax = 1, -0.2, 0.5
events = mne.read_events(event_fname)[0:100]
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                    picks=picks, baseline=(None, 0), preload=True,
                    reject=dict(grad=4000e-13, mag=4e-12))


def test_compute_csd():
    """Test computing cross-spectral density from epochs
    """
    # Check that wrong parameters are recognized
    assert_raises(ValueError, compute_csd, epochs, mode='notamode')
    assert_raises(ValueError, compute_csd, epochs, fmin=20, fmax=10)
    assert_raises(ValueError, compute_csd, epochs, tmin=0.15, tmax=0.1)
    assert_raises(ValueError, compute_csd, epochs, tmin=0, tmax=10)
    assert_raises(ValueError, compute_csd, epochs, tmin=10, tmax=11)

    data_csd_mt = compute_csd(epochs, mode='multitaper', fmin=8, fmax=12,
                              tmin=0.04, tmax=0.15)
    data_csd_fourier = compute_csd(epochs, mode='fourier', fmin=8, fmax=12,
                                   tmin=0.04, tmax=0.15)

    # Check shape of the CSD matrix
    n_chan = len(data_csd_mt.ch_names)
    assert_equal(data_csd_mt.data.shape, (n_chan, n_chan))
    assert_equal(data_csd_fourier.data.shape, (n_chan, n_chan))

    # Check if the CSD matrix is hermitian
    assert_array_equal(np.tril(data_csd_mt.data).T.conj(),
                       np.triu(data_csd_mt.data))
    assert_array_equal(np.tril(data_csd_fourier.data).T.conj(),
                       np.triu(data_csd_fourier.data))

    # Computing induced power for comparison
    epochs.crop(tmin=0.04, tmax=0.15)
    power, _ = induced_power(epochs.get_data(), epochs.info['sfreq'], [10],
                             n_cycles=0.6)
    power = np.mean(power, 2)

    # Maximum PSD should occur for specific channel
    max_ch_power = power.argmax()
    max_ch_mt = data_csd_mt.data.diagonal().argmax()
    max_ch_fourier = data_csd_fourier.data.diagonal().argmax()
    assert_equal(max_ch_mt, max_ch_power)
    assert_equal(max_ch_fourier, max_ch_power)

    # Maximum CSD should occur for specific channel
    ch_csd_mt = [np.abs(data_csd_mt.data[max_ch_power][i])
                 if i != max_ch_power else 0 for i in range(n_chan)]
    max_ch_csd_mt = np.argmax(ch_csd_mt)
    ch_csd_fourier = [np.abs(data_csd_fourier.data[max_ch_power][i])
                      if i != max_ch_power else 0 for i in range(n_chan)]
    max_ch_csd_fourier = np.argmax(ch_csd_fourier)
    assert_equal(max_ch_csd_mt, max_ch_csd_fourier)
