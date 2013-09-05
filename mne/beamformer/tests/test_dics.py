import warnings
import os.path as op
import copy as cp

from nose.tools import assert_true, assert_raises
import numpy as np
from numpy.testing import assert_array_equal

import mne
from mne.datasets import sample
from mne.beamformer import dics, dics_epochs, dics_source_power
from mne.time_frequency import compute_epochs_csd

# Note that this is the first test file, this will apply to all subsequent
# tests in a full nosetest:
warnings.simplefilter("always")  # ensure we can verify expected warnings

data_path = sample.data_path()
fname_data = op.join(data_path, 'MEG', 'sample', 'sample_audvis-ave.fif')
fname_raw = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')
fname_fwd = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis-meg-oct-6-fwd.fif')
fname_fwd_vol = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis-meg-vol-7-fwd.fif')
fname_event = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw-eve.fif')
label = 'Aud-lh'
fname_label = op.join(data_path, 'MEG', 'sample', 'labels', '%s.label' % label)

# preloading raw here increases mem requirements by 400 mb for all nosetests
# that include this file's parent directory :(


def read_data():
    """Read in data used in tests
    """
    label = mne.read_label(fname_label)
    events = mne.read_events(fname_event)[:10]
    raw = mne.fiff.Raw(fname_raw, preload=False)
    forward = mne.read_forward_solution(fname_fwd)
    forward_surf_ori = mne.read_forward_solution(fname_fwd, surf_ori=True)
    forward_fixed = mne.read_forward_solution(fname_fwd, force_fixed=True,
                                              surf_ori=True)
    forward_vol = mne.read_forward_solution(fname_fwd_vol, surf_ori=True)

    event_id, tmin, tmax = 1, -0.11, 0.15

    # Setup for reading the raw data
    raw.info['bads'] = ['MEG 2443', 'EEG 053']  # 2 bads channels

    # Set up pick list: MEG - bad channels
    left_temporal_channels = mne.read_selection('Left-temporal')
    picks = mne.fiff.pick_types(raw.info, meg=True, eeg=False,
                                stim=True, eog=True, exclude='bads',
                                selection=left_temporal_channels)

    # Read epochs
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                        picks=picks, baseline=(None, 0), preload=True,
                        reject=dict(grad=4000e-13, mag=4e-12, eog=150e-6))
    epochs.resample(200, npad=0, n_jobs=2)
    evoked = epochs.average()

    # Computing the data and noise cross-spectral density matrices
    data_csd = compute_epochs_csd(epochs, mode='multitaper', tmin=0.04,
                                  tmax=None, fmin=8, fmax=12)
    noise_csd = compute_epochs_csd(epochs, mode='multitaper', tmin=None,
                                   tmax=0.0, fmin=8, fmax=12)

    return raw, epochs, evoked, data_csd, noise_csd, label, forward,\
        forward_surf_ori, forward_fixed, forward_vol


def test_dics():
    """Test DICS with evoked data and single trials
    """
    raw, epochs, evoked, data_csd, noise_csd, label, forward,\
        forward_surf_ori, forward_fixed, forward_vol = read_data()

    stc = dics(evoked, forward, noise_csd=noise_csd, data_csd=data_csd,
               label=label)

    stc_pow = np.sum(stc.data, axis=1)
    idx = np.argmax(stc_pow)
    max_stc = stc.data[idx]
    tmax = stc.times[np.argmax(max_stc)]

    assert_true(0.09 < tmax < 0.11)
    assert_true(10 < np.max(max_stc) < 11)

    # Test picking normal orientation
    stc_normal = dics(evoked, forward_surf_ori, noise_csd, data_csd,
                      pick_ori="normal", label=label)

    # The amplitude of normal orientation results should always be smaller than
    # free orientation results
    assert_true((np.abs(stc_normal.data) <= stc.data).all())

    # Test if fixed forward operator is detected when picking normal
    # orientation
    assert_raises(ValueError, dics_epochs, epochs, forward_fixed, noise_csd,
                  data_csd, pick_ori="normal")

    # Test if non-surface oriented forward operator is detected when picking
    # normal orientation
    assert_raises(ValueError, dics_epochs, epochs, forward, noise_csd,
                  data_csd, pick_ori="normal")

    # Test if volume forward operator is detected when picking normal
    # orientation
    assert_raises(ValueError, dics_epochs, epochs, forward_vol, noise_csd,
                  data_csd, pick_ori="normal")

    # Now test single trial using fixed orientation forward solution
    # so we can compare it to the evoked solution
    stcs = dics_epochs(epochs, forward_fixed, noise_csd, data_csd, reg=0.01,
                       label=label)

    # Testing returning of generator
    stcs_ = dics_epochs(epochs, forward_fixed, noise_csd, data_csd, reg=0.01,
                        return_generator=True, label=label)
    assert_array_equal(stcs[0].data, stcs_.next().data)

    # Test whether correct number of trials was returned
    epochs.drop_bad_epochs()
    assert_true(len(epochs.events) == len(stcs))

    # Average the single trial estimates
    stc_avg = np.zeros_like(stcs[0].data)
    for this_stc in stcs:
        stc_avg += this_stc.data
    stc_avg /= len(stcs)

    idx = np.argmax(np.max(stc_avg, axis=1))
    max_stc = stc_avg[idx]
    tmax = stc.times[np.argmax(max_stc)]

    assert_true(0.045 < tmax < 0.055)  # odd due to limited number of epochs
    assert_true(17.5 < np.max(max_stc) < 18.5)


def test_dics_source_power():
    """Test DICS source power computation
    """
    raw, epochs, evoked, data_csd, noise_csd, label, forward,\
        forward_surf_ori, forward_fixed, forward_vol = read_data()

    stc_source_power = dics_source_power(epochs.info, forward, noise_csd,
                                         data_csd, label=label)

    max_source_idx = np.argmax(stc_source_power.data)
    max_source_power = np.max(stc_source_power.data)

    # TODO: Maybe these could be more directly compared to dics() results?
    assert_true(max_source_idx == 18)
    assert_true(1.05 < max_source_power < 1.15)

    # Test picking normal orientation and using a list of CSD matrices
    stc_normal = dics_source_power(epochs.info, forward_surf_ori,
                                   [noise_csd] * 2, [data_csd] * 2,
                                   pick_ori="normal", label=label)

    assert_true(stc_normal.data.shape == (stc_source_power.data.shape[0], 2))

    # The normal orientation results should always be smaller than free
    # orientation results
    assert_true((np.abs(stc_normal.data[:, 0]) <=
                 stc_source_power.data[:, 0]).all())

    # Test if fixed forward operator is detected when picking normal
    # orientation
    assert_raises(ValueError, dics_source_power, raw.info, forward_fixed,
                  noise_csd, data_csd, pick_ori="normal")

    # Test if non-surface oriented forward operator is detected when picking
    # normal orientation
    assert_raises(ValueError, dics_source_power, raw.info, forward, noise_csd,
                  data_csd, pick_ori="normal")

    # Test if volume forward operator is detected when picking normal
    # orientation
    assert_raises(ValueError, dics_source_power, epochs.info, forward_vol,
                  noise_csd, data_csd, pick_ori="normal")

    # Test detection of different number of CSD matrices provided
    assert_raises(ValueError, dics_source_power, epochs.info, forward,
                  [noise_csd] * 2, [data_csd] * 3)

    # Test detection of different frequencies in noise and data CSD objects
    noise_csd.frequencies = [1, 2]
    data_csd.frequencies = [1, 2, 3]
    assert_raises(ValueError, dics_source_power, epochs.info, forward,
                  noise_csd, data_csd)

    # Test detection of uneven frequency spacing
    data_csds = [cp.deepcopy(data_csd) for i in range(3)]
    frequencies = [1, 3, 4]
    for freq, data_csd in zip(frequencies, data_csds):
        data_csd.frequencies = [freq]
    noise_csds = data_csds
    with warnings.catch_warnings(True) as w:
        dics_source_power(epochs.info, forward, noise_csds, data_csds)
    assert len(w) == 1
