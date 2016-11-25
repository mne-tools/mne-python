# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Cathy Nangini <cnangini@gmail.com>
#          Mainak Jas <mainak@neuro.hut.fi>
#
# License: Simplified BSD

import os.path as op
import warnings

import numpy as np
from numpy.testing import assert_raises

from mne import (read_events, read_cov, read_source_spaces, read_evokeds,
                 read_dipole, SourceEstimate)
from mne.datasets import testing
from mne.filter import create_filter
from mne.io import read_raw_fif
from mne.minimum_norm import read_inverse_operator
from mne.viz import (plot_bem, plot_events, plot_source_spectrogram,
                     plot_snr_estimate, plot_filter)
from mne.utils import (requires_nibabel, run_tests_if_main, slow_test,
                       requires_version)

# Set our plotters to test mode
import matplotlib
matplotlib.use('Agg')  # for testing don't use X server

warnings.simplefilter('always')  # enable b/c these tests throw warnings

data_path = testing.data_path(download=False)
subjects_dir = op.join(data_path, 'subjects')
src_fname = op.join(subjects_dir, 'sample', 'bem', 'sample-oct-6-src.fif')
inv_fname = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-meg-eeg-oct-4-meg-inv.fif')
evoked_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis-ave.fif')
dip_fname = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc_set1.dip')
base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')
cov_fname = op.join(base_dir, 'test-cov.fif')
event_fname = op.join(base_dir, 'test-eve.fif')


def _get_raw():
    """Get raw data."""
    return read_raw_fif(raw_fname, preload=True)


def _get_events():
    """Get events."""
    return read_events(event_fname)


@requires_version('scipy', '0.16')
def test_plot_filter():
    """Test filter plotting."""
    import matplotlib.pyplot as plt
    l_freq, h_freq, sfreq = 2., 40., 1000.
    data = np.zeros(5000)
    freq = [0, 2, 40, 50, 500]
    gain = [0, 1, 1, 0, 0]
    h = create_filter(data, sfreq, l_freq, h_freq)
    plot_filter(h, sfreq)
    plt.close('all')
    plot_filter(h, sfreq, freq, gain)
    plt.close('all')
    iir = create_filter(data, sfreq, l_freq, h_freq, method='iir')
    plot_filter(iir, sfreq)
    plt.close('all')
    plot_filter(iir, sfreq,  freq, gain)
    plt.close('all')
    iir_ba = create_filter(data, sfreq, l_freq, h_freq, method='iir',
                           iir_params=dict(output='ba'))
    plot_filter(iir_ba, sfreq,  freq, gain)
    plt.close('all')
    plot_filter(h, sfreq, freq, gain, fscale='linear')
    plt.close('all')


def test_plot_cov():
    """Test plotting of covariances."""
    raw = _get_raw()
    cov = read_cov(cov_fname)
    with warnings.catch_warnings(record=True):  # bad proj
        fig1, fig2 = cov.plot(raw.info, proj=True, exclude=raw.ch_names[6:])


@testing.requires_testing_data
@requires_nibabel()
def test_plot_bem():
    """Test plotting of BEM contours."""
    assert_raises(IOError, plot_bem, subject='bad-subject',
                  subjects_dir=subjects_dir)
    assert_raises(ValueError, plot_bem, subject='sample',
                  subjects_dir=subjects_dir, orientation='bad-ori')
    plot_bem(subject='sample', subjects_dir=subjects_dir,
             orientation='sagittal', slices=[25, 50])
    plot_bem(subject='sample', subjects_dir=subjects_dir,
             orientation='coronal', slices=[25, 50],
             brain_surfaces='white')
    plot_bem(subject='sample', subjects_dir=subjects_dir,
             orientation='coronal', slices=[25, 50], src=src_fname)


def test_plot_events():
    """Test plotting events."""
    event_labels = {'aud_l': 1, 'aud_r': 2, 'vis_l': 3, 'vis_r': 4}
    color = {1: 'green', 2: 'yellow', 3: 'red', 4: 'c'}
    raw = _get_raw()
    events = _get_events()
    plot_events(events, raw.info['sfreq'], raw.first_samp)
    plot_events(events, raw.info['sfreq'], raw.first_samp, equal_spacing=False)
    # Test plotting events without sfreq
    plot_events(events, first_samp=raw.first_samp)
    warnings.simplefilter('always', UserWarning)
    with warnings.catch_warnings(record=True):
        plot_events(events, raw.info['sfreq'], raw.first_samp,
                    event_id=event_labels)
        plot_events(events, raw.info['sfreq'], raw.first_samp,
                    color=color)
        plot_events(events, raw.info['sfreq'], raw.first_samp,
                    event_id=event_labels, color=color)
        assert_raises(ValueError, plot_events, events, raw.info['sfreq'],
                      raw.first_samp, event_id={'aud_l': 1}, color=color)
        assert_raises(ValueError, plot_events, events, raw.info['sfreq'],
                      raw.first_samp, event_id={'aud_l': 111}, color=color)


@testing.requires_testing_data
def test_plot_source_spectrogram():
    """Test plotting of source spectrogram."""
    sample_src = read_source_spaces(op.join(subjects_dir, 'sample',
                                            'bem', 'sample-oct-6-src.fif'))

    # dense version
    vertices = [s['vertno'] for s in sample_src]
    n_times = 5
    n_verts = sum(len(v) for v in vertices)
    stc_data = np.ones((n_verts, n_times))
    stc = SourceEstimate(stc_data, vertices, 1, 1)
    plot_source_spectrogram([stc, stc], [[1, 2], [3, 4]])
    assert_raises(ValueError, plot_source_spectrogram, [], [])
    assert_raises(ValueError, plot_source_spectrogram, [stc, stc],
                  [[1, 2], [3, 4]], tmin=0)
    assert_raises(ValueError, plot_source_spectrogram, [stc, stc],
                  [[1, 2], [3, 4]], tmax=7)


@slow_test
@testing.requires_testing_data
def test_plot_snr():
    """Test plotting SNR estimate."""
    inv = read_inverse_operator(inv_fname)
    evoked = read_evokeds(evoked_fname, baseline=(None, 0))[0]
    plot_snr_estimate(evoked, inv)


@testing.requires_testing_data
def test_plot_dipole_amplitudes():
    """Test plotting dipole amplitudes."""
    dipoles = read_dipole(dip_fname)
    dipoles.plot_amplitudes(show=False)

run_tests_if_main()
