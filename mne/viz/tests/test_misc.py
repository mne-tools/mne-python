# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Cathy Nangini <cnangini@gmail.com>
#          Mainak Jas <mainak@neuro.hut.fi>
#
# License: Simplified BSD

import os.path as op

import numpy as np
from numpy.testing import assert_array_equal
import pytest
import matplotlib.pyplot as plt

from mne import (read_events, read_cov, read_source_spaces, read_evokeds,
                 read_dipole, SourceEstimate, pick_events)
from mne.chpi import compute_chpi_snr
from mne.datasets import testing
from mne.filter import create_filter
from mne.io import read_raw_fif
from mne.minimum_norm import read_inverse_operator
from mne.viz import (plot_bem, plot_events, plot_source_spectrogram,
                     plot_snr_estimate, plot_filter, plot_csd, plot_chpi_snr)
from mne.viz.misc import _handle_event_colors
from mne.viz.utils import _get_color_list
from mne.utils import requires_nibabel
from mne.time_frequency import CrossSpectralDensity

data_path = testing.data_path(download=False)
subjects_dir = op.join(data_path, 'subjects')
src_fname = op.join(subjects_dir, 'sample', 'bem', 'sample-oct-6-src.fif')
inv_fname = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-meg-eeg-oct-4-meg-inv.fif')
evoked_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis-ave.fif')
dip_fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_trunc_set1.dip')
chpi_fif_fname = op.join(data_path, 'SSS', 'test_move_anon_raw.fif')
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


def test_plot_filter():
    """Test filter plotting."""
    l_freq, h_freq, sfreq = 2., 40., 1000.
    data = np.zeros(5000)
    freq = [0, 2, 40, 50, 500]
    gain = [0, 1, 1, 0, 0]
    h = create_filter(data, sfreq, l_freq, h_freq, fir_design='firwin2')
    plot_filter(h, sfreq)
    plt.close('all')
    plot_filter(h, sfreq, freq, gain)
    plt.close('all')
    iir = create_filter(data, sfreq, l_freq, h_freq, method='iir')
    plot_filter(iir, sfreq)
    plt.close('all')
    plot_filter(iir, sfreq, freq, gain)
    plt.close('all')
    iir_ba = create_filter(data, sfreq, l_freq, h_freq, method='iir',
                           iir_params=dict(output='ba'))
    plot_filter(iir_ba, sfreq, freq, gain)
    plt.close('all')
    fig = plot_filter(h, sfreq, freq, gain, fscale='linear')
    assert len(fig.axes) == 3
    plt.close('all')
    fig = plot_filter(h, sfreq, freq, gain, fscale='linear',
                      plot=('time', 'delay'))
    assert len(fig.axes) == 2
    plt.close('all')
    fig = plot_filter(h, sfreq, freq, gain, fscale='linear',
                      plot=['magnitude', 'delay'])
    assert len(fig.axes) == 2
    plt.close('all')
    fig = plot_filter(h, sfreq, freq, gain, fscale='linear',
                      plot='magnitude')
    assert len(fig.axes) == 1
    plt.close('all')
    fig = plot_filter(h, sfreq, freq, gain, fscale='linear',
                      plot=('magnitude'))
    assert len(fig.axes) == 1
    plt.close('all')
    with pytest.raises(ValueError, match='Invalid value for the .plot'):
        plot_filter(h, sfreq, freq, gain, plot=('turtles'))
    _, axes = plt.subplots(1)
    fig = plot_filter(h, sfreq, freq, gain, plot=('magnitude'), axes=axes)
    assert len(fig.axes) == 1
    _, axes = plt.subplots(2)
    fig = plot_filter(h, sfreq, freq, gain, plot=('magnitude', 'delay'),
                      axes=axes)
    assert len(fig.axes) == 2
    plt.close('all')
    _, axes = plt.subplots(1)
    with pytest.raises(ValueError, match='Length of axes'):
        plot_filter(h, sfreq, freq, gain,
                    plot=('magnitude', 'delay'), axes=axes)


def test_plot_cov():
    """Test plotting of covariances."""
    raw = _get_raw()
    cov = read_cov(cov_fname)
    with pytest.warns(RuntimeWarning, match='projection'):
        fig1, fig2 = cov.plot(raw.info, proj=True, exclude=raw.ch_names[6:])


@testing.requires_testing_data
@requires_nibabel()
def test_plot_bem():
    """Test plotting of BEM contours."""
    with pytest.raises(IOError, match='MRI file .* not found'):
        plot_bem(subject='bad-subject', subjects_dir=subjects_dir)
    with pytest.raises(ValueError, match="Invalid value for the 'orientation"):
        plot_bem(subject='sample', subjects_dir=subjects_dir,
                 orientation='bad-ori')
    with pytest.raises(ValueError, match="sorted 1D array"):
        plot_bem(subject='sample', subjects_dir=subjects_dir, slices=[0, 500])
    fig = plot_bem(subject='sample', subjects_dir=subjects_dir,
                   orientation='sagittal', slices=[25, 50])
    assert len(fig.axes) == 2
    assert len(fig.axes[0].collections) == 3  # 3 BEM surfaces ...
    fig = plot_bem(subject='sample', subjects_dir=subjects_dir,
                   orientation='coronal', brain_surfaces='white')
    assert len(fig.axes[0].collections) == 5  # 3 BEM surfaces + 2 hemis
    fig = plot_bem(subject='sample', subjects_dir=subjects_dir,
                   orientation='coronal', slices=[25, 50], src=src_fname)
    assert len(fig.axes[0].collections) == 4  # 3 BEM surfaces + 1 src contour
    with pytest.raises(ValueError, match='MRI coordinates, got head'):
        plot_bem(subject='sample', subjects_dir=subjects_dir,
                 src=inv_fname)


def test_event_colors():
    """Test color assignment."""
    events = pick_events(_get_events(), include=[1, 2])
    unique_events = set(events[:, 2])
    # make sure defaults work
    colors = _handle_event_colors(None, unique_events, dict())
    default_colors = _get_color_list()
    assert colors[1] == default_colors[0]
    # make sure custom color overrides default
    colors = _handle_event_colors(color_dict=dict(foo='k', bar='#facade'),
                                  unique_events=unique_events,
                                  event_id=dict(foo=1, bar=2))
    assert colors[1] == 'k'
    assert colors[2] == '#facade'


def test_plot_events():
    """Test plotting events."""
    event_labels = {'aud_l': 1, 'aud_r': 2, 'vis_l': 3, 'vis_r': 4}
    color = {1: 'green', 2: 'yellow', 3: 'red', 4: 'c'}
    raw = _get_raw()
    events = _get_events()
    fig = plot_events(events, raw.info['sfreq'], raw.first_samp)
    assert fig.axes[0].get_legend() is not None  # legend even with no event_id
    plot_events(events, raw.info['sfreq'], raw.first_samp, equal_spacing=False)
    # Test plotting events without sfreq
    plot_events(events, first_samp=raw.first_samp)
    with pytest.warns(RuntimeWarning, match='will be ignored'):
        fig = plot_events(events, raw.info['sfreq'], raw.first_samp,
                          event_id=event_labels)
    assert fig.axes[0].get_legend() is not None
    with pytest.warns(RuntimeWarning, match='Color was not assigned'):
        plot_events(events, raw.info['sfreq'], raw.first_samp,
                    color=color)
    with pytest.warns(RuntimeWarning, match=r'vent \d+ missing from event_id'):
        plot_events(events, raw.info['sfreq'], raw.first_samp,
                    event_id=event_labels, color=color)
    multimatch = r'event \d+ missing from event_id|in the color dict but is'
    with pytest.warns(RuntimeWarning, match=multimatch):
        plot_events(events, raw.info['sfreq'], raw.first_samp,
                    event_id={'aud_l': 1}, color=color)
    extra_id = {'missing': 111}
    with pytest.raises(ValueError, match='from event_id is not present in'):
        plot_events(events, raw.info['sfreq'], raw.first_samp,
                    event_id=extra_id)
    with pytest.raises(RuntimeError, match='No usable event IDs'):
        plot_events(events, raw.info['sfreq'], raw.first_samp,
                    event_id=extra_id, on_missing='ignore')
    extra_id = {'aud_l': 1, 'missing': 111}
    with pytest.warns(RuntimeWarning, match='from event_id is not present in'):
        plot_events(events, raw.info['sfreq'], raw.first_samp,
                    event_id=extra_id, on_missing='warn')
    with pytest.warns(RuntimeWarning, match='event 2 missing'):
        plot_events(events, raw.info['sfreq'], raw.first_samp,
                    event_id=extra_id, on_missing='ignore')
    events = events[events[:, 2] == 1]
    assert len(events) > 0
    plot_events(events, raw.info['sfreq'], raw.first_samp,
                event_id=extra_id, on_missing='ignore')
    with pytest.raises(ValueError, match='No events'):
        plot_events(np.empty((0, 3)))


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
    pytest.raises(ValueError, plot_source_spectrogram, [], [])
    pytest.raises(ValueError, plot_source_spectrogram, [stc, stc],
                  [[1, 2], [3, 4]], tmin=0)
    pytest.raises(ValueError, plot_source_spectrogram, [stc, stc],
                  [[1, 2], [3, 4]], tmax=7)


@pytest.mark.slowtest
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


def test_plot_csd():
    """Test plotting of CSD matrices."""
    csd = CrossSpectralDensity([1, 2, 3], ['CH1', 'CH2'],
                               frequencies=[(10, 20)], n_fft=1,
                               tmin=0, tmax=1,)
    plot_csd(csd, mode='csd')  # Plot cross-spectral density
    plot_csd(csd, mode='coh')  # Plot coherence


@pytest.mark.slowtest  # Slow on Azure
@testing.requires_testing_data
def test_plot_chpi_snr():
    """Test plotting cHPI SNRs."""
    raw = read_raw_fif(chpi_fif_fname, allow_maxshield='yes')
    result = compute_chpi_snr(raw)
    # test figure creation
    fig = plot_chpi_snr(result)
    assert len(fig.axes) == len(result) - 2
    assert len(fig.axes[0].lines) == len(result['freqs'])
    assert len(fig.legends) == 1
    texts = [entry.get_text() for entry in fig.legends[0].get_texts()]
    assert len(texts) == len(result['freqs'])
    freqs = [float(text.split()[0]) for text in texts]
    assert_array_equal(freqs, result['freqs'])
    # test user-passed axes
    _, axs = plt.subplots(2, 3)
    _ = plot_chpi_snr(result, axes=axs.ravel())
    # test error
    _, axs = plt.subplots(5)
    with pytest.raises(ValueError, match='a list of 6 axes, got length 5'):
        _ = plot_chpi_snr(result, axes=axs.ravel())
