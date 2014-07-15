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

from mne import io, read_events, read_cov, read_source_spaces
from mne import SourceEstimate
from mne.datasets import sample

from mne.viz import plot_cov, plot_bem, plot_events
from mne.viz import plot_source_spectrogram


warnings.simplefilter('always')  # enable b/c these tests throw warnings

# Set our plotters to test mode
import matplotlib
matplotlib.use('Agg')  # for testing don't use X server
import matplotlib.pyplot as plt


data_dir = sample.data_path(download=False)
subjects_dir = op.join(data_dir, 'subjects')
ecg_fname = op.join(data_dir, 'MEG', 'sample', 'sample_audvis_ecg_proj.fif')

base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')
cov_fname = op.join(base_dir, 'test-cov.fif')
event_name = op.join(base_dir, 'test-eve.fif')


def _get_raw():
    return io.Raw(raw_fname, preload=True)


def _get_events():
    return read_events(event_name)


def test_plot_cov():
    """Test plotting of covariances
    """
    raw = _get_raw()
    cov = read_cov(cov_fname)
    fig1, fig2 = plot_cov(cov, raw.info, proj=True, exclude=raw.ch_names[6:])
    plt.close('all')


@sample.requires_sample_data
def test_plot_bem():
    """Test plotting of BEM contours
    """
    assert_raises(IOError, plot_bem, subject='bad-subject',
                  subjects_dir=subjects_dir)
    assert_raises(ValueError, plot_bem, subject='sample',
                  subjects_dir=subjects_dir, orientation='bad-ori')
    plot_bem(subject='sample', subjects_dir=subjects_dir,
             orientation='sagittal', slices=[50, 100])


def test_plot_events():
    """Test plotting events
    """
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


@sample.requires_sample_data
def test_plot_source_spectrogram():
    """Test plotting of source spectrogram
    """
    sample_src = read_source_spaces(op.join(data_dir, 'subjects', 'sample',
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
