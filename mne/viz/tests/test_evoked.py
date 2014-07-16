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

# Set our plotters to test mode
import matplotlib
matplotlib.use('Agg')  # for testing don't use X server
import matplotlib.pyplot as plt

from mne import io, read_events, Epochs
from mne import pick_types
from mne.layouts import read_layout
from mne.datasets import sample


warnings.simplefilter('always')  # enable b/c these tests throw warnings


data_dir = sample.data_path(download=False)
subjects_dir = op.join(data_dir, 'subjects')
ecg_fname = op.join(data_dir, 'MEG', 'sample', 'sample_audvis_ecg_proj.fif')

base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
evoked_fname = op.join(base_dir, 'test-ave.fif')
raw_fname = op.join(base_dir, 'test_raw.fif')
cov_fname = op.join(base_dir, 'test-cov.fif')
event_name = op.join(base_dir, 'test-eve.fif')
event_id, tmin, tmax = 1, -0.1, 0.1
n_chan = 6
layout = read_layout('Vectorview-all')


def _get_raw():
    return io.Raw(raw_fname, preload=False)


def _get_events():
    return read_events(event_name)


def _get_picks(raw):
    return pick_types(raw.info, meg=True, eeg=False, stim=False,
                      ecg=False, eog=False, exclude='bads')


def _get_epochs():
    raw = _get_raw()
    events = _get_events()
    picks = _get_picks(raw)
    # Use a subset of channels for plotting speed
    picks = np.round(np.linspace(0, len(picks) + 1, n_chan)).astype(int)
    epochs = Epochs(raw, events[:5], event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0))
    return epochs


def _get_epochs_delayed_ssp():
    raw = _get_raw()
    events = _get_events()
    picks = _get_picks(raw)
    reject = dict(mag=4e-12)
    epochs_delayed_ssp = Epochs(raw, events[:10], event_id, tmin, tmax,
                                picks=picks, baseline=(None, 0),
                                proj='delayed', reject=reject)
    return epochs_delayed_ssp


def test_plot_evoked():
    """Test plotting of evoked
    """
    evoked = _get_epochs().average()
    with warnings.catch_warnings(record=True):
        evoked.plot(proj=True, hline=[1])
        # plot with bad channels excluded
        evoked.plot(exclude='bads')
        evoked.plot(exclude=evoked.info['bads'])  # does the same thing

        # test selective updating of dict keys is working.
        evoked.plot(hline=[1], units=dict(mag='femto foo'))
        evoked_delayed_ssp = _get_epochs_delayed_ssp().average()
        evoked_delayed_ssp.plot(proj='interactive')
        evoked_delayed_ssp.apply_proj()
        assert_raises(RuntimeError, evoked_delayed_ssp.plot,
                      proj='interactive')
        evoked_delayed_ssp.info['projs'] = []
        assert_raises(RuntimeError, evoked_delayed_ssp.plot,
                      proj='interactive')
        assert_raises(RuntimeError, evoked_delayed_ssp.plot,
                      proj='interactive', axes='foo')

        evoked.plot_image(proj=True)
        # plot with bad channels excluded
        evoked.plot_image(exclude='bads')
        evoked.plot_image(exclude=evoked.info['bads'])  # does the same thing
        plt.close('all')
