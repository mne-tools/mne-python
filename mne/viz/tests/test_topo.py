# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
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
from mne import pick_channels_evoked
from mne.layouts import read_layout
from mne.datasets import sample
from mne.time_frequency.tfr import AverageTFR

from mne.viz import plot_topo, plot_topo_image_epochs


warnings.simplefilter('always')  # enable b/c these tests throw warnings


data_dir = sample.data_path(download=False)
subjects_dir = op.join(data_dir, 'subjects')
ecg_fname = op.join(data_dir, 'MEG', 'sample', 'sample_audvis_ecg_proj.fif')

base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
evoked_fname = op.join(base_dir, 'test-ave.fif')
raw_fname = op.join(base_dir, 'test_raw.fif')
event_name = op.join(base_dir, 'test-eve.fif')
event_id, tmin, tmax = 1, -0.2, 0.2
layout = read_layout('Vectorview-all')


def _get_raw():
    return io.Raw(raw_fname, preload=False)


def _get_events():
    return read_events(event_name)


def _get_picks(raw):
    return [0, 1, 2, 6, 7, 8, 12, 13, 14]  # take a only few channels


def _get_epochs():
    raw = _get_raw()
    events = _get_events()
    picks = _get_picks(raw)
    epochs = Epochs(raw, events[:10], event_id, tmin, tmax, picks=picks,
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


def test_plot_topo():
    """Test plotting of ERP topography
    """
    # Show topography
    evoked = _get_epochs().average()
    plot_topo(evoked, layout)
    warnings.simplefilter('always', UserWarning)
    picked_evoked = pick_channels_evoked(evoked, evoked.ch_names[:3])

    # test scaling
    with warnings.catch_warnings(record=True):
        for ylim in [dict(mag=[-600, 600]), None]:
            plot_topo([picked_evoked] * 2, layout, ylim=ylim)

        for evo in [evoked, [evoked, picked_evoked]]:
            assert_raises(ValueError, plot_topo, evo, layout, color=['y', 'b'])

        evoked_delayed_ssp = _get_epochs_delayed_ssp().average()
        ch_names = evoked_delayed_ssp.ch_names[:3]  # make it faster
        picked_evoked_delayed_ssp = pick_channels_evoked(evoked_delayed_ssp,
                                                         ch_names)
        plot_topo(picked_evoked_delayed_ssp, layout, proj='interactive')


def test_plot_topo_image_epochs():
    """Test plotting of epochs image topography
    """
    title = 'ERF images - MNE sample data'
    epochs = _get_epochs()
    plot_topo_image_epochs(epochs, layout, sigma=0.5, vmin=-200, vmax=200,
                           colorbar=True, title=title)
    plt.close('all')


def test_plot_tfr_topo():
    """Test plotting of TFR data
    """
    epochs = _get_epochs()
    n_freqs = 3
    nave = 1
    data = np.random.randn(len(epochs.ch_names), n_freqs, len(epochs.times))
    tfr = AverageTFR(epochs.info, data, epochs.times, np.arange(n_freqs), nave)
    tfr.plot_topo(baseline=(None, 0), mode='ratio', title='Average power',
                  vmin=0., vmax=14.)
    tfr.plot([4], baseline=(None, 0), mode='ratio')
