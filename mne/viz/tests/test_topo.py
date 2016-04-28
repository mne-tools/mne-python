# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#
# License: Simplified BSD

import os.path as op
import warnings
from collections import namedtuple

import numpy as np
from numpy.testing import assert_raises

from mne import io, read_events, Epochs
from mne import pick_channels_evoked
from mne.channels import read_layout
from mne.time_frequency.tfr import AverageTFR
from mne.utils import run_tests_if_main

from mne.viz import (plot_topo_image_epochs, _get_presser,
                     mne_analyze_colormap, plot_evoked_topo)
from mne.viz.utils import _fake_click
from mne.viz.topo import _plot_update_evoked_topo_proj, iter_topography

# Set our plotters to test mode
import matplotlib
matplotlib.use('Agg')  # for testing don't use X server

warnings.simplefilter('always')  # enable b/c these tests throw warnings


base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
evoked_fname = op.join(base_dir, 'test-ave.fif')
raw_fname = op.join(base_dir, 'test_raw.fif')
event_name = op.join(base_dir, 'test-eve.fif')
event_id, tmin, tmax = 1, -0.2, 0.2
layout = read_layout('Vectorview-all')


def _get_raw():
    return io.read_raw_fif(raw_fname, preload=False)


def _get_events():
    return read_events(event_name)


def _get_picks(raw):
    return [0, 1, 2, 6, 7, 8, 306, 340, 341, 342]  # take a only few channels


def _get_epochs():
    raw = _get_raw()
    events = _get_events()
    picks = _get_picks(raw)
    with warnings.catch_warnings(record=True):  # bad proj
        epochs = Epochs(raw, events[:10], event_id, tmin, tmax, picks=picks,
                        baseline=(None, 0), verbose='error')
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
    import matplotlib.pyplot as plt
    # Show topography
    evoked = _get_epochs().average()
    plot_evoked_topo(evoked)  # should auto-find layout
    # Test jointplot
    evoked.plot_joint()
    evoked.plot_joint(title='test', ts_args=dict(spatial_colors=True),
                      topomap_args=dict(colorbar=True, times=[0.]))

    warnings.simplefilter('always', UserWarning)
    picked_evoked = evoked.copy().pick_channels(evoked.ch_names[:3])
    picked_evoked_eeg = evoked.copy().pick_types(meg=False, eeg=True)
    picked_evoked_eeg.pick_channels(picked_evoked_eeg.ch_names[:3])

    # test scaling
    with warnings.catch_warnings(record=True):
        for ylim in [dict(mag=[-600, 600]), None]:
            plot_evoked_topo([picked_evoked] * 2, layout, ylim=ylim)

        for evo in [evoked, [evoked, picked_evoked]]:
            assert_raises(ValueError, plot_evoked_topo, evo, layout,
                          color=['y', 'b'])

        evoked_delayed_ssp = _get_epochs_delayed_ssp().average()
        ch_names = evoked_delayed_ssp.ch_names[:3]  # make it faster
        picked_evoked_delayed_ssp = pick_channels_evoked(evoked_delayed_ssp,
                                                         ch_names)
        fig = plot_evoked_topo(picked_evoked_delayed_ssp, layout,
                               proj='interactive')
        func = _get_presser(fig)
        event = namedtuple('Event', ['inaxes', 'xdata', 'ydata'])
        func(event(inaxes=fig.axes[0], xdata=fig.axes[0]._mne_axs[0].pos[0],
                   ydata=fig.axes[0]._mne_axs[0].pos[1]))
        func(event(inaxes=fig.axes[0], xdata=0, ydata=0))
        params = dict(evokeds=[picked_evoked_delayed_ssp],
                      times=picked_evoked_delayed_ssp.times,
                      fig=fig, projs=picked_evoked_delayed_ssp.info['projs'])
        bools = [True] * len(params['projs'])
        _plot_update_evoked_topo_proj(params, bools)
    # should auto-generate layout
    plot_evoked_topo(picked_evoked_eeg.copy(),
                     fig_background=np.zeros((4, 3, 3)), proj=True)
    picked_evoked.plot_topo(merge_grads=True)  # Test RMS plot of grad pairs
    plt.close('all')
    for ax, idx in iter_topography(evoked.info):
        ax.plot(evoked.data[idx], color='red')
    plt.close('all')


def test_plot_topo_image_epochs():
    """Test plotting of epochs image topography
    """
    import matplotlib.pyplot as plt
    title = 'ERF images - MNE sample data'
    epochs = _get_epochs()
    cmap = mne_analyze_colormap(format='matplotlib')
    fig = plot_topo_image_epochs(epochs, sigma=0.5, vmin=-200, vmax=200,
                                 colorbar=True, title=title, cmap=cmap)
    _fake_click(fig, fig.axes[2], (0.08, 0.64))
    plt.close('all')


def test_plot_tfr_topo():
    """Test plotting of TFR data
    """
    epochs = _get_epochs()
    n_freqs = 3
    nave = 1
    data = np.random.RandomState(0).randn(len(epochs.ch_names),
                                          n_freqs, len(epochs.times))
    tfr = AverageTFR(epochs.info, data, epochs.times, np.arange(n_freqs), nave)
    tfr.plot_topo(baseline=(None, 0), mode='ratio', title='Average power',
                  vmin=0., vmax=14., show=False)
    tfr.plot([4], baseline=(None, 0), mode='ratio', show=False, title='foo')

run_tests_if_main()
