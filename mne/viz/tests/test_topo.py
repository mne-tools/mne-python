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
from numpy.testing import assert_raises, assert_equal

from mne import read_events, Epochs, pick_channels_evoked
from mne.channels import read_layout
from mne.io import read_raw_fif
from mne.time_frequency.tfr import AverageTFR
from mne.utils import run_tests_if_main

from mne.viz import (plot_topo_image_epochs, _get_presser,
                     mne_analyze_colormap, plot_evoked_topo)
from mne.viz.utils import _fake_click
from mne.viz.topo import (_plot_update_evoked_topo_proj, iter_topography,
                          _imshow_tfr)

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


def _get_events():
    """Get events."""
    return read_events(event_name)


def _get_picks(raw):
    """Get picks."""
    return [0, 1, 2, 6, 7, 8, 306, 340, 341, 342]  # take a only few channels


def _get_epochs():
    """Get epochs."""
    raw = read_raw_fif(raw_fname)
    raw.add_proj([], remove_existing=True)
    events = _get_events()
    picks = _get_picks(raw)
    # bad proj warning
    epochs = Epochs(raw, events[:10], event_id, tmin, tmax, picks=picks)
    return epochs


def _get_epochs_delayed_ssp():
    """Get epochs with delayed SSP."""
    raw = read_raw_fif(raw_fname)
    events = _get_events()
    picks = _get_picks(raw)
    reject = dict(mag=4e-12)
    epochs_delayed_ssp = Epochs(
        raw, events[:10], event_id, tmin, tmax, picks=picks,
        proj='delayed', reject=reject)
    return epochs_delayed_ssp


def test_plot_topo():
    """Test plotting of ERP topography."""
    import matplotlib.pyplot as plt
    # Show topography
    evoked = _get_epochs().average()
    # should auto-find layout
    plot_evoked_topo([evoked, evoked], merge_grads=True)
    # Test jointplot
    evoked.plot_joint()

    def return_inds(d):  # to test function kwarg to zorder arg of evoked.plot
        return list(range(d.shape[0]))
    ts_args = dict(spatial_colors=True, zorder=return_inds)
    evoked.plot_joint(title='test', ts_args=ts_args,
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
    """Test plotting of epochs image topography."""
    import matplotlib.pyplot as plt
    title = 'ERF images - MNE sample data'
    epochs = _get_epochs()
    epochs.load_data()
    cmap = mne_analyze_colormap(format='matplotlib')
    data_min = epochs._data.min()
    fig = plot_topo_image_epochs(epochs, sigma=0.5, vmin=-200, vmax=200,
                                 colorbar=True, title=title, cmap=cmap)
    assert_equal(epochs._data.min(), data_min)
    _fake_click(fig, fig.axes[2], (0.08, 0.64))
    plt.close('all')


def test_plot_tfr_topo():
    """Test plotting of TFR data."""
    import matplotlib.pyplot as plt

    epochs = _get_epochs()
    n_freqs = 3
    nave = 1
    data = np.random.RandomState(0).randn(len(epochs.ch_names),
                                          n_freqs, len(epochs.times))
    tfr = AverageTFR(epochs.info, data, epochs.times, np.arange(n_freqs), nave)
    fig = tfr.plot_topo(baseline=(None, 0), mode='ratio',
                        title='Average power', vmin=0., vmax=14.)

    # test opening tfr by clicking
    num_figures_before = len(plt.get_fignums())
    _fake_click(fig, fig.axes[-1], (0.08, 0.65))
    assert_equal(num_figures_before + 1, len(plt.get_fignums()))
    plt.close('all')

    tfr.plot([4], baseline=(None, 0), mode='ratio', show=False, title='foo')
    assert_raises(ValueError, tfr.plot, [4], yscale='lin', show=False)

    # nonuniform freqs
    freqs = np.logspace(*np.log10([3, 10]), num=3)
    tfr = AverageTFR(epochs.info, data, epochs.times, freqs, nave)
    fig = tfr.plot([4], baseline=(None, 0), mode='mean', vmax=14., show=False)
    assert_equal(fig.axes[0].get_yaxis().get_scale(), 'log')

    # one timesample
    tfr = AverageTFR(epochs.info, data[:, :, [0]], epochs.times[[1]],
                     freqs, nave)
    tfr.plot([4], baseline=None, vmax=14., show=False, yscale='linear')

    # one freqency bin, log scale required: as it doesn't make sense
    # to plot log scale for one value, we test whether yscale is set to linear
    vmin, vmax = 0., 2.
    fig, ax = plt.subplots()
    tmin, tmax = epochs.times[0], epochs.times[-1]
    _imshow_tfr(ax, 3, tmin, tmax, vmin, vmax, None, tfr=data[:, [0], :],
                freq=freqs[[-1]], x_label=None, y_label=None,
                colorbar=False, cmap=('RdBu_r', True), yscale='log')
    fig = plt.gcf()
    assert_equal(fig.axes[0].get_yaxis().get_scale(), 'linear')

    # ValueError when freq[0] == 0 and yscale == 'log'
    these_freqs = freqs[:3].copy()
    these_freqs[0] = 0
    assert_raises(ValueError, _imshow_tfr, ax, 3, tmin, tmax, vmin, vmax,
                  None, tfr=data[:, :3, :], freq=these_freqs, x_label=None,
                  y_label=None, colorbar=False, cmap=('RdBu_r', True),
                  yscale='log')

run_tests_if_main()
