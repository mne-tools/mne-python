# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Jaakko Leppakangas <jaeilepp@student.jyu.fi>
#
# License: Simplified BSD

import os.path as op
import warnings
from nose.tools import assert_raises

import numpy as np
from numpy.testing import assert_equal

from mne import read_events, Epochs, pick_types
from mne.channels import read_layout
from mne.io import read_raw_fif
from mne.utils import run_tests_if_main, requires_version
from mne.viz import plot_drop_log
from mne.viz.utils import _fake_click

# Set our plotters to test mode
import matplotlib
matplotlib.use('Agg')  # for testing don't use X server

warnings.simplefilter('always')  # enable b/c these tests throw warnings


base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
evoked_fname = op.join(base_dir, 'test-ave.fif')
raw_fname = op.join(base_dir, 'test_raw.fif')
cov_fname = op.join(base_dir, 'test-cov.fif')
event_name = op.join(base_dir, 'test-eve.fif')
event_id, tmin, tmax = 1, -0.1, 1.0
n_chan = 15
layout = read_layout('Vectorview-all')


def _get_picks(raw):
    """Get picks."""
    return pick_types(raw.info, meg=True, eeg=False, stim=False,
                      ecg=False, eog=False, exclude='bads')


def _get_epochs():
    """Get epochs."""
    raw = read_raw_fif(raw_fname)
    events = read_events(event_name)
    picks = _get_picks(raw)
    # Use a subset of channels for plotting speed
    picks = np.round(np.linspace(0, len(picks) + 1, n_chan)).astype(int)
    with warnings.catch_warnings(record=True):  # bad proj
        epochs = Epochs(raw, events[:5], event_id, tmin, tmax, picks=picks)
    return epochs


def _get_epochs_delayed_ssp():
    """Get epochs with delayed SSP."""
    raw = read_raw_fif(raw_fname)
    events = read_events(event_name)
    picks = _get_picks(raw)
    reject = dict(mag=4e-12)
    epochs_delayed_ssp = Epochs(
        raw, events[:10], event_id, tmin, tmax, picks=picks,
        proj='delayed', reject=reject)
    return epochs_delayed_ssp


def test_plot_epochs():
    """Test epoch plotting."""
    import matplotlib.pyplot as plt
    epochs = _get_epochs()
    epochs.info.normalize_proj()  # avoid warnings
    epochs.plot(scalings=None, title='Epochs')
    plt.close('all')
    fig = epochs[0].plot(picks=[0, 2, 3], scalings=None)
    fig.canvas.key_press_event('escape')
    plt.close('all')
    fig = epochs.plot()
    fig.canvas.key_press_event('left')
    fig.canvas.key_press_event('right')
    fig.canvas.scroll_event(0.5, 0.5, -0.5)  # scroll down
    fig.canvas.scroll_event(0.5, 0.5, 0.5)  # scroll up
    fig.canvas.key_press_event('up')
    fig.canvas.key_press_event('down')
    fig.canvas.key_press_event('pageup')
    fig.canvas.key_press_event('pagedown')
    fig.canvas.key_press_event('-')
    fig.canvas.key_press_event('+')
    fig.canvas.key_press_event('=')
    fig.canvas.key_press_event('b')
    fig.canvas.key_press_event('f11')
    fig.canvas.key_press_event('home')
    fig.canvas.key_press_event('?')
    fig.canvas.key_press_event('h')
    fig.canvas.key_press_event('o')
    fig.canvas.key_press_event('end')
    fig.canvas.resize_event()
    fig.canvas.close_event()  # closing and epoch dropping
    plt.close('all')
    assert_raises(RuntimeError, epochs.plot, picks=[])
    plt.close('all')
    with warnings.catch_warnings(record=True):
        fig = epochs.plot(events=epochs.events)
        # test mouse clicks
        x = fig.get_axes()[0].get_xlim()[1] / 2
        y = fig.get_axes()[0].get_ylim()[0] / 2
        data_ax = fig.get_axes()[0]
        n_epochs = len(epochs)
        _fake_click(fig, data_ax, [x, y], xform='data')  # mark a bad epoch
        _fake_click(fig, data_ax, [x, y], xform='data')  # unmark a bad epoch
        _fake_click(fig, data_ax, [0.5, 0.999])  # click elsewhere in 1st axes
        _fake_click(fig, data_ax, [-0.1, 0.9])  # click on y-label
        _fake_click(fig, data_ax, [-0.1, 0.9], button=3)
        _fake_click(fig, fig.get_axes()[2], [0.5, 0.5])  # change epochs
        _fake_click(fig, fig.get_axes()[3], [0.5, 0.5])  # change channels
        fig.canvas.close_event()  # closing and epoch dropping
        assert(n_epochs - 1 == len(epochs))
        plt.close('all')
    epochs.plot_sensors()  # Test plot_sensors
    plt.close('all')


def test_plot_epochs_image():
    """Test plotting of epochs image."""
    import matplotlib.pyplot as plt
    epochs = _get_epochs()
    epochs.plot_image(picks=[1, 2])
    overlay_times = [0.1]
    epochs.plot_image(order=[0], overlay_times=overlay_times, vmin=0.01)
    epochs.plot_image(overlay_times=overlay_times, vmin=-0.001, vmax=0.001)
    assert_raises(ValueError, epochs.plot_image,
                  overlay_times=[0.1, 0.2])
    assert_raises(ValueError, epochs.plot_image,
                  order=[0, 1])
    with warnings.catch_warnings(record=True) as w:
        epochs.plot_image(overlay_times=[1.1])
        warnings.simplefilter('always')
    assert_equal(len(w), 1)

    plt.close('all')


def test_plot_drop_log():
    """Test plotting a drop log."""
    import matplotlib.pyplot as plt
    epochs = _get_epochs()
    assert_raises(ValueError, epochs.plot_drop_log)
    epochs.drop_bad()

    warnings.simplefilter('always', UserWarning)
    with warnings.catch_warnings(record=True):
        epochs.plot_drop_log()

        plot_drop_log([['One'], [], []])
        plot_drop_log([['One'], ['Two'], []])
        plot_drop_log([['One'], ['One', 'Two'], []])
    plt.close('all')


@requires_version('scipy', '0.12')
def test_plot_psd_epochs():
    """Test plotting epochs psd (+topomap)."""
    import matplotlib.pyplot as plt
    epochs = _get_epochs()
    epochs.plot_psd()
    assert_raises(RuntimeError, epochs.plot_psd_topomap,
                  bands=[(0, 0.01, 'foo')])  # no freqs in range
    epochs.plot_psd_topomap()
    plt.close('all')


run_tests_if_main()
