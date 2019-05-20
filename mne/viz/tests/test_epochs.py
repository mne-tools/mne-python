# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Jaakko Leppakangas <jaeilepp@student.jyu.fi>
#
# License: Simplified BSD

import os.path as op

import numpy as np
import pytest
import matplotlib.pyplot as plt

from mne import (read_events, Epochs, pick_types, read_cov, create_info,
                 EpochsArray)
from mne.channels import read_layout
from mne.io import read_raw_fif
from mne.utils import run_tests_if_main
from mne.viz import plot_drop_log
from mne.viz.utils import _fake_click

base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
evoked_fname = op.join(base_dir, 'test-ave.fif')
raw_fname = op.join(base_dir, 'test_raw.fif')
cov_fname = op.join(base_dir, 'test-cov.fif')
event_name = op.join(base_dir, 'test-eve.fif')
event_id, tmin, tmax = 1, -0.1, 1.0
layout = read_layout('Vectorview-all')


def _get_epochs(stop=5, meg=True, eeg=False, n_chan=20):
    """Get epochs."""
    raw = read_raw_fif(raw_fname)
    events = read_events(event_name)
    picks = pick_types(raw.info, meg=meg, eeg=eeg, stim=False,
                       ecg=False, eog=False, exclude='bads')
    # Use a subset of channels for plotting speed
    picks = np.round(np.linspace(0, len(picks) + 1, n_chan)).astype(int)
    with pytest.warns(RuntimeWarning, match='projection'):
        epochs = Epochs(raw, events[:stop], event_id, tmin, tmax, picks=picks,
                        proj=False)
    epochs.info.normalize_proj()  # avoid warnings
    return epochs


def test_plot_epochs(capsys):
    """Test epoch plotting."""
    epochs = _get_epochs().load_data()
    assert len(epochs.events) == 1
    epochs.info['lowpass'] = 10.  # allow heavy decim during plotting
    epochs.plot(scalings=None, title='Epochs')
    plt.close('all')
    # covariance / whitening
    cov = read_cov(cov_fname)
    assert len(cov['names']) == 366  # all channels
    assert cov['bads'] == []
    assert epochs.info['bads'] == []  # all good
    with pytest.warns(RuntimeWarning, match='projection'):
        epochs.plot(noise_cov=cov)
    plt.close('all')
    # add a channel to the epochs.info['bads']
    epochs.info['bads'] = [epochs.ch_names[0]]
    with pytest.warns(RuntimeWarning, match='projection'):
        epochs.plot(noise_cov=cov)
    plt.close('all')
    # add a channel to cov['bads']
    cov['bads'] = [epochs.ch_names[1]]
    with pytest.warns(RuntimeWarning, match='projection'):
        epochs.plot(noise_cov=cov)
    plt.close('all')
    # have a data channels missing from the covariance
    cov['names'] = cov['names'][:306]
    cov['data'] = cov['data'][:306][:306]
    with pytest.warns(RuntimeWarning, match='projection'):
        epochs.plot(noise_cov=cov)
    plt.close('all')
    # other options
    fig = epochs[0].plot(picks=[0, 2, 3], scalings=None)
    fig.canvas.key_press_event('escape')
    plt.close('all')
    keystotest = ['b', 'b', 'left', 'right', 'up', 'down',
                  'pageup', 'pagedown', '-', '+', '=',
                  'f11', 'home', '?', 'h', 'o', 'end']
    fig = epochs.plot()
    with pytest.warns(None):  # sometimes matplotlib warns about limits
        for key in keystotest:
            fig.canvas.key_press_event(key)
    fig.canvas.scroll_event(0.5, 0.5, -0.5)  # scroll down
    fig.canvas.scroll_event(0.5, 0.5, 0.5)  # scroll up
    fig.canvas.resize_event()
    fig.canvas.close_event()  # closing and epoch dropping
    plt.close('all')
    pytest.raises(ValueError, epochs.plot, picks=[])
    plt.close('all')
    fig = epochs.plot(events=epochs.events)
    # test mouse clicks
    data_ax = fig.get_axes()[0]
    x = data_ax.get_xlim()[1] / 2
    y = data_ax.get_ylim()[0] / 2
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
    # gh-5906
    epochs = _get_epochs(None).load_data()
    epochs.load_data()
    assert len(epochs) == 7
    epochs.info['bads'] = [epochs.ch_names[0]]
    capsys.readouterr()
    fig = epochs.plot(n_epochs=3)
    data_ax = fig.get_axes()[0]
    _fake_click(fig, data_ax, [-0.1, 0.9])  # click on y-label
    fig.canvas.key_press_event('right')  # move right
    x = fig.get_axes()[0].get_xlim()[1] / 6.
    y = fig.get_axes()[0].get_ylim()[0] / 2
    _fake_click(fig, data_ax, [x, y], xform='data')  # mark a bad epoch
    fig.canvas.key_press_event('left')  # move back
    out, err = capsys.readouterr()
    assert 'out of bounds' not in out
    assert 'out of bounds' not in err
    fig.canvas.close_event()
    assert len(epochs) == 6
    plt.close('all')


def test_plot_epochs_image():
    """Test plotting of epochs image."""
    epochs = _get_epochs()
    epochs.plot_image(picks=[1, 2])
    epochs.plot_image(picks='mag')
    overlay_times = [0.1]
    epochs.plot_image(picks=[1], order=[0], overlay_times=overlay_times,
                      vmin=0.01, title="test"
                      )
    epochs.plot_image(picks=[1], overlay_times=overlay_times, vmin=-0.001,
                      vmax=0.001)
    pytest.raises(ValueError, epochs.plot_image,
                  picks=[1], overlay_times=[0.1, 0.2])
    pytest.raises(ValueError, epochs.plot_image,
                  picks=[1], order=[0, 1])
    pytest.raises(ValueError, epochs.plot_image, axes=dict(), group_by=list(),
                  combine='mean')
    pytest.raises(ValueError, epochs.plot_image, axes=list(), group_by=dict(),
                  combine='mean')
    pytest.raises(ValueError, epochs.plot_image, group_by='error',
                  picks=[1, 2])
    pytest.raises(ValueError, epochs.plot_image, units={"hi": 1},
                  scalings={"ho": 1})
    assert len(epochs.plot_image(picks=["eeg", "mag", "grad"])) < 6
    epochs.load_data().pick_types(meg='mag')
    epochs.info.normalize_proj()
    epochs.plot_image(group_by='type', combine='mean')
    epochs.plot_image(group_by={"1": [1, 2], "2": [1, 2]}, combine='mean')
    epochs.plot_image(vmin=lambda x: x.min())
    pytest.raises(ValueError, epochs.plot_image, axes=1, fig=2)
    ts_args = dict(show_sensors=False)
    with pytest.warns(RuntimeWarning, match='fall outside'):
        epochs.plot_image(overlay_times=[1.1], combine="gfp", ts_args=ts_args)
    pytest.raises(ValueError, epochs.plot_image, combine='error',
                  ts_args=ts_args)
    with pytest.raises(NotImplementedError, match='currently'):
        epochs.plot_image(ts_args=dict(invert_y=True))

    plt.close('all')


def test_plot_drop_log():
    """Test plotting a drop log."""
    epochs = _get_epochs()
    pytest.raises(ValueError, epochs.plot_drop_log)
    epochs.drop_bad()
    epochs.plot_drop_log()
    plot_drop_log([['One'], [], []])
    plot_drop_log([['One'], ['Two'], []])
    plot_drop_log([['One'], ['One', 'Two'], []])
    plt.close('all')


def test_plot_butterfly():
    """Test butterfly view in epochs browse window."""
    rng = np.random.RandomState(0)
    n_epochs, n_channels, n_times = 50, 30, 20
    sfreq = 1000.
    data = np.sin(rng.randn(n_epochs, n_channels, n_times))
    events = np.array([np.arange(n_epochs), [0] * n_epochs, np.ones([n_epochs],
                       dtype=np.int)]).T
    chanlist = ['eeg' if chan < n_channels // 3 else 'ecog'
                if chan < n_channels // 2 else 'seeg'
                for chan in range(n_channels)]
    info = create_info(n_channels, sfreq, chanlist)
    epochs = EpochsArray(data, info, events)
    fig = epochs.plot(butterfly=True)
    keystotest = ['b', 'b', 'left', 'right', 'up', 'down',
                  'pageup', 'pagedown', '-', '+', '=',
                  'f11', 'home', '?', 'h', 'o', 'end']
    for key in keystotest:
        fig.canvas.key_press_event(key)
    fig.canvas.scroll_event(0.5, 0.5, -0.5)  # scroll down
    fig.canvas.scroll_event(0.5, 0.5, 0.5)  # scroll up
    fig.canvas.resize_event()
    fig.canvas.close_event()  # closing and epoch dropping
    plt.close('all')


def test_plot_psd_epochs():
    """Test plotting epochs psd (+topomap)."""
    epochs = _get_epochs()
    epochs.plot_psd()
    pytest.raises(RuntimeError, epochs.plot_psd_topomap,
                  bands=[(0, 0.01, 'foo')])  # no freqs in range
    epochs.plot_psd_topomap()
    plt.close('all')


run_tests_if_main()
