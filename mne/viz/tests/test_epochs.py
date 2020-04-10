# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Jaakko Leppakangas <jaeilepp@student.jyu.fi>
#          Daniel McCloy <dan.mccloy@gmail.com>
#
# License: Simplified BSD

import os.path as op

import numpy as np
import pytest
import matplotlib.pyplot as plt

from mne import (read_events, Epochs, pick_types, read_cov, create_info,
                 EpochsArray)
from mne.channels import read_layout
from mne.io import read_raw_fif, read_raw_ctf
from mne.utils import run_tests_if_main
from mne.viz import plot_drop_log
from mne.viz.utils import _fake_click
from mne.datasets import testing
from mne.event import make_fixed_length_events


base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
evoked_fname = op.join(base_dir, 'test-ave.fif')
raw_fname = op.join(base_dir, 'test_raw.fif')
cov_fname = op.join(base_dir, 'test-cov.fif')
event_name = op.join(base_dir, 'test-eve.fif')
event_id, tmin, tmax = 1, -0.1, 1.0
layout = read_layout('Vectorview-all')
test_base_dir = testing.data_path(download=False)
ctf_fname = op.join(test_base_dir, 'CTF', 'testdata_ctf.ds')


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


def test_plot_epochs_basic(capsys):
    """Test epoch plotting."""
    epochs = _get_epochs().load_data()
    assert len(epochs.events) == 1
    epochs.info['lowpass'] = 10.  # allow heavy decim during plotting
    fig = epochs.plot(scalings=None, title='Epochs')
    ticks = [x.get_text() for x in fig.axes[0].get_xticklabels()]
    assert ticks == ['0']
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
    # check the epochs color plotting
    epoch_colors = [['r'] * len(epochs.ch_names) for _ in
                    range(len(epochs.events))]
    epochs.plot(epoch_colors=epoch_colors)
    with pytest.raises(TypeError, match='must be an instance of'):
        epochs.plot(epoch_colors='r')
    with pytest.raises(ValueError, match='must be list of len'):
        epochs.plot(epoch_colors=['r'] * len(epochs.ch_names))
    epoch_colors[0] = None
    with pytest.raises(TypeError, match='must be an instance of'):
        epochs.plot(epoch_colors=epoch_colors)
    epoch_colors[0] = ['r'] * 5
    with pytest.raises(ValueError, match='epoch_colors for the 0th epoch has'
                       ' length'):
        epochs.plot(epoch_colors=epoch_colors)
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


def test_plot_epochs_nodata():
    """Test plotting of epochs when no data channels are present."""
    data = np.random.RandomState(0).randn(10, 2, 1000)
    info = create_info(2, 1000., 'stim')
    epochs = EpochsArray(data, info)
    with pytest.raises(ValueError, match='consider passing picks explicitly'):
        epochs.plot()


def test_plot_epochs_image():
    """Test plotting of epochs image.

    Note that some of these tests that should pass are triggering MPL
    UserWarnings about tight_layout not being applied ("tight_layout cannot
    make axes width small enough to accommodate all axes decorations"). Calling
    `plt.close('all')` just before the offending test seems to prevent this
    warning, though it's unclear why.
    """
    plt.close('all')
    epochs = _get_epochs()
    figs = epochs.plot_image()
    assert len(figs) == 2  # one fig per ch_type (test data has mag, grad)
    assert len(plt.get_fignums()) == 2
    figs = epochs.plot_image()
    assert len(figs) == 2
    assert len(plt.get_fignums()) == 4  # should create new figures
    epochs.plot_image(picks='mag', sigma=0.1)
    epochs.plot_image(picks=[0, 1], combine='mean',
                      ts_args=dict(show_sensors=False))
    epochs.plot_image(picks=[1], order=[0], overlay_times=[0.1], vmin=0.01,
                      title='test')
    plt.close('all')
    epochs.plot_image(picks=[1], overlay_times=[0.1], vmin=-0.001, vmax=0.001)
    plt.close('all')
    epochs.plot_image(picks=[1], vmin=lambda x: x.min())
    # test providing figure
    fig, axs = plt.subplots(3, 1)
    epochs.plot_image(picks=[1], fig=fig)
    # test providing axes instance
    epochs.plot_image(picks=[1], axes=axs[0], evoked=False, colorbar=False)
    plt.close('all')
    # test order=callable
    epochs.plot_image(picks=[0, 1],
                      order=lambda times, data: np.arange(len(data))[::-1])
    # test warning
    with pytest.warns(RuntimeWarning, match='Only one channel in group'):
        epochs.plot_image(picks=[1], combine='mean')
    # group_by should be a dict
    with pytest.raises(TypeError, match="dict or None"):
        epochs.plot_image(group_by='foo')
    # units and scalings keys must match
    with pytest.raises(ValueError, match='Scalings and units must have the'):
        epochs.plot_image(units=dict(hi=1), scalings=dict(ho=1))
    plt.close('all')
    # test invert_y
    epochs.plot_image(ts_args=dict(invert_y=True))
    # can't combine different sensor types
    with pytest.raises(ValueError, match='Cannot combine sensors of differ'):
        epochs.plot_image(group_by=dict(foo=[0, 1, 2]))
    # can't pass both fig and axes
    with pytest.raises(ValueError, match='one of "fig" or "axes" must be'):
        epochs.plot_image(fig='foo', axes='bar')
    # wrong number of axes in fig
    with pytest.raises(ValueError, match='"fig" must contain . axes, got .'):
        epochs.plot_image(fig=plt.figure())
    # only 1 group allowed when fig is passed
    with pytest.raises(ValueError, match='"group_by" can only have one group'):
        fig, axs = plt.subplots(3, 1)
        epochs.plot_image(fig=fig, group_by=dict(foo=[0, 1], bar=[5, 6]))
        del fig, axs
    plt.close('all')
    # must pass correct number of axes (1, 2, or 3)
    with pytest.raises(ValueError, match='is a list, can only plot one group'):
        fig, axs = plt.subplots(1, 3)
        epochs.plot_image(axes=axs)
    for length, kwargs in ([3, dict()],
                           [2, dict(evoked=False)],
                           [2, dict(colorbar=False)],
                           [1, dict(evoked=False, colorbar=False)]):
        fig, axs = plt.subplots(1, length + 1)
        epochs.plot_image(picks='mag', axes=axs[:length], **kwargs)
        with pytest.raises(ValueError, match='"axes" must be length ., got .'):
            epochs.plot_image(picks='mag', axes=axs, **kwargs)
    plt.close('all')
    # mismatch between axes dict keys and group_by dict keys
    with pytest.raises(ValueError, match='must match the keys in "group_by"'):
        epochs.plot_image(axes=dict())
    # wrong number of axes in dict
    match = 'each value in "axes" must be a list of . axes, got .'
    with pytest.raises(ValueError, match=match):
        epochs.plot_image(axes=dict(foo=axs[:2], bar=axs[:3]),
                          group_by=dict(foo=[0, 1], bar=[5, 6]))
    # bad value of "combine"
    with pytest.raises(ValueError, match='"combine" must be None, a callable'):
        epochs.plot_image(combine='foo')
    # mismatched picks and overlay_times
    with pytest.raises(ValueError, match='size of overlay_times parameter'):
        epochs.plot_image(picks=[1], overlay_times=[0.1, 0.2])
    # bad overlay times
    with pytest.warns(RuntimeWarning, match='fall outside'):
        epochs.plot_image(overlay_times=[999.])
    # mismatched picks and order
    with pytest.raises(ValueError, match='must match the length of the data'):
        epochs.plot_image(picks=[1], order=[0, 1])
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
    epochs.load_data()
    epochs.plot_psd(average=True, spatial_colors=False)
    epochs.plot_psd(average=False, spatial_colors=True)
    epochs.plot_psd(average=False, spatial_colors=False)
    pytest.raises(RuntimeError, epochs.plot_psd_topomap,
                  bands=[(0, 0.01, 'foo')])  # no freqs in range
    epochs.plot_psd_topomap()

    # with a flat channel
    err_str = 'for channel %s' % epochs.ch_names[2]
    epochs.get_data()[0, 2, :] = 0
    for dB in [True, False]:
        with pytest.warns(UserWarning, match=err_str):
            epochs.plot_psd(dB=dB)

    plt.close('all')


@testing.requires_testing_data
def test_plot_epochs_ctf():
    """Test of basic CTF plotting."""
    raw = read_raw_ctf(ctf_fname, preload=True)
    raw.pick_channels(['UDIO001', 'UPPT001', 'SCLK01-177',
                       'BG1-4304', 'MLC11-4304', 'MLC11-4304',
                       'EEG058', 'UADC007-4302'])
    evts = make_fixed_length_events(raw)
    epochs = Epochs(raw, evts, preload=True)
    epochs.plot()
    plt.close('all')

    # test butterfly
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


@testing.requires_testing_data
def test_plot_psd_epochs_ctf():
    """Test plotting CTF epochs psd (+topomap)."""
    raw = read_raw_ctf(ctf_fname, preload=True)
    evts = make_fixed_length_events(raw)
    epochs = Epochs(raw, evts, preload=True)
    pytest.raises(RuntimeError, epochs.plot_psd_topomap,
                  bands=[(0, 0.01, 'foo')])  # no freqs in range
    epochs.plot_psd_topomap()

    # EEG060 is flat in this dataset
    for dB in [True, False]:
        with pytest.warns(UserWarning, match='for channel EEG060'):
            epochs.plot_psd(dB=dB)
    epochs.drop_channels(['EEG060'])
    epochs.plot_psd(spatial_colors=False, average=False)
    plt.close('all')


run_tests_if_main()
