# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Jaakko Leppakangas <jaeilepp@student.jyu.fi>
#          Daniel McCloy <dan@mccloy.info>
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
from mne.utils import run_tests_if_main, _click_ch_name, _close_event
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
                        proj=False, preload=False)
    epochs.info.normalize_proj()  # avoid warnings
    return epochs


@pytest.fixture()
def epochs():
    """Get minimal, pre-loaded epochs data suitable for most tests."""
    return _get_epochs().load_data()


def test_plot_epochs_not_preloaded():
    """Test plotting non-preloaded epochs."""
    epochs = _get_epochs()
    assert epochs._data is None
    epochs.plot()
    assert epochs._data is None


def test_plot_epochs_basic(epochs, capsys):
    """Test epoch plotting."""
    assert len(epochs.events) == 1
    epochs.info['lowpass'] = 10.  # allow heavy decim during plotting
    fig = epochs.plot(scalings=None, title='Epochs')
    ticks = [x.get_text() for x in fig.mne.ax_main.get_xticklabels(minor=True)]
    assert ticks == ['2']
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
    # have a data channel missing from the covariance
    cov['names'] = cov['names'][:306]
    cov['data'] = cov['data'][:306][:306]
    with pytest.warns(RuntimeWarning, match='projection'):
        epochs.plot(noise_cov=cov)
    plt.close('all')
    # other options
    fig = epochs[0].plot(picks=[0, 2, 3], scalings=None)
    fig.canvas.key_press_event('escape')
    with pytest.raises(ValueError, match='No appropriate channels found'):
        epochs.plot(picks=[])
    # gh-5906
    epochs = _get_epochs(None).load_data()
    epochs.load_data()
    assert len(epochs) == 7
    epochs.info['bads'] = [epochs.ch_names[0]]
    capsys.readouterr()
    # test title error handling
    with pytest.raises(TypeError, match='title must be None or a string, got'):
        epochs.plot(title=7)
    # test auto-generated title, and selection mode
    epochs.plot(group_by='selection', title='')


@pytest.mark.parametrize('scalings', (dict(mag=1e-12, grad=1e-11, stim='auto'),
                                      None, 'auto'))
def test_plot_epochs_scalings(epochs, scalings):
    """Test the valid options for scalings."""
    epochs.plot(scalings=scalings)


def test_plot_epochs_colors(epochs):
    """Test epoch_colors, for compatibility with autoreject."""
    epoch_colors = [['r'] * len(epochs.ch_names) for _ in
                    range(len(epochs.events))]
    epochs.plot(epoch_colors=epoch_colors)
    with pytest.raises(ValueError, match='length equal to the number of epo'):
        epochs.plot(epoch_colors=[['r'], ['b']])  # epochs obj has only 1 epoch
    with pytest.raises(ValueError, match=r'epoch colors for epoch \d+ has'):
        epochs.plot(epoch_colors=[['r']])  # need 1 color for each channel
    # also test event_colors
    with pytest.warns(DeprecationWarning, match='replaced by event_color in'):
        epochs.plot(event_colors='r')
    with pytest.warns(DeprecationWarning,
                      match='in 0.23. Since you passed values for both'):
        epochs.plot(event_colors='r', event_color='b')


def test_plot_epochs_scale_bar(epochs):
    """Test scale bar for epochs."""
    fig = epochs.plot()
    fig.canvas.key_press_event('s')  # default is to not show scalebars
    ax = fig.mne.ax_main
    assert len(ax.texts) == 2  # only mag & grad in this instance
    texts = tuple(t.get_text().strip() for t in ax.texts)
    wants = ('800.0 fT/cm', '2000.0 fT')
    assert texts == wants


def test_plot_epochs_clicks(epochs, capsys):
    """Test plot_epochs mouse interaction."""
    fig = epochs.plot(events=epochs.events)
    data_ax = fig.mne.ax_main
    x = fig.mne.traces[0].get_xdata()[3]
    y = fig.mne.traces[0].get_ydata()[3]
    n_epochs = len(epochs)
    epoch_num = fig.mne.inst.selection[0]
    # test (un)marking bad epochs
    _fake_click(fig, data_ax, [x, y], xform='data')  # mark a bad epoch
    assert epoch_num in fig.mne.bad_epochs
    _fake_click(fig, data_ax, [x, y], xform='data')  # unmark it
    assert epoch_num not in fig.mne.bad_epochs
    _fake_click(fig, data_ax, [x, y], xform='data')  # mark it bad again
    assert epoch_num in fig.mne.bad_epochs
    # test vline
    fig.canvas.key_press_event('escape')  # close and drop epochs
    _close_event(fig)  # XXX workaround, MPL Agg doesn't trigger close event
    assert(n_epochs - 1 == len(epochs))
    # test marking bad channels
    epochs = _get_epochs(None).load_data()  # need more than 1 epoch this time
    fig = epochs.plot(n_epochs=3)
    data_ax = fig.mne.ax_main
    first_ch = data_ax.get_yticklabels()[0].get_text()
    assert first_ch not in fig.mne.info['bads']
    _click_ch_name(fig, ch_index=0, button=1)  # click ch name to mark bad
    assert first_ch in fig.mne.info['bads']
    # test clicking scrollbars
    _fake_click(fig, fig.mne.ax_vscroll, [0.5, 0.5])
    _fake_click(fig, fig.mne.ax_hscroll, [0.5, 0.5])
    # test moving bad epoch offscreen
    fig.canvas.key_press_event('right')  # move right
    x = fig.mne.traces[0].get_xdata()[-3]
    y = fig.mne.traces[0].get_ydata()[-3]
    _fake_click(fig, data_ax, [x, y], xform='data')  # mark a bad epoch
    fig.canvas.key_press_event('left')  # move back
    out, err = capsys.readouterr()
    assert 'out of bounds' not in out
    assert 'out of bounds' not in err
    fig.canvas.key_press_event('escape')
    _close_event(fig)  # XXX workaround, MPL Agg doesn't trigger close event
    assert len(epochs) == 6
    # test rightclick → image plot
    fig = epochs.plot()
    _click_ch_name(fig, ch_index=0, button=3)  # show image plot
    assert len(fig.mne.child_figs) == 1
    # test scroll wheel
    fig.canvas.scroll_event(0.5, 0.5, -0.5)  # scroll down
    fig.canvas.scroll_event(0.5, 0.5, 0.5)  # scroll up


def test_plot_epochs_keypresses():
    """Test plot_epochs keypress interaction."""
    epochs = _get_epochs(stop=15).load_data()  # we need more than 1 epoch
    epochs.drop_bad(dict(mag=4e-12))  # for histogram plot coverage
    fig = epochs.plot(n_epochs=3)
    data_ax = fig.mne.ax_main
    # make sure green vlines are visible first (for coverage)
    sample_idx = len(epochs.times) // 2  # halfway through the first epoch
    x = fig.mne.traces[0].get_xdata()[sample_idx]
    y = (fig.mne.traces[0].get_ydata()[sample_idx]
         + fig.mne.traces[1].get_ydata()[sample_idx]) / 2
    _fake_click(fig, data_ax, [x, y], xform='data')  # click between traces
    # test keys
    keys = ('pagedown', 'down', 'up', 'down', 'right', 'left', '-', '+', '=',
            'd', 'd', 'pageup', 'home', 'shift+right', 'end', 'shift+left',
            'z', 'z', 's', 's', 'f11', '?', 'h', 'j', 'b')
    for key in keys * 2:  # test twice → once in normal, once in butterfly view
        fig.canvas.key_press_event(key)
    _fake_click(fig, data_ax, [x, y], xform='data', button=3)  # remove vlines


def test_epochs_plot_sensors(epochs):
    """Test sensor plotting."""
    epochs.plot_sensors()


def test_plot_epochs_nodata():
    """Test plotting of epochs when no data channels are present."""
    data = np.random.RandomState(0).randn(10, 2, 1000)
    info = create_info(2, 1000., 'stim')
    epochs = EpochsArray(data, info)
    with pytest.raises(ValueError, match='consider passing picks explicitly'):
        epochs.plot()


def test_plot_epochs_image(epochs):
    """Test plotting of epochs image.

    Note that some of these tests that should pass are triggering MPL
    UserWarnings about tight_layout not being applied ("tight_layout cannot
    make axes width small enough to accommodate all axes decorations"). Calling
    `plt.close('all')` just before the offending test seems to prevent this
    warning, though it's unclear why.
    """
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
    epochs = _get_epochs()  # not loaded
    with pytest.raises(ValueError, match='bad epochs have not yet been'):
        epochs.plot_drop_log()
    epochs.drop_bad()
    epochs.plot_drop_log()
    plot_drop_log((('One',), (), ()))
    plot_drop_log((('One',), ('Two',), ()))
    plot_drop_log((('One',), ('One', 'Two'), ()))
    for arg in ([], ([],), (1,)):
        with pytest.raises(TypeError, match='tuple of tuple of str'):
            plot_drop_log(arg)
    plt.close('all')


def test_plot_psd_epochs(epochs):
    """Test plotting epochs psd (+topomap)."""
    epochs.plot_psd(average=True, spatial_colors=False)
    epochs.plot_psd(average=False, spatial_colors=True)
    epochs.plot_psd(average=False, spatial_colors=False)
    # test plot_psd_topomap errors
    with pytest.raises(RuntimeError, match='No frequencies in band'):
        epochs.plot_psd_topomap(bands=[(0, 0.01, 'foo')])
    plt.close('all')
    # test defaults
    fig = epochs.plot_psd_topomap()
    assert len(fig.axes) == 10  # default: 5 bands (δ, θ, α, β, γ) + colorbars
    # test joint vlim
    fig = epochs.plot_psd_topomap(vlim='joint')
    vmin_0 = fig.axes[0].images[0].norm.vmin
    vmax_0 = fig.axes[0].images[0].norm.vmax
    assert all(vmin_0 == ax.images[0].norm.vmin for ax in fig.axes[1:5])
    assert all(vmax_0 == ax.images[0].norm.vmax for ax in fig.axes[1:5])
    # test support for single-bin bands
    fig = epochs.plot_psd_topomap(bands=[(20, '20 Hz'), (15, 25, '15-25 Hz')])
    # test with a flat channel
    err_str = 'for channel %s' % epochs.ch_names[2]
    epochs.get_data()[0, 2, :] = 0
    for dB in [True, False]:
        with pytest.warns(UserWarning, match=err_str):
            epochs.plot_psd(dB=dB)


def test_plot_psdtopo_nirs(fnirs_epochs):
    """Test plotting of PSD topography for nirs data."""
    bands = [(0.2, '0.2 Hz'), (0.4, '0.4 Hz'), (0.8, '0.8 Hz')]
    fig = fnirs_epochs.plot_psd_topomap(bands=bands)
    assert len(fig.axes) == 6  # 3 band x (plot + cmap)


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
    keys = ('b', 'b', 'pagedown', 'down', 'up', 'down', 'right', 'left', '-',
            '+', '=', 'd', 'd', 'pageup', 'home', 'end', 'z', 'z', 's', 's',
            'f11', '?', 'h', 'j')
    for key in keys:
        fig.canvas.key_press_event(key)
    fig.canvas.scroll_event(0.5, 0.5, -0.5)  # scroll down
    fig.canvas.scroll_event(0.5, 0.5, 0.5)  # scroll up
    fig.canvas.resize_event()
    fig.canvas.key_press_event('escape')  # close and drop epochs


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


run_tests_if_main()
