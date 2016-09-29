# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: Simplified BSD

import numpy as np
import os.path as op
import warnings

from numpy.testing import assert_raises, assert_equal

from mne import read_events, pick_types, Annotations
from mne.io import read_raw_fif
from mne.utils import requires_version, run_tests_if_main
from mne.viz.utils import _fake_click
from mne.viz import plot_raw, plot_sensors

# Set our plotters to test mode
import matplotlib
matplotlib.use('Agg')  # for testing don't use X server

warnings.simplefilter('always')  # enable b/c these tests throw warnings

base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')
event_name = op.join(base_dir, 'test-eve.fif')


def _get_raw():
    """Get raw data."""
    raw = read_raw_fif(raw_fname, preload=True, add_eeg_ref=False)
    # Throws a warning about a changed unit.
    with warnings.catch_warnings(record=True):
        raw.set_channel_types({raw.ch_names[0]: 'ias'})
    raw.pick_channels(raw.ch_names[:9])
    raw.info.normalize_proj()  # Fix projectors after subselection
    return raw


def _get_events():
    """Get events."""
    return read_events(event_name)


@requires_version('matplotlib', '1.2')
def test_plot_raw():
    """Test plotting of raw data."""
    import matplotlib.pyplot as plt
    raw = _get_raw()
    events = _get_events()
    plt.close('all')  # ensure all are closed
    with warnings.catch_warnings(record=True):
        fig = raw.plot(events=events, show_options=True)
        # test mouse clicks
        x = fig.get_axes()[0].lines[1].get_xdata().mean()
        y = fig.get_axes()[0].lines[1].get_ydata().mean()
        data_ax = fig.get_axes()[0]
        _fake_click(fig, data_ax, [x, y], xform='data')  # mark a bad channel
        _fake_click(fig, data_ax, [x, y], xform='data')  # unmark a bad channel
        _fake_click(fig, data_ax, [0.5, 0.999])  # click elsewhere in 1st axes
        _fake_click(fig, data_ax, [-0.1, 0.9])  # click on y-label
        _fake_click(fig, fig.get_axes()[1], [0.5, 0.5])  # change time
        _fake_click(fig, fig.get_axes()[2], [0.5, 0.5])  # change channels
        _fake_click(fig, fig.get_axes()[3], [0.5, 0.5])  # open SSP window
        fig.canvas.button_press_event(1, 1, 1)  # outside any axes
        fig.canvas.scroll_event(0.5, 0.5, -0.5)  # scroll down
        fig.canvas.scroll_event(0.5, 0.5, 0.5)  # scroll up
        # sadly these fail when no renderer is used (i.e., when using Agg):
        # ssp_fig = set(plt.get_fignums()) - set([fig.number])
        # assert_equal(len(ssp_fig), 1)
        # ssp_fig = plt.figure(list(ssp_fig)[0])
        # ax = ssp_fig.get_axes()[0]  # only one axis is used
        # t = [c for c in ax.get_children() if isinstance(c,
        #      matplotlib.text.Text)]
        # pos = np.array(t[0].get_position()) + 0.01
        # _fake_click(ssp_fig, ssp_fig.get_axes()[0], pos, xform='data')  # off
        # _fake_click(ssp_fig, ssp_fig.get_axes()[0], pos, xform='data')  # on
        #  test keypresses
        fig.canvas.key_press_event('escape')
        fig.canvas.key_press_event('down')
        fig.canvas.key_press_event('up')
        fig.canvas.key_press_event('right')
        fig.canvas.key_press_event('left')
        fig.canvas.key_press_event('o')
        fig.canvas.key_press_event('-')
        fig.canvas.key_press_event('+')
        fig.canvas.key_press_event('=')
        fig.canvas.key_press_event('pageup')
        fig.canvas.key_press_event('pagedown')
        fig.canvas.key_press_event('home')
        fig.canvas.key_press_event('end')
        fig.canvas.key_press_event('?')
        fig.canvas.key_press_event('f11')
        fig.canvas.key_press_event('escape')
        # Color setting
        assert_raises(KeyError, raw.plot, event_color={0: 'r'})
        assert_raises(TypeError, raw.plot, event_color={'foo': 'r'})
        annot = Annotations([10, 10 + raw.first_samp / raw.info['sfreq']],
                            [10, 10], ['test', 'test'], raw.info['meas_date'])
        raw.annotations = annot
        fig = plot_raw(raw, events=events, event_color={-1: 'r', 998: 'b'})
        plt.close('all')
        for order in ['position', 'selection', range(len(raw.ch_names))[::-4],
                      [1, 2, 4, 6]]:
            fig = raw.plot(order=order)
            x = fig.get_axes()[0].lines[1].get_xdata()[10]
            y = fig.get_axes()[0].lines[1].get_ydata()[10]
            _fake_click(fig, data_ax, [x, y], xform='data')  # mark bad
            fig.canvas.key_press_event('down')  # change selection
            _fake_click(fig, fig.get_axes()[2], [0.5, 0.5])  # change channels
            if order in ('position', 'selection'):
                sel_fig = plt.figure(1)
                topo_ax = sel_fig.axes[1]
                _fake_click(sel_fig, topo_ax, [-0.425, 0.20223853],
                            xform='data')
                fig.canvas.key_press_event('down')
                fig.canvas.key_press_event('up')
                fig.canvas.scroll_event(0.5, 0.5, -1)  # scroll down
                fig.canvas.scroll_event(0.5, 0.5, 1)  # scroll up
                _fake_click(sel_fig, topo_ax, [-0.5, 0.], xform='data')
                _fake_click(sel_fig, topo_ax, [0.5, 0.], xform='data',
                            kind='motion')
                _fake_click(sel_fig, topo_ax, [0.5, 0.5], xform='data',
                            kind='motion')
                _fake_click(sel_fig, topo_ax, [-0.5, 0.5], xform='data',
                            kind='release')

            plt.close('all')
        # test if meas_date has only one element
        raw.info['meas_date'] = np.array([raw.info['meas_date'][0]],
                                         dtype=np.int32)
        raw.annotations = Annotations([1 + raw.first_samp / raw.info['sfreq']],
                                      [5], ['bad'])
        raw.plot()
        plt.close('all')


@requires_version('scipy', '0.10')
def test_plot_raw_filtered():
    """Test filtering of raw plots."""
    raw = _get_raw()
    assert_raises(ValueError, raw.plot, lowpass=raw.info['sfreq'] / 2.)
    assert_raises(ValueError, raw.plot, highpass=0)
    assert_raises(ValueError, raw.plot, lowpass=1, highpass=1)
    assert_raises(ValueError, raw.plot, lowpass=1, filtorder=0)
    assert_raises(ValueError, raw.plot, clipping='foo')
    raw.plot(lowpass=1, clipping='transparent')
    raw.plot(highpass=1, clipping='clamp')
    raw.plot(highpass=1, lowpass=2)


@requires_version('scipy', '0.12')
def test_plot_raw_psd():
    """Test plotting of raw psds."""
    import matplotlib.pyplot as plt
    raw = _get_raw()
    # normal mode
    raw.plot_psd(tmax=2.0)
    # specific mode
    picks = pick_types(raw.info, meg='mag', eeg=False)[:4]
    raw.plot_psd(picks=picks, area_mode='range')
    ax = plt.axes()
    # if ax is supplied:
    assert_raises(ValueError, raw.plot_psd, ax=ax)
    raw.plot_psd(picks=picks, ax=ax)
    plt.close('all')
    ax = plt.axes()
    assert_raises(ValueError, raw.plot_psd, ax=ax)
    ax = [ax, plt.axes()]
    raw.plot_psd(ax=ax)
    plt.close('all')
    # topo psd
    raw.plot_psd_topo()
    plt.close('all')
    # with a flat channel
    raw[5, :] = 0
    assert_raises(ValueError, raw.plot_psd)


@requires_version('matplotlib', '1.2')
def test_plot_sensors():
    """Test plotting of sensor array."""
    import matplotlib.pyplot as plt
    raw = _get_raw()
    fig = raw.plot_sensors('3d')
    _fake_click(fig, fig.gca(), (-0.08, 0.67))
    raw.plot_sensors('topomap', ch_type='mag')
    ax = plt.subplot(111)
    raw.plot_sensors(ch_groups='position', axes=ax)
    raw.plot_sensors(ch_groups='selection')
    raw.plot_sensors(ch_groups=[[0, 1, 2], [3, 4]])
    assert_raises(ValueError, raw.plot_sensors, ch_groups='asd')
    assert_raises(TypeError, plot_sensors, raw)  # needs to be info
    assert_raises(ValueError, plot_sensors, raw.info, kind='sasaasd')
    plt.close('all')
    fig, sels = raw.plot_sensors('select', show_names=True)
    ax = fig.axes[0]

    # Click with no sensors
    _fake_click(fig, ax, (0., 0.), xform='data')
    _fake_click(fig, ax, (0, 0.), xform='data', kind='release')
    assert_equal(len(fig.lasso.selection), 0)

    # Lasso with 1 sensor
    _fake_click(fig, ax, (-0.5, 0.5), xform='data')
    plt.draw()
    _fake_click(fig, ax, (0., 0.5), xform='data', kind='motion')
    _fake_click(fig, ax, (0., 0.), xform='data', kind='motion')
    fig.canvas.key_press_event('control')
    _fake_click(fig, ax, (-0.5, 0.), xform='data', kind='release')
    assert_equal(len(fig.lasso.selection), 1)

    _fake_click(fig, ax, (-0.09, -0.43), xform='data')  # single selection
    assert_equal(len(fig.lasso.selection), 2)
    _fake_click(fig, ax, (-0.09, -0.43), xform='data')  # deselect
    assert_equal(len(fig.lasso.selection), 1)
    plt.close('all')

run_tests_if_main()
