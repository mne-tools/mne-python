# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: Simplified BSD

import os.path as op
import warnings

from numpy.testing import assert_raises

from mne import io, read_events, pick_types
from mne.utils import requires_scipy_version, run_tests_if_main
from mne.viz.utils import _fake_click

# Set our plotters to test mode
import matplotlib
matplotlib.use('Agg')  # for testing don't use X server

warnings.simplefilter('always')  # enable b/c these tests throw warnings

base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')
event_name = op.join(base_dir, 'test-eve.fif')


def _get_raw():
    raw = io.Raw(raw_fname, preload=True)
    raw.pick_channels(raw.ch_names[:9])
    return raw


def _get_events():
    return read_events(event_name)


def test_plot_raw():
    """Test plotting of raw data
    """
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
        _fake_click(fig, fig.get_axes()[1], [0.5, 0.5])  # change time
        _fake_click(fig, fig.get_axes()[2], [0.5, 0.5])  # change channels
        _fake_click(fig, fig.get_axes()[3], [0.5, 0.5])  # open SSP window
        fig.canvas.button_press_event(1, 1, 1)  # outside any axes
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
        fig.canvas.key_press_event('escape')
        # Color setting
        assert_raises(KeyError, raw.plot, event_color={0: 'r'})
        assert_raises(TypeError, raw.plot, event_color={'foo': 'r'})
        fig = raw.plot(events=events, event_color={-1: 'r', 998: 'b'})
        plt.close('all')


@requires_scipy_version('0.10')
def test_plot_raw_filtered():
    """Test filtering of raw plots
    """
    raw = _get_raw()
    assert_raises(ValueError, raw.plot, lowpass=raw.info['sfreq'] / 2.)
    assert_raises(ValueError, raw.plot, highpass=0)
    assert_raises(ValueError, raw.plot, lowpass=1, highpass=1)
    assert_raises(ValueError, raw.plot, lowpass=1, filtorder=0)
    assert_raises(ValueError, raw.plot, clipping='foo')
    raw.plot(lowpass=1, clipping='transparent')
    raw.plot(highpass=1, clipping='clamp')
    raw.plot(highpass=1, lowpass=2)


@requires_scipy_version('0.12')
def test_plot_raw_psd():
    """Test plotting of raw psds
    """
    import matplotlib.pyplot as plt
    raw = _get_raw()
    # normal mode
    raw.plot_psd(tmax=2.0)
    # specific mode
    picks = pick_types(raw.info, meg='mag', eeg=False)[:4]
    raw.plot_psd(picks=picks, area_mode='range')
    ax = plt.axes()
    # if ax is supplied, picks must be, too:
    assert_raises(ValueError, raw.plot_psd, ax=ax)
    raw.plot_psd(picks=picks, ax=ax)
    plt.close('all')


run_tests_if_main()
