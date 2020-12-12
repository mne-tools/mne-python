# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: Simplified BSD

import numpy as np
import os.path as op
import itertools

from numpy.testing import assert_allclose
import pytest
import matplotlib
import matplotlib.pyplot as plt

from mne import read_events, pick_types, Annotations, create_info
from mne.datasets import testing
from mne.io import read_raw_fif, read_raw_ctf, RawArray
from mne.utils import (run_tests_if_main, _dt_to_stamp, _click_ch_name,
                       _close_event)
from mne.viz.utils import _fake_click
from mne.annotations import _sync_onset
from mne.viz import plot_raw, plot_sensors

ctf_dir = op.join(testing.data_path(download=False), 'CTF')
ctf_fname_continuous = op.join(ctf_dir, 'testdata_ctf.ds')

base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')
event_name = op.join(base_dir, 'test-eve.fif')
cov_fname = op.join(base_dir, 'test-cov.fif')


@pytest.fixture()
def raw():
    """Get raw data."""
    raw = read_raw_fif(raw_fname, preload=True)
    # Throws a warning about a changed unit.
    with pytest.warns(RuntimeWarning, match='unit'):
        raw.set_channel_types({raw.ch_names[0]: 'ias'})
    raw.pick_channels(raw.ch_names[:9])
    raw.info.normalize_proj()  # Fix projectors after subselection
    return raw


def _get_events():
    """Get events."""
    return read_events(event_name)


def _annotation_helper(raw, events=False):
    """Test interactive annotations."""
    # Some of our checks here require modern mpl to work properly
    n_anns = len(raw.annotations)
    plt.close('all')

    if events:
        events = np.array([[raw.first_samp + 100, 0, 1],
                           [raw.first_samp + 300, 0, 3]])
        n_events = len(events)
    else:
        events = None
        n_events = 0
    fig = raw.plot(events=events)
    assert len(plt.get_fignums()) == 1
    data_ax = fig.mne.ax_main
    fig.canvas.key_press_event('a')  # annotation mode
    assert len(plt.get_fignums()) == 2
    # +2 from the scale bars
    n_scale = 2
    assert len(data_ax.texts) == n_anns + n_events + n_scale
    # modify description to create label "BAD test"
    ann_fig = fig.mne.fig_annotation
    for key in ['backspace'] + list(' test;'):  # semicolon is ignored
        ann_fig.canvas.key_press_event(key)
    ann_fig.canvas.key_press_event('enter')

    # change annotation label
    for ix in (-1, 0):
        xy = ann_fig.mne.radio_ax.buttons.circles[ix].center
        _fake_click(ann_fig, ann_fig.mne.radio_ax, xy, xform='data')

    # draw annotation
    _fake_click(fig, data_ax, [1., 1.], xform='data', button=1, kind='press')
    _fake_click(fig, data_ax, [5., 1.], xform='data', button=1, kind='motion')
    _fake_click(fig, data_ax, [5., 1.], xform='data', button=1, kind='release')
    assert len(raw.annotations.onset) == n_anns + 1
    assert len(raw.annotations.duration) == n_anns + 1
    assert len(raw.annotations.description) == n_anns + 1
    assert raw.annotations.description[n_anns] == 'BAD test'
    assert len(data_ax.texts) == n_anns + 1 + n_events + n_scale
    onset = raw.annotations.onset[n_anns]
    want_onset = _sync_onset(raw, 1., inverse=True)
    assert_allclose(onset, want_onset)
    assert_allclose(raw.annotations.duration[n_anns], 4.)
    # test hover event
    fig.canvas.key_press_event('p')  # first turn on draggable mode
    assert fig.mne.draggable_annotations
    hover_kwargs = dict(xform='data', button=None, kind='motion')
    _fake_click(fig, data_ax, [4.6, 1.], **hover_kwargs)  # well inside ann.
    _fake_click(fig, data_ax, [4.9, 1.], **hover_kwargs)  # almost at edge
    assert fig.mne.annotation_hover_line is not None
    _fake_click(fig, data_ax, [5.5, 1.], **hover_kwargs)  # well outside ann.
    assert fig.mne.annotation_hover_line is None
    # more tests of hover line
    _fake_click(fig, data_ax, [4.6, 1.], **hover_kwargs)  # well inside ann.
    _fake_click(fig, data_ax, [4.9, 1.], **hover_kwargs)  # almost at edge
    assert fig.mne.annotation_hover_line is not None
    fig.canvas.key_press_event('p')  # turn off draggable mode, then move a bit
    _fake_click(fig, data_ax, [4.95, 1.], **hover_kwargs)
    assert fig.mne.annotation_hover_line is None
    fig.canvas.key_press_event('p')  # turn draggable mode back on
    # modify annotation from end (duration 4 → 1.5)
    _fake_click(fig, data_ax, [4.9, 1.], xform='data', button=None,
                kind='motion')  # ease up to it
    _fake_click(fig, data_ax, [5., 1.], xform='data', button=1, kind='press')
    _fake_click(fig, data_ax, [2.5, 1.], xform='data', button=1, kind='motion')
    _fake_click(fig, data_ax, [2.5, 1.], xform='data', button=1,
                kind='release')
    assert raw.annotations.onset[n_anns] == onset
    assert_allclose(raw.annotations.duration[n_anns], 1.5)  # 4 → 1.5
    # modify annotation from beginning (duration 1.5 → 2.0)
    _fake_click(fig, data_ax, [1., 1.], xform='data', button=1, kind='press')
    _fake_click(fig, data_ax, [0.5, 1.], xform='data', button=1, kind='motion')
    _fake_click(fig, data_ax, [0.5, 1.], xform='data', button=1,
                kind='release')
    assert_allclose(raw.annotations.onset[n_anns], onset - 0.5, atol=1e-10)
    assert_allclose(raw.annotations.duration[n_anns], 2.0)  # 1.5 → 2.0
    assert len(raw.annotations.onset) == n_anns + 1
    assert len(raw.annotations.duration) == n_anns + 1
    assert len(raw.annotations.description) == n_anns + 1
    assert raw.annotations.description[n_anns] == 'BAD test'
    assert len(fig.axes[0].texts) == n_anns + 1 + n_events + n_scale
    fig.canvas.key_press_event('shift+right')
    assert len(fig.axes[0].texts) == n_scale
    fig.canvas.key_press_event('shift+left')
    assert len(fig.axes[0].texts) == n_anns + 1 + n_events + n_scale

    # draw another annotation merging the two
    _fake_click(fig, data_ax, [5.5, 1.], xform='data', button=1, kind='press')
    _fake_click(fig, data_ax, [2., 1.], xform='data', button=1, kind='motion')
    _fake_click(fig, data_ax, [2., 1.], xform='data', button=1, kind='release')
    # delete the annotation
    assert len(raw.annotations.onset) == n_anns + 1
    assert len(raw.annotations.duration) == n_anns + 1
    assert len(raw.annotations.description) == n_anns + 1
    assert_allclose(raw.annotations.onset[n_anns], onset - 0.5, atol=1e-10)
    assert_allclose(raw.annotations.duration[n_anns], 5.0)
    assert len(fig.axes[0].texts) == n_anns + 1 + n_events + n_scale
    # Delete
    _fake_click(fig, data_ax, [1.5, 1.], xform='data', button=3, kind='press')
    # exit, re-enter, then exit a different way
    fig.canvas.key_press_event('a')  # exit
    fig.canvas.key_press_event('a')  # enter
    fig.mne.fig_annotation.canvas.key_press_event('escape')  # exit again
    assert len(raw.annotations.onset) == n_anns
    assert len(fig.axes[0].texts) == n_anns + n_events + n_scale
    fig.canvas.key_press_event('shift+right')
    assert len(fig.axes[0].texts) == n_scale
    fig.canvas.key_press_event('shift+left')
    assert len(fig.axes[0].texts) == n_anns + n_events + n_scale
    plt.close('all')


def _proj_status(ax):
    return [line.get_visible()
            for line in ax.findobj(matplotlib.lines.Line2D)][::2]


def _child_fig_helper(fig, key, attr):
    # Spawn and close child figs of raw.plot()
    num_figs = len(plt.get_fignums())
    assert getattr(fig.mne, attr) is None
    # spawn
    fig.canvas.key_press_event(key)
    assert len(fig.mne.child_figs) == 1
    assert len(plt.get_fignums()) == num_figs + 1
    child_fig = getattr(fig.mne, attr)
    assert child_fig is not None
    # close via main window toggle
    fig.canvas.key_press_event(key)
    _close_event(child_fig)
    assert len(fig.mne.child_figs) == 0
    assert len(plt.get_fignums()) == num_figs
    assert getattr(fig.mne, attr) is None
    # spawn again
    fig.canvas.key_press_event(key)
    assert len(fig.mne.child_figs) == 1
    assert len(plt.get_fignums()) == num_figs + 1
    child_fig = getattr(fig.mne, attr)
    assert child_fig is not None
    # close via child window
    child_fig.canvas.key_press_event(child_fig.mne.close_key)
    _close_event(child_fig)
    assert len(fig.mne.child_figs) == 0
    assert len(plt.get_fignums()) == num_figs
    assert getattr(fig.mne, attr) is None


def test_scale_bar():
    """Test scale bar for raw."""
    sfreq = 1000.
    t = np.arange(10000) / sfreq
    data = np.sin(2 * np.pi * 10. * t)
    # ± 1000 fT, 400 fT/cm, 20 µV
    data = data * np.array([[1000e-15, 400e-13, 20e-6]]).T
    info = create_info(3, sfreq, ('mag', 'grad', 'eeg'))
    raw = RawArray(data, info)
    fig = raw.plot()
    ax = fig.mne.ax_main
    assert len(ax.texts) == 3  # our labels
    texts = tuple(t.get_text().strip() for t in ax.texts)
    wants = ('800.0 fT/cm', '2000.0 fT', '40.0 µV')
    assert texts == wants
    assert len(ax.lines) == 7  # 1 green vline, 3 data, 3 scalebars
    for data, bar in zip(fig.mne.traces, fig.mne.scalebars.values()):
        y = data.get_ydata()
        y_lims = [y.min(), y.max()]
        bar_lims = bar.get_ydata()
        assert_allclose(y_lims, bar_lims, atol=1e-4)
    plt.close('all')


def test_plot_raw_selection(raw):
    """Test selection mode of plot_raw()."""
    raw.info['lowpass'] = 10.  # allow heavy decim during plotting
    plt.close('all')           # ensure all are closed
    assert len(plt.get_fignums()) == 0
    fig = raw.plot(group_by='selection', proj=False)
    assert len(plt.get_fignums()) == 2
    sel_fig = fig.mne.fig_selection
    buttons = sel_fig.mne.radio_ax.buttons
    assert sel_fig is not None
    # test changing selection with arrow keys
    sel_dict = fig.mne.ch_selections
    assert len(fig.mne.traces) == len(sel_dict['Left-temporal'])  # 6
    sel_fig.canvas.key_press_event('down')
    assert len(fig.mne.traces) == len(sel_dict['Left-frontal'])  # 3
    sel_fig.canvas.key_press_event('down')
    assert len(fig.mne.traces) == len(sel_dict['Misc'])  # 1
    sel_fig.canvas.key_press_event('down')  # ignored; no custom sel defined
    assert len(fig.mne.traces) == len(sel_dict['Misc'])  # 1
    # switch to butterfly mode
    sel_fig.canvas.key_press_event('b')
    assert len(fig.mne.traces) == len(np.concatenate(list(sel_dict.values())))
    assert fig.mne.butterfly
    # test clicking on radio buttons → should cancel butterfly mode
    xy = buttons.circles[0].center
    _fake_click(sel_fig, sel_fig.mne.radio_ax, xy, xform='data')
    assert len(fig.mne.traces) == len(sel_dict['Left-temporal'])  # 6
    assert not fig.mne.butterfly
    # test clicking on "custom" when not defined: should be no-op
    before_state = buttons.value_selected
    xy = buttons.circles[-1].center
    _fake_click(sel_fig, sel_fig.mne.radio_ax, xy, xform='data')
    assert len(fig.mne.traces) == len(sel_dict['Left-temporal'])  # unchanged
    assert buttons.value_selected == before_state                 # unchanged
    # test marking bad channel in selection mode → should make sensor red
    assert sel_fig.lasso.ec[:, 0].sum() == 0   # R of RGBA zero for all chans
    _click_ch_name(fig, ch_index=1, button=1)  # mark bad
    assert sel_fig.lasso.ec[:, 0].sum() == 1   # one channel red
    _click_ch_name(fig, ch_index=1, button=1)  # mark good
    assert sel_fig.lasso.ec[:, 0].sum() == 0   # all channels black
    # test lasso
    sel_fig._set_custom_selection()  # lasso empty → should do nothing
    sensor_ax = sel_fig.mne.sensor_ax
    # Lasso with 1 mag/grad sensor unit (upper left)
    _fake_click(sel_fig, sensor_ax, (0, 1), xform='ax')
    _fake_click(sel_fig, sensor_ax, (0.65, 1), xform='ax', kind='motion')
    _fake_click(sel_fig, sensor_ax, (0.65, 0.7), xform='ax', kind='motion')
    _fake_click(sel_fig, sensor_ax, (0, 0.7), xform='ax', kind='release')
    want = ['MEG 0121', 'MEG 0122', 'MEG 0123']
    assert sorted(want) == sorted(sel_fig.lasso.selection)
    # test joint closing of selection & data windows
    sel_fig.canvas.key_press_event(sel_fig.mne.close_key)
    _close_event(sel_fig)
    assert len(plt.get_fignums()) == 0


def test_plot_raw_ssp_interaction(raw):
    """Test SSP projector UI of plot_raw()."""
    raw.info['lowpass'] = 10.  # allow heavy decim during plotting
    # apply some (not all) projs to test our proj UI (greyed out applied projs)
    projs = raw.info['projs'][-2:]
    raw.del_proj([-2, -1])
    raw.apply_proj()
    raw.add_proj(projs)
    fig = raw.plot()
    # open SSP window
    _fake_click(fig, fig.mne.ax_proj, [0.5, 0.5])
    assert len(plt.get_fignums()) == 2
    ssp_fig = fig.mne.fig_proj
    t = ssp_fig.mne.proj_checkboxes.labels
    ax = ssp_fig.mne.proj_checkboxes.ax
    assert _proj_status(ax) == [True, True, True]
    # this should have no effect (proj 0 is already applied)
    assert t[0].get_text().endswith('(already applied)')
    pos = np.array(t[0].get_position()) + 0.01
    _fake_click(ssp_fig, ssp_fig.get_axes()[0], pos, xform='data')
    assert _proj_status(ax) == [True, True, True]
    # this should work (proj 1 not applied)
    pos = np.array(t[1].get_position()) + 0.01
    _fake_click(ssp_fig, ax, pos, xform='data')
    assert _proj_status(ax) == [True, False, True]
    # turn it back on
    _fake_click(ssp_fig, ax, pos, xform='data')
    assert _proj_status(ax) == [True, True, True]
    # toggle all off (button axes need both press and release)
    _fake_click(ssp_fig, ssp_fig.mne.proj_all.ax, [0.5, 0.5])
    _fake_click(ssp_fig, ssp_fig.mne.proj_all.ax, [0.5, 0.5], kind='release')
    assert _proj_status(ax) == [True, False, False]
    # turn all on
    _fake_click(ssp_fig, ssp_fig.mne.proj_all.ax, [0.5, 0.5])  # all on
    _fake_click(ssp_fig, ssp_fig.mne.proj_all.ax, [0.5, 0.5], kind='release')
    assert fig.mne.projector is not None  # on
    assert _proj_status(ax) == [True, True, True]


def test_plot_raw_child_figures(raw):
    """Test spawning and closing of child figures."""
    raw.info['lowpass'] = 10.  # allow heavy decim during plotting
    plt.close('all')  # make sure we start clean
    assert len(plt.get_fignums()) == 0
    fig = raw.plot()
    assert len(plt.get_fignums()) == 1
    # test child fig toggles
    _child_fig_helper(fig, '?', 'fig_help')
    _child_fig_helper(fig, 'j', 'fig_proj')
    _child_fig_helper(fig, 'a', 'fig_annotation')
    assert len(fig.mne.child_figs) == 0  # make sure the helper cleaned up
    assert len(plt.get_fignums()) == 1
    # test right-click → channel location popup
    fig.canvas.draw()
    _click_ch_name(fig, ch_index=2, button=3)
    assert len(fig.mne.child_figs) == 1
    assert len(plt.get_fignums()) == 2
    fig.mne.child_figs[0].canvas.key_press_event('escape')
    _close_event(fig.mne.child_figs[0])
    assert len(plt.get_fignums()) == 1
    # test right-click on non-data channel
    ix = raw.get_channel_types().index('ias')  # find the shielding channel
    trace_ix = fig.mne.ch_order.tolist().index(ix)  # get its plotting position
    assert len(fig.mne.child_figs) == 0
    assert len(plt.get_fignums()) == 1
    fig.canvas.draw()
    _click_ch_name(fig, ch_index=trace_ix, button=3)  # should be no-op
    assert len(fig.mne.child_figs) == 0
    assert len(plt.get_fignums()) == 1
    # test resize of main window
    width, height = fig.canvas.manager.canvas.get_width_height()
    fig.canvas.manager.canvas.resize(width // 2, height // 2)
    plt.close('all')


def test_plot_raw_keypresses(raw):
    """Test keypress interactivity of plot_raw()."""
    raw.info['lowpass'] = 10.  # allow heavy decim during plotting
    fig = raw.plot()
    # test twice → once in normal, once in butterfly view.
    # NB: keys a, j, and ? are tested in test_plot_raw_child_figures()
    keys = ('pagedown', 'down', 'up', 'down', 'right', 'left', '-', '+', '=',
            'd', 'd', 'pageup', 'home', 'end', 'z', 'z', 's', 's', 'f11', 'b')
    # test for group_by='original'
    for key in 2 * keys + ('escape',):
        fig.canvas.key_press_event(key)
    # test for group_by='selection'
    fig = plot_raw(raw, group_by='selection')
    for key in 2 * keys + ('escape',):
        fig.canvas.key_press_event(key)


def test_plot_raw_traces(raw):
    """Test plotting of raw data."""
    raw.info['lowpass'] = 10.  # allow heavy decim during plotting
    events = _get_events()
    plt.close('all')  # ensure all are closed
    fig = raw.plot(events=events, order=[1, 7, 5, 2, 3], n_channels=3,
                   group_by='original')
    assert hasattr(fig, 'mne')  # make sure fig.mne param object is present
    assert len(fig.axes) == 5

    # setup
    x = fig.mne.traces[0].get_xdata()[5]
    y = fig.mne.traces[0].get_ydata()[5]
    data_ax = fig.mne.ax_main
    hscroll = fig.mne.ax_hscroll
    vscroll = fig.mne.ax_vscroll
    # test marking bad channels
    label = fig.mne.ax_main.get_yticklabels()[0].get_text()
    assert label not in fig.mne.info['bads']
    _fake_click(fig, data_ax, [x, y], xform='data')  # click data to mark bad
    assert label in fig.mne.info['bads']
    _fake_click(fig, data_ax, [x, y], xform='data')  # click data to unmark bad
    assert label not in fig.mne.info['bads']
    _click_ch_name(fig, ch_index=0, button=1)        # click name to mark bad
    assert label in fig.mne.info['bads']
    # test other kinds of clicks
    _fake_click(fig, data_ax, [0.5, 0.999])  # click elsewhere (add vline)
    _fake_click(fig, data_ax, [0.5, 0.999], button=3)  # remove vline
    _fake_click(fig, hscroll, [0.5, 0.5])  # change time
    _fake_click(fig, hscroll, [0.5, 0.5])  # shouldn't change time this time
    # test scrolling through channels
    labels = [label.get_text() for label in data_ax.get_yticklabels()]
    assert labels == [raw.ch_names[1], raw.ch_names[7], raw.ch_names[5]]
    _fake_click(fig, vscroll, [0.5, 0.01])  # change channels to end
    labels = [label.get_text() for label in data_ax.get_yticklabels()]
    assert labels == [raw.ch_names[5], raw.ch_names[2], raw.ch_names[3]]
    for _ in (0, 0):
        # first click changes channels to mid; second time shouldn't change
        _fake_click(fig, vscroll, [0.5, 0.5])
        labels = [label.get_text() for label in data_ax.get_yticklabels()]
        assert labels == [raw.ch_names[7], raw.ch_names[5], raw.ch_names[2]]
        assert len(plt.get_fignums()) == 1

    # test clicking a channel name in butterfly mode
    bads = fig.mne.info['bads'].copy()
    fig.canvas.key_press_event('b')
    _click_ch_name(fig, ch_index=0, button=1)  # should be no-op
    assert fig.mne.info['bads'] == bads        # unchanged
    fig.canvas.key_press_event('b')

    # test starting up in zen mode
    fig = plot_raw(raw, show_scrollbars=False)
    # test order, title, & show_options kwargs
    with pytest.raises(ValueError, match='order should be array-like; got'):
        raw.plot(order='foo')
    with pytest.raises(TypeError, match='title must be None or a string, got'):
        raw.plot(title=1)
    raw.plot(show_options=True)
    plt.close('all')

    # Color setting
    with pytest.raises(KeyError, match='must be strictly positive, or -1'):
        raw.plot(event_color={0: 'r'})
    with pytest.raises(TypeError, match='event_color key must be an int, got'):
        raw.plot(event_color={'foo': 'r'})
    annot = Annotations([10, 10 + raw.first_samp / raw.info['sfreq']],
                        [10, 10], ['test', 'test'], raw.info['meas_date'])
    with pytest.warns(RuntimeWarning, match='outside data range'):
        raw.set_annotations(annot)
    fig = plot_raw(raw, events=events, event_color={-1: 'r', 998: 'b'})
    plt.close('all')
    for group_by, order in zip(['position', 'selection'],
                               [np.arange(len(raw.ch_names))[::-3],
                                [1, 2, 4, 6]]):
        fig = raw.plot(group_by=group_by, order=order)
        x = fig.get_axes()[0].lines[1].get_xdata()[10]
        y = fig.get_axes()[0].lines[1].get_ydata()[10]
        _fake_click(fig, data_ax, [x, y], xform='data')  # mark bad
        fig.canvas.key_press_event('down')  # change selection
        _fake_click(fig, fig.get_axes()[2], [0.5, 0.5])  # change channels
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
    # test if meas_date is off
    raw.set_meas_date(_dt_to_stamp(raw.info['meas_date'])[0])
    annot = Annotations([1 + raw.first_samp / raw.info['sfreq']],
                        [5], ['bad'])
    with pytest.warns(RuntimeWarning, match='outside data range'):
        raw.set_annotations(annot)
    with pytest.warns(None):  # sometimes projection
        raw.plot(group_by='position', order=np.arange(8))
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        if hasattr(fig, 'radio'):  # Get access to selection fig.
            break
    for key in ['down', 'up', 'escape']:
        fig.canvas.key_press_event(key)

    raw._data[:] = np.nan
    # this should (at least) not die, the output should pretty clearly show
    # that there is a problem so probably okay to just plot something blank
    with pytest.warns(None):
        raw.plot(scalings='auto')

    plt.close('all')


@testing.requires_testing_data
def test_plot_raw_white():
    """Test plotting whitened raw data."""
    raw = read_raw_fif(raw_fname).crop(0, 1).load_data()
    fig = raw.plot(noise_cov=cov_fname)
    # toggle whitening
    fig.canvas.key_press_event('w')
    fig.canvas.key_press_event('w')
    plt.close('all')


@testing.requires_testing_data
def test_plot_ref_meg():
    """Test plotting ref_meg."""
    raw_ctf = read_raw_ctf(ctf_fname_continuous).crop(0, 1).load_data()
    raw_ctf.plot()
    plt.close('all')
    pytest.raises(ValueError, raw_ctf.plot, group_by='selection')


def test_plot_misc_auto():
    """Test plotting of data with misc auto scaling."""
    data = np.random.RandomState(0).randn(1, 1000)
    raw = RawArray(data, create_info(1, 1000., 'misc'))
    raw.plot()
    plt.close('all')


def test_plot_annotations(raw):
    """Test annotation mode of the plotter."""
    raw.info['lowpass'] = 10.
    _annotation_helper(raw)
    _annotation_helper(raw, events=True)

    annot = Annotations([42], [1], 'test', raw.info['meas_date'])
    with pytest.warns(RuntimeWarning, match='expanding outside'):
        raw.set_annotations(annot)
    _annotation_helper(raw)
    # test annotation visibility toggle
    fig = raw.plot()
    assert len(fig.mne.annotations) == 1
    assert len(fig.mne.annotation_texts) == 1
    fig.canvas.key_press_event('a')  # start annotation mode
    checkboxes = fig.mne.show_hide_annotation_checkboxes
    checkboxes.set_active(0)
    assert len(fig.mne.annotations) == 0
    assert len(fig.mne.annotation_texts) == 0
    checkboxes.set_active(0)
    assert len(fig.mne.annotations) == 1
    assert len(fig.mne.annotation_texts) == 1


@pytest.mark.parametrize('filtorder', (0, 2))  # FIR, IIR
def test_plot_raw_filtered(filtorder, raw):
    """Test filtering of raw plots."""
    with pytest.raises(ValueError, match='lowpass.*Nyquist'):
        raw.plot(lowpass=raw.info['sfreq'] / 2., filtorder=filtorder)
    with pytest.raises(ValueError, match='highpass must be > 0'):
        raw.plot(highpass=0, filtorder=filtorder)
    with pytest.raises(ValueError, match='Filter order must be'):
        raw.plot(lowpass=1, filtorder=-1)
    with pytest.raises(ValueError, match="Invalid value for the 'clipping'"):
        raw.plot(clipping='foo')
    raw.plot(lowpass=40, clipping='transparent', filtorder=filtorder)
    raw.plot(highpass=1, clipping='clamp', filtorder=filtorder)
    raw.plot(lowpass=40, butterfly=True, filtorder=filtorder)
    plt.close('all')


def test_plot_raw_psd(raw):
    """Test plotting of raw psds."""
    raw_orig = raw.copy()
    # normal mode
    fig = raw.plot_psd(average=False)
    fig.canvas.resize_event()
    # specific mode
    picks = pick_types(raw.info, meg='mag', eeg=False)[:4]
    raw.plot_psd(tmax=None, picks=picks, area_mode='range', average=False,
                 spatial_colors=True)
    raw.plot_psd(tmax=20., color='yellow', dB=False, line_alpha=0.4,
                 n_overlap=0.1, average=False)
    plt.close('all')
    # one axes supplied
    ax = plt.axes()
    raw.plot_psd(tmax=None, picks=picks, ax=ax, average=True)
    plt.close('all')
    # two axes supplied
    _, axs = plt.subplots(2)
    raw.plot_psd(tmax=None, ax=axs, average=True)
    plt.close('all')
    # need 2, got 1
    ax = plt.axes()
    with pytest.raises(ValueError, match='of length 2, while the length is 1'):
        raw.plot_psd(ax=ax, average=True)
    plt.close('all')
    # topo psd
    ax = plt.subplot()
    raw.plot_psd_topo(axes=ax)
    plt.close('all')
    # with channel information not available
    for idx in range(len(raw.info['chs'])):
        raw.info['chs'][idx]['loc'] = np.zeros(12)
    with pytest.warns(RuntimeWarning, match='locations not available'):
        raw.plot_psd(spatial_colors=True, average=False)
    # with a flat channel
    raw[5, :] = 0
    for dB, estimate in itertools.product((True, False),
                                          ('power', 'amplitude')):
        with pytest.warns(UserWarning, match='[Infinite|Zero]'):
            fig = raw.plot_psd(average=True, dB=dB, estimate=estimate)
        # check grad axes
        title = fig.axes[0].get_title()
        ylabel = fig.axes[0].get_ylabel()
        ends_dB = ylabel.endswith('mathrm{(dB)}$')
        unit = '(fT/cm)²/Hz' if estimate == 'power' else r'fT/cm/\sqrt{Hz}'
        assert title == 'Gradiometers', title
        assert unit in ylabel, ylabel
        if dB:
            assert ends_dB, ylabel
        else:
            assert not ends_dB, ylabel
        # check mag axes
        title = fig.axes[1].get_title()
        ylabel = fig.axes[1].get_ylabel()
        unit = 'fT²/Hz' if estimate == 'power' else r'fT/\sqrt{Hz}'
        assert title == 'Magnetometers', title
        assert unit in ylabel, ylabel
    # test reject_by_annotation
    raw = raw_orig
    raw.set_annotations(Annotations([1, 5], [3, 3], ['test', 'test']))
    raw.plot_psd(reject_by_annotation=True)
    raw.plot_psd(reject_by_annotation=False)
    plt.close('all')

    # test fmax value checking
    with pytest.raises(ValueError, match='must not exceed ½ the sampling'):
        raw.plot_psd(fmax=50000)

    # test xscale value checking
    with pytest.raises(ValueError, match="Invalid value for the 'xscale'"):
        raw.plot_psd(xscale='blah')

    # gh-5046
    raw = read_raw_fif(raw_fname, preload=True).crop(0, 1)
    picks = pick_types(raw.info, meg=True)
    raw.plot_psd(picks=picks, average=False)
    raw.plot_psd(picks=picks, average=True)
    plt.close('all')
    raw.set_channel_types({'MEG 0113': 'hbo', 'MEG 0112': 'hbr',
                           'MEG 0122': 'fnirs_cw_amplitude',
                           'MEG 0123': 'fnirs_od'},
                          verbose='error')
    fig = raw.plot_psd()
    assert len(fig.axes) == 10
    plt.close('all')

    # gh-7631
    data = 1e-3 * np.random.rand(2, 100)
    info = create_info(['CH1', 'CH2'], 100)
    raw = RawArray(data, info)
    picks = pick_types(raw.info, misc=True)
    raw.plot_psd(picks=picks, spatial_colors=False)
    plt.close('all')


def test_plot_sensors(raw):
    """Test plotting of sensor array."""
    plt.close('all')
    fig = raw.plot_sensors('3d')
    _fake_click(fig, fig.gca(), (-0.08, 0.67))
    raw.plot_sensors('topomap', ch_type='mag',
                     show_names=['MEG 0111', 'MEG 0131'])
    plt.close('all')
    ax = plt.subplot(111)
    raw.plot_sensors(ch_groups='position', axes=ax)
    raw.plot_sensors(ch_groups='selection', to_sphere=False)
    raw.plot_sensors(ch_groups=[[0, 1, 2], [3, 4]])
    pytest.raises(ValueError, raw.plot_sensors, ch_groups='asd')
    pytest.raises(TypeError, plot_sensors, raw)  # needs to be info
    pytest.raises(ValueError, plot_sensors, raw.info, kind='sasaasd')
    plt.close('all')
    fig, sels = raw.plot_sensors('select', show_names=True)
    ax = fig.axes[0]

    # Click with no sensors
    _fake_click(fig, ax, (0., 0.), xform='data')
    _fake_click(fig, ax, (0, 0.), xform='data', kind='release')
    assert fig.lasso.selection == []

    # Lasso with 1 sensor (upper left)
    _fake_click(fig, ax, (0, 1), xform='ax')
    fig.canvas.draw()
    assert fig.lasso.selection == []
    _fake_click(fig, ax, (0.65, 1), xform='ax', kind='motion')
    _fake_click(fig, ax, (0.65, 0.7), xform='ax', kind='motion')
    fig.canvas.key_press_event('control')
    _fake_click(fig, ax, (0, 0.7), xform='ax', kind='release')
    assert fig.lasso.selection == ['MEG 0121']

    # check that point appearance changes
    fc = fig.lasso.collection.get_facecolors()
    ec = fig.lasso.collection.get_edgecolors()
    assert (fc[:, -1] == [0.5, 1., 0.5]).all()
    assert (ec[:, -1] == [0.25, 1., 0.25]).all()

    _fake_click(fig, ax, (0.7, 1), xform='ax', kind='motion')
    xy = ax.collections[0].get_offsets()
    _fake_click(fig, ax, xy[2], xform='data')  # single selection
    assert fig.lasso.selection == ['MEG 0121', 'MEG 0131']
    _fake_click(fig, ax, xy[2], xform='data')  # deselect
    assert fig.lasso.selection == ['MEG 0121']
    plt.close('all')

    raw.info['dev_head_t'] = None  # like empty room
    with pytest.warns(RuntimeWarning, match='identity'):
        raw.plot_sensors()


run_tests_if_main()
