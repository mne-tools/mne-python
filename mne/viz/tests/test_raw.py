# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: Simplified BSD

import itertools

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
import matplotlib
import matplotlib.pyplot as plt

from mne import pick_types, Annotations, create_info
from mne.datasets import testing
from mne.utils import get_config, set_config
from mne.io import RawArray
from mne.utils import _dt_to_stamp
from mne.viz.utils import _fake_click
from mne.annotations import _sync_onset
from mne.viz import plot_raw, plot_sensors


def _annotation_helper(raw, browse_backend, events=False):
    """Test interactive annotations."""
    # Some of our checks here require modern mpl to work properly
    n_anns = len(raw.annotations)
    browse_backend._close_all()

    if events:
        events = np.array([[raw.first_samp + 100, 0, 1],
                           [raw.first_samp + 300, 0, 3]])
        n_events = len(events)
    else:
        events = None
        n_events = 0
    fig = raw.plot(events=events)
    assert browse_backend._get_n_figs() == 1
    data_ax = fig.mne.ax_main
    fig._fake_keypress('a')  # annotation mode
    # ToDo: This will be different in pyqtgraph because it handles annotations
    #  from the toolbar.
    assert browse_backend._get_n_figs() == 2
    # +3 from the scale bars
    n_scale = 3
    assert len(data_ax.texts) == n_anns + n_events + n_scale
    # modify description to create label "BAD test"
    ann_fig = fig.mne.fig_annotation
    # semicolon is ignored
    for key in ['backspace'] + list(' test;') + ['enter']:
        fig._fake_keypress(key, fig=ann_fig)

    # change annotation label
    for ix in (-1, 0):
        xy = ann_fig.mne.radio_ax.buttons.circles[ix].center
        fig._fake_click(xy, ann_fig, ann_fig.mne.radio_ax,
                        xform='data')

    # draw annotation
    fig._fake_click((1., 1.), xform='data', button=1, kind='press')
    fig._fake_click((5., 1.), xform='data', button=1, kind='motion')
    fig._fake_click((5., 1.), xform='data', button=1, kind='release')
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
    fig._fake_keypress('p')  # first turn on draggable mode
    assert fig.mne.draggable_annotations
    hover_kwargs = dict(xform='data', button=None, kind='motion')
    fig._fake_click((4.6, 1.), **hover_kwargs)  # well inside ann.
    fig._fake_click((4.9, 1.), **hover_kwargs)  # almost at edge
    assert fig.mne.annotation_hover_line is not None
    fig._fake_click((5.5, 1.), **hover_kwargs)  # well outside ann.
    assert fig.mne.annotation_hover_line is None
    # more tests of hover line
    fig._fake_click((4.6, 1.), **hover_kwargs)  # well inside ann.
    fig._fake_click((4.9, 1.), **hover_kwargs)  # almost at edge
    assert fig.mne.annotation_hover_line is not None
    fig._fake_keypress('p')  # turn off draggable mode, then move a bit
    fig._fake_click((4.95, 1.), **hover_kwargs)
    assert fig.mne.annotation_hover_line is None
    fig._fake_keypress('p')  # turn draggable mode back on
    # modify annotation from end (duration 4 → 1.5)
    fig._fake_click((4.9, 1.), xform='data', button=None,
                    kind='motion')  # ease up to it
    fig._fake_click((5., 1.), xform='data', button=1, kind='press')
    fig._fake_click((2.5, 1.), xform='data', button=1, kind='motion')
    fig._fake_click((2.5, 1.), xform='data', button=1,
                    kind='release')
    assert raw.annotations.onset[n_anns] == onset
    assert_allclose(raw.annotations.duration[n_anns], 1.5)  # 4 → 1.5
    # modify annotation from beginning (duration 1.5 → 2.0)
    fig._fake_click((1., 1.), xform='data', button=1, kind='press')
    fig._fake_click((0.5, 1.), xform='data', button=1, kind='motion')
    fig._fake_click((0.5, 1.), xform='data', button=1,
                    kind='release')
    assert_allclose(raw.annotations.onset[n_anns], onset - 0.5, atol=1e-10)
    assert_allclose(raw.annotations.duration[n_anns], 2.0)  # 1.5 → 2.0
    assert len(raw.annotations.onset) == n_anns + 1
    assert len(raw.annotations.duration) == n_anns + 1
    assert len(raw.annotations.description) == n_anns + 1
    assert raw.annotations.description[n_anns] == 'BAD test'
    assert len(fig.axes[0].texts) == n_anns + 1 + n_events + n_scale
    fig._fake_keypress('shift+right')
    assert len(fig.axes[0].texts) == n_scale
    fig._fake_keypress('shift+left')
    assert len(fig.axes[0].texts) == n_anns + 1 + n_events + n_scale

    # draw another annotation merging the two
    fig._fake_click((5.5, 1.), xform='data', button=1, kind='press')
    fig._fake_click((2., 1.), xform='data', button=1, kind='motion')
    fig._fake_click((2., 1.), xform='data', button=1, kind='release')
    # delete the annotation
    assert len(raw.annotations.onset) == n_anns + 1
    assert len(raw.annotations.duration) == n_anns + 1
    assert len(raw.annotations.description) == n_anns + 1
    assert_allclose(raw.annotations.onset[n_anns], onset - 0.5, atol=1e-10)
    assert_allclose(raw.annotations.duration[n_anns], 5.0)
    assert len(fig.axes[0].texts) == n_anns + 1 + n_events + n_scale
    # Delete
    fig._fake_click((1.5, 1.), xform='data', button=3, kind='press')
    # exit, re-enter, then exit a different way
    fig._fake_keypress('a')  # exit
    fig._fake_keypress('a')  # enter
    fig._fake_keypress('escape', fig=fig.mne.fig_annotation)  # exit again
    assert len(raw.annotations.onset) == n_anns
    assert len(fig.axes[0].texts) == n_anns + n_events + n_scale
    fig._fake_keypress('shift+right')
    assert len(fig.axes[0].texts) == n_scale
    fig._fake_keypress('shift+left')
    assert len(fig.axes[0].texts) == n_anns + n_events + n_scale


def _proj_status(ax):
    return [line.get_visible()
            for line in ax.findobj(matplotlib.lines.Line2D)][::2]


def _child_fig_helper(fig, key, attr, browse_backend):
    # Spawn and close child figs of raw.plot()
    num_figs = browse_backend._get_n_figs()
    assert getattr(fig.mne, attr) is None
    # spawn
    fig._fake_keypress(key)
    assert len(fig.mne.child_figs) == 1
    assert browse_backend._get_n_figs() == num_figs + 1
    child_fig = getattr(fig.mne, attr)
    assert child_fig is not None
    # close via main window toggle
    fig._fake_keypress(key)
    fig._close_event(child_fig)
    assert len(fig.mne.child_figs) == 0
    assert browse_backend._get_n_figs() == num_figs
    assert getattr(fig.mne, attr) is None
    # spawn again
    fig._fake_keypress(key)
    assert len(fig.mne.child_figs) == 1
    assert browse_backend._get_n_figs() == num_figs + 1
    child_fig = getattr(fig.mne, attr)
    assert child_fig is not None
    # close via child window
    fig._fake_keypress(child_fig.mne.close_key, fig=child_fig)
    fig._close_event(child_fig)
    assert len(fig.mne.child_figs) == 0
    assert browse_backend._get_n_figs() == num_figs
    assert getattr(fig.mne, attr) is None


def test_scale_bar(browse_backend):
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
    assert len(ax.texts) == 4  # empty vline-text + ch_type scale-bars
    # ToDo: This might be solved differently in pyqtgraph.
    texts = tuple(t.get_text().strip() for t in ax.texts)
    wants = ('', '800.0 fT/cm', '2000.0 fT', '40.0 µV')
    assert texts == wants
    assert len(ax.lines) == 7  # 1 green vline, 3 data, 3 scalebars
    for data, bar in zip(fig.mne.traces, fig.mne.scalebars.values()):
        y = data.get_ydata()
        y_lims = [y.min(), y.max()]
        bar_lims = bar.get_ydata()
        assert_allclose(y_lims, bar_lims, atol=1e-4)


def test_plot_raw_selection(raw, browse_backend):
    """Test selection mode of plot_raw()."""
    raw.info['lowpass'] = 10.  # allow heavy decim during plotting
    browse_backend._close_all()           # ensure all are closed
    assert browse_backend._get_n_figs() == 0
    fig = raw.plot(group_by='selection', proj=False)
    assert browse_backend._get_n_figs() == 2
    sel_fig = fig.mne.fig_selection
    # ToDo: These gui-elements might differ in pyqtgraph.
    buttons = sel_fig.mne.radio_ax.buttons
    assert sel_fig is not None
    # test changing selection with arrow keys
    sel_dict = fig.mne.ch_selections
    assert len(fig.mne.traces) == len(sel_dict['Left-temporal'])  # 6
    fig._fake_keypress('down', fig=sel_fig)
    assert len(fig.mne.traces) == len(sel_dict['Left-frontal'])  # 3
    fig._fake_keypress('down', fig=sel_fig)
    assert len(fig.mne.traces) == len(sel_dict['Misc'])  # 1
    fig._fake_keypress('down', fig=sel_fig)  # ignored; no custom sel defined
    assert len(fig.mne.traces) == len(sel_dict['Misc'])  # 1
    # switch to butterfly mode
    fig._fake_keypress('b', fig=sel_fig)
    assert len(fig.mne.traces) == len(np.concatenate(list(sel_dict.values())))
    assert fig.mne.butterfly
    # test clicking on radio buttons → should cancel butterfly mode
    xy = buttons.circles[0].center
    fig._fake_click(xy, sel_fig, sel_fig.mne.radio_ax, xform='data')
    assert len(fig.mne.traces) == len(sel_dict['Left-temporal'])  # 6
    assert not fig.mne.butterfly
    # test clicking on "custom" when not defined: should be no-op
    before_state = buttons.value_selected
    xy = buttons.circles[-1].center
    fig._fake_click(xy, sel_fig, sel_fig.mne.radio_ax, xform='data')
    assert len(fig.mne.traces) == len(sel_dict['Left-temporal'])  # unchanged
    assert buttons.value_selected == before_state                 # unchanged
    # test marking bad channel in selection mode → should make sensor red
    assert sel_fig.lasso.ec[:, 0].sum() == 0   # R of RGBA zero for all chans
    fig._click_ch_name(ch_index=1, button=1)  # mark bad
    assert sel_fig.lasso.ec[:, 0].sum() == 1   # one channel red
    fig._click_ch_name(ch_index=1, button=1)  # mark good
    assert sel_fig.lasso.ec[:, 0].sum() == 0   # all channels black
    # test lasso
    sel_fig._set_custom_selection()  # lasso empty → should do nothing
    sensor_ax = sel_fig.mne.sensor_ax
    # Lasso with 1 mag/grad sensor unit (upper left)
    fig._fake_click((0, 1), sel_fig,
                    sensor_ax, xform='ax')
    fig._fake_click((0.65, 1), sel_fig, sensor_ax,
                    xform='ax', kind='motion')
    fig._fake_click((0.65, 0.7), sel_fig, sensor_ax,
                    xform='ax', kind='motion')
    fig._fake_click((0, 0.7), sel_fig, sensor_ax,
                    xform='ax', kind='release')
    want = ['MEG 0121', 'MEG 0122', 'MEG 0123']
    assert sorted(want) == sorted(sel_fig.lasso.selection)
    # test joint closing of selection & data windows
    fig._fake_keypress(sel_fig.mne.close_key, fig=sel_fig)
    fig._close_event(sel_fig)
    assert browse_backend._get_n_figs() == 0


def test_plot_raw_ssp_interaction(raw, browse_backend):
    """Test SSP projector UI of plot_raw()."""
    raw.info['lowpass'] = 10.  # allow heavy decim during plotting
    # apply some (not all) projs to test our proj UI (greyed out applied projs)
    projs = raw.info['projs'][-2:]
    raw.del_proj([-2, -1])
    raw.apply_proj()
    raw.add_proj(projs)
    fig = raw.plot()
    # open SSP window
    fig._fake_click((0.5, 0.5), ax=fig.mne.ax_proj)
    assert browse_backend._get_n_figs() == 2
    ssp_fig = fig.mne.fig_proj
    # ToDo: These gui-elements might differ in pyqtgraph.
    t = ssp_fig.mne.proj_checkboxes.labels
    ax = ssp_fig.mne.proj_checkboxes.ax
    assert _proj_status(ax) == [True, True, True]
    # this should have no effect (proj 0 is already applied)
    assert t[0].get_text().endswith('(already applied)')
    pos = np.array(t[0].get_position()) + 0.01
    fig._fake_click(pos, ssp_fig, ax, xform='data')
    assert _proj_status(ax) == [True, True, True]
    # this should work (proj 1 not applied)
    pos = np.array(t[1].get_position()) + 0.01
    fig._fake_click(pos, ssp_fig, ax, xform='data')
    assert _proj_status(ax) == [True, False, True]
    # turn it back on
    fig._fake_click(pos, ssp_fig, ax, xform='data')
    assert _proj_status(ax) == [True, True, True]
    # toggle all off (button axes need both press and release)
    fig._fake_click((0.5, 0.5), ssp_fig, ssp_fig.mne.proj_all.ax)
    fig._fake_click((0.5, 0.5), ssp_fig,
                    ssp_fig.mne.proj_all.ax, kind='release')
    assert _proj_status(ax) == [True, False, False]
    fig._fake_keypress('J')
    assert _proj_status(ax) == [True, True, True]
    fig._fake_keypress('J')
    assert _proj_status(ax) == [True, False, False]
    # turn all on
    fig._fake_click((0.5, 0.5), ssp_fig, ssp_fig.mne.proj_all.ax)  # all on
    fig._fake_click((0.5, 0.5), ssp_fig, ssp_fig.mne.proj_all.ax,
                    kind='release')
    assert fig.mne.projector is not None  # on
    assert _proj_status(ax) == [True, True, True]


def test_plot_raw_child_figures(raw, browse_backend):
    """Test spawning and closing of child figures."""
    raw.info['lowpass'] = 10.  # allow heavy decim during plotting
    browse_backend._close_all()  # make sure we start clean
    assert browse_backend._get_n_figs() == 0
    fig = raw.plot()
    assert browse_backend._get_n_figs() == 1
    # test child fig toggles
    _child_fig_helper(fig, '?', 'fig_help', browse_backend)
    _child_fig_helper(fig, 'j', 'fig_proj', browse_backend)
    # ToDo: This figure won't be there with pyqtgraph.
    _child_fig_helper(fig, 'a', 'fig_annotation', browse_backend)
    assert len(fig.mne.child_figs) == 0  # make sure the helper cleaned up
    assert browse_backend._get_n_figs() == 1
    # test right-click → channel location popup
    fig._redraw()
    fig._click_ch_name(ch_index=2, button=3)
    assert len(fig.mne.child_figs) == 1
    assert browse_backend._get_n_figs() == 2
    fig._fake_keypress('escape', fig=fig.mne.child_figs[0])
    fig._close_event(fig.mne.child_figs[0])
    assert browse_backend._get_n_figs() == 1
    # test right-click on non-data channel
    ix = raw.get_channel_types().index('ias')  # find the shielding channel
    trace_ix = fig.mne.ch_order.tolist().index(ix)  # get its plotting position
    assert len(fig.mne.child_figs) == 0
    assert browse_backend._get_n_figs() == 1
    fig._redraw()
    fig._click_ch_name(ch_index=trace_ix, button=3)  # should be no-op
    assert len(fig.mne.child_figs) == 0
    assert browse_backend._get_n_figs() == 1
    # test resize of main window
    fig._resize_by_factor(0.5)


def test_plot_raw_keypresses(raw, browse_backend):
    """Test keypress interactivity of plot_raw()."""
    raw.info['lowpass'] = 10.  # allow heavy decim during plotting
    fig = raw.plot()
    # test twice → once in normal, once in butterfly view.
    # NB: keys a, j, and ? are tested in test_plot_raw_child_figures()
    keys = ('pagedown', 'down', 'up', 'down', 'right', 'left', '-', '+', '=',
            'd', 'd', 'pageup', 'home', 'end', 'z', 'z', 's', 's', 'f11', 't',
            'b')
    # test for group_by='original'
    for key in 2 * keys + ('escape',):
        fig._fake_keypress(key)
    # test for group_by='selection'
    fig = plot_raw(raw, group_by='selection')
    for key in 2 * keys + ('escape',):
        fig._fake_keypress(key)


def test_plot_raw_traces(raw, events, browse_backend):
    """Test plotting of raw data."""
    raw.info['lowpass'] = 10.  # allow heavy decim during plotting
    fig = raw.plot(events=events, order=[1, 7, 5, 2, 3], n_channels=3,
                   group_by='original')
    assert hasattr(fig, 'mne')  # make sure fig.mne param object is present
    assert len(fig.axes) == 5

    # setup
    x = fig.mne.traces[0].get_xdata()[5]
    y = fig.mne.traces[0].get_ydata()[5]
    data_ax = fig.mne.ax_main
    # ToDo: The interaction with scrollbars will be different in pyqtgraph.
    hscroll = fig.mne.ax_hscroll
    vscroll = fig.mne.ax_vscroll
    # test marking bad channels
    label = fig.mne.ax_main.get_yticklabels()[0].get_text()
    assert label not in fig.mne.info['bads']
    # click data to mark bad
    fig._fake_click((x, y), xform='data')
    assert label in fig.mne.info['bads']
    # click data to unmark bad
    fig._fake_click((x, y), xform='data')
    assert label not in fig.mne.info['bads']
    # click name to mark bad
    fig._click_ch_name(ch_index=0, button=1)
    assert label in fig.mne.info['bads']
    # test other kinds of clicks
    fig._fake_click((0.5, 0.999))  # click elsewhere (add vline)
    fig._fake_click((0.5, 0.999), button=3)  # remove vline
    fig._fake_click((0.5, 0.5), ax=hscroll)  # change time
    fig._fake_click((0.5, 0.5), ax=hscroll)  # shouldn't change time this time
    # test scrolling through channels
    labels = [label.get_text() for label in data_ax.get_yticklabels()]
    assert labels == [raw.ch_names[1], raw.ch_names[7], raw.ch_names[5]]
    fig._fake_click((0.5, 0.01), ax=vscroll)  # change channels to end
    labels = [label.get_text() for label in data_ax.get_yticklabels()]
    assert labels == [raw.ch_names[5], raw.ch_names[2], raw.ch_names[3]]
    for _ in (0, 0):
        # first click changes channels to mid; second time shouldn't change
        fig._fake_click((0.5, 0.5), ax=vscroll)
        labels = [label.get_text() for label in data_ax.get_yticklabels()]
        assert labels == [raw.ch_names[7], raw.ch_names[5], raw.ch_names[2]]
        assert browse_backend._get_n_figs() == 1

    # test clicking a channel name in butterfly mode
    bads = fig.mne.info['bads'].copy()
    fig._fake_keypress('b')
    fig._click_ch_name(ch_index=0, button=1)  # should be no-op
    assert fig.mne.info['bads'] == bads        # unchanged
    fig._fake_keypress('b')

    # test starting up in zen mode
    fig = plot_raw(raw, show_scrollbars=False)
    # test order, title, & show_options kwargs
    with pytest.raises(ValueError, match='order should be array-like; got'):
        raw.plot(order='foo')
    with pytest.raises(TypeError, match='title must be None or a string, got'):
        raw.plot(title=1)
    raw.plot(show_options=True)
    browse_backend._close_all()

    # annotations outside data range
    annot = Annotations([10, 10 + raw.first_samp / raw.info['sfreq']],
                        [10, 10], ['test', 'test'], raw.info['meas_date'])
    with pytest.warns(RuntimeWarning, match='outside data range'):
        raw.set_annotations(annot)

    # Color setting
    with pytest.raises(KeyError, match='must be strictly positive, or -1'):
        raw.plot(event_color={0: 'r'})
    with pytest.raises(TypeError, match='event_color key must be an int, got'):
        raw.plot(event_color={'foo': 'r'})
    fig = plot_raw(raw, events=events, event_color={-1: 'r', 998: 'b'})
    browse_backend._close_all()


@pytest.mark.parametrize('group_by', ('position', 'selection'))
def test_plot_raw_groupby(raw, browse_backend, group_by):
    """Test group-by plotting of raw data."""
    raw.info['lowpass'] = 10.  # allow heavy decim during plotting
    order = (np.arange(len(raw.ch_names))[::-3] if group_by == 'position' else
             [1, 2, 4, 6])
    fig = raw.plot(group_by=group_by, order=order)
    x = fig.mne.traces[0].get_xdata()[10]
    y = fig.mne.traces[0].get_ydata()[10]
    fig._fake_keypress('down')  # change selection
    fig._fake_click((x, y), xform='data')  # mark bad
    fig._fake_click((0.5, 0.5), ax=fig.mne.ax_vscroll)  # change channels
    sel_fig = fig.mne.fig_selection
    topo_ax = sel_fig.mne.sensor_ax
    fig._fake_click([-0.425, 0.20223853], sel_fig, topo_ax, xform='data')
    fig._fake_keypress('down')
    fig._fake_keypress('up')
    fig._fake_scroll(0.5, 0.5, -1)  # scroll down
    fig._fake_scroll(0.5, 0.5, 1)  # scroll up
    fig._fake_click([-0.5, 0.], sel_fig, topo_ax, xform='data')
    fig._fake_click((0.5, 0.), sel_fig, topo_ax, xform='data',
                    kind='motion')
    fig._fake_click((0.5, 0.5), sel_fig, topo_ax, xform='data',
                    kind='motion')
    fig._fake_click([-0.5, 0.5], sel_fig, topo_ax, xform='data',
                    kind='release')
    browse_backend._close_all()


def test_plot_raw_meas_date(raw, browse_backend):
    """Test effect of mismatched meas_date in raw.plot()."""
    raw.set_meas_date(_dt_to_stamp(raw.info['meas_date'])[0])
    annot = Annotations([1 + raw.first_samp / raw.info['sfreq']], [5], ['bad'])
    with pytest.warns(RuntimeWarning, match='outside data range'):
        raw.set_annotations(annot)
    with pytest.warns(None):  # sometimes projection
        raw.plot(group_by='position', order=np.arange(8))
    fig = raw.plot()
    for key in ['down', 'up', 'escape']:
        fig._fake_keypress(key, fig=fig.mne.fig_selection)
    browse_backend._close_all()


def test_plot_raw_nan(raw, browse_backend):
    """Test plotting all NaNs."""
    raw._data[:] = np.nan
    # this should (at least) not die, the output should pretty clearly show
    # that there is a problem so probably okay to just plot something blank
    with pytest.warns(None):
        raw.plot(scalings='auto')
    browse_backend._close_all()


@testing.requires_testing_data
def test_plot_raw_white(raw_orig, noise_cov_io, browse_backend):
    """Test plotting whitened raw data."""
    raw_orig.crop(0, 1)
    fig = raw_orig.plot(noise_cov=noise_cov_io)
    # toggle whitening
    fig._fake_keypress('w')
    fig._fake_keypress('w')
    browse_backend._close_all()


@testing.requires_testing_data
def test_plot_ref_meg(raw_ctf, browse_backend):
    """Test plotting ref_meg."""
    raw_ctf.crop(0, 1)
    raw_ctf.plot()
    browse_backend._close_all()
    pytest.raises(ValueError, raw_ctf.plot, group_by='selection')


def test_plot_misc_auto(browse_backend):
    """Test plotting of data with misc auto scaling."""
    data = np.random.RandomState(0).randn(1, 1000)
    raw = RawArray(data, create_info(1, 1000., 'misc'))
    raw.plot()
    browse_backend._close_all()


@pytest.mark.slowtest
def test_plot_annotations(raw, browse_backend):
    """Test annotation mode of the plotter."""
    raw.info['lowpass'] = 10.
    _annotation_helper(raw, browse_backend)
    _annotation_helper(raw, browse_backend, events=True)

    annot = Annotations([42], [1], 'test', raw.info['meas_date'])
    with pytest.warns(RuntimeWarning, match='expanding outside'):
        raw.set_annotations(annot)
    _annotation_helper(raw, browse_backend)
    # test annotation visibility toggle
    fig = raw.plot()
    assert len(fig.mne.annotations) == 1
    assert len(fig.mne.annotation_texts) == 1
    fig._fake_keypress('a')  # start annotation mode
    # ToDo: This will be different in pyqtgraph (toolbar).
    checkboxes = fig.mne.show_hide_annotation_checkboxes
    checkboxes.set_active(0)
    assert len(fig.mne.annotations) == 0
    assert len(fig.mne.annotation_texts) == 0
    checkboxes.set_active(0)
    assert len(fig.mne.annotations) == 1
    assert len(fig.mne.annotation_texts) == 1


@pytest.mark.parametrize('hide_which', ([], [0], [1], [0, 1]))
def test_remove_annotations(raw, hide_which, browse_backend):
    """Test that right-click doesn't remove hidden annotation spans."""
    ann = Annotations(onset=[2, 1], duration=[1, 3],
                      description=['foo', 'bar'])
    raw.set_annotations(ann)
    assert len(raw.annotations) == 2
    fig = raw.plot()
    fig._fake_keypress('a')  # start annotation mode
    # ToDo: This will be different in pyqtgraph (toolbar).
    checkboxes = fig.mne.show_hide_annotation_checkboxes
    for which in hide_which:
        checkboxes.set_active(which)
    fig._fake_click((2.5, 0.1), xform='data', button=3)
    assert len(raw.annotations) == len(hide_which)


@pytest.mark.parametrize('filtorder', (0, 2))  # FIR, IIR
def test_plot_raw_filtered(filtorder, raw, browse_backend):
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
    # shouldn't break if all shown are non-data
    RawArray(np.zeros((1, 100)), create_info(1, 20., 'stim')).plot(lowpass=5)


def test_plot_raw_psd(raw, raw_orig):
    """Test plotting of raw psds."""
    raw_unchanged = raw.copy()
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
    raw = raw_unchanged
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
    raw = raw_orig.crop(0, 1)
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


@pytest.mark.parametrize('cfg_value', (None, '0.1,0.1'))
def test_min_window_size(raw, cfg_value, browse_backend):
    """Test minimum window plot size."""
    old_cfg = get_config('MNE_BROWSE_RAW_SIZE')
    set_config('MNE_BROWSE_RAW_SIZE', cfg_value)
    fig = raw.plot()
    # 8 × 8 inches is default minimum size
    assert_array_equal(fig.get_size_inches(), (8, 8))
    set_config('MNE_BROWSE_RAW_SIZE', old_cfg)


def test_scalings_int(browse_backend):
    """Test that auto scalings access samples using integers."""
    raw = RawArray(np.zeros((1, 500)), create_info(1, 1000., 'eeg'))
    raw.plot(scalings='auto')


@pytest.mark.parametrize('dur, n_dec', [(20, 1), (4.2, 2), (0.01, 4)])
def test_clock_xticks(raw, dur, n_dec, browse_backend):
    """Test if decimal seconds of xticks have appropriate length."""
    fig = raw.plot(duration=dur, time_format='clock')
    fig.canvas.draw()
    ticklabels = fig.mne.ax_main.get_xticklabels()
    tick_texts = [tl.get_text() for tl in ticklabels]
    assert tick_texts[0].startswith('19:01:53')
    if len(tick_texts[0].split('.')) > 1:
        assert len(tick_texts[0].split('.')[1]) == n_dec
