# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: Simplified BSD

import itertools
import os
from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import backend_bases
import numpy as np
import pytest
from numpy.testing import assert_allclose

from mne import Annotations, create_info, pick_types
from mne.annotations import _sync_onset
from mne.datasets import testing
from mne.io import RawArray
from mne.io.pick import _DATA_CH_TYPES_ORDER_DEFAULT, _PICK_TYPES_DATA_DICT
from mne.utils import (_dt_to_stamp, _record_warnings, get_config, set_config,
                       _assert_no_instances)
from mne.viz import plot_raw, plot_sensors
from mne.viz.utils import _fake_click, _fake_keypress


def _annotation_helper(raw, browse_backend, events=False):
    """Test interactive annotations."""
    ismpl = browse_backend.name == 'matplotlib'
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
    if ismpl:
        assert browse_backend._get_n_figs() == 1

    fig._fake_keypress('a')  # annotation mode
    ann_fig = fig.mne.fig_annotation
    if ismpl:
        assert browse_backend._get_n_figs() == 2
        # +3 from the scale bars
        n_scale = 3
        assert len(fig.mne.ax_main.texts) == n_anns + n_events + n_scale
    else:
        assert ann_fig.isVisible()

    # modify description to create label "BAD test"
    # semicolon is ignored
    if ismpl:
        for key in ['backspace'] + list(' test;') + ['enter']:
            fig._fake_keypress(key, fig=ann_fig)
        # change annotation label
        for ix in (-1, 0):
            xy = ann_fig.mne.radio_ax.buttons.circles[ix].center
            fig._fake_click(xy, fig=ann_fig, ax=ann_fig.mne.radio_ax,
                            xform='data')
    else:
        # The modal dialogs of the Qt-backend would block the test,
        # thus a new description will be added programmatically.
        ann_fig._add_description('BAD test')

    # draw annotation
    fig._fake_click((1., 1.), add_points=[(5., 1.)], xform='data', button=1,
                    kind='drag')
    if ismpl:
        assert len(fig.mne.ax_main.texts) == n_anns + 1 + n_events + n_scale
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
    assert len(raw.annotations.onset) == n_anns + 1
    assert len(raw.annotations.duration) == n_anns + 1
    assert len(raw.annotations.description) == n_anns + 1
    assert raw.annotations.description[n_anns] == 'BAD test'
    onset = raw.annotations.onset[n_anns]
    want_onset = _sync_onset(raw, 1., inverse=True)
    # pyqtgraph: during the transformation from pixel-coordinates
    # to scene-coordinates when the click is simulated on QGraphicsView
    # with QTest, there seems to happen a rounding of pixels to integers
    # internally. This deviatian also seems to change between runs
    # (maybe device-dependent?).
    atol = 1e-10 if ismpl else 2e-2
    assert_allclose(onset, want_onset, atol=atol)
    assert_allclose(raw.annotations.duration[n_anns], 4., atol=atol)
    # modify annotation from end (duration 4 → 1.5)
    fig._fake_click((4.9, 1.), xform='data', button=1,
                    kind='motion')  # ease up to it
    fig._fake_click((5., 1.), add_points=[(2.5, 1.)], xform='data',
                    button=1, kind='drag')
    assert raw.annotations.onset[n_anns] == onset
    # 4 → 1.5
    assert_allclose(raw.annotations.duration[n_anns], 1.5, atol=atol)
    # modify annotation from beginning (duration 1.5 → 2.0)
    fig._fake_click((1., 1.), add_points=[(0.5, 1.)], xform='data', button=1,
                    kind='drag')
    assert_allclose(raw.annotations.onset[n_anns], onset - 0.5, atol=atol)
    # 1.5 → 2.0
    assert_allclose(raw.annotations.duration[n_anns], 2.0, atol=atol)
    assert len(raw.annotations.onset) == n_anns + 1
    assert len(raw.annotations.duration) == n_anns + 1
    assert len(raw.annotations.description) == n_anns + 1
    assert raw.annotations.description[n_anns] == 'BAD test'
    if ismpl:
        assert len(fig.axes[0].texts) == n_anns + 1 + n_events + n_scale
        fig._fake_keypress('shift+right')
        assert len(fig.axes[0].texts) == n_scale
        fig._fake_keypress('shift+left')
        assert len(fig.axes[0].texts) == n_anns + 1 + n_events + n_scale

    # draw another annotation merging the two
    fig._fake_click((5.5, 1.), add_points=[(2., 1.)],
                    xform='data', button=1, kind='drag')
    # delete the annotation
    assert len(raw.annotations.onset) == n_anns + 1
    assert len(raw.annotations.duration) == n_anns + 1
    assert len(raw.annotations.description) == n_anns + 1
    assert_allclose(raw.annotations.onset[n_anns], onset - 0.5, atol=atol)
    assert_allclose(raw.annotations.duration[n_anns], 5.0, atol=atol)
    if ismpl:
        assert len(fig.axes[0].texts) == n_anns + 1 + n_events + n_scale
    # Delete
    fig._fake_click((1.5, 1.), xform='data', button=3, kind='press')
    # exit, re-enter, then exit a different way
    fig._fake_keypress('a')  # exit
    fig._fake_keypress('a')  # enter
    assert len(raw.annotations.onset) == n_anns
    if ismpl:
        fig._fake_keypress('escape', fig=fig.mne.fig_annotation)  # exit again
        assert len(fig.axes[0].texts) == n_anns + n_events + n_scale
        fig._fake_keypress('shift+right')
        assert len(fig.axes[0].texts) == n_scale
        fig._fake_keypress('shift+left')
        assert len(fig.axes[0].texts) == n_anns + n_events + n_scale


def _proj_status(ssp_fig, browse_backend):
    if browse_backend.name == 'matplotlib':
        ax = ssp_fig.mne.proj_checkboxes.ax
        return [line.get_visible() for line
                in ax.findobj(matplotlib.lines.Line2D)][::2]
    else:
        return [chkbx.isChecked() for chkbx in ssp_fig.checkboxes]


def _proj_label(ssp_fig, browse_backend):
    if browse_backend.name == 'matplotlib':
        return [lb.get_text() for lb in ssp_fig.mne.proj_checkboxes.labels]
    else:
        return [chkbx.text() for chkbx in ssp_fig.checkboxes]


def _proj_click(idx, fig, browse_backend):
    ssp_fig = fig.mne.fig_proj
    if browse_backend.name == 'matplotlib':
        pos = np.array(ssp_fig.mne.proj_checkboxes.
                       labels[idx].get_position()) + 0.01

        fig._fake_click(pos, fig=ssp_fig, ax=ssp_fig.mne.proj_checkboxes.ax,
                        xform='data')
    else:
        # _fake_click on QCheckBox is inconsistent across platforms
        # (also see comment in test_plot_raw_selection).
        ssp_fig._proj_changed(not fig.mne.projs_on[idx], idx)
        # Update Checkbox
        ssp_fig.checkboxes[idx].setChecked(bool(fig.mne.projs_on[idx]))


def _proj_click_all(fig, browse_backend):
    ssp_fig = fig.mne.fig_proj
    if browse_backend.name == 'matplotlib':
        fig._fake_click((0.5, 0.5), fig=ssp_fig, ax=ssp_fig.mne.proj_all.ax)
        fig._fake_click((0.5, 0.5), fig=ssp_fig, ax=ssp_fig.mne.proj_all.ax,
                        kind='release')
    else:
        # _fake_click on QPushButton is inconsistent across platforms.
        ssp_fig.toggle_all()


def _spawn_child_fig(fig, attr, browse_backend, key):
    # starting state
    n_figs = browse_backend._get_n_figs()
    n_children = len(fig.mne.child_figs)
    # spawn the child fig
    fig._fake_keypress(key)
    # make sure the figure was actually spawned
    assert len(fig.mne.child_figs) == n_children + 1
    assert browse_backend._get_n_figs() == n_figs + 1
    # make sure the parent fig knows the child fig's name
    child_fig = getattr(fig.mne, attr)
    assert child_fig is not None
    return child_fig


def _destroy_child_fig(fig, child_fig, attr, browse_backend, key, key_target):
    # starting state
    n_figs = browse_backend._get_n_figs()
    n_children = len(fig.mne.child_figs)
    # destroy child fig (_close_event is MPL agg backend workaround)
    fig._fake_keypress(key, fig=key_target)
    fig._close_event(child_fig)
    # make sure the figure was actually destroyed
    assert len(fig.mne.child_figs) == n_children - 1
    assert browse_backend._get_n_figs() == n_figs - 1
    assert getattr(fig.mne, attr) is None


def _child_fig_helper(fig, key, attr, browse_backend):
    # Spawn and close child figs of raw.plot()
    assert getattr(fig.mne, attr) is None
    # spawn, then close via main window toggle
    child_fig = _spawn_child_fig(fig, attr, browse_backend, key)
    _destroy_child_fig(fig, child_fig, attr, browse_backend, key,
                       key_target=fig)
    # spawn again, then close via child window's close key
    child_fig = _spawn_child_fig(fig, attr, browse_backend, key)
    _destroy_child_fig(fig, child_fig, attr, browse_backend,
                       key=child_fig.mne.close_key,
                       key_target=child_fig)


def test_scale_bar(browser_backend):
    """Test scale bar for raw."""
    ismpl = browser_backend.name == 'matplotlib'
    sfreq = 1000.
    t = np.arange(10000) / sfreq
    data = np.sin(2 * np.pi * 10. * t)
    # ± 1000 fT, 400 fT/cm, 20 µV
    data = data * np.array([[1000e-15, 400e-13, 20e-6]]).T
    info = create_info(3, sfreq, ('mag', 'grad', 'eeg'))
    raw = RawArray(data, info)
    fig = raw.plot()
    texts = fig._get_scale_bar_texts()
    assert len(texts) == 3  # ch_type scale-bars
    wants = ('800.0 fT/cm', '2000.0 fT', '40.0 µV')
    assert texts == wants
    if ismpl:
        # 1 green vline, 3 data, 3 scalebars
        assert len(fig.mne.ax_main.lines) == 7
    else:
        assert len(fig.mne.scalebars) == 3
    for data, bar in zip(fig.mne.traces, fig.mne.scalebars.values()):
        y = data.get_ydata()
        y_lims = [y.min(), y.max()]
        bar_lims = bar.get_ydata()
        assert_allclose(y_lims, bar_lims, atol=1e-4)


def test_plot_raw_selection(raw, browser_backend):
    """Test selection mode of plot_raw()."""
    ismpl = browser_backend.name == 'matplotlib'
    with raw.info._unlock():
        raw.info['lowpass'] = 10.  # allow heavy decim during plotting
    browser_backend._close_all()           # ensure all are closed
    assert browser_backend._get_n_figs() == 0
    fig = raw.plot(group_by='selection', proj=False)
    assert browser_backend._get_n_figs() == 2
    sel_fig = fig.mne.fig_selection
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

    # ToDo: For Qt-backend the framework around RawTraceItem makes
    #  it difficult to show the same channel multiple times which is why
    #  it is currently not implemented.
    #  This would be relevant if you wanted to plot several selections in
    #  butterfly-mode which have some channels in common.
    sel_picks = len(np.concatenate(list(sel_dict.values())))
    if ismpl:
        assert len(fig.mne.traces) == sel_picks
    else:
        assert len(fig.mne.traces) == sel_picks - 1
    assert fig.mne.butterfly
    # test clicking on radio buttons → should cancel butterfly mode
    if ismpl:
        xy = sel_fig.mne.radio_ax.buttons.circles[0].center
        fig._fake_click(xy, fig=sel_fig, ax=sel_fig.mne.radio_ax, xform='data')
    else:
        # For an unknown reason test-clicking on checkboxes is inconsistent
        # across platforms.
        # (QTest.mouseClick works isolated on all platforms but somehow
        # not in this context. _fake_click isn't working on linux)
        sel_fig._chkbx_changed(list(sel_fig.chkbxs.keys())[0])
    assert len(fig.mne.traces) == len(sel_dict['Left-temporal'])  # 6
    assert not fig.mne.butterfly
    # test clicking on "custom" when not defined: should be no-op
    if ismpl:
        before_state = sel_fig.mne.radio_ax.buttons.value_selected
        xy = sel_fig.mne.radio_ax.buttons.circles[-1].center
        fig._fake_click(xy, fig=sel_fig, ax=sel_fig.mne.radio_ax, xform='data')
        lasso = sel_fig.lasso
        sensor_ax = sel_fig.mne.sensor_ax
        assert sel_fig.mne.radio_ax.buttons.value_selected == before_state
    else:
        before_state = sel_fig.mne.old_selection
        chkbx = sel_fig.chkbxs[list(sel_fig.chkbxs.keys())[-1]]
        fig._fake_click((0.5, 0.5), fig=chkbx)
        lasso = sel_fig.channel_fig.lasso
        sensor_ax = sel_fig.channel_widget
        assert before_state == sel_fig.mne.old_selection          # unchanged
    assert len(fig.mne.traces) == len(sel_dict['Left-temporal'])  # unchanged
    # test marking bad channel in selection mode → should make sensor red
    assert lasso.ec[:, 0].sum() == 0   # R of RGBA zero for all chans
    fig._click_ch_name(ch_index=1, button=1)  # mark bad
    assert lasso.ec[:, 0].sum() == 1   # one channel red
    fig._click_ch_name(ch_index=1, button=1)  # mark good
    assert lasso.ec[:, 0].sum() == 0   # all channels black
    # test lasso
    # Testing lasso-interactivity of sensor-plot within Qt-backend
    # with QTest doesn't seem to work.
    want = ['MEG 0111', 'MEG 0112', 'MEG 0113', 'MEG 0131', 'MEG 0132',
            'MEG 0133']
    assert want == sorted(fig.mne.ch_names[fig.mne.picks])
    want = ['MEG 0121', 'MEG 0122', 'MEG 0123']
    if ismpl:
        sel_fig._set_custom_selection()  # lasso empty → should do nothing
        # Lasso with 1 mag/grad sensor unit (upper left)
        fig._fake_click((0, 1), add_points=[(0.65, 1), (0.65, 0.7), (0, 0.7)],
                        fig=sel_fig, ax=sensor_ax, xform='ax', kind='drag')
    else:
        lasso.selection = want
        sel_fig._set_custom_selection()
    assert sorted(want) == sorted(fig.mne.ch_names[fig.mne.picks])
    # test joint closing of selection & data windows
    fig._fake_keypress(sel_fig.mne.close_key, fig=sel_fig)
    fig._close_event(sel_fig)
    assert browser_backend._get_n_figs() == 0


def test_plot_raw_ssp_interaction(raw, browser_backend):
    """Test SSP projector UI of plot_raw()."""
    with raw.info._unlock():
        raw.info['lowpass'] = 10.  # allow heavy decim during plotting
    # apply some (not all) projs to test our proj UI (greyed out applied projs)
    projs = raw.info['projs'][-2:]
    raw.del_proj([-2, -1])
    raw.apply_proj()
    raw.add_proj(projs)
    fig = raw.plot()
    # open SSP window
    fig._fake_keypress('j')
    assert browser_backend._get_n_figs() == 2
    ssp_fig = fig.mne.fig_proj
    assert _proj_status(ssp_fig, browser_backend) == [True, True, True]
    # this should have no effect (proj 0 is already applied)
    assert _proj_label(ssp_fig,
                       browser_backend)[0].endswith('(already applied)')
    _proj_click(0, fig, browser_backend)
    assert _proj_status(ssp_fig, browser_backend) == [True, True, True]
    # this should work (proj 1 not applied)
    _proj_click(1, fig, browser_backend)
    assert _proj_status(ssp_fig, browser_backend) == [True, False, True]
    # turn it back on
    _proj_click(1, fig, browser_backend)
    assert _proj_status(ssp_fig, browser_backend) == [True, True, True]
    # toggle all off (button axes need both press and release)
    _proj_click_all(fig, browser_backend)
    assert _proj_status(ssp_fig, browser_backend) == [True, False, False]
    fig._fake_keypress('J')
    assert _proj_status(ssp_fig, browser_backend) == [True, True, True]
    fig._fake_keypress('J')
    assert _proj_status(ssp_fig, browser_backend) == [True, False, False]
    # turn all on
    _proj_click_all(fig, browser_backend)
    assert fig.mne.projector is not None  # on
    assert _proj_status(ssp_fig, browser_backend) == [True, True, True]


def test_plot_raw_child_figures(raw, browser_backend):
    """Test spawning and closing of child figures."""
    ismpl = browser_backend.name == 'matplotlib'
    with raw.info._unlock():
        raw.info['lowpass'] = 10.  # allow heavy decim during plotting
    # make sure we start clean
    assert browser_backend._get_n_figs() == 0
    fig = raw.plot()
    assert browser_backend._get_n_figs() == 1
    # test child fig toggles
    _child_fig_helper(fig, '?', 'fig_help', browser_backend)
    _child_fig_helper(fig, 'j', 'fig_proj', browser_backend)
    if ismpl:  # in mne-qt-browser, annotation is a dock-widget, not a window
        _child_fig_helper(fig, 'a', 'fig_annotation', browser_backend)
    # test right-click → channel location popup
    fig._redraw()
    fig._click_ch_name(ch_index=2, button=3)
    assert len(fig.mne.child_figs) == 1
    assert browser_backend._get_n_figs() == 2
    fig._fake_keypress('escape', fig=fig.mne.child_figs[0])
    if ismpl:
        fig._close_event(fig.mne.child_figs[0])
    assert len(fig.mne.child_figs) == 0
    assert browser_backend._get_n_figs() == 1
    # test right-click on non-data channel
    ix = raw.get_channel_types().index('ias')  # find the shielding channel
    trace_ix = fig.mne.ch_order.tolist().index(ix)  # get its plotting position
    fig._redraw()
    fig._click_ch_name(ch_index=trace_ix, button=3)  # should be no-op
    assert len(fig.mne.child_figs) == 0
    assert browser_backend._get_n_figs() == 1
    # test resize of main window
    fig._resize_by_factor(0.5)


def test_orphaned_annot_fig(raw, browser_backend):
    """Test that annotation window is not orphaned (GH #10454)."""
    if browser_backend.name != 'matplotlib':
        return
    assert browser_backend._get_n_figs() == 0
    fig = raw.plot()
    _spawn_child_fig(fig, 'fig_annotation', browser_backend, 'a')
    fig._fake_keypress(key=fig.mne.close_key)
    fig._close_event()
    assert len(fig.mne.child_figs) == 0
    assert browser_backend._get_n_figs() == 0


def _monkeypatch_fig(fig, browser_backend):
    if browser_backend.name == 'matplotlib':
        fig.canvas.manager.full_screen_toggle = lambda: None
    else:
        # Monkeypatch the Qt methods
        def _full():
            fig.isFullScreen = lambda: True

        def _norm():
            fig.isFullScreen = lambda: False

        fig.showFullScreen = _full
        fig.showNormal = _norm


def test_plot_raw_keypresses(raw, browser_backend, monkeypatch):
    """Test keypress interactivity of plot_raw()."""
    with raw.info._unlock():
        raw.info['lowpass'] = 10.  # allow heavy decim during plotting
    fig = raw.plot()
    # test twice → once in normal, once in butterfly view.
    # NB: keys a, j, and ? are tested in test_plot_raw_child_figures()
    keys = ('pagedown', 'down', 'up', 'down', 'right', 'left', '-', '+', '=',
            'd', 'd', 'pageup', 'home', 'end', 'z', 'z', 's', 's', 'f11', 't',
            'b')
    # Avoid annoying fullscreen issues by monkey-patching our handlers
    _monkeypatch_fig(fig, browser_backend)
    # test for group_by='original'
    for key in 2 * keys + ('escape',):
        fig._fake_keypress(key)
    # test for group_by='selection'
    fig = plot_raw(raw, group_by='selection')
    _monkeypatch_fig(fig, browser_backend)
    for key in 2 * keys + ('escape',):
        fig._fake_keypress(key)


def test_plot_raw_traces(raw, events, browser_backend):
    """Test plotting of raw data."""
    ismpl = browser_backend.name == 'matplotlib'
    with raw.info._unlock():
        raw.info['lowpass'] = 10.  # allow heavy decim during plotting
    fig = raw.plot(events=events, order=[1, 7, 5, 2, 3], n_channels=3,
                   group_by='original')
    assert hasattr(fig, 'mne')  # make sure fig.mne param object is present
    if ismpl:
        assert len(fig.axes) == 5

    # setup
    x = fig.mne.traces[0].get_xdata()[5]
    y = fig.mne.traces[0].get_ydata()[5]
    hscroll = fig.mne.ax_hscroll
    vscroll = fig.mne.ax_vscroll
    # test marking bad channels
    label = fig._get_ticklabels('y')[0]
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
    fig._fake_click((0.5, 0.98))  # click elsewhere (add vline)
    assert fig.mne.vline_visible is True
    fig._fake_click((0.5, 0.98), button=3)  # remove vline
    assert fig.mne.vline_visible is False
    fig._fake_click((0.5, 0.5), ax=hscroll)  # change time
    t_start = fig.mne.t_start
    fig._fake_click((0.5, 0.5), ax=hscroll)  # shouldn't change time this time
    assert round(t_start, 6) == round(fig.mne.t_start, 6)
    # test scrolling through channels
    labels = fig._get_ticklabels('y')
    assert labels == [raw.ch_names[1], raw.ch_names[7], raw.ch_names[5]]
    fig._fake_click((0.5, 0.05), ax=vscroll)  # change channels to end
    labels = fig._get_ticklabels('y')
    assert labels == [raw.ch_names[5], raw.ch_names[2], raw.ch_names[3]]
    for _ in (0, 0):
        # first click changes channels to mid; second time shouldn't change
        # This needs to be changed for Qt, because there scrollbars are
        # drawn differently (value of slider at lower end, not at middle)
        yclick = 0.5 if ismpl else 0.7
        fig._fake_click((0.5, yclick), ax=vscroll)
        labels = fig._get_ticklabels('y')
        assert labels == [raw.ch_names[7], raw.ch_names[5], raw.ch_names[2]]

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
    browser_backend._close_all()

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
    plot_raw(raw, events=events, event_color={-1: 'r', 998: 'b'})


@pytest.mark.parametrize('group_by', ('position', 'selection'))
def test_plot_raw_groupby(raw, browser_backend, group_by):
    """Test group-by plotting of raw data."""
    with raw.info._unlock():
        raw.info['lowpass'] = 10.  # allow heavy decim during plotting
    order = (np.arange(len(raw.ch_names))[::-3] if group_by == 'position' else
             [1, 2, 4, 6])
    fig = raw.plot(group_by=group_by, order=order)
    x = fig.mne.traces[0].get_xdata()[10]
    y = fig.mne.traces[0].get_ydata()[10]
    fig._fake_keypress('down')  # change selection
    fig._fake_click((x, y), xform='data')  # mark bad
    fig._fake_click((0.5, 0.5), ax=fig.mne.ax_vscroll)  # change channels
    if browser_backend.name == 'matplotlib':
        # Test lasso-selection
        # (test difficult with Qt-backend, set plot_raw_selection)
        sel_fig = fig.mne.fig_selection
        topo_ax = sel_fig.mne.sensor_ax
        fig._fake_click([-0.425, 0.20223853], fig=sel_fig, ax=topo_ax,
                        xform='data')
        fig._fake_click((-0.5, 0.), add_points=[(0.5, 0.),
                                                (0.5, 0.5),
                                                (-0.5, 0.5)],
                        fig=sel_fig, ax=topo_ax, xform='data', kind='drag')
        fig._fake_keypress('down')
        fig._fake_keypress('up')
    fig._fake_keypress('up')
    fig._fake_scroll(0.5, 0.5, -1)  # scroll down
    fig._fake_scroll(0.5, 0.5, 1)  # scroll up


def test_plot_raw_meas_date(raw, browser_backend):
    """Test effect of mismatched meas_date in raw.plot()."""
    raw.set_meas_date(_dt_to_stamp(raw.info['meas_date'])[0])
    annot = Annotations([1 + raw.first_samp / raw.info['sfreq']], [5], ['bad'])
    with pytest.warns(RuntimeWarning, match='outside data range'):
        raw.set_annotations(annot)
    with _record_warnings():  # sometimes projection
        raw.plot(group_by='position', order=np.arange(8))
    fig = raw.plot()
    for key in ['down', 'up', 'escape']:
        fig._fake_keypress(key, fig=fig.mne.fig_selection)


def test_plot_raw_nan(raw, browser_backend):
    """Test plotting all NaNs."""
    raw._data[:] = np.nan
    # this should (at least) not die, the output should pretty clearly show
    # that there is a problem so probably okay to just plot something blank
    with _record_warnings():
        raw.plot(scalings='auto')


@testing.requires_testing_data
def test_plot_raw_white(raw_orig, noise_cov_io, browser_backend):
    """Test plotting whitened raw data."""
    raw_orig.crop(0, 1)
    fig = raw_orig.plot(noise_cov=noise_cov_io)
    # toggle whitening
    fig._fake_keypress('w')
    fig._fake_keypress('w')


@testing.requires_testing_data
def test_plot_ref_meg(raw_ctf, browser_backend):
    """Test plotting ref_meg."""
    raw_ctf.crop(0, 1)
    raw_ctf.plot()
    pytest.raises(ValueError, raw_ctf.plot, group_by='selection')


def test_plot_misc_auto(browser_backend):
    """Test plotting of data with misc auto scaling."""
    data = np.random.RandomState(0).randn(1, 1000)
    raw = RawArray(data, create_info(1, 1000., 'misc'))
    raw.plot()
    raw = RawArray(data, create_info(1, 1000., 'dipole'))
    raw.plot(order=[0])  # plot, even though it's not "data"
    browser_backend._close_all()


@pytest.mark.slowtest
def test_plot_annotations(raw, browser_backend):
    """Test annotation mode of the plotter."""
    ismpl = browser_backend.name == 'matplotlib'
    with raw.info._unlock():
        raw.info['lowpass'] = 10.
    _annotation_helper(raw, browser_backend)
    _annotation_helper(raw, browser_backend, events=True)

    annot = Annotations([42], [1], 'test', raw.info['meas_date'])
    with pytest.warns(RuntimeWarning, match='expanding outside'):
        raw.set_annotations(annot)
    _annotation_helper(raw, browser_backend)
    # test annotation visibility toggle
    fig = raw.plot()
    if ismpl:
        assert len(fig.mne.annotations) == 1
        assert len(fig.mne.annotation_texts) == 1
    else:
        assert len(fig.mne.regions) == 1
    fig._fake_keypress('a')  # start annotation mode
    if ismpl:
        checkboxes = fig.mne.show_hide_annotation_checkboxes
        checkboxes.set_active(0)
        assert len(fig.mne.annotations) == 0
        assert len(fig.mne.annotation_texts) == 0
        checkboxes.set_active(0)
        assert len(fig.mne.annotations) == 1
        assert len(fig.mne.annotation_texts) == 1
    else:
        fig.mne.visible_annotations['test'] = False
        fig._update_regions_visible()
        assert not fig.mne.regions[0].isVisible()
        fig.mne.visible_annotations['test'] = True
        fig._update_regions_visible()
        assert fig.mne.regions[0].isVisible()


@pytest.mark.parametrize('hide_which', ([], [0], [1], [0, 1]))
def test_remove_annotations(raw, hide_which, browser_backend):
    """Test that right-click doesn't remove hidden annotation spans."""
    descriptions = ['foo', 'bar']
    ann = Annotations(onset=[2, 1], duration=[1, 3],
                      description=descriptions)
    raw.set_annotations(ann)
    assert len(raw.annotations) == 2
    fig = raw.plot()
    fig._fake_keypress('a')  # start annotation mode
    if browser_backend.name == 'matplotlib':
        checkboxes = fig.mne.show_hide_annotation_checkboxes
        for which in hide_which:
            checkboxes.set_active(which)
    else:
        for hide_idx in hide_which:
            hide_key = descriptions[hide_idx]
            fig.mne.visible_annotations[hide_key] = False
        fig._update_regions_visible()
    fig._fake_click((2.5, 0.1), xform='data', button=3)
    assert len(raw.annotations) == len(hide_which)


@pytest.mark.parametrize('filtorder', (0, 2))  # FIR, IIR
def test_plot_raw_filtered(filtorder, raw, browser_backend):
    """Test filtering of raw plots."""
    # Opening that many plots can cause a Segmentation fault
    # if multithreading is activated in Qt-backend
    pg_kwargs = {'precompute': False}
    with pytest.raises(ValueError, match='lowpass.*Nyquist'):
        raw.plot(lowpass=raw.info['sfreq'] / 2., filtorder=filtorder,
                 **pg_kwargs)
    with pytest.raises(ValueError, match='highpass must be > 0'):
        raw.plot(highpass=0, filtorder=filtorder, **pg_kwargs)
    with pytest.raises(ValueError, match='Filter order must be'):
        raw.plot(lowpass=1, filtorder=-1, **pg_kwargs)
    with pytest.raises(ValueError, match="Invalid value for the 'clipping'"):
        raw.plot(clipping='foo', **pg_kwargs)
    raw.plot(lowpass=40, clipping='transparent', filtorder=filtorder,
             **pg_kwargs)
    raw.plot(highpass=1, clipping='clamp', filtorder=filtorder, **pg_kwargs)
    raw.plot(lowpass=40, butterfly=True, filtorder=filtorder, **pg_kwargs)
    # shouldn't break if all shown are non-data
    RawArray(np.zeros((1, 100)), create_info(1, 20., 'stim')).plot(lowpass=5)


def test_plot_raw_psd(raw, raw_orig):
    """Test plotting of raw psds."""
    raw_unchanged = raw.copy()
    # normal mode
    fig = raw.plot_psd(average=False)
    fig.canvas.callbacks.process(
        'resize_event',
        backend_bases.ResizeEvent('resize_event', fig.canvas))
    # specific mode
    picks = pick_types(raw.info, meg='mag', eeg=False)[:4]
    raw.plot_psd(tmax=None, picks=picks, area_mode='range', average=False,
                 spatial_colors=True)
    raw.plot_psd(tmax=20., color='yellow', dB=False, line_alpha=0.4,
                 average=False)
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
    with pytest.raises(ValueError, match='of length 2.*the length is 1'):
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
    n_times = sfreq = n_fft = 100
    data = 1e-3 * np.random.rand(2, n_times)
    info = create_info(['CH1', 'CH2'], sfreq)  # ch_types defaults to 'misc'
    raw = RawArray(data, info)
    picks = pick_types(raw.info, misc=True)
    raw.plot_psd(picks=picks, spatial_colors=False, n_fft=n_fft)
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
    _fake_keypress(fig, 'control')
    _fake_click(fig, ax, (0, 0.7), xform='ax', kind='release', key='control')
    assert fig.lasso.selection == ['MEG 0121']

    # check that point appearance changes
    fc = fig.lasso.collection.get_facecolors()
    ec = fig.lasso.collection.get_edgecolors()
    assert (fc[:, -1] == [0.5, 1., 0.5]).all()
    assert (ec[:, -1] == [0.25, 1., 0.25]).all()

    _fake_click(fig, ax, (0.7, 1), xform='ax', kind='motion', key='control')
    xy = ax.collections[0].get_offsets()
    _fake_click(fig, ax, xy[2], xform='data', key='control')  # single sel
    assert fig.lasso.selection == ['MEG 0121', 'MEG 0131']
    _fake_click(fig, ax, xy[2], xform='data', key='control')  # deselect
    assert fig.lasso.selection == ['MEG 0121']
    plt.close('all')

    raw.info['dev_head_t'] = None  # like empty room
    with pytest.warns(RuntimeWarning, match='identity'):
        raw.plot_sensors()

    # Test plotting with sphere='eeglab'
    info = create_info(
        ch_names=['Fpz', 'Oz', 'T7', 'T8'],
        sfreq=100,
        ch_types='eeg'
    )
    data = 1e-6 * np.random.rand(4, 100)
    raw_eeg = RawArray(data=data, info=info)
    raw_eeg.set_montage('biosemi64')
    raw_eeg.plot_sensors(sphere='eeglab')

    # Should work with "FPz" as well
    raw_eeg.rename_channels({'Fpz': 'FPz'})
    raw_eeg.plot_sensors(sphere='eeglab')

    # Should still work without Fpz/FPz, as long as we still have Oz
    raw_eeg.drop_channels('FPz')
    raw_eeg.plot_sensors(sphere='eeglab')

    # Should raise if Oz is missing too, as we cannot reconstruct Fpz anymore
    raw_eeg.drop_channels('Oz')
    with pytest.raises(ValueError, match='could not find: Fpz'):
        raw_eeg.plot_sensors(sphere='eeglab')

    # Should raise if we don't have a montage
    chs = deepcopy(raw_eeg.info['chs'])
    raw_eeg.set_montage(None)
    with raw_eeg.info._unlock():
        raw_eeg.info['chs'] = chs
    with pytest.raises(ValueError, match='No montage was set'):
        raw_eeg.plot_sensors(sphere='eeglab')


@pytest.mark.parametrize('cfg_value', (None, '0.1,0.1'))
def test_min_window_size(raw, cfg_value, browser_backend):
    """Test minimum window plot size."""
    old_cfg = get_config('MNE_BROWSE_RAW_SIZE')
    set_config('MNE_BROWSE_RAW_SIZE', cfg_value)
    fig = raw.plot()
    # For an unknown reason, the Windows-CI is a bit off
    # (on local Windows 10 the size is exactly as expected).
    atol = 0 if not os.name == 'nt' else 0.2
    # 8 × 8 inches is default minimum size.
    assert_allclose(fig._get_size(), (8, 8), atol=atol)
    set_config('MNE_BROWSE_RAW_SIZE', old_cfg)


def test_scalings_int(browser_backend):
    """Test that auto scalings access samples using integers."""
    raw = RawArray(np.zeros((1, 500)), create_info(1, 1000., 'eeg'))
    raw.plot(scalings='auto')


@pytest.mark.parametrize('dur, n_dec', [(20, 1), (1.8, 2), (0.01, 4)])
def test_clock_xticks(raw, dur, n_dec, browser_backend):
    """Test if decimal seconds of xticks have appropriate length."""
    fig = raw.plot(duration=dur, time_format='clock')
    fig._redraw()
    tick_texts = fig._get_ticklabels('x')
    assert tick_texts[0].startswith('19:01:53')
    if len(tick_texts[0].split('.')) > 1:
        assert len(tick_texts[0].split('.')[1]) == n_dec


def test_plotting_order_consistency():
    """Test that our internal variables have some consistency."""
    pick_data_set = set(_PICK_TYPES_DATA_DICT)
    pick_data_set.remove('meg')
    pick_data_set.remove('fnirs')
    missing = pick_data_set.difference(set(_DATA_CH_TYPES_ORDER_DEFAULT))
    assert missing == set()


def test_plotting_temperature_gsr(browser_backend):
    """Test that we can plot temperature and GSR."""
    data = np.random.RandomState(0).randn(2, 1000)
    data[0] += 37  # deg C
    # no idea what the scale should be for GSR
    info = create_info(2, 1000., ['temperature', 'gsr'])
    raw = RawArray(data, info)
    fig = raw.plot()
    tick_texts = fig._get_ticklabels('y')
    assert len(tick_texts) == 2


@pytest.mark.pgtest
def test_plotting_memory_garbage_collection(raw, pg_backend):
    """Test that memory can be garbage collected properly."""
    pytest.importorskip('mne_qt_browser', minversion='0.4')
    raw.plot().close()
    import mne_qt_browser
    from mne_qt_browser._pg_figure import MNEQtBrowser
    assert len(mne_qt_browser._browser_instances) == 0
    _assert_no_instances(MNEQtBrowser, 'after closing')
