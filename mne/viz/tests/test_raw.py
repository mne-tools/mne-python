# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: Simplified BSD

import numpy as np
import os.path as op
import itertools
from distutils.version import LooseVersion

from numpy.testing import assert_allclose
import pytest
import matplotlib
import matplotlib.pyplot as plt

from mne import read_events, pick_types, Annotations, create_info
from mne.datasets import testing
from mne.io import read_raw_fif, read_raw_ctf, RawArray
from mne.utils import run_tests_if_main, _dt_to_stamp
from mne.viz.utils import _fake_click, _annotation_radio_clicked, _sync_onset
from mne.viz import plot_raw, plot_sensors

ctf_dir = op.join(testing.data_path(download=False), 'CTF')
ctf_fname_continuous = op.join(ctf_dir, 'testdata_ctf.ds')

base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')
event_name = op.join(base_dir, 'test-eve.fif')
cov_fname = op.join(base_dir, 'test-cov.fif')


def _get_raw():
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
    mpl_good_enough = LooseVersion(matplotlib.__version__) >= '2.0'
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
    # modify description
    ann_fig = fig.mne.fig_annotation
    for key in ' test':
        ann_fig.canvas.key_press_event(key)
    ann_fig.canvas.key_press_event('enter')

    # XXX: _fake_click raises an error on Agg backend
    _annotation_radio_clicked('', ann_fig.mne.radio_ax.buttons, data_ax.selector)

    # draw annotation
    fig.canvas.key_press_event('p')  # use snap mode
    _fake_click(fig, data_ax, [1., 1.], xform='data', button=1, kind='press')
    _fake_click(fig, data_ax, [5., 1.], xform='data', button=1, kind='motion')
    _fake_click(fig, data_ax, [5., 1.], xform='data', button=1, kind='release')
    assert len(raw.annotations.onset) == n_anns + 1
    assert len(raw.annotations.duration) == n_anns + 1
    assert len(raw.annotations.description) == n_anns + 1
    assert raw.annotations.description[n_anns] == 'BAD_ test'
    assert len(data_ax.texts) == n_anns + 1 + n_events + n_scale
    onset = raw.annotations.onset[n_anns]
    want_onset = _sync_onset(raw, 1., inverse=True)
    assert_allclose(onset, want_onset)
    assert_allclose(raw.annotations.duration[n_anns], 4.)
    # hover event
    _fake_click(fig, data_ax, [4.5, 1.], xform='data', button=None,
                kind='motion')
    _fake_click(fig, data_ax, [4.7, 1.], xform='data', button=None,
                kind='motion')
    # modify annotation from end
    _fake_click(fig, data_ax, [5., 1.], xform='data', button=1, kind='press')
    _fake_click(fig, data_ax, [2.5, 1.], xform='data', button=1, kind='motion')
    _fake_click(fig, data_ax, [2.5, 1.], xform='data', button=1,
                kind='release')
    if mpl_good_enough:
        assert raw.annotations.onset[n_anns] == onset
        assert_allclose(raw.annotations.duration[n_anns], 1.5)  # 4->1.5
    # modify annotation from beginning
    _fake_click(fig, data_ax, [1., 1.], xform='data', button=1, kind='press')
    _fake_click(fig, data_ax, [0.5, 1.], xform='data', button=1, kind='motion')
    _fake_click(fig, data_ax, [0.5, 1.], xform='data', button=1,
                kind='release')
    if mpl_good_enough:
        assert_allclose(raw.annotations.onset[n_anns], onset - 0.5, atol=1e-10)
        assert_allclose(raw.annotations.duration[n_anns], 2.0)
    assert len(raw.annotations.onset) == n_anns + 1
    assert len(raw.annotations.duration) == n_anns + 1
    assert len(raw.annotations.description) == n_anns + 1
    assert raw.annotations.description[n_anns] == 'BAD_ test'
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
    if mpl_good_enough:
        assert_allclose(raw.annotations.onset[n_anns], onset - 0.5, atol=1e-10)
        assert_allclose(raw.annotations.duration[n_anns], 5.0)
    assert len(fig.axes[0].texts) == n_anns + 1 + n_events + n_scale
    # Delete
    _fake_click(fig, data_ax, [1.5, 1.], xform='data', button=3,
                kind='press')
    fig.canvas.key_press_event('a')  # exit annotation mode
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
    wants = ('2000.0 fT', '800.0 fT/cm', '40.0 µV')
    assert texts == wants
    assert len(ax.lines) == 7  # 1 green vline, 3 data, 3 scalebars
    for data, bar in zip(fig.mne.traces, fig.mne.scalebars.values()):
        y = data.get_ydata()
        y_lims = [y.min(), y.max()]
        bar_lims = bar.get_ydata()
        assert_allclose(y_lims, bar_lims, atol=1e-4)
    plt.close('all')


def test_plot_raw_traces():
    """Test plotting of raw data."""
    raw = _get_raw()
    raw.info['lowpass'] = 10.  # allow heavy decim during plotting
    events = _get_events()
    plt.close('all')  # ensure all are closed
    assert len(plt.get_fignums()) == 0
    fig = raw.plot(events=events, order=[1, 7, 5, 2, 3], n_channels=3,
                   group_by='original')
    assert len(plt.get_fignums()) == 1

    # make sure fig.mne param object is present
    assert hasattr(fig, 'mne')

    # test mouse clicks
    x = fig.get_axes()[0].lines[1].get_xdata().mean()
    y = fig.get_axes()[0].lines[1].get_ydata().mean()
    data_ax = fig.mne.ax_main
    hscroll = fig.mne.ax_hscroll
    vscroll = fig.mne.ax_vscroll
    assert len(fig.axes) == 5

    _fake_click(fig, data_ax, [x, y], xform='data')  # mark a bad channel
    _fake_click(fig, data_ax, [x, y], xform='data')  # unmark a bad channel
    _fake_click(fig, data_ax, [0.5, 0.999])  # click elsewhere in 1st axes
    _fake_click(fig, data_ax, [-0.1, 0.9])  # click on y-label
    _fake_click(fig, hscroll, [0.5, 0.5])  # change time
    labels = [label.get_text() for label in data_ax.get_yticklabels()]
    assert labels == [raw.ch_names[1], raw.ch_names[7], raw.ch_names[5]]
    _fake_click(fig, vscroll, [0.5, 0.01])  # change channels to end
    labels = [label.get_text() for label in data_ax.get_yticklabels()]
    assert labels == [raw.ch_names[5], raw.ch_names[2], raw.ch_names[3]]
    _fake_click(fig, vscroll, [0.5, 0.5])  # change channels to mid
    labels = [label.get_text() for label in data_ax.get_yticklabels()]
    assert labels == [raw.ch_names[7], raw.ch_names[5], raw.ch_names[2]]
    assert len(plt.get_fignums()) == 1
    # open SSP window
    _fake_click(fig, fig.mne.ax_proj, [0.5, 0.5])
    _fake_click(fig, fig.mne.ax_proj, [0.5, 0.5], kind='release')
    assert len(plt.get_fignums()) == 2
    ssp_fig = fig.mne.fig_proj
    fig.canvas.button_press_event(1, 1, 1)  # outside any axes
    fig.canvas.scroll_event(0.5, 0.5, -0.5)  # scroll down
    fig.canvas.scroll_event(0.5, 0.5, 0.5)  # scroll up

    ax = ssp_fig.mne.proj_checkboxes.ax
    assert _proj_status(ax) == [True] * 3
    t = [c for c in ax.get_children() if isinstance(c, matplotlib.text.Text)]
    pos = np.array(t[0].get_position()) + 0.01
    _fake_click(ssp_fig, ssp_fig.get_axes()[0], pos, xform='data')  # off
    assert _proj_status(ax) == [False, True, True]
    _fake_click(ssp_fig, ssp_fig.get_axes()[0], pos, xform='data')  # on
    assert _proj_status(ax) == [True] * 3
    _fake_click(ssp_fig, ssp_fig.get_axes()[1], [0.5, 0.5])  # all off
    _fake_click(ssp_fig, ssp_fig.get_axes()[1], [0.5, 0.5], kind='release')
    assert _proj_status(ax) == [False] * 3
    assert fig.mne.projector is None  # actually off
    _fake_click(ssp_fig, ssp_fig.get_axes()[1], [0.5, 0.5])  # all on
    _fake_click(ssp_fig, ssp_fig.get_axes()[1], [0.5, 0.5], kind='release')
    assert fig.mne.projector is not None  # on
    assert _proj_status(ax) == [True] * 3

    # test keypresses
    # test for group_by='original'
    for key in ['down', 'up', 'right', 'left', 'o', '-', '+', '=', 'd', 'd',
                'pageup', 'pagedown', 'home', 'end', '?', 'f11', 'z',
                'escape']:
        fig.canvas.key_press_event(key)

    # test for group_by='selection'
    fig = plot_raw(raw, events=events, group_by='selection')
    for key in ['b', 'down', 'up', 'right', 'left', 'o', '-', '+', '=', 'd',
                'd', 'pageup', 'pagedown', 'home', 'end', '?', 'f11', 'b', 'z',
                'escape']:
        fig.canvas.key_press_event(key)

    # test zen mode
    fig = plot_raw(raw, show_scrollbars=False)

    # Color setting
    pytest.raises(KeyError, raw.plot, event_color={0: 'r'})
    pytest.raises(TypeError, raw.plot, event_color={'foo': 'r'})
    annot = Annotations([10, 10 + raw.first_samp / raw.info['sfreq']],
                        [10, 10], ['test', 'test'], raw.info['meas_date'])
    with pytest.warns(RuntimeWarning, match='outside data range'):
        raw.set_annotations(annot)
    fig = plot_raw(raw, events=events, event_color={-1: 'r', 998: 'b'})
    plt.close('all')
    for group_by, order in zip(['position', 'selection'],
                               [np.arange(len(raw.ch_names))[::-3],
                                [1, 2, 4, 6]]):
        with pytest.warns(None):  # sometimes projection
            fig = raw.plot(group_by=group_by, order=order)
        x = fig.get_axes()[0].lines[1].get_xdata()[10]
        y = fig.get_axes()[0].lines[1].get_ydata()[10]
        with pytest.warns(None):  # old mpl (at least 2.0) can warn
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
    raw.plot(noise_cov=cov_fname)
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


def test_plot_annotations():
    """Test annotation mode of the plotter."""
    raw = _get_raw()
    raw.info['lowpass'] = 10.
    _annotation_helper(raw)
    _annotation_helper(raw, events=True)

    annot = Annotations([42], [1], 'test', raw.info['meas_date'])
    with pytest.warns(RuntimeWarning, match='expanding outside'):
        raw.set_annotations(annot)
    _annotation_helper(raw)
    plt.close('all')


@pytest.mark.parametrize('filtorder', (0, 2))  # FIR, IIR
def test_plot_raw_filtered(filtorder):
    """Test filtering of raw plots."""
    raw = _get_raw()
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


def test_plot_raw_psd():
    """Test plotting of raw psds."""
    raw = _get_raw()
    # normal mode
    raw.plot_psd(average=False)
    # specific mode
    picks = pick_types(raw.info, meg='mag', eeg=False)[:4]
    raw.plot_psd(tmax=None, picks=picks, area_mode='range', average=False,
                 spatial_colors=True)
    raw.plot_psd(tmax=20., color='yellow', dB=False, line_alpha=0.4,
                 n_overlap=0.1, average=False)
    plt.close('all')
    ax = plt.axes()
    # if ax is supplied:
    pytest.raises(ValueError, raw.plot_psd, ax=ax, average=True)
    raw.plot_psd(tmax=None, picks=picks, ax=ax, average=True)
    plt.close('all')
    ax = plt.axes()
    with pytest.raises(ValueError, match='2 axes must be supplied, got 1'):
        raw.plot_psd(ax=ax, average=True)
    plt.close('all')
    ax = plt.subplots(2)[1]
    raw.plot_psd(tmax=None, ax=ax, average=True)
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
        ylabel = fig.axes[1].get_ylabel()
        ends_dB = ylabel.endswith('mathrm{(dB)}$')
        if dB:
            assert ends_dB, ylabel
        else:
            assert not ends_dB, ylabel
        if estimate == 'amplitude':
            assert r'fT/cm/\sqrt{Hz}' in ylabel, ylabel
        else:
            assert estimate == 'power'
            assert '(fT/cm)²/Hz' in ylabel, ylabel
        ylabel = fig.axes[0].get_ylabel()
        if estimate == 'amplitude':
            assert r'fT/\sqrt{Hz}' in ylabel
        else:
            assert 'fT²/Hz' in ylabel
    # test reject_by_annotation
    raw = _get_raw()
    raw.set_annotations(Annotations([1, 5], [3, 3], ['test', 'test']))
    raw.plot_psd(reject_by_annotation=True)
    raw.plot_psd(reject_by_annotation=False)
    plt.close('all')

    # test fmax value checking
    with pytest.raises(ValueError, match='not exceed one half the sampling'):
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


def test_plot_sensors():
    """Test plotting of sensor array."""
    raw = _get_raw()
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
    plt.draw()
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
