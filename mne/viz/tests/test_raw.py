# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: Simplified BSD

import numpy as np
import os.path as op
import warnings
import itertools
from distutils.version import LooseVersion

from numpy.testing import assert_raises, assert_allclose

from mne import read_events, pick_types, Annotations
from mne.datasets import testing
from mne.io import read_raw_fif, read_raw_ctf
from mne.utils import requires_version, run_tests_if_main
from mne.viz.utils import _fake_click, _annotation_radio_clicked, _sync_onset
from mne.viz import plot_raw, plot_sensors

# Set our plotters to test mode
import matplotlib
matplotlib.use('Agg')  # for testing don't use X server

warnings.simplefilter('always')  # enable b/c these tests throw warnings

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
    with warnings.catch_warnings(record=True):
        raw.set_channel_types({raw.ch_names[0]: 'ias'})
    raw.pick_channels(raw.ch_names[:9])
    raw.info.normalize_proj()  # Fix projectors after subselection
    return raw


def _get_events():
    """Get events."""
    return read_events(event_name)


def _annotation_helper(raw, events=False):
    """Helper for testing interactive annotations."""
    import matplotlib.pyplot as plt
    # Some of our checks here require modern mpl to work properly
    mpl_good_enough = LooseVersion(matplotlib.__version__) >= '2.0'
    n_anns = 0 if raw.annotations is None else len(raw.annotations.onset)
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
    data_ax = fig.axes[0]
    fig.canvas.key_press_event('a')  # annotation mode
    assert len(plt.get_fignums()) == 2
    assert len(fig.axes[0].texts) == n_anns + n_events
    # modify description
    ann_fig = plt.gcf()
    for key in ' test':
        ann_fig.canvas.key_press_event(key)
    ann_fig.canvas.key_press_event('enter')

    ann_fig = plt.gcf()
    # XXX: _fake_click raises an error on Agg backend
    _annotation_radio_clicked('', ann_fig.radio, data_ax.selector)

    # draw annotation
    _fake_click(fig, data_ax, [1., 1.], xform='data', button=1, kind='press')
    _fake_click(fig, data_ax, [5., 1.], xform='data', button=1, kind='motion')
    _fake_click(fig, data_ax, [5., 1.], xform='data', button=1, kind='release')
    assert len(raw.annotations.onset) == n_anns + 1
    assert len(raw.annotations.duration) == n_anns + 1
    assert len(raw.annotations.description) == n_anns + 1
    assert raw.annotations.description[n_anns] == 'BAD_ test'
    assert len(fig.axes[0].texts) == n_anns + 1 + n_events
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
        assert_allclose(raw.annotations.duration[n_anns], 1.5)
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
    assert len(fig.axes[0].texts) == n_anns + 1 + n_events
    fig.canvas.key_press_event('shift+right')
    assert len(fig.axes[0].texts) == 0
    fig.canvas.key_press_event('shift+left')
    assert len(fig.axes[0].texts) == n_anns + 1 + n_events

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
    assert len(fig.axes[0].texts) == n_anns + 1 + n_events
    # Delete
    _fake_click(fig, data_ax, [1.5, 1.], xform='data', button=3, kind='press')
    fig.canvas.key_press_event('a')  # exit annotation mode
    assert len(raw.annotations.onset) == n_anns
    assert len(fig.axes[0].texts) == n_anns + n_events
    fig.canvas.key_press_event('shift+right')
    assert len(fig.axes[0].texts) == 0
    fig.canvas.key_press_event('shift+left')
    assert len(fig.axes[0].texts) == n_anns + n_events
    plt.close('all')


def test_plot_raw():
    """Test plotting of raw data."""
    import matplotlib.pyplot as plt
    raw = _get_raw()
    raw.info['lowpass'] = 10.  # allow heavy decim during plotting
    events = _get_events()
    plt.close('all')  # ensure all are closed
    with warnings.catch_warnings(record=True):
        fig = raw.plot(events=events, show_options=True, order=[1, 7, 3],
                       group_by='original')
        # test mouse clicks
        x = fig.get_axes()[0].lines[1].get_xdata().mean()
        y = fig.get_axes()[0].lines[1].get_ydata().mean()
        data_ax = fig.axes[0]

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
        for key in ['down', 'up', 'right', 'left', 'o', '-', '+', '=',
                    'pageup', 'pagedown', 'home', 'end', '?', 'f11', 'escape']:
            fig.canvas.key_press_event(key)
        fig = plot_raw(raw, events=events, group_by='selection')
        for key in ['b', 'down', 'up', 'right', 'left', 'o', '-', '+', '=',
                    'pageup', 'pagedown', 'home', 'end', '?', 'f11', 'b',
                    'escape']:
            fig.canvas.key_press_event(key)
        # Color setting
        assert_raises(KeyError, raw.plot, event_color={0: 'r'})
        assert_raises(TypeError, raw.plot, event_color={'foo': 'r'})
        annot = Annotations([10, 10 + raw.first_samp / raw.info['sfreq']],
                            [10, 10], ['test', 'test'], raw.info['meas_date'])
        raw.annotations = annot
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
        # test if meas_date has only one element
        raw.info['meas_date'] = np.array([raw.info['meas_date'][0]],
                                         dtype=np.int32)
        raw.annotations = Annotations([1 + raw.first_samp / raw.info['sfreq']],
                                      [5], ['bad'])
        raw.plot(group_by='position', order=np.arange(8))
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            if hasattr(fig, 'radio'):  # Get access to selection fig.
                break
        for key in ['down', 'up', 'escape']:
            fig.canvas.key_press_event(key)
        plt.close('all')


@testing.requires_testing_data
def test_plot_raw_white():
    """Test plotting whitened raw data."""
    import matplotlib.pyplot as plt
    raw = read_raw_fif(raw_fname).crop(0, 1).load_data()
    raw.plot(noise_cov=cov_fname)
    plt.close('all')


@testing.requires_testing_data
def test_plot_ref_meg():
    """Test plotting ref_meg."""
    import matplotlib.pyplot as plt
    raw_ctf = read_raw_ctf(ctf_fname_continuous).crop(0, 1).load_data()
    raw_ctf.plot()
    plt.close('all')
    assert_raises(ValueError, raw_ctf.plot, group_by='selection')


def test_plot_annotations():
    """Test annotation mode of the plotter."""
    raw = _get_raw()
    raw.info['lowpass'] = 10.
    _annotation_helper(raw)
    _annotation_helper(raw, events=True)

    with warnings.catch_warnings(record=True):  # cut off
        raw.annotations = Annotations([42], [1], 'test', raw.info['meas_date'])
    _annotation_helper(raw)


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
    raw.plot(highpass=1, lowpass=2, butterfly=True)


@requires_version('scipy', '0.12')
def test_plot_raw_psd():
    """Test plotting of raw psds."""
    import matplotlib.pyplot as plt
    raw = _get_raw()
    # normal mode
    raw.plot_psd(average=False)
    # specific mode
    picks = pick_types(raw.info, meg='mag', eeg=False)[:4]
    raw.plot_psd(tmax=np.inf, picks=picks, area_mode='range', average=False,
                 spatial_colors=True)
    raw.plot_psd(tmax=20., color='yellow', dB=False, line_alpha=0.4,
                 n_overlap=0.1, average=False)
    plt.close('all')
    ax = plt.axes()
    # if ax is supplied:
    assert_raises(ValueError, raw.plot_psd, ax=ax, average=True)
    assert_raises(ValueError, raw.plot_psd, average=True, spatial_colors=True)
    raw.plot_psd(tmax=np.inf, picks=picks, ax=ax, average=True)
    plt.close('all')
    ax = plt.axes()
    assert_raises(ValueError, raw.plot_psd, ax=ax, average=True)
    plt.close('all')
    ax = plt.subplots(2)[1]
    raw.plot_psd(tmax=np.inf, ax=ax, average=True)
    plt.close('all')
    # topo psd
    ax = plt.subplot()
    raw.plot_psd_topo(axes=ax)
    plt.close('all')
    # with channel information not available
    for idx in range(len(raw.info['chs'])):
        raw.info['chs'][idx]['loc'] = np.zeros(12)
    with warnings.catch_warnings(record=True):  # missing channel locations
        raw.plot_psd(spatial_colors=True, average=False)
    # with a flat channel
    raw[5, :] = 0
    with warnings.catch_warnings(record=True) as w:
        for dB, estimate in itertools.product((True, False),
                                              ('power', 'amplitude')):
            raw.plot_psd(average=True, dB=dB, estimate=estimate)
    assert len(w) == 4
    # test reject_by_annotation
    raw = _get_raw()
    raw.annotations = Annotations([1, 5], [3, 3], ['test', 'test'])
    raw.plot_psd(reject_by_annotation=True)
    raw.plot_psd(reject_by_annotation=False)

    # gh-5046
    raw = read_raw_fif(raw_fname, preload=True).crop(0, 1)
    picks = pick_types(raw.info)
    raw.plot_psd(picks=picks, average=False)
    raw.plot_psd(picks=picks, average=True)
    plt.close('all')


def test_plot_sensors():
    """Test plotting of sensor array."""
    import matplotlib.pyplot as plt
    raw = _get_raw()
    fig = raw.plot_sensors('3d')
    _fake_click(fig, fig.gca(), (-0.08, 0.67))
    raw.plot_sensors('topomap', ch_type='mag',
                     show_names=['MEG 0111', 'MEG 0131'])
    plt.close('all')
    ax = plt.subplot(111)
    raw.plot_sensors(ch_groups='position', axes=ax)
    raw.plot_sensors(ch_groups='selection', to_sphere=False)
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
    assert len(fig.lasso.selection) == 0

    # Lasso with 1 sensor
    _fake_click(fig, ax, (-0.5, 0.5), xform='data')
    plt.draw()
    _fake_click(fig, ax, (0., 0.5), xform='data', kind='motion')
    _fake_click(fig, ax, (0., 0.), xform='data', kind='motion')
    fig.canvas.key_press_event('control')
    _fake_click(fig, ax, (-0.5, 0.), xform='data', kind='release')
    assert len(fig.lasso.selection) == 1

    _fake_click(fig, ax, (-0.09, -0.43), xform='data')  # single selection
    assert len(fig.lasso.selection) == 2
    _fake_click(fig, ax, (-0.09, -0.43), xform='data')  # deselect
    assert len(fig.lasso.selection) == 1
    plt.close('all')


run_tests_if_main()
