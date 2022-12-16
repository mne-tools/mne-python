# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Robert Luke <mail@robertluke.net>
#
# License: Simplified BSD

import os.path as op
from functools import partial

import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_almost_equal
import pytest
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from mne import (read_evokeds, read_proj, make_fixed_length_events, Epochs,
                 compute_proj_evoked, find_layout, pick_types, create_info,
                 read_cov, EvokedArray)
from mne.io.proj import make_eeg_average_ref_proj, Projection
from mne.io import read_raw_fif, read_info, RawArray
from mne.io.constants import FIFF
from mne.io.pick import pick_info, channel_indices_by_type, _picks_to_idx
from mne.io.compensator import get_current_comp
from mne.channels import (read_layout, make_dig_montage, make_standard_montage,
                          find_ch_adjacency)
from mne.datasets import testing
from mne.preprocessing import compute_bridged_electrodes
from mne.time_frequency.tfr import AverageTFR

from mne.viz import plot_evoked_topomap, plot_projs_topomap, topomap
from mne.viz.topomap import (_get_pos_outlines, _onselect, plot_topomap,
                             plot_arrowmap, plot_psds_topomap,
                             plot_bridged_electrodes, plot_ch_adjacency)
from mne.viz.utils import (_find_peaks, _fake_click, _fake_keypress,
                           _fake_scroll)
from mne.utils import requires_sklearn, check_version

data_dir = testing.data_path(download=False)
subjects_dir = op.join(data_dir, 'subjects')
ecg_fname = op.join(data_dir, 'MEG', 'sample', 'sample_audvis_ecg-proj.fif')
triux_fname = op.join(data_dir, 'SSS', 'TRIUX', 'triux_bmlhus_erm_raw.fif')

base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
evoked_fname = op.join(base_dir, 'test-ave.fif')
raw_fname = op.join(base_dir, 'test_raw.fif')
event_name = op.join(base_dir, 'test-eve.fif')
ctf_fname = op.join(base_dir, 'test_ctf_comp_raw.fif')
layout = read_layout('Vectorview-all')
cov_fname = op.join(base_dir, 'test-cov.fif')


@pytest.mark.parametrize('constrained_layout', (False, True))
def test_plot_topomap_interactive(constrained_layout):
    """Test interactive topomap projection plotting."""
    evoked = read_evokeds(evoked_fname, baseline=(None, 0))[0]
    evoked.pick_types(meg='mag')
    with evoked.info._unlock():
        evoked.info['projs'] = []
    assert not evoked.proj
    evoked.add_proj(compute_proj_evoked(evoked, n_mag=1))

    plt.close('all')
    fig, ax = plt.subplots(constrained_layout=constrained_layout)
    canvas = fig.canvas

    kwargs = dict(vlim=(-240, 240), times=[0.1], colorbar=False, axes=ax,
                  res=8, time_unit='s')
    evoked.copy().plot_topomap(proj=False, **kwargs)
    canvas.draw()
    image_noproj = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    assert len(plt.get_fignums()) == 1

    ax.clear()
    evoked.copy().plot_topomap(proj=True, **kwargs)
    canvas.draw()
    image_proj = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    assert not np.array_equal(image_noproj, image_proj)
    assert len(plt.get_fignums()) == 1

    ax.clear()
    evoked.copy().plot_topomap(proj='interactive', **kwargs)
    canvas.draw()
    image_interactive = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    assert_array_equal(image_noproj, image_interactive)
    assert not np.array_equal(image_proj, image_interactive)
    assert len(plt.get_fignums()) == 2

    proj_fig = plt.figure(plt.get_fignums()[-1])
    assert len(proj_fig.axes[0].lines) == 2
    for line in proj_fig.axes[0].lines:
        assert not line.get_visible()
    _fake_click(proj_fig, proj_fig.axes[0], [0.5, 0.5], xform='data')
    assert len(proj_fig.axes[0].lines) == 2
    proj_fig.canvas.draw_idle()
    for li, line in enumerate(proj_fig.axes[0].lines):
        assert line.get_visible(), f'line {li} not visible'
    canvas.draw()
    image_interactive_click = np.frombuffer(
        canvas.tostring_rgb(), dtype='uint8')
    corr = np.corrcoef(
        image_proj.ravel(), image_interactive_click.ravel())[0, 1]
    assert 0.99 < corr <= 1
    corr = np.corrcoef(
        image_noproj.ravel(), image_interactive_click.ravel())[0, 1]
    assert 0.85 < corr < 0.9

    _fake_click(proj_fig, proj_fig.axes[0], [0.5, 0.5], xform='data')
    canvas.draw()
    image_interactive_click = np.frombuffer(
        canvas.tostring_rgb(), dtype='uint8')
    corr = np.corrcoef(
        image_noproj.ravel(), image_interactive_click.ravel())[0, 1]
    assert 0.99 < corr <= 1
    corr = np.corrcoef(
        image_proj.ravel(), image_interactive_click.ravel())[0, 1]
    assert 0.85 < corr < 0.9


@testing.requires_testing_data
def test_plot_projs_topomap():
    """Test plot_projs_topomap."""
    projs = read_proj(ecg_fname)
    info = read_info(raw_fname)
    fast_test = {"res": 8, "contours": 0, "sensors": False}
    plot_projs_topomap(projs, info=info, colorbar=True, **fast_test)
    plt.close('all')
    ax = plt.subplot(111)
    projs[3].plot_topomap(info)
    plot_projs_topomap(projs[:1], info, axes=ax, **fast_test)  # test axes
    plt.close('all')
    triux_info = read_info(triux_fname)
    plot_projs_topomap(triux_info['projs'][-1:], triux_info, **fast_test)
    plt.close('all')
    plot_projs_topomap(triux_info['projs'][:1], triux_info, **fast_test)
    plt.close('all')
    eeg_avg = make_eeg_average_ref_proj(info)
    eeg_avg.plot_topomap(info, **fast_test)
    plt.close('all')
    # test vlims
    for vlim in ('joint', (-1, 1), (None, 0.5), (0.5, None), (None, None)):
        plot_projs_topomap(projs[:-1], info, vlim=vlim, colorbar=True)
    plt.close('all')

    eeg_proj = make_eeg_average_ref_proj(info)
    info_meg = pick_info(info, pick_types(info, meg=True, eeg=False))
    with pytest.raises(ValueError, match='No channel names in info match p'):
        plot_projs_topomap([eeg_proj], info_meg)


def test_plot_topomap_animation(capsys):
    """Test topomap plotting."""
    # evoked
    evoked = read_evokeds(evoked_fname, 'Left Auditory',
                          baseline=(None, 0))

    # Test animation
    _, anim = evoked.animate_topomap(ch_type='grad', times=[0, 0.1],
                                     butterfly=False, time_unit='s',
                                     verbose='debug')
    anim._func(1)  # _animate has to be tested separately on 'Agg' backend.
    out, _ = capsys.readouterr()
    assert 'extrapolation mode local to 0' in out
    plt.close('all')


@pytest.mark.filterwarnings('ignore:.*No contour levels.*:UserWarning')
def test_plot_topomap_animation_nirs(fnirs_evoked, capsys):
    """Test topomap plotting for nirs data."""
    fig, anim = fnirs_evoked.animate_topomap(ch_type='hbo', verbose='debug')
    anim._func(1)  # _animate has to be tested separately on 'Agg' backend.
    out, _ = capsys.readouterr()
    assert 'extrapolation mode head to 0' in out
    assert len(fig.axes) == 2
    plt.close('all')


def test_plot_evoked_topomap_errors(evoked, monkeypatch):
    """Test error handling for evoked topomap plots."""
    # simplify data and set some params to make the test really fast
    evoked.pick(['EEG 001', 'EEG 002'])
    fast_func = partial(evoked.plot_topomap, res=8, contours=0, sensors=False)
    fast_func_onetime = partial(fast_func, times=0.1)
    # wrong channel type
    with pytest.raises(ValueError, match="No channels of type 'mag'"):
        fast_func(ch_type='mag')
    # bad times
    with pytest.raises(ValueError, match='Times should be between 0.0 and'):
        fast_func(times=[-100])
    with pytest.raises(ValueError, match='times must be 1D, got 2 dimensions'):
        fast_func(times=[[0]])
    # times / average mismatch
    with pytest.raises(ValueError, match='3 time points.*2 periods for aver'):
        fast_func([0.05, 0.1, 0.15], ch_type='eeg', average=[0.01, 0.02])
    # average
    with pytest.raises(ValueError, match='number of seconds.* got -1000.0'):
        fast_func_onetime(average=-1e3)
    with pytest.raises(TypeError, match='number of seconds.* got type:'):
        fast_func_onetime(average='x')
    # image_interp
    with pytest.raises(RuntimeError, match='`image_interp` must be'):
        fast_func_onetime(image_interp='bilinear')
    # border
    with pytest.raises(TypeError, match='be an instance of numeric or str'):
        fast_func_onetime(extrapolate='head', border=[1, 2, 3])
    with pytest.raises(ValueError, match="allowed value.*'mean'.*got 'fancy'"):
        fast_func_onetime(extrapolate='head', border='fancy')
    # projs
    with pytest.raises(RuntimeError, match='Projs are already applied.'):
        fast_func_onetime(proj='interactive')
    # too many subplots
    with monkeypatch.context() as m:  # speed it up by not actually plotting
        m.setattr(topomap, '_plot_topomap',
                  lambda *args, **kwargs: (None, None, None))
        with pytest.warns(RuntimeWarning, match='More than 25 topomaps plots'):
            fast_func([0.1] * 26, colorbar=False)
    # missing channel locations
    with evoked.info._unlock():
        for ch in evoked.info['chs']:
            ch['loc'][:3] = 0.
    with pytest.raises(ValueError, match="points.*doesn't match.*channels."):
        evoked.plot_topomap()
    with evoked.info._unlock():
        evoked.info['dig'] = None
    with pytest.raises(RuntimeError, match='No digitization points found.'):
        evoked.plot_topomap()


@pytest.mark.parametrize('units, scalings, expected_unit', [
    (None, None, 'µV'),
    ('foo', None, 'foo'),
    (None, 7., 'AU'),  # non-default scaling → "AU"
])
def test_plot_evoked_topomap_units(evoked, units, scalings, expected_unit):
    """Test that colorbar units respect scalings correctly."""
    evoked.pick(['EEG 001', 'EEG 002', 'EEG 003'])
    fig = evoked.plot_topomap(times=0.1, res=8, contours=0, sensors=False,
                              units=units, scalings=scalings)
    # ideally we'd do this:
    #     cbar = [ax for ax in fig.axes if hasattr(ax, '_colorbar')]
    #     assert len(cbar) == 1
    #     cbar = cbar[0]
    #     assert cbar.get_title() == expected_unit
    # ...but not all matplotlib versions support it, and we can't use
    # @requires_version because it's hard figure out exactly which MPL version
    # is the cutoff since it relies on a private attribute. So for now we just
    # do this:
    for ax in fig.axes:
        if hasattr(ax, '_colorbar'):
            assert ax.get_title() == expected_unit


@pytest.mark.parametrize('extrapolate', ('box', 'local', 'head'))
def test_plot_evoked_topomap_extrapolation(evoked, extrapolate):
    """Test topomap extrapolation options."""
    evoked.pick(['EEG 001', 'EEG 002', 'EEG 003'])
    evoked.plot_topomap(times=0.1, extrapolate=extrapolate, res=8, contours=0,
                        sensors=False)


def test_plot_evoked_topomap_border():
    """Test topomap extrapolation border values."""
    # make some fake sensor locations: 25 sensors at distances of 0.2 to 1.0
    # in steps of 0.2 in the ±x, ±y, and +z directions
    ch_pos = np.array([[[r, 0, 0],   # +x
                        [-r, 0, 0],  # -x
                        [0, r, 0],   # +y
                        [0, -r, 0],  # -y
                        [0, 0, r],   # +z
                        ] for r in np.linspace(0.2, 1, 5)]).reshape(-1, 3)
    info = create_info(len(ch_pos), 250, 'eeg')
    ch_pos_dict = {name: pos for name, pos in zip(info['ch_names'], ch_pos)}
    dig = make_dig_montage(ch_pos_dict, coord_frame='head')
    info.set_montage(dig)
    # simulate data
    data = np.full(len(ch_pos), 5)
    kwargs = dict(res=15, extrapolate='head', sphere=1, sensors=False)
    idx = kwargs['res'] // 2

    # when border=0...
    img, _ = plot_topomap(data, info, border=0, **kwargs)
    img_data = img.get_array().data
    # middle pixel should exactly equal sensor data:
    assert_equal(img_data[idx, idx], data[0])
    # corner pixel should be close(ish) to zero:
    assert img_data[0, 0] < 1.5

    # when border='mean'...
    img, _ = plot_topomap(data, info, border='mean', **kwargs)
    img_data = img.get_array().data
    # middle pixel should exactly equal sensor data:
    assert_equal(img_data[idx, idx], data[0])
    # and corner pixel should *also* be very close to sensor data:
    assert_almost_equal(img_data[idx, idx], data[0], decimal=9)


@pytest.mark.slowtest
def test_plot_topomap_basic():
    """Test basics of topomap plotting."""
    evoked = read_evokeds(evoked_fname, 'Left Auditory',
                          baseline=(None, 0))
    res = 8
    fast_test = dict(res=res, contours=0, sensors=False, time_unit='s')
    fast_test_noscale = dict(res=res, contours=0, sensors=False)
    ev_bad = evoked.copy().pick_types(meg=False, eeg=True)
    ev_bad.pick_channels(ev_bad.ch_names[:2])
    plt_topomap = partial(ev_bad.plot_topomap, **fast_test)
    plt_topomap(times=ev_bad.times[:2] - 1e-6)  # auto, plots EEG
    evoked.plot_topomap([0.1], ch_type='eeg', scalings=1, res=res,
                        contours=[-100, 0, 100], time_unit='ms')

    # test channel placement when only 'grad' are picked:
    # ---------------------------------------------------
    info_grad = evoked.copy().pick('grad').info
    n_grads = len(info_grad['ch_names'])
    data = np.random.randn(n_grads)
    img, _ = plot_topomap(data, info_grad)

    # check that channels are scattered around x == 0
    pos = img.axes.collections[-1].get_offsets()
    prop_channels_on_the_right = (pos[:, 0] > 0).mean()
    assert prop_channels_on_the_right < 0.6

    # other:
    # ------
    plt_topomap = partial(evoked.plot_topomap, **fast_test)
    plt.close('all')
    axes = [plt.subplot(221), plt.subplot(222)]
    plt_topomap(axes=axes, colorbar=False)
    plt.close('all')
    plt_topomap(times=[-0.1, 0.2])
    plt.close('all')
    evoked_grad = evoked.copy().crop(0, 0).pick_types(meg='grad')
    mask = np.zeros((204, 1), bool)
    mask[[0, 3, 5, 6]] = True
    names = []

    def proc_names(x):
        names.append(x)
        return x[4:]

    evoked_grad.plot_topomap(ch_type='grad', times=[0], mask=mask,
                             show_names=proc_names, **fast_test)
    want_names = np.array(evoked_grad.ch_names)[mask.squeeze()].tolist()
    assert_equal([f'{name[:-1]}x' for name in want_names],
                 ['MEG 011x', 'MEG 012x', 'MEG 013x', 'MEG 014x'])
    mask = np.zeros_like(evoked.data, dtype=bool)
    mask[[1, 5], :] = True
    plt_topomap(ch_type='mag', outlines=None)
    times = [0.1]
    plt_topomap(times, ch_type='grad', mask=mask)
    plt_topomap(times, ch_type='planar1')
    plt_topomap(times, ch_type='planar2')
    plt_topomap(times, ch_type='grad', mask=mask, show_names=True,
                mask_params={'marker': 'x'})
    plt.close('all')

    p = plt_topomap(times, ch_type='grad', image_interp='cubic',
                    show_names=lambda x: x.replace('MEG', ''))
    subplot = [
        x for x in p.get_children()
        if any(t in str(type(x)) for t in ('Axes', 'Subplot'))]
    assert len(subplot) >= 1, [type(x) for x in p.get_children()]
    subplot = subplot[0]

    have_all = all('MEG' not in x.get_text()
                   for x in subplot.get_children()
                   if isinstance(x, matplotlib.text.Text))
    assert have_all

    # Plot array
    for ch_type in ('mag', 'grad'):
        evoked_ = evoked.copy().pick_types(eeg=False, meg=ch_type)
        plot_topomap(evoked_.data[:, 0], evoked_.info, **fast_test_noscale)
    # fail with multiple channel types
    pytest.raises(ValueError, plot_topomap, evoked.data[0, :], evoked.info)

    # Test title
    def get_texts(p):
        return [x.get_text() for x in p.get_children() if
                isinstance(x, matplotlib.text.Text)]

    p = plt_topomap(times, ch_type='eeg', average=0.01)
    assert_equal(len(get_texts(p)), 0)
    plt.close('all')

    # Test averaging with a scalar input
    averaging_times = [ev_bad.times[0], times[0], ev_bad.times[-1]]
    p = plt_topomap(averaging_times, ch_type='eeg', average=0.01)

    expected_ax_titles = (
        '-0.200 – -0.195 s',  # clipped on the left
        '0.095 – 0.105 s',    # full range
        '0.494 – 0.499 s'     # clipped on the right
    )
    for idx, expected_title in enumerate(expected_ax_titles):
        assert p.axes[idx].get_title() == expected_title

    # Test averaging with an array-like input
    averaging_durations = [0.01, 0.02, None]
    p = plt_topomap(
        averaging_times, ch_type='eeg', average=averaging_durations
    )
    expected_ax_titles = (
        '-0.200 – -0.195 s',  # clipped on the left
        '0.090 – 0.110 s',    # full range
        '0.499 s'             # No averaging
    )
    for idx, expected_title in enumerate(expected_ax_titles):
        assert p.axes[idx].get_title() == expected_title

    del averaging_times, expected_ax_titles, expected_title

    # delaunay triangulation warning
    plt_topomap(times, ch_type='mag')

    # change to no-proj mode
    evoked = read_evokeds(evoked_fname, 'Left Auditory',
                          baseline=(None, 0), proj=False)
    fig1 = evoked.plot_topomap('interactive', ch_type='mag',
                               proj='interactive', **fast_test)
    _fake_click(fig1, fig1.axes[1], (0.5, 0.5))  # click slider
    data_max = np.max(fig1.axes[0].images[0]._A)
    fig2 = plt.gcf()
    _fake_click(fig2, fig2.axes[0], (0.075, 0.775))  # toggle projector
    # make sure projector gets toggled
    assert (np.max(fig1.axes[0].images[0]._A) != data_max)

    for ch in evoked.info['chs']:
        if ch['coil_type'] == FIFF.FIFFV_COIL_EEG:
            ch['loc'].fill(0)

    # Remove extra digitization point, so EEG digitization points
    # correspond with the EEG electrodes
    del evoked.info['dig'][85]

    # Pass custom outlines without patch
    eeg_picks = pick_types(evoked.info, meg=False, eeg=True)
    pos, outlines = _get_pos_outlines(evoked.info, eeg_picks, 0.1)
    evoked.plot_topomap(times, ch_type='eeg', outlines=outlines, **fast_test)
    plt.close('all')

    # Test interactive cmap
    fig = plot_evoked_topomap(evoked, times=[0., 0.1], ch_type='eeg',
                              cmap=('Reds', True), **fast_test)
    _fake_keypress(fig, 'up')
    _fake_keypress(fig, ' ')
    _fake_keypress(fig, 'down')
    cbar = fig.get_axes()[0].CB  # Fake dragging with mouse.
    ax = cbar.cbar.ax
    _fake_click(fig, ax, (0.1, 0.1))
    _fake_click(fig, ax, (0.1, 0.2), kind='motion')
    _fake_click(fig, ax, (0.1, 0.3), kind='release')

    _fake_click(fig, ax, (0.1, 0.1), button=3)
    _fake_click(fig, ax, (0.1, 0.2), button=3, kind='motion')
    _fake_click(fig, ax, (0.1, 0.3), kind='release')

    _fake_scroll(fig, 0.5, 0.5, -0.5)  # scroll down
    _fake_scroll(fig, 0.5, 0.5, 0.5)  # scroll up

    plt.close('all')

    # Pass custom outlines with patch callable
    def patch():
        return Circle((0.5, 0.4687), radius=.46,
                      clip_on=True, transform=plt.gca().transAxes)
    outlines['patch'] = patch
    plot_evoked_topomap(evoked, times, ch_type='eeg', outlines=outlines,
                        **fast_test)

    # Test error messages for invalid pos parameter
    n_channels = len(pos)
    data = np.ones(n_channels)
    pos_1d = np.zeros(n_channels)
    pos_3d = np.zeros((n_channels, 2, 2))
    pytest.raises(ValueError, plot_topomap, data, pos_1d)
    pytest.raises(ValueError, plot_topomap, data, pos_3d)
    pytest.raises(ValueError, plot_topomap, data, pos[:3, :])

    pos_x = pos[:, :1]
    pos_xyz = np.c_[pos, np.zeros(n_channels)[:, np.newaxis]]
    pytest.raises(ValueError, plot_topomap, data, pos_x)
    pytest.raises(ValueError, plot_topomap, data, pos_xyz)

    # An #channels x 4 matrix should work though. In this case (x, y, width,
    # height) is assumed.
    pos_xywh = np.c_[pos, np.zeros((n_channels, 2))]
    plot_topomap(data, pos_xywh)
    plt.close('all')

    # Test peak finder
    axes = [plt.subplot(131), plt.subplot(132)]
    evoked.plot_topomap(times='peaks', axes=axes, **fast_test)
    plt.close('all')
    evoked.data = np.zeros(evoked.data.shape)
    evoked.data[50][1] = 1
    assert_array_equal(_find_peaks(evoked, 10), evoked.times[1])
    evoked.data[80][100] = 1
    assert_array_equal(_find_peaks(evoked, 10), evoked.times[[1, 100]])
    evoked.data[2][95] = 2
    assert_array_equal(_find_peaks(evoked, 10), evoked.times[[1, 95]])
    assert_array_equal(_find_peaks(evoked, 1), evoked.times[95])

    # Test excluding bads channels
    evoked_grad.info['bads'] += [evoked_grad.info['ch_names'][0]]
    orig_bads = evoked_grad.info['bads']
    evoked_grad.plot_topomap(ch_type='grad', times=[0], time_unit='ms')
    assert_array_equal(evoked_grad.info['bads'], orig_bads)
    plt.close('all')


def test_plot_tfr_topomap():
    """Test plotting of TFR data."""
    raw = read_raw_fif(raw_fname)
    times = np.linspace(-0.1, 0.1, 200)
    res = 8
    n_freqs = 3
    nave = 1
    rng = np.random.RandomState(42)
    picks = [93, 94, 96, 97, 21, 22, 24, 25, 129, 130, 315, 316, 2, 5, 8, 11]
    info = pick_info(raw.info, picks)
    data = rng.randn(len(picks), n_freqs, len(times))

    # test complex numbers
    tfr = AverageTFR(info, data * (1 + 1j), times, np.arange(n_freqs), nave)
    tfr.plot_topomap(ch_type='mag', tmin=0.05, tmax=0.150, fmin=0, fmax=10,
                     res=res, contours=0)

    # test real numbers
    tfr = AverageTFR(info, data, times, np.arange(n_freqs), nave)
    tfr.plot_topomap(ch_type='mag', tmin=0.05, tmax=0.150, fmin=0, fmax=10,
                     res=res, contours=0)

    eclick = matplotlib.backend_bases.MouseEvent(
        'button_press_event', plt.gcf().canvas, 0, 0, 1)
    eclick.xdata = eclick.ydata = 0.1
    eclick.inaxes = plt.gca()
    erelease = matplotlib.backend_bases.MouseEvent(
        'button_release_event', plt.gcf().canvas, 0.9, 0.9, 1)
    erelease.xdata = 0.3
    erelease.ydata = 0.2
    pos = np.array([[0.11, 0.11], [0.25, 0.5], [0.0, 0.2], [0.2, 0.39]])
    _onselect(eclick, erelease, tfr, pos, 'grad', 1, 3, 1, 3, 'RdBu_r', list())
    _onselect(eclick, erelease, tfr, pos, 'mag', 1, 3, 1, 3, 'RdBu_r', list())
    eclick.xdata = eclick.ydata = 0.
    erelease.xdata = erelease.ydata = 0.9
    tfr._onselect(eclick, erelease, None, 'mean', None)
    plt.close('all')

    # test plot_psds_topomap
    info = raw.info.copy()
    chan_inds = channel_indices_by_type(info)
    info = pick_info(info, chan_inds['grad'][:4])

    fig, axes = plt.subplots()
    freqs = np.arange(3., 9.5)
    bands = [(4, 8, 'Theta')]
    psd = np.random.rand(len(info['ch_names']), freqs.shape[0])
    plot_psds_topomap(psd, freqs, info, bands=bands, axes=[axes])


def test_ctf_plotting():
    """Test CTF topomap plotting."""
    raw = read_raw_fif(ctf_fname, preload=True)
    assert raw.compensation_grade == 3
    events = make_fixed_length_events(raw, duration=0.01)
    assert len(events) > 10
    evoked = Epochs(raw, events, tmin=0, tmax=0.01, baseline=None).average()
    assert get_current_comp(evoked.info) == 3
    # smoke test that compensation does not matter
    evoked.plot_topomap(time_unit='s')
    # better test that topomaps can still be used without plotting ref
    evoked.pick_types(meg=True, ref_meg=False)
    evoked.plot_topomap()


@pytest.mark.slowtest  # can be slow on OSX
@testing.requires_testing_data
def test_plot_arrowmap(evoked):
    """Test arrowmap plotting."""
    with pytest.raises(ValueError, match='Multiple channel types'):
        plot_arrowmap(evoked.data[:, 0], evoked.info)
    evoked_eeg = evoked.copy().pick('eeg')
    with pytest.raises(ValueError, match='Multiple channel types'):
        plot_arrowmap(evoked_eeg.data[:, 0], evoked.info)
    evoked_mag = evoked.copy().pick('mag')
    evoked_grad = evoked.pick('grad', exclude='bads')
    plot_arrowmap(evoked_mag.data[:, 0], info_from=evoked_mag.info)
    plot_arrowmap(evoked_grad.data[:, 0], info_from=evoked_grad.info,
                  info_to=evoked_mag.info)


@testing.requires_testing_data
def test_plot_topomap_neuromag122():
    """Test topomap plotting."""
    res = 8
    fast_test = dict(res=res, contours=0, sensors=False)
    evoked = read_evokeds(evoked_fname, 'Left Auditory',
                          baseline=(None, 0))
    evoked.pick_types(meg='grad')
    evoked.pick_channels(evoked.ch_names[:122])
    ch_names = ['MEG %03d' % k for k in range(1, 123)]
    for c in evoked.info['chs']:
        c['coil_type'] = FIFF.FIFFV_COIL_NM_122
    evoked.rename_channels({c_old: c_new for (c_old, c_new) in
                            zip(evoked.ch_names, ch_names)})
    layout = find_layout(evoked.info)
    assert layout.kind.startswith('Neuromag_122')
    evoked.plot_topomap(times=[0.1], **fast_test)

    proj = Projection(active=False,
                      desc="test", kind=1,
                      data=dict(nrow=1, ncol=122,
                                row_names=None,
                                col_names=evoked.ch_names, data=np.ones(122)),
                      explained_var=0.5)

    plot_projs_topomap([proj], evoked.info, **fast_test)


def test_plot_topomap_bads():
    """Test plotting topomap with bad channels (gh-7213)."""
    import matplotlib.pyplot as plt
    data = np.random.RandomState(0).randn(3, 1000)
    raw = RawArray(data, create_info(3, 1000., 'eeg'))
    ch_pos_dict = {name: pos for name, pos in zip(raw.ch_names, np.eye(3))}
    raw.info.set_montage(make_dig_montage(ch_pos_dict, coord_frame='head'))
    for count in range(3):
        raw.info['bads'] = raw.ch_names[:count]
        raw.info._check_consistency()
        plot_topomap(data[:, 0], raw.info)
    plt.close('all')


def test_plot_topomap_channel_distance():
    """
    Test topomap plotting with spread out channels (gh-9511, gh-9526).

    Test topomap plotting when the distance between channels is greater than
    the head radius.
    """
    ch_names = ['TP9', 'AF7', 'AF8', 'TP10']

    info = create_info(ch_names, 100, ch_types='eeg')
    evoked = EvokedArray(np.random.randn(4, 10) * 1e-6, info)
    ten_five = make_standard_montage("standard_1005")
    evoked.set_montage(ten_five)

    evoked.plot_topomap(sphere=0.05, res=8)
    plt.close('all')


def test_plot_topomap_bads_grad():
    """Test plotting topomap with bad gradiometer channels (gh-8802)."""
    import matplotlib.pyplot as plt
    data = np.random.RandomState(0).randn(203)
    info = read_info(evoked_fname)
    info['bads'] = ['MEG 2242']
    picks = pick_types(info, meg='grad')
    info = pick_info(info, picks)
    assert len(info['chs']) == 203
    plot_topomap(data, info, res=8)
    plt.close('all')


def test_plot_topomap_nirs_overlap(fnirs_epochs):
    """Test plotting nirs topomap with overlapping channels (gh-7414)."""
    fig = fnirs_epochs['A'].average(picks='hbo').plot_topomap()
    assert len(fig.axes) == 5
    plt.close('all')


@requires_sklearn
def test_plot_topomap_nirs_ica(fnirs_epochs):
    """Test plotting nirs ica topomap."""
    from mne.preprocessing import ICA
    fnirs_epochs = fnirs_epochs.load_data().pick(picks='hbo')
    fnirs_epochs = fnirs_epochs.pick(picks=range(30))

    # fake high-pass filtering and hide the fact that the epochs were
    # baseline corrected
    with fnirs_epochs.info._unlock():
        fnirs_epochs.info['highpass'] = 1.0
    fnirs_epochs.baseline = None

    ica = ICA().fit(fnirs_epochs)
    fig = ica.plot_components()
    assert len(fig[0].axes) == 20
    plt.close('all')


def test_plot_cov_topomap():
    """Test plotting a covariance topomap."""
    cov = read_cov(cov_fname)
    info = read_info(evoked_fname)
    cov.plot_topomap(info)
    cov.plot_topomap(info, noise_cov=cov)
    plt.close('all')


def test_plot_topomap_cnorm():
    """Test colormap normalization."""
    if check_version("matplotlib", "3.2.0"):
        from matplotlib.colors import TwoSlopeNorm
    else:
        from matplotlib.colors import DivergingNorm as TwoSlopeNorm
    from matplotlib.colors import PowerNorm

    rng = np.random.default_rng(42)
    v = rng.uniform(low=-1, high=2.5, size=64)
    v[:3] = [-1, 0, 2.5]

    montage = make_standard_montage("biosemi64")
    info = create_info(montage.ch_names, 256, "eeg").set_montage("biosemi64")
    cnorm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=2.5)

    # pass only cnorm, no vmin/vmax
    plot_topomap(v, info, cnorm=cnorm)

    # pass cnorm and vmin
    with pytest.warns(RuntimeWarning, match='implicitly defines vmin=-1'):
        plot_topomap(v, info, vlim=(-10, None), cnorm=cnorm)

    # pass cnorm and vmax
    with pytest.warns(RuntimeWarning, match='implicitly defines .* vmax=2.5'):
        plot_topomap(v, info, vlim=(None, 10), cnorm=cnorm)

    # try another subclass of mpl.colors.Normalize
    plot_topomap(v, info, cnorm=PowerNorm(0.5))


def test_plot_bridged_electrodes():
    """Test plotting of bridged electrodes."""
    rng = np.random.default_rng(42)
    montage = make_standard_montage("biosemi64")
    info = create_info(montage.ch_names, 256, "eeg").set_montage("biosemi64")
    bridged_idx = [(0, 1), (2, 3)]
    n_epochs = 10
    ed_matrix = np.zeros((n_epochs, len(info.ch_names),
                          len(info.ch_names))) * np.nan
    triu_idx = np.triu_indices(len(info.ch_names), 1)
    for i in range(n_epochs):
        ed_matrix[i][triu_idx] = rng.random() + rng.random(triu_idx[0].size)
    fig = plot_bridged_electrodes(info, bridged_idx, ed_matrix,
                                  topomap_args=dict(names=info.ch_names,
                                                    vlim=(None, 1)))
    # two bridged lines plus head outlines
    assert len(fig.axes[0].lines) == 6
    # test with sphere="eeglab"
    fig = plot_bridged_electrodes(
        info, bridged_idx, ed_matrix,
        topomap_args=dict(names=info.ch_names, sphere="eeglab", vlim=(None, 1))
    )

    with pytest.raises(RuntimeError, match='Expected'):
        plot_bridged_electrodes(info, bridged_idx, np.zeros((5, 6, 7)))

    # test with multiple channel types
    raw = read_raw_fif(raw_fname, preload=True)
    picks = _picks_to_idx(raw.info, "eeg")
    raw._data[picks[0]] = raw._data[picks[1]]  # artificially bridge electrodes
    bridged_idx, ed_matrix = compute_bridged_electrodes(raw)
    plot_bridged_electrodes(raw.info, bridged_idx, ed_matrix)


def test_plot_ch_adjacency():
    """Test plotting of adjacency matrix."""
    xyz_pos = np.array([[-0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0., 0., 0.12],
                        [-0.1, -0.1, 0.1], [0.1, -0.1, 0.1]])

    info = create_info(list('abcde'), 23, ch_types='eeg')
    montage = make_dig_montage(
        ch_pos={ch: pos for ch, pos in zip(info.ch_names, xyz_pos)},
        coord_frame='head')
    info.set_montage(montage)

    # construct adjacency
    adj_sparse, ch_names = find_ch_adjacency(info, 'eeg')

    # plot adjacency
    fig = plot_ch_adjacency(info, adj_sparse, ch_names, kind='2d', edit=True)

    # find channel positions
    collection = fig.axes[0].collections[0]
    pos = collection.get_offsets().data

    # get adjacency lines
    lines = fig.axes[0].lines[4:]  # (first four lines are head outlines)

    # make sure lines match adjacency relations in the matrix
    for line in lines:
        x, y = line.get_data()
        ch_idx = [np.where((pos == [[x[ix], y[ix]]]).all(axis=1))[0][0]
                  for ix in range(2)]
        assert adj_sparse[ch_idx[0], ch_idx[1]]

    # make sure additional point is generated after clicking a channel
    _fake_click(fig, fig.axes[0], pos[0], xform='data')
    collections = fig.axes[0].collections
    assert len(collections) == 2

    # make sure the point is green
    green = matplotlib.colors.to_rgba('tab:green')
    assert (collections[1].get_facecolor() == green).all()

    # make sure adjacency entry is modified after second click on another node
    assert adj_sparse[0, 1]
    assert adj_sparse[1, 0]
    n_lines_before = len(lines)
    _fake_click(fig, fig.axes[0], pos[1], xform='data')

    assert not adj_sparse[0, 1]
    assert not adj_sparse[1, 0]

    # and there is one line less
    lines = fig.axes[0].lines[4:]
    n_lines_after = len(lines)
    assert n_lines_after == n_lines_before - 1

    # make sure there is still one green point ...
    collections = fig.axes[0].collections
    assert len(collections) == 2
    assert (collections[1].get_facecolor() == green).all()

    # ... but its at a different location
    point_pos = collections[1].get_offsets().data
    assert (point_pos == pos[1]).all()

    # check that clicking again removes the green selection point
    _fake_click(fig, fig.axes[0], pos[1], xform='data')
    collections = fig.axes[0].collections
    assert len(collections) == 1

    # clicking the points again adds a green line
    _fake_click(fig, fig.axes[0], pos[1], xform='data')
    _fake_click(fig, fig.axes[0], pos[0], xform='data')

    lines = fig.axes[0].lines[4:]
    assert len(lines) == n_lines_after + 1
    assert lines[-1].get_color() == 'tab:green'

    # smoke test for 3d option
    adj = adj_sparse.toarray()
    fig = plot_ch_adjacency(info, adj, ch_names, kind='3d')

    # test errors
    # -----------
    # number of channels in the adjacency matrix and info must match
    msg = ("``adjacency`` must have the same number of rows as the number of "
           "channels in ``info``")
    with pytest.raises(ValueError, match=msg):
        plot_ch_adjacency(info, adj_sparse, ch_names[:3], kind='2d')

    # edition mode only available for 2d plot
    msg = "Editing a 3d adjacency plot is not supported."
    with pytest.raises(ValueError, match=msg):
        plot_ch_adjacency(info, adj, ch_names, kind='3d', edit=True)
