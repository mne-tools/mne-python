# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Cathy Nangini <cnangini@gmail.com>
#          Mainak Jas <mainak@neuro.hut.fi>
#          Jona Sassenhagen <jona.sassenhagen@gmail.com>
#          Daniel McCloy <dan.mccloy@gmail.com>
#
# License: Simplified BSD

import os.path as op

import numpy as np
from numpy.testing import assert_allclose
import pytest
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.cm import get_cmap

import mne
from mne import (read_events, Epochs, read_cov, compute_covariance,
                 make_fixed_length_events, compute_proj_evoked)
from mne.io import read_raw_fif
from mne.utils import run_tests_if_main, catch_logging, requires_version
from mne.viz import plot_compare_evokeds, plot_evoked_white
from mne.viz.utils import _fake_click
from mne.datasets import testing
from mne.io.constants import FIFF
from mne.stats.parametric import _parametric_ci

base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
evoked_fname = op.join(base_dir, 'test-ave.fif')
raw_fname = op.join(base_dir, 'test_raw.fif')
raw_sss_fname = op.join(base_dir, 'test_chpi_raw_sss.fif')
cov_fname = op.join(base_dir, 'test-cov.fif')
event_name = op.join(base_dir, 'test-eve.fif')
event_id, tmin, tmax = 1, -0.1, 0.1

# Use a subset of channels for plotting speed
# make sure we have a magnetometer and a pair of grad pairs for topomap.
default_picks = (0, 1, 2, 3, 4, 6, 7, 61, 122, 183, 244, 305,
                 315, 316, 317, 318)  # EEG channels
sel = (0, 7)


def _get_epochs(picks=default_picks):
    """Get epochs."""
    raw = read_raw_fif(raw_fname)
    raw.add_proj([], remove_existing=True)
    events = read_events(event_name)
    epochs = Epochs(raw, events[:5], event_id, tmin, tmax, picks=picks,
                    decim=10, verbose='error')
    epochs.info['bads'] = [
        epochs.ch_names[-5],  # MEG
        epochs.ch_names[-1]   # EEG
    ]
    epochs.info.normalize_proj()
    return epochs


def _get_epochs_delayed_ssp():
    """Get epochs with delayed SSP."""
    raw = read_raw_fif(raw_fname)
    events = read_events(event_name)
    reject = dict(mag=4e-12)
    epochs_delayed_ssp = Epochs(raw, events[:10], event_id, tmin, tmax,
                                picks=default_picks, proj='delayed',
                                reject=reject, verbose='error')
    epochs_delayed_ssp.info.normalize_proj()
    return epochs_delayed_ssp


def test_plot_evoked_cov():
    """Test plot_evoked with noise_cov."""
    evoked = _get_epochs().average()
    cov = read_cov(cov_fname)
    cov['projs'] = []  # avoid warnings
    with pytest.warns(RuntimeWarning, match='No average EEG reference'):
        evoked.plot(noise_cov=cov, time_unit='s')
    with pytest.raises(TypeError, match='Covariance'):
        evoked.plot(noise_cov=1., time_unit='s')
    with pytest.raises(IOError, match='No such file'):
        evoked.plot(noise_cov='nonexistent-cov.fif', time_unit='s')
    raw = read_raw_fif(raw_sss_fname)
    events = make_fixed_length_events(raw)
    epochs = Epochs(raw, events, picks=default_picks)
    cov = compute_covariance(epochs)
    evoked_sss = epochs.average()
    with pytest.warns(RuntimeWarning, match='relative scaling'):
        evoked_sss.plot(noise_cov=cov, time_unit='s')
    plt.close('all')


@pytest.mark.slowtest
def test_plot_evoked():
    """Test evoked.plot."""
    epochs = _get_epochs()
    evoked = epochs.average()
    assert evoked.proj is False
    fig = evoked.plot(proj=True, hline=[1], exclude=[], window_title='foo',
                      time_unit='s')
    amplitudes = _get_amplitudes(fig)
    assert len(amplitudes) == len(default_picks)
    assert evoked.proj is False
    # Test a click
    ax = fig.get_axes()[0]
    line = ax.lines[0]
    _fake_click(fig, ax,
                [line.get_xdata()[0], line.get_ydata()[0]], 'data')
    _fake_click(fig, ax,
                [ax.get_xlim()[0], ax.get_ylim()[1]], 'data')
    # plot with bad channels excluded & spatial_colors & zorder
    evoked.plot(exclude='bads', time_unit='s')

    # test selective updating of dict keys is working.
    evoked.plot(hline=[1], units=dict(mag='femto foo'), time_unit='s')
    evoked_delayed_ssp = _get_epochs_delayed_ssp().average()
    evoked_delayed_ssp.plot(proj='interactive', time_unit='s')
    evoked_delayed_ssp.apply_proj()
    pytest.raises(RuntimeError, evoked_delayed_ssp.plot,
                  proj='interactive', time_unit='s')
    evoked_delayed_ssp.info['projs'] = []
    pytest.raises(RuntimeError, evoked_delayed_ssp.plot,
                  proj='interactive', time_unit='s')
    pytest.raises(RuntimeError, evoked_delayed_ssp.plot,
                  proj='interactive', axes='foo', time_unit='s')
    plt.close('all')

    # test `gfp='only'`: GFP (EEG) and RMS (MEG)
    fig, ax = plt.subplots(3)
    evoked.plot(gfp='only', time_unit='s', axes=ax)

    assert len(ax[0].lines) == len(ax[1].lines) == len(ax[2].lines) == 1

    assert ax[0].get_title() == 'EEG (3 channels)'
    assert ax[0].texts[0].get_text() == 'GFP'

    assert ax[1].get_title() == 'Gradiometers (9 channels)'
    assert ax[1].texts[0].get_text() == 'RMS'

    assert ax[2].get_title() == 'Magnetometers (2 channels)'
    assert ax[1].texts[0].get_text() == 'RMS'

    plt.close('all')

    # Test invalid `gfp`
    with pytest.raises(ValueError):
        evoked.plot(gfp='foo', time_unit='s')

    # plot with bad channels excluded, spatial_colors, zorder & pos. layout
    evoked.rename_channels({'MEG 0133': 'MEG 0000'})
    evoked.plot(exclude=evoked.info['bads'], spatial_colors=True, gfp=True,
                zorder='std', time_unit='s')
    evoked.plot(exclude=[], spatial_colors=True, zorder='unsorted',
                time_unit='s')
    pytest.raises(TypeError, evoked.plot, zorder='asdf', time_unit='s')
    plt.close('all')

    evoked.plot_sensors()  # Test plot_sensors
    plt.close('all')

    evoked.pick_channels(evoked.ch_names[:4])
    with catch_logging() as log_file:
        evoked.plot(verbose=True, time_unit='s')
    assert 'Need more than one' in log_file.getvalue()


@requires_version('matplotlib', '2.2')
def test_constrained_layout():
    """Test that we handle constrained layouts correctly."""
    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    assert fig.get_constrained_layout()
    evoked = mne.read_evokeds(evoked_fname)[0]
    evoked.pick(evoked.ch_names[:2])
    evoked.plot(axes=ax)  # smoke test that it does not break things
    assert fig.get_constrained_layout()
    plt.close('all')


def _get_amplitudes(fig):
    amplitudes = [line.get_ydata() for ax in fig.axes
                  for line in ax.get_lines()]
    amplitudes = np.array(
        [line for line in amplitudes if isinstance(line, np.ndarray)])
    return amplitudes


@pytest.mark.parametrize('picks, rlims, avg_proj', [
    (default_picks[:-4], (0.59, 0.61), False),  # MEG
    (np.arange(340, 360), (0.56, 0.57), True),  # EEG
    (np.arange(340, 360), (0.79, 0.81), False),  # EEG
])
def test_plot_evoked_reconstruct(picks, rlims, avg_proj):
    """Test proj="reconstruct"."""
    evoked = _get_epochs(picks=picks).average()
    if avg_proj:
        evoked.set_eeg_reference(projection=True).apply_proj()
        assert len(evoked.info['projs']) == 1
        assert evoked.proj is True
    else:
        assert len(evoked.info['projs']) == 0
        assert evoked.proj is False
    fig = evoked.plot(proj=True, hline=[1], exclude=[], window_title='foo',
                      time_unit='s')
    amplitudes = _get_amplitudes(fig)
    assert len(amplitudes) == len(picks)
    assert evoked.proj is avg_proj

    fig = evoked.plot(proj='reconstruct', exclude=[])
    amplitudes_recon = _get_amplitudes(fig)
    if avg_proj is False:
        assert_allclose(amplitudes, amplitudes_recon)
    proj = compute_proj_evoked(evoked.copy().crop(None, 0).apply_proj())
    evoked.add_proj(proj)
    assert len(evoked.info['projs']) == 2 if len(picks) == 3 else 4
    fig = evoked.plot(proj=True, exclude=[])
    amplitudes_proj = _get_amplitudes(fig)
    fig = evoked.plot(proj='reconstruct', exclude=[])
    amplitudes_recon = _get_amplitudes(fig)
    assert len(amplitudes_recon) == len(picks)
    norm = np.linalg.norm(amplitudes)
    norm_proj = np.linalg.norm(amplitudes_proj)
    norm_recon = np.linalg.norm(amplitudes_recon)
    r = np.dot(amplitudes_recon.ravel(), amplitudes.ravel()) / (
        norm_recon * norm)
    assert rlims[0] < r < rlims[1]
    assert 1.05 * norm_proj < norm_recon
    if not avg_proj:
        assert norm_proj < norm * 0.9

    cov = read_cov(cov_fname)
    with pytest.raises(ValueError, match='Cannot use proj="reconstruct"'):
        evoked.plot(noise_cov=cov, proj='reconstruct')
    plt.close('all')


def test_plot_evoked_image():
    """Test plot_evoked_image."""
    evoked = _get_epochs().average()
    evoked.plot_image(proj=True, time_unit='ms')

    # fail nicely on NaN
    evoked_nan = evoked.copy()
    evoked_nan.data[:, 0] = np.nan
    pytest.raises(ValueError, evoked_nan.plot)
    with np.errstate(invalid='ignore'):
        pytest.raises(ValueError, evoked_nan.plot_image)
        pytest.raises(ValueError, evoked_nan.plot_joint)

    # test mask
    evoked.plot_image(picks=[1, 2], mask=evoked.data > 0, time_unit='s')
    evoked.plot_image(picks=[1, 2], mask_cmap=None, colorbar=False,
                      mask=np.ones(evoked.data.shape).astype(bool),
                      time_unit='s')
    with pytest.warns(RuntimeWarning, match='not adding contour'):
        evoked.plot_image(picks=[1, 2], mask=None, mask_style="both",
                          time_unit='s')
    with pytest.raises(ValueError, match='must have the same shape'):
        evoked.plot_image(mask=evoked.data[1:, 1:] > 0, time_unit='s')

    # plot with bad channels excluded
    evoked.plot_image(exclude='bads', cmap='interactive', time_unit='s')
    plt.close('all')

    with pytest.raises(ValueError, match='not unique'):
        evoked.plot_image(picks=[0, 0], time_unit='s')  # duplicates

    ch_names = evoked.ch_names[3:5]
    picks = [evoked.ch_names.index(ch) for ch in ch_names]
    evoked.plot_image(show_names="all", time_unit='s', picks=picks)
    yticklabels = plt.gca().get_yticklabels()
    for tick_target, tick_observed in zip(ch_names, yticklabels):
        assert tick_target in str(tick_observed)
    evoked.plot_image(show_names=True, time_unit='s')

    # test groupby
    evoked.plot_image(group_by=dict(sel=sel), axes=dict(sel=plt.axes()))
    plt.close('all')
    for group_by, axes in (("something", dict()), (dict(), "something")):
        pytest.raises(ValueError, evoked.plot_image, group_by=group_by,
                      axes=axes)


def test_plot_white():
    """Test plot_white."""
    cov = read_cov(cov_fname)
    cov['method'] = 'empirical'
    cov['projs'] = []  # avoid warnings
    evoked = _get_epochs().average()
    evoked.set_eeg_reference('average')  # Avoid warnings

    # test rank param.
    evoked.plot_white(cov, rank={'mag': 101, 'grad': 201, 'eeg': 10},
                      time_unit='s')
    fig = evoked.plot_white(cov, rank={'mag': 101}, time_unit='s')  # test rank
    evoked.plot_white(cov, rank={'grad': 201}, time_unit='s', axes=fig.axes)
    with pytest.raises(ValueError, match=r'must have shape \(4,\), got \(2,'):
        evoked.plot_white(cov, axes=fig.axes[:2])
    with pytest.raises(ValueError, match='When not using SSS'):
        evoked.plot_white(cov, rank={'meg': 306})
    evoked.plot_white([cov, cov], time_unit='s')
    plt.close('all')

    fig = plot_evoked_white(evoked, [cov, cov])
    assert len(fig.axes) == 3 * 2
    axes = np.array(fig.axes).reshape(3, 2)
    plot_evoked_white(evoked, [cov, cov], axes=axes)
    with pytest.raises(ValueError, match=r'have shape \(3, 2\), got'):
        plot_evoked_white(evoked, [cov, cov], axes=axes[:, :1])

    # Hack to test plotting of maxfiltered data
    evoked_sss = evoked.copy()
    sss = dict(sss_info=dict(in_order=80, components=np.arange(80)))
    evoked_sss.info['proc_history'] = [dict(max_info=sss)]
    evoked_sss.plot_white(cov, rank={'meg': 64})
    with pytest.raises(ValueError, match='When using SSS'):
        evoked_sss.plot_white(cov, rank={'grad': 201})
    evoked_sss.plot_white(cov, time_unit='s')
    plt.close('all')


def test_plot_compare_evokeds():
    """Test plot_compare_evokeds."""
    evoked = _get_epochs().average()
    # test defaults
    figs = plot_compare_evokeds(evoked)
    assert len(figs) == 3
    # test picks, combine, and vlines (1-channel pick also shows sensor inset)
    picks = ['MEG 0113', 'mag'] + 2 * [['MEG 0113', 'MEG 0112']] + [[0, 1]]
    vlines = [[0.1, 0.2], []] + 3 * ['auto']
    combine = [None, 'mean', 'std', None, lambda x: np.min(x, axis=1)]
    title = ['MEG 0113', '(mean)', '(std. dev.)', '(GFP)', 'MEG 0112']
    for _p, _v, _c, _t in zip(picks, vlines, combine, title):
        fig = plot_compare_evokeds(evoked, picks=_p, vlines=_v, combine=_c)
        assert fig[0].axes[0].get_title().endswith(_t)
    # test passing more than one evoked
    red, blue = evoked.copy(), evoked.copy()
    red.data *= 1.5
    blue.data /= 1.5
    evoked_dict = {'aud/l': blue, 'aud/r': red, 'vis': evoked}
    huge_dict = {'cond{}'.format(i): ev for i, ev in enumerate([evoked] * 11)}
    plot_compare_evokeds(evoked_dict)                           # dict
    plot_compare_evokeds([[red, evoked], [blue, evoked]])       # list of lists
    figs = plot_compare_evokeds({'cond': [blue, red, evoked]})  # dict of list
    # test that confidence bands are plausible
    for fig in figs:
        extents = fig.axes[0].collections[0].get_paths()[0].get_extents()
        xlim, ylim = extents.get_points().T
        assert np.allclose(xlim, evoked.times[[0, -1]])
        line = fig.axes[0].lines[0]
        xvals = line.get_xdata()
        assert np.allclose(xvals, evoked.times)
        yvals = line.get_ydata()
        assert (yvals < ylim[1]).all()
        assert (yvals > ylim[0]).all()
    plt.close('all')
    # test other CI args
    for _ci in (None, False, 0.5,
                lambda x: np.stack([x.mean(axis=0) + 1, x.mean(axis=0) - 1])):
        plot_compare_evokeds({'cond': [blue, red, evoked]}, ci=_ci)
    with pytest.raises(TypeError, match='"ci" must be None, bool, float or'):
        plot_compare_evokeds(evoked, ci='foo')
    # test sensor inset, legend location, and axis inversion & truncation
    plot_compare_evokeds(evoked_dict, invert_y=True, legend='upper left',
                         show_sensors='center', truncate_xaxis=False,
                         truncate_yaxis=False)
    plot_compare_evokeds(evoked, ylim=dict(mag=(-50, 50)), truncate_yaxis=True)
    plt.close('all')
    # test styles
    plot_compare_evokeds(evoked_dict, colors=['b', 'r', 'g'],
                         linestyles=[':', '-', '--'], split_legend=True)
    style_dict = dict(aud=dict(alpha=0.3), vis=dict(linewidth=3, c='k'))
    plot_compare_evokeds(evoked_dict, styles=style_dict, colors={'aud/r': 'r'},
                         linestyles=dict(vis='dotted'), ci=False)
    plot_compare_evokeds(evoked_dict, colors=list(range(3)))
    plt.close('all')
    # test colormap
    cmap = get_cmap('viridis')
    plot_compare_evokeds(evoked_dict, cmap=cmap, colors=dict(aud=0.4, vis=0.9))
    plot_compare_evokeds(evoked_dict, cmap=cmap, colors=dict(aud=1, vis=2))
    plot_compare_evokeds(evoked_dict, cmap=('cmap title', 'inferno'),
                         linestyles=['-', ':', '--'])
    plt.close('all')
    # test warnings
    with pytest.warns(RuntimeWarning, match='in "picks"; cannot combine'):
        plot_compare_evokeds(evoked, picks=[0], combine='median')
    plt.close('all')
    # test errors
    with pytest.raises(TypeError, match='"evokeds" must be a dict, list'):
        plot_compare_evokeds('foo')
    with pytest.raises(ValueError, match=r'keys in "styles" \(.*\) must '):
        plot_compare_evokeds(evoked_dict, styles=dict(foo='foo', bar='bar'))
    with pytest.raises(ValueError, match='colors in the default color cycle'):
        plot_compare_evokeds(huge_dict, colors=None)
    with pytest.raises(TypeError, match='"cmap" is specified, then "colors"'):
        plot_compare_evokeds(evoked_dict, cmap='Reds', colors={'aud/l': 'foo',
                                                               'aud/r': 'bar',
                                                               'vis': 'baz'})
    plt.close('all')
    for kwargs in [dict(colors=[0, 1]), dict(linestyles=['-', ':'])]:
        match = r'but there are only \d* (colors|linestyles). Please specify'
        with pytest.raises(ValueError, match=match):
            plot_compare_evokeds(evoked_dict, **kwargs)
    for kwargs in [dict(colors='foo'), dict(linestyles='foo')]:
        match = r'"(colors|linestyles)" must be a dict, list, or None; got '
        with pytest.raises(TypeError, match=match):
            plot_compare_evokeds(evoked_dict, **kwargs)
    for kwargs in [dict(colors=dict(foo='f')), dict(linestyles=dict(foo='f'))]:
        match = r'If "(colors|linestyles)" is a dict its keys \(.*\) must '
        with pytest.raises(ValueError, match=match):
            plot_compare_evokeds(evoked_dict, **kwargs)
    for kwargs in [dict(legend='foo'), dict(show_sensors='foo')]:
        with pytest.raises(ValueError, match='not a legal MPL loc, please'):
            plot_compare_evokeds(evoked_dict, **kwargs)
    with pytest.raises(TypeError, match='an instance of list or tuple'):
        plot_compare_evokeds(evoked_dict, vlines='foo')
    with pytest.raises(ValueError, match='"truncate_yaxis" must be bool or '):
        plot_compare_evokeds(evoked_dict, truncate_yaxis='foo')
    plt.close('all')
    # test axes='topo'
    figs = plot_compare_evokeds(evoked_dict, axes='topo', legend=True)
    for fig in figs:
        assert len(fig.axes[0].lines) == len(evoked_dict)
    # test with (fake) CSD data
    csd = _get_epochs(picks=np.arange(315, 320)).average()  # 5 EEG chs
    for entry in csd.info['chs']:
        entry['coil_type'] = FIFF.FIFFV_COIL_EEG_CSD
        entry['unit'] = FIFF.FIFF_UNIT_V_M2
    plot_compare_evokeds(csd, picks='csd', axes='topo')
    # old tests
    red.info['chs'][0]['loc'][:2] = 0  # test plotting channel at zero
    plot_compare_evokeds([red, blue], picks=[0],
                         ci=lambda x: [x.std(axis=0), -x.std(axis=0)])
    plot_compare_evokeds([list(evoked_dict.values())], picks=[0],
                         ci=_parametric_ci)
    # smoke test for tmin >= 0 (from mailing list)
    red.crop(0.01, None)
    assert len(red.times) > 2
    plot_compare_evokeds(red)
    # plot a flat channel
    red.data = np.zeros_like(red.data)
    plot_compare_evokeds(red)
    # smoke test for one time point (not useful but should not fail)
    red.crop(0.02, 0.02)
    assert len(red.times) == 1
    plot_compare_evokeds(red)
    # now that we've cropped `red`:
    with pytest.raises(ValueError, match='not contain the same time instants'):
        plot_compare_evokeds(evoked_dict)
    plt.close('all')


def test_plot_compare_evokeds_neuromag122():
    """Test topomap plotting."""
    evoked = mne.read_evokeds(evoked_fname, 'Left Auditory',
                              baseline=(None, 0))
    evoked.pick_types(meg='grad')
    evoked.pick_channels(evoked.ch_names[:122])
    ch_names = ['MEG %03d' % k for k in range(1, 123)]
    for c in evoked.info['chs']:
        c['coil_type'] = FIFF.FIFFV_COIL_NM_122
    evoked.rename_channels({c_old: c_new for (c_old, c_new) in
                            zip(evoked.ch_names, ch_names)})
    mne.viz.plot_compare_evokeds([evoked, evoked])


@testing.requires_testing_data
def test_plot_ctf():
    """Test plotting of CTF evoked."""
    ctf_dir = op.join(testing.data_path(download=False), 'CTF')
    raw_fname = op.join(ctf_dir, 'testdata_ctf.ds')

    raw = mne.io.read_raw_ctf(raw_fname, preload=True)
    events = np.array([[200, 0, 1]])
    event_id = 1
    tmin, tmax = -0.1, 0.5  # start and end of an epoch in sec.
    picks = mne.pick_types(raw.info, meg=True, stim=True, eog=True,
                           ref_meg=True, exclude='bads')[::20]
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                        picks=picks, preload=True, decim=10, verbose='error')
    evoked = epochs.average()
    evoked.plot_joint(times=[0.1])
    mne.viz.plot_compare_evokeds([evoked, evoked])

    # make sure axes position is "almost" unchanged
    # when axes were passed to plot_joint by the user
    times = [0.1, 0.2, 0.3]
    fig = plt.figure()

    # create custom axes for topomaps, colorbar and the timeseries
    gs = gridspec.GridSpec(3, 7, hspace=0.5, top=0.8)
    topo_axes = [fig.add_subplot(gs[0, idx * 2:(idx + 1) * 2])
                 for idx in range(len(times))]
    topo_axes.append(fig.add_subplot(gs[0, -1]))
    ts_axis = fig.add_subplot(gs[1:, 1:-1])

    def get_axes_midpoints(axes):
        midpoints = list()
        for ax in axes[:-1]:
            pos = ax.get_position()
            midpoints.append([pos.x0 + (pos.width * 0.5),
                              pos.y0 + (pos.height * 0.5)])
        return np.array(midpoints)

    midpoints_before = get_axes_midpoints(topo_axes)
    evoked.plot_joint(times=times, ts_args={'axes': ts_axis},
                      topomap_args={'axes': topo_axes}, title=None)
    midpoints_after = get_axes_midpoints(topo_axes)
    assert (np.linalg.norm(midpoints_before - midpoints_after) < 0.1).all()


run_tests_if_main()
