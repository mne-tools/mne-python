# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Cathy Nangini <cnangini@gmail.com>
#          Mainak Jas <mainak@neuro.hut.fi>
#          Jona Sassenhagen <jona.sassenhagen@gmail.com>
#
# License: Simplified BSD

import os.path as op
import warnings

import numpy as np
from numpy.testing import assert_raises, assert_allclose
from nose.tools import assert_true
import pytest

import mne
from mne import (read_events, Epochs, pick_types, read_cov, compute_covariance,
                 make_fixed_length_events)
from mne.channels import read_layout
from mne.io import read_raw_fif
from mne.utils import run_tests_if_main, catch_logging
from mne.viz.evoked import _line_plot_onselect, plot_compare_evokeds
from mne.viz.utils import _fake_click
from mne.stats import _parametric_ci
from mne.datasets import testing

# Set our plotters to test mode
import matplotlib
matplotlib.use('Agg')  # for testing don't use X server

warnings.simplefilter('always')  # enable b/c these tests throw warnings

base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
evoked_fname = op.join(base_dir, 'test-ave.fif')
raw_fname = op.join(base_dir, 'test_raw.fif')
raw_sss_fname = op.join(base_dir, 'test_chpi_raw_sss.fif')
cov_fname = op.join(base_dir, 'test-cov.fif')
event_name = op.join(base_dir, 'test-eve.fif')
event_id, tmin, tmax = 1, -0.1, 0.1
n_chan = 6
layout = read_layout('Vectorview-all')


def _get_picks(raw):
    """Get picks."""
    return pick_types(raw.info, meg=True, eeg=False, stim=False,
                      ecg=False, eog=False, exclude='bads')


def _get_epochs():
    """Get epochs."""
    raw = read_raw_fif(raw_fname)
    raw.add_proj([], remove_existing=True)
    events = read_events(event_name)
    picks = _get_picks(raw)
    # Use a subset of channels for plotting speed
    picks = picks[np.round(np.linspace(0, len(picks) - 1, n_chan)).astype(int)]
    # make sure we have a magnetometer and a pair of grad pairs for topomap.
    picks = np.concatenate([[2, 3, 4, 6, 7], picks])
    epochs = Epochs(raw, events[:5], event_id, tmin, tmax, picks=picks)
    epochs.info['bads'] = [epochs.ch_names[-1]]
    return epochs


def _get_epochs_delayed_ssp():
    """Get epochs with delayed SSP."""
    raw = read_raw_fif(raw_fname)
    events = read_events(event_name)
    picks = _get_picks(raw)
    reject = dict(mag=4e-12)
    epochs_delayed_ssp = Epochs(raw, events[:10], event_id, tmin, tmax,
                                picks=picks, proj='delayed', reject=reject)
    return epochs_delayed_ssp


def test_plot_evoked_cov():
    """Test plot_evoked with noise_cov."""
    import matplotlib.pyplot as plt
    evoked = _get_epochs().average()
    cov = read_cov(cov_fname)
    cov['projs'] = []  # avoid warnings
    evoked.plot(noise_cov=cov, time_unit='s')
    with pytest.raises(TypeError, match='Covariance'):
        evoked.plot(noise_cov=1., time_unit='s')
    with pytest.raises(IOError, match='No such file'):
        evoked.plot(noise_cov='nonexistent-cov.fif', time_unit='s')
    raw = read_raw_fif(raw_sss_fname)
    events = make_fixed_length_events(raw)
    epochs = Epochs(raw, events)
    cov = compute_covariance(epochs)
    evoked_sss = epochs.average()
    with warnings.catch_warnings(record=True) as w:
        evoked_sss.plot(noise_cov=cov, time_unit='s')
    plt.close('all')
    assert any('relative scal' in str(ww.message) for ww in w)


@pytest.mark.slowtest
def test_plot_evoked():
    """Test plotting of evoked."""
    import matplotlib.pyplot as plt
    rng = np.random.RandomState(0)
    evoked = _get_epochs().average()
    fig = evoked.plot(proj=True, hline=[1], exclude=[], window_title='foo',
                      time_unit='s')
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
    assert_raises(RuntimeError, evoked_delayed_ssp.plot,
                  proj='interactive', time_unit='s')
    evoked_delayed_ssp.info['projs'] = []
    assert_raises(RuntimeError, evoked_delayed_ssp.plot,
                  proj='interactive', time_unit='s')
    assert_raises(RuntimeError, evoked_delayed_ssp.plot,
                  proj='interactive', axes='foo', time_unit='s')
    plt.close('all')

    # test GFP only
    evoked.plot(gfp='only', time_unit='s')
    assert_raises(ValueError, evoked.plot, gfp='foo', time_unit='s')

    evoked.plot_image(proj=True, time_unit='ms')
    # test mask
    evoked.plot_image(picks=[1, 2], mask=evoked.data > 0, time_unit='s')
    evoked.plot_image(picks=[1, 2], mask_cmap=None, colorbar=False,
                      mask=np.ones(evoked.data.shape).astype(bool),
                      time_unit='s')

    with warnings.catch_warnings(record=True) as w:
        evoked.plot_image(picks=[1, 2], mask=None, mask_style="both",
                          time_unit='s')
    assert len(w) == 2
    assert_raises(ValueError, evoked.plot_image, mask=evoked.data[1:, 1:] > 0,
                  time_unit='s')

    # plot with bad channels excluded
    evoked.plot_image(exclude='bads', cmap='interactive', time_unit='s')
    evoked.plot_image(exclude=evoked.info['bads'], time_unit='s')  # same thing
    plt.close('all')

    assert_raises(ValueError, evoked.plot_image, picks=[0, 0],
                  time_unit='s')  # duplicates

    evoked.plot_topo()  # should auto-find layout
    _line_plot_onselect(0, 200, ['mag', 'grad'], evoked.info, evoked.data,
                        evoked.times)
    plt.close('all')

    cov = read_cov(cov_fname)
    cov['method'] = 'empirical'
    cov['projs'] = []  # avoid warnings
    # test rank param.
    evoked.plot_white(cov, rank={'mag': 101, 'grad': 201}, time_unit='s')
    evoked.plot_white(cov, rank={'mag': 101}, time_unit='s')  # test rank param
    evoked.plot_white(cov, rank={'grad': 201}, time_unit='s')
    assert_raises(
        ValueError, evoked.plot_white, cov,
        rank={'mag': 101, 'grad': 201, 'meg': 306}, time_unit='s')
    assert_raises(
        ValueError, evoked.plot_white, cov, rank={'meg': 306}, time_unit='s')

    evoked.plot_white([cov, cov], time_unit='s')

    # plot_compare_evokeds: test condition contrast, CI, color assignment
    plot_compare_evokeds(evoked.copy().pick_types(meg='mag'))
    plot_compare_evokeds(
        evoked.copy().pick_types(meg='grad'), picks=[1, 2],
        show_sensors="upper right", show_legend="upper left")
    evokeds = [evoked.copy() for _ in range(10)]
    for evoked in evokeds:
        evoked.data += (rng.randn(*evoked.data.shape) *
                        np.std(evoked.data, axis=-1, keepdims=True))
    for picks in ([0], [1], [2], [0, 2], [1, 2], [0, 1, 2],):
        figs = plot_compare_evokeds([evokeds], picks=picks, ci=0.95)
        if not isinstance(figs, list):
            figs = [figs]
        for fig in figs:
            ext = fig.axes[0].collections[0].get_paths()[0].get_extents()
            xs, ylim = ext.get_points().T
            assert_allclose(xs, evoked.times[[0, -1]])
            line = fig.axes[0].lines[0]
            xs = line.get_xdata()
            assert_allclose(xs, evoked.times)
            ys = line.get_ydata()
            assert (ys < ylim[1]).all()
            assert (ys > ylim[0]).all()
        plt.close('all')

    evoked.rename_channels({'MEG 2142': "MEG 1642"})
    assert len(plot_compare_evokeds(evoked)) == 2
    colors = dict(red='r', blue='b')
    linestyles = dict(red='--', blue='-')
    red, blue = evoked.copy(), evoked.copy()
    red.data *= 1.1
    blue.data *= 0.9
    plot_compare_evokeds([red, blue], picks=3)  # list of evokeds
    plot_compare_evokeds([red, blue], picks=3, truncate_yaxis=True)
    plot_compare_evokeds([[red, evoked], [blue, evoked]],
                         picks=3)  # list of lists
    # test picking & plotting grads
    contrast = dict()
    contrast["red/stim"] = list((evoked.copy(), red))
    contrast["blue/stim"] = list((evoked.copy(), blue))
    # test a bunch of params at once
    for evokeds_ in (evoked.copy().pick_types(meg='mag'), contrast,
                     [red, blue], [[red, evoked], [blue, evoked]]):
        plot_compare_evokeds(evokeds_, picks=0, ci=True)  # also tests CI
    plt.close('all')
    # test styling +  a bunch of other params at once
    colors, linestyles = dict(red='r', blue='b'), dict(red='--', blue='-')
    plot_compare_evokeds(contrast, colors=colors, linestyles=linestyles,
                         picks=[0, 2], vlines=[.01, -.04], invert_y=True,
                         truncate_yaxis=False, ylim=dict(mag=(-10, 10)),
                         styles={"red/stim": {"linewidth": 1}},
                         show_sensors=True)
    # various bad styles
    params = [dict(picks=3, colors=dict(fake=1)),
              dict(picks=3, styles=dict(fake=1)), dict(picks=3, gfp=True),
              dict(picks=3, show_sensors="a"),
              dict(colors=dict(red=10., blue=-2))]
    for param in params:
        assert_raises(ValueError, plot_compare_evokeds, evoked, **param)
    assert_raises(TypeError, plot_compare_evokeds, evoked, picks='str')
    assert_raises(TypeError, plot_compare_evokeds, evoked, vlines='x')
    plt.close('all')
    # `evoked` must contain Evokeds
    assert_raises(ValueError, plot_compare_evokeds, [[1, 2], [3, 4]])
    # `ci` must be float or None
    assert_raises(TypeError, plot_compare_evokeds, contrast, ci='err')
    # test all-positive ylim
    contrast["red/stim"], contrast["blue/stim"] = red, blue
    plot_compare_evokeds(contrast, picks=[0], colors=['r', 'b'],
                         ylim=dict(mag=(1, 10)), ci=_parametric_ci,
                         truncate_yaxis='max_ticks', show_sensors=False,
                         show_legend=False)

    # sequential colors
    evokeds = (evoked, blue, red)
    contrasts = {"a{}/b".format(ii): ev for ii, ev in
                 enumerate(evokeds)}
    colors = {"a" + str(ii): ii for ii, _ in enumerate(evokeds)}
    contrasts["a1/c"] = evoked.copy()
    for split in (True, False):
        for linestyles in (["-"], {"b": "-", "c": ":"}):
            plot_compare_evokeds(
                contrasts, colors=colors, picks=[0], cmap='Reds',
                split_legend=split, linestyles=linestyles,
                ci=False, show_sensors=False)
    colors = {"a" + str(ii): ii / len(evokeds)
              for ii, _ in enumerate(evokeds)}
    plot_compare_evokeds(
        contrasts, colors=colors, picks=[0], cmap='Reds',
        split_legend=split, linestyles=linestyles, ci=False,
        show_sensors=False)
    red.info["chs"][0]["loc"][:2] = 0  # test plotting channel at zero
    plot_compare_evokeds(red, picks=[0],
                         ci=lambda x: [x.std(axis=0), -x.std(axis=0)])
    plot_compare_evokeds([red, blue], picks=[0], cmap="summer", ci=None,
                         split_legend=None)
    plot_compare_evokeds([red, blue], cmap=None, split_legend=True)
    assert_raises(ValueError, plot_compare_evokeds, [red] * 20)
    assert_raises(ValueError, plot_compare_evokeds, contrasts,
                  cmap='summer')

    plt.close('all')

    # Hack to test plotting of maxfiltered data
    evoked_sss = evoked.copy()
    sss = dict(sss_info=dict(in_order=80, components=np.arange(80)))
    evoked_sss.info['proc_history'] = [dict(max_info=sss)]
    evoked_sss.plot_white(cov, rank={'meg': 64}, time_unit='s')
    assert_raises(
        ValueError, evoked_sss.plot_white, cov, rank={'grad': 201},
        time_unit='s')
    evoked_sss.plot_white(cov, time_unit='s')

    # plot with bad channels excluded, spatial_colors, zorder & pos. layout
    evoked.rename_channels({'MEG 0133': 'MEG 0000'})
    evoked.plot(exclude=evoked.info['bads'], spatial_colors=True, gfp=True,
                zorder='std', time_unit='s')
    evoked.plot(exclude=[], spatial_colors=True, zorder='unsorted',
                time_unit='s')
    assert_raises(TypeError, evoked.plot, zorder='asdf', time_unit='s')
    plt.close('all')

    evoked.plot_sensors()  # Test plot_sensors
    plt.close('all')

    evoked.pick_channels(evoked.ch_names[:4])
    with catch_logging() as log_file:
        evoked.plot(verbose=True, time_unit='s')
    assert_true('Need more than one' in log_file.getvalue())


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
                           ref_meg=True, exclude='bads')
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                        picks=picks, preload=True)
    evoked = epochs.average()
    evoked.plot_joint(times=[0.1])
    mne.viz.plot_compare_evokeds([evoked, evoked])

run_tests_if_main()
