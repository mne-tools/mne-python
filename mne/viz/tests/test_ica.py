# Authors: Denis Engemann <denis.engemann@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: Simplified BSD

import os.path as op

import numpy as np
from numpy.testing import assert_equal, assert_array_equal
import pytest
import matplotlib.pyplot as plt

from mne import read_events, Epochs, read_cov, pick_types
from mne.io import read_raw_fif
from mne.preprocessing import ICA, create_ecg_epochs, create_eog_epochs
from mne.utils import run_tests_if_main, requires_sklearn
from mne.viz.ica import _create_properties_layout, plot_ica_properties
from mne.viz.utils import _fake_click

base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
evoked_fname = op.join(base_dir, 'test-ave.fif')
raw_fname = op.join(base_dir, 'test_raw.fif')
cov_fname = op.join(base_dir, 'test-cov.fif')
event_name = op.join(base_dir, 'test-eve.fif')
event_id, tmin, tmax = 1, -0.1, 0.2
raw_ctf_fname = op.join(base_dir, 'test_ctf_raw.fif')


def _get_raw(preload=False):
    """Get raw data."""
    return read_raw_fif(raw_fname, preload=preload)


def _get_events():
    """Get events."""
    return read_events(event_name)


def _get_picks(raw):
    """Get picks."""
    return [0, 1, 2, 6, 7, 8, 12, 13, 14]  # take a only few channels


def _get_epochs():
    """Get epochs."""
    raw = _get_raw()
    events = _get_events()
    picks = _get_picks(raw)
    with pytest.warns(RuntimeWarning, match='projection'):
        epochs = Epochs(raw, events[:10], event_id, tmin, tmax, picks=picks)
    return epochs


@requires_sklearn
def test_plot_ica_components():
    """Test plotting of ICA solutions."""
    res = 8
    fast_test = {"res": res, "contours": 0, "sensors": False}
    raw = _get_raw()
    ica = ICA(noise_cov=read_cov(cov_fname), n_components=2,
              max_pca_components=3, n_pca_components=3)
    ica_picks = _get_picks(raw)
    with pytest.warns(RuntimeWarning, match='projection'):
        ica.fit(raw, picks=ica_picks)

    for components in [0, [0], [0, 1], [0, 1] * 2, None]:
        ica.plot_components(components, image_interp='bilinear',
                            colorbar=True, **fast_test)
    plt.close('all')

    # test interactive mode (passing 'inst' arg)
    ica.plot_components([0, 1], image_interp='bilinear', inst=raw, res=16)
    fig = plt.gcf()

    # test title click
    # ----------------
    lbl = fig.axes[1].get_label()
    ica_idx = int(lbl[-3:])
    titles = [ax.title for ax in fig.axes]
    title_pos_midpoint = (titles[1].get_window_extent().extents
                          .reshape((2, 2)).mean(axis=0))
    # first click adds to exclude
    _fake_click(fig, fig.axes[1], title_pos_midpoint, xform='pix')
    assert ica_idx in ica.exclude
    # clicking again removes from exclude
    _fake_click(fig, fig.axes[1], title_pos_midpoint, xform='pix')
    assert ica_idx not in ica.exclude

    # test topo click
    # ---------------
    _fake_click(fig, fig.axes[1], (0., 0.), xform='data')

    c_fig = plt.gcf()
    labels = [ax.get_label() for ax in c_fig.axes]

    for l in ['topomap', 'image', 'erp', 'spectrum', 'variance']:
        assert (l in labels)

    topomap_ax = c_fig.axes[labels.index('topomap')]
    title = topomap_ax.get_title()
    assert (lbl == title)

    ica.info = None
    with pytest.raises(RuntimeError, match='fit the ICA'):
        ica.plot_components(1, ch_type='mag')
    plt.close('all')


@requires_sklearn
def test_plot_ica_properties():
    """Test plotting of ICA properties."""
    res = 8
    raw = _get_raw(preload=True)
    raw.add_proj([], remove_existing=True)
    events = _get_events()
    picks = _get_picks(raw)[:6]
    pick_names = [raw.ch_names[k] for k in picks]
    raw.pick_channels(pick_names)
    reject = dict(grad=4000e-13, mag=4e-12)

    epochs = Epochs(raw, events[:10], event_id, tmin, tmax,
                    baseline=(None, 0), preload=True)

    ica = ICA(noise_cov=read_cov(cov_fname), n_components=2, max_iter=1,
              max_pca_components=2, n_pca_components=2, random_state=0)
    with pytest.warns(RuntimeWarning, match='projection'):
        ica.fit(raw)

    # test _create_properties_layout
    fig, ax = _create_properties_layout()
    assert_equal(len(ax), 5)

    topoargs = dict(topomap_args={'res': res, 'contours': 0, "sensors": False})
    ica.plot_properties(raw, picks=0, **topoargs)
    ica.plot_properties(epochs, picks=1, dB=False, plot_std=1.5, **topoargs)
    ica.plot_properties(epochs, picks=1, image_args={'sigma': 1.5},
                        topomap_args={'res': 10, 'colorbar': True},
                        psd_args={'fmax': 65.}, plot_std=False,
                        figsize=[4.5, 4.5], reject=reject)
    plt.close('all')

    pytest.raises(TypeError, ica.plot_properties, epochs, dB=list('abc'))
    pytest.raises(TypeError, ica.plot_properties, ica)
    pytest.raises(TypeError, ica.plot_properties, [0.2])
    pytest.raises(TypeError, plot_ica_properties, epochs, epochs)
    pytest.raises(TypeError, ica.plot_properties, epochs,
                  psd_args='not dict')
    pytest.raises(ValueError, ica.plot_properties, epochs, plot_std=[])

    fig, ax = plt.subplots(2, 3)
    ax = ax.ravel()[:-1]
    ica.plot_properties(epochs, picks=1, axes=ax, **topoargs)
    fig = ica.plot_properties(raw, picks=[0, 1], **topoargs)
    assert_equal(len(fig), 2)
    pytest.raises(TypeError, plot_ica_properties, epochs, ica, picks=[0, 1],
                  axes=ax)
    pytest.raises(ValueError, ica.plot_properties, epochs, axes='not axes')
    plt.close('all')

    # Test merging grads.
    pick_names = raw.ch_names[:15:2] + raw.ch_names[1:15:2]
    raw = _get_raw(preload=True).pick_channels(pick_names)
    raw.info.normalize_proj()
    ica = ICA(random_state=0, max_iter=1)
    with pytest.warns(UserWarning, match='did not converge'):
        ica.fit(raw)
    ica.plot_properties(raw)
    plt.close('all')

    # Test handling of zeros
    raw._data[:] = 0
    with pytest.warns(None):  # Usually UserWarning: Infinite value .* for epo
        ica.plot_properties(raw)
    ica = ICA(random_state=0, max_iter=1)
    epochs.pick_channels(pick_names)
    with pytest.warns(UserWarning, match='did not converge'):
        ica.fit(epochs)
    epochs._data[0] = 0
    with pytest.warns(None):  # Usually UserWarning: Infinite value .* for epo
        ica.plot_properties(epochs)
    plt.close('all')


@requires_sklearn
def test_plot_ica_sources():
    """Test plotting of ICA panel."""
    raw = read_raw_fif(raw_fname).crop(0, 1).load_data()
    picks = _get_picks(raw)
    epochs = _get_epochs()
    raw.pick_channels([raw.ch_names[k] for k in picks])
    ica_picks = pick_types(raw.info, meg=True, eeg=False, stim=False,
                           ecg=False, eog=False, exclude='bads')
    ica = ICA(n_components=2, max_pca_components=3, n_pca_components=3)
    ica.fit(raw, picks=ica_picks)
    ica.exclude = [1]
    fig = ica.plot_sources(raw)
    fig.canvas.key_press_event('escape')
    # Sadly close_event isn't called on Agg backend and the test always passes.
    assert_array_equal(ica.exclude, [1])
    plt.close('all')

    # dtype can change int->np.int after load, test it explicitly
    ica.n_components_ = np.int64(ica.n_components_)
    fig = ica.plot_sources(raw)
    # also test mouse clicks
    data_ax = fig.axes[0]
    assert len(plt.get_fignums()) == 1
    _fake_click(fig, data_ax, [-0.1, 0.9])  # click on y-label
    assert len(plt.get_fignums()) == 2
    ica.exclude = [1]
    ica.plot_sources(raw)

    raw.info['bads'] = ['MEG 0113']
    with pytest.raises(RuntimeError, match="Raw doesn't match fitted data"):
        ica.plot_sources(inst=raw)
    ica.plot_sources(epochs)
    epochs.info['bads'] = ['MEG 0113']
    with pytest.raises(RuntimeError, match="Epochs don't match fitted data"):
        ica.plot_sources(inst=epochs)
    epochs.info['bads'] = []
    ica.plot_sources(epochs.average())
    evoked = epochs.average()
    fig = ica.plot_sources(evoked)
    # Test a click
    ax = fig.get_axes()[0]
    line = ax.lines[0]
    _fake_click(fig, ax,
                [line.get_xdata()[0], line.get_ydata()[0]], 'data')
    _fake_click(fig, ax,
                [ax.get_xlim()[0], ax.get_ylim()[1]], 'data')
    # plot with bad channels excluded
    ica.exclude = [0]
    ica.plot_sources(evoked)
    ica.labels_ = dict(eog=[0])
    ica.labels_['eog/0/crazy-channel'] = [0]
    ica.plot_sources(evoked)  # now with labels
    with pytest.raises(ValueError, match='must be of Raw or Epochs type'):
        ica.plot_sources('meeow')
    plt.close('all')


@requires_sklearn
def test_plot_ica_overlay():
    """Test plotting of ICA cleaning."""
    raw = _get_raw(preload=True)
    picks = _get_picks(raw)
    ica = ICA(noise_cov=read_cov(cov_fname), n_components=2,
              max_pca_components=3, n_pca_components=3, random_state=0)
    # can't use info.normalize_proj here because of how and when ICA and Epochs
    # objects do picking of Raw data
    with pytest.warns(RuntimeWarning, match='projection'):
        ica.fit(raw, picks=picks)
    # don't test raw, needs preload ...
    with pytest.warns(RuntimeWarning, match='projection'):
        ecg_epochs = create_ecg_epochs(raw, picks=picks)
    ica.plot_overlay(ecg_epochs.average())
    with pytest.warns(RuntimeWarning, match='projection'):
        eog_epochs = create_eog_epochs(raw, picks=picks)
    ica.plot_overlay(eog_epochs.average())
    pytest.raises(TypeError, ica.plot_overlay, raw[:2, :3][0])
    pytest.raises(TypeError, ica.plot_overlay, raw, exclude=2)
    ica.plot_overlay(raw)
    plt.close('all')

    # smoke test for CTF
    raw = read_raw_fif(raw_ctf_fname)
    raw.apply_gradient_compensation(3)
    picks = pick_types(raw.info, meg=True, ref_meg=False)
    ica = ICA(n_components=2, max_pca_components=3, n_pca_components=3)
    ica.fit(raw, picks=picks)
    with pytest.warns(RuntimeWarning, match='longer than'):
        ecg_epochs = create_ecg_epochs(raw)
    ica.plot_overlay(ecg_epochs.average())
    plt.close('all')


@requires_sklearn
def test_plot_ica_scores():
    """Test plotting of ICA scores."""
    raw = _get_raw()
    picks = _get_picks(raw)
    ica = ICA(noise_cov=read_cov(cov_fname), n_components=2,
              max_pca_components=3, n_pca_components=3)
    with pytest.warns(RuntimeWarning, match='projection'):
        ica.fit(raw, picks=picks)
    ica.labels_ = dict()
    ica.labels_['eog/0/foo'] = 0
    ica.labels_['eog'] = 0
    ica.labels_['ecg'] = 1
    ica.plot_scores([0.3, 0.2], axhline=[0.1, -0.1])
    ica.plot_scores([0.3, 0.2], axhline=[0.1, -0.1], labels='foo')
    ica.plot_scores([0.3, 0.2], axhline=[0.1, -0.1], labels='eog')
    ica.plot_scores([0.3, 0.2], axhline=[0.1, -0.1], labels='ecg')
    pytest.raises(
        ValueError,
        ica.plot_scores,
        [0.3, 0.2], axhline=[0.1, -0.1], labels=['one', 'one-too-many'])
    pytest.raises(ValueError, ica.plot_scores, [0.2])
    plt.close('all')


@requires_sklearn
def test_plot_instance_components():
    """Test plotting of components as instances of raw and epochs."""
    raw = _get_raw()
    picks = _get_picks(raw)
    ica = ICA(noise_cov=read_cov(cov_fname), n_components=2,
              max_pca_components=3, n_pca_components=3)
    with pytest.warns(RuntimeWarning, match='projection'):
        ica.fit(raw, picks=picks)
    ica.exclude = [0]
    fig = ica.plot_sources(raw, title='Components')
    for key in ['down', 'up', 'right', 'left', 'o', '-', '+', '=', 'pageup',
                'pagedown', 'home', 'end', 'f11', 'b']:
        fig.canvas.key_press_event(key)
    ax = fig.get_axes()[0]
    line = ax.lines[0]
    _fake_click(fig, ax, [line.get_xdata()[0], line.get_ydata()[0]],
                'data')
    _fake_click(fig, ax, [-0.1, 0.9])  # click on y-label
    fig.canvas.key_press_event('escape')
    plt.close('all')
    epochs = _get_epochs()
    fig = ica.plot_sources(epochs, title='Components')
    for key in ['down', 'up', 'right', 'left', 'o', '-', '+', '=', 'pageup',
                'pagedown', 'home', 'end', 'f11', 'b']:
        fig.canvas.key_press_event(key)
    # Test a click
    ax = fig.get_axes()[0]
    line = ax.lines[0]
    _fake_click(fig, ax, [line.get_xdata()[0], line.get_ydata()[0]], 'data')
    _fake_click(fig, ax, [-0.1, 0.9])  # click on y-label
    fig.canvas.key_press_event('escape')
    plt.close('all')


run_tests_if_main()
