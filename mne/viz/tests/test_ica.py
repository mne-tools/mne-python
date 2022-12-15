# Authors: Denis Engemann <denis.engemann@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: Simplified BSD

import os.path as op
import sys

import numpy as np
from numpy.testing import assert_equal, assert_array_equal
import pytest
import matplotlib.pyplot as plt

from mne import (read_events, Epochs, read_cov, pick_types, Annotations,
                 make_fixed_length_events)
from mne.io import read_raw_fif
from mne.preprocessing import ICA, create_ecg_epochs, create_eog_epochs
from mne.utils import (requires_sklearn, catch_logging, _record_warnings)
from mne.viz.ica import _create_properties_layout, plot_ica_properties
from mne.viz.utils import _fake_click, _fake_keypress

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
    ica = ICA(noise_cov=read_cov(cov_fname), n_components=2)
    ica_picks = _get_picks(raw)
    with pytest.warns(RuntimeWarning, match='projection'):
        ica.fit(raw, picks=ica_picks)

    for components in [0, [0], [0, 1], [0, 1] * 2, None]:
        ica.plot_components(components, image_interp='cubic',
                            colorbar=True, **fast_test)
    plt.close('all')

    # test interactive mode (passing 'inst' arg)
    with catch_logging() as log:
        ica.plot_components([0, 1], image_interp='cubic', inst=raw, res=16,
                            verbose='debug', ch_type='grad')
    log = log.getvalue()
    assert 'grad data' in log
    assert 'extrapolation mode local to mean' in log
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

    for label in ['topomap', 'image', 'erp', 'spectrum', 'variance']:
        assert label in labels

    topomap_ax = c_fig.axes[labels.index('topomap')]
    title = topomap_ax.get_title()
    assert (lbl.split(' ')[0] == title.split(' ')[0])

    ica.info = None
    with pytest.raises(RuntimeError, match='fit the ICA'):
        ica.plot_components(1, ch_type='mag')


@pytest.mark.slowtest
@requires_sklearn
def test_plot_ica_properties():
    """Test plotting of ICA properties."""
    raw = _get_raw(preload=True).crop(0, 5)
    raw.add_proj([], remove_existing=True)
    with raw.info._unlock():
        raw.info['highpass'] = 1.0  # fake high-pass filtering
    events = make_fixed_length_events(raw)
    picks = _get_picks(raw)[:6]
    pick_names = [raw.ch_names[k] for k in picks]
    raw.pick_channels(pick_names)
    reject = dict(grad=4000e-13, mag=4e-12)

    epochs = Epochs(raw, events[:3], event_id, tmin, tmax,
                    baseline=(None, 0), preload=True)

    ica = ICA(noise_cov=read_cov(cov_fname), n_components=2, max_iter=1,
              random_state=0)
    with pytest.warns(RuntimeWarning, match='projection'):
        ica.fit(raw)

    # test _create_properties_layout
    fig, ax = _create_properties_layout()
    assert_equal(len(ax), 5)
    with pytest.raises(ValueError, match='specify both fig and figsize'):
        _create_properties_layout(figsize=(2, 2), fig=fig)

    topoargs = dict(topomap_args={'res': 4, 'contours': 0, "sensors": False})
    with catch_logging() as log:
        ica.plot_properties(raw, picks=0, verbose='debug', **topoargs)
    log = log.getvalue()
    assert raw.ch_names[0] == 'MEG 0113'
    assert 'extrapolation mode local to mean' in log, log
    ica.plot_properties(epochs, picks=1, dB=False, plot_std=1.5, **topoargs)
    fig = ica.plot_properties(epochs, picks=1, image_args={'sigma': 1.5},
                              topomap_args=dict(res=4, colorbar=True),
                              psd_args={'fmax': 65.}, plot_std=False,
                              log_scale=True, figsize=[4.5, 4.5],
                              reject=reject)[0]

    # test keypresses
    ax_labels = [ax.get_label() for ax in fig.axes]

    # test topomap change type
    ax = fig.axes[ax_labels.index('topomap')]
    assert ax.get_title() == 'ICA001 (mag)'
    _fake_keypress(fig, 't')
    assert ax.get_title() == 'ICA001 (grad)'
    _fake_keypress(fig, 't')
    assert ax.get_title() == 'ICA001 (mag)'

    # test log scale
    ax = fig.axes[ax_labels.index('spectrum')]
    assert ax.get_xscale() == 'log'
    _fake_keypress(fig, 'l')
    assert ax.get_xscale() == 'linear'
    _fake_keypress(fig, 'l')
    assert ax.get_xscale() == 'log'

    plt.close('all')

    with pytest.raises(TypeError, match='must be an instance'):
        ica.plot_properties(epochs, dB=list('abc'))
    with pytest.raises(TypeError, match='must be an instance'):
        ica.plot_properties(ica)
    with pytest.raises(TypeError, match='must be an instance'):
        ica.plot_properties([0.2])
    with pytest.raises(TypeError, match='must be an instance'):
        plot_ica_properties(epochs, epochs)
    with pytest.raises(TypeError, match='must be an instance'):
        ica.plot_properties(epochs, psd_args='not dict')
    with pytest.raises(TypeError, match='must be an instance'):
        ica.plot_properties(epochs, plot_std=[])

    fig, ax = plt.subplots(2, 3)
    ax = ax.ravel()[:-1]
    ica.plot_properties(epochs, picks=1, axes=ax, **topoargs)
    pytest.raises(TypeError, plot_ica_properties, epochs, ica, picks=[0, 1],
                  axes=ax)
    pytest.raises(ValueError, ica.plot_properties, epochs, axes='not axes')
    plt.close('all')

    # Test merging grads.
    pick_names = raw.ch_names[:15:2] + raw.ch_names[1:15:2]
    raw = _get_raw(preload=True).pick_channels(pick_names).crop(0, 5)
    raw.info.normalize_proj()
    ica = ICA(random_state=0, max_iter=1)
    with pytest.warns(UserWarning, match='did not converge'):
        ica.fit(raw)
    ica.plot_properties(raw)
    plt.close('all')

    # Test handling of zeros
    ica = ICA(random_state=0, max_iter=1)
    epochs.pick_channels(pick_names)
    with pytest.warns(UserWarning, match='did not converge'):
        ica.fit(epochs)
    epochs._data[0] = 0
    # Usually UserWarning: Infinite value .* for epo
    with _record_warnings():
        ica.plot_properties(epochs, **topoargs)
    plt.close('all')

    # Test Raw with annotations
    annot = Annotations(onset=[1], duration=[1], description=['BAD'])
    raw_annot = _get_raw(preload=True).set_annotations(annot).crop(0, 8)
    raw_annot.pick(np.arange(10))
    raw_annot.del_proj()

    with pytest.warns(UserWarning, match='did not converge'):
        ica.fit(raw_annot)
    # drop bad data segments
    fig = ica.plot_properties(raw_annot, picks=[0, 1], **topoargs)
    assert_equal(len(fig), 2)
    # don't drop
    ica.plot_properties(raw_annot, reject_by_annotation=False, **topoargs)


@requires_sklearn
def test_plot_ica_sources(raw_orig, browser_backend, monkeypatch):
    """Test plotting of ICA panel."""
    raw = raw_orig.copy().crop(0, 1)
    picks = _get_picks(raw)
    epochs = _get_epochs()
    raw.pick_channels([raw.ch_names[k] for k in picks])
    ica_picks = pick_types(raw.info, meg=True, eeg=False, stim=False,
                           ecg=False, eog=False, exclude='bads')
    ica = ICA(n_components=2)
    ica.fit(raw, picks=ica_picks)
    ica.exclude = [1]
    if sys.platform == 'darwin':  # unknown transformation bug
        monkeypatch.setenv('MNE_BROWSE_RAW_SIZE', '20,20')
    fig = ica.plot_sources(raw)
    assert browser_backend._get_n_figs() == 1
    # change which component is in ICA.exclude (click data trace to remove
    # current one; click name to add other one)
    fig._redraw()
    assert_array_equal(ica.exclude, [1])
    assert fig.mne.info['bads'] == [ica._ica_names[1]]
    x = fig.mne.traces[1].get_xdata()[5]
    y = fig.mne.traces[1].get_ydata()[5]
    fig._fake_click((x, y), xform='data')  # exclude = []
    assert fig.mne.info['bads'] == []
    assert_array_equal(ica.exclude, [1])  # unchanged
    fig._click_ch_name(ch_index=0, button=1)  # exclude = [0]
    assert fig.mne.info['bads'] == [ica._ica_names[0]]
    assert_array_equal(ica.exclude, [1])
    fig._fake_keypress(fig.mne.close_key)
    fig._close_event()
    assert browser_backend._get_n_figs() == 0
    assert_array_equal(ica.exclude, [0])
    # test when picks does not include ica.exclude.
    ica.plot_sources(raw, picks=[1])
    assert browser_backend._get_n_figs() == 1
    browser_backend._close_all()

    # dtype can change int->np.int64 after load, test it explicitly
    ica.n_components_ = np.int64(ica.n_components_)

    # test clicks on y-label (need >2 secs for plot_properties() to work)
    long_raw = raw_orig.crop(0, 5)
    fig = ica.plot_sources(long_raw)
    assert browser_backend._get_n_figs() == 1
    fig._redraw()
    fig._click_ch_name(ch_index=0, button=3)
    assert len(fig.mne.child_figs) == 1
    assert browser_backend._get_n_figs() == 2
    # close child fig directly (workaround for mpl issue #18609)
    fig._fake_keypress('escape', fig=fig.mne.child_figs[0])
    assert browser_backend._get_n_figs() == 1
    fig._fake_keypress(fig.mne.close_key)
    assert browser_backend._get_n_figs() == 0
    del long_raw

    # test with annotations
    orig_annot = raw.annotations
    raw.set_annotations(Annotations([0.2], [0.1], 'Test'))
    fig = ica.plot_sources(raw)
    if browser_backend.name == 'matplotlib':
        assert len(fig.mne.ax_main.collections) == 1
        assert len(fig.mne.ax_hscroll.collections) == 1
    else:
        assert len(fig.mne.regions) == 1
    raw.set_annotations(orig_annot)

    # test error handling
    raw_ = raw.copy().load_data()
    raw_.drop_channels('MEG 0113')
    with pytest.raises(RuntimeError, match="Raw doesn't match fitted data"), \
         pytest.warns(RuntimeWarning, match='could not be picked'):
        ica.plot_sources(inst=raw_)
    epochs_ = epochs.copy().load_data()
    epochs_.drop_channels('MEG 0113')
    with pytest.raises(RuntimeError, match="Epochs don't match fitted data"), \
         pytest.warns(RuntimeWarning, match='could not be picked'):
        ica.plot_sources(inst=epochs_)
    del raw_
    del epochs_

    # test w/ epochs and evokeds
    ica.plot_sources(epochs)
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

    # pretend find_bads_eog() yielded some results
    ica.labels_ = {
        'eog': [0],
        'eog/0/crazy-channel': [0]
    }
    ica.plot_sources(evoked)  # now with labels

    # pass an invalid inst
    with pytest.raises(ValueError, match='must be of Raw or Epochs type'):
        ica.plot_sources('meeow')


@pytest.mark.slowtest
@requires_sklearn
def test_plot_ica_overlay():
    """Test plotting of ICA cleaning."""
    raw = _get_raw(preload=True)
    with raw.info._unlock():
        raw.info['highpass'] = 1.0  # fake high-pass filtering
    picks = _get_picks(raw)
    ica = ICA(noise_cov=read_cov(cov_fname), n_components=2, random_state=0)
    # overlay plotting requires a fitted ICA
    with pytest.raises(RuntimeError, match='need to fit'):
        ica.plot_overlay(inst=raw)
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
    ica.plot_overlay(eog_epochs.average(), n_pca_components=2)
    pytest.raises(TypeError, ica.plot_overlay, raw[:2, :3][0])
    pytest.raises(TypeError, ica.plot_overlay, raw, exclude=2)
    ica.plot_overlay(raw)
    plt.close('all')

    # smoke test for CTF
    raw = read_raw_fif(raw_ctf_fname)
    raw.apply_gradient_compensation(3)
    with raw.info._unlock():
        raw.info['highpass'] = 1.0  # fake high-pass filtering
    picks = pick_types(raw.info, meg=True, ref_meg=False)
    ica = ICA(n_components=2, )
    ica.fit(raw, picks=picks)
    with pytest.warns(RuntimeWarning, match='longer than'):
        ecg_epochs = create_ecg_epochs(raw)
    ica.plot_overlay(ecg_epochs.average())


def _get_geometry(fig):
    try:
        return fig.axes[0].get_subplotspec().get_geometry()  # pragma: no cover
    except AttributeError:  # MPL < 3.4 (probably)
        return fig.axes[0].get_geometry()  # pragma: no cover


@requires_sklearn
def test_plot_ica_scores():
    """Test plotting of ICA scores."""
    raw = _get_raw()
    picks = _get_picks(raw)
    ica = ICA(noise_cov=read_cov(cov_fname), n_components=2)
    with pytest.warns(RuntimeWarning, match='projection'):
        ica.fit(raw, picks=picks)
    ica.plot_scores([0.3, 0.2], axhline=[0.1, -0.1], figsize=(6.4, 2.7))
    ica.plot_scores([[0.3, 0.2], [0.3, 0.2]], axhline=[0.1, -0.1])

    # check labels
    ica.labels_ = dict()
    ica.labels_['eog'] = 0
    ica.labels_['ecg'] = 1
    ica.plot_scores([0.3, 0.2], axhline=[0.1, -0.1], labels='eog')
    ica.plot_scores([0.3, 0.2], axhline=[0.1, -0.1], labels='ecg')
    ica.labels_['eog/0/foo'] = 0
    ica.labels_['ecg/1/bar'] = 0
    ica.plot_scores([0.3, 0.2], axhline=[0.1, -0.1], labels='foo')
    ica.plot_scores([0.3, 0.2], axhline=[0.1, -0.1], labels='eog')
    ica.plot_scores([0.3, 0.2], axhline=[0.1, -0.1], labels='ecg')

    # check setting number of columns
    fig = ica.plot_scores([[0.3, 0.2], [0.3, 0.2], [0.3, 0.2]],
                          axhline=[0.1, -0.1])
    assert 2 == _get_geometry(fig)[1]
    fig = ica.plot_scores([[0.3, 0.2], [0.3, 0.2]], axhline=[0.1, -0.1],
                          n_cols=1)
    assert 1 == _get_geometry(fig)[1]

    # only use 1 column (even though 2 were requested)
    fig = ica.plot_scores([0.3, 0.2], axhline=[0.1, -0.1], n_cols=2)
    assert 1 == _get_geometry(fig)[1]

    with pytest.raises(ValueError, match='Need as many'):
        ica.plot_scores([0.3, 0.2], axhline=[0.1, -0.1],
                        labels=['one', 'one-too-many'])
    with pytest.raises(ValueError, match='The length of'):
        ica.plot_scores([0.2])


@requires_sklearn
def test_plot_instance_components(browser_backend):
    """Test plotting of components as instances of raw and epochs."""
    raw = _get_raw()
    picks = _get_picks(raw)
    ica = ICA(noise_cov=read_cov(cov_fname), n_components=2)
    with pytest.warns(RuntimeWarning, match='projection'):
        ica.fit(raw, picks=picks)
    ica.exclude = [0]
    fig = ica.plot_sources(raw, title='Components')
    keys = ('home', 'home', 'end', 'down', 'up', 'right', 'left', '-', '+',
            '=', 'd', 'd', 'pageup', 'pagedown', 'z', 'z', 's', 's', 'b')
    for key in keys:
        fig._fake_keypress(key)
    x = fig.mne.traces[0].get_xdata()[0]
    y = fig.mne.traces[0].get_ydata()[0]
    fig._fake_click((x, y), xform='data')
    fig._click_ch_name(ch_index=0, button=1)
    fig._fake_keypress('escape')
    browser_backend._close_all()

    epochs = _get_epochs()
    fig = ica.plot_sources(epochs, title='Components')
    for key in keys:
        fig._fake_keypress(key)
    # Test a click
    x = fig.mne.traces[0].get_xdata()[0]
    y = fig.mne.traces[0].get_ydata()[0]
    fig._fake_click((x, y), xform='data')
    fig._click_ch_name(ch_index=0, button=1)
    fig._fake_keypress('escape')
