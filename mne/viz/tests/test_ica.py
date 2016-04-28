# Authors: Denis Engemann <denis.engemann@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: Simplified BSD

import os.path as op
import warnings

from numpy.testing import assert_raises

from mne import io, read_events, Epochs, read_cov
from mne import pick_types
from mne.utils import run_tests_if_main, requires_sklearn
from mne.viz.utils import _fake_click
from mne.preprocessing import ICA, create_ecg_epochs, create_eog_epochs

# Set our plotters to test mode
import matplotlib
matplotlib.use('Agg')  # for testing don't use X server

warnings.simplefilter('always')  # enable b/c these tests throw warnings

base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
evoked_fname = op.join(base_dir, 'test-ave.fif')
raw_fname = op.join(base_dir, 'test_raw.fif')
cov_fname = op.join(base_dir, 'test-cov.fif')
event_name = op.join(base_dir, 'test-eve.fif')
event_id, tmin, tmax = 1, -0.1, 0.2


def _get_raw(preload=False):
    return io.read_raw_fif(raw_fname, preload=preload)


def _get_events():
    return read_events(event_name)


def _get_picks(raw):
    return [0, 1, 2, 6, 7, 8, 12, 13, 14]  # take a only few channels


def _get_epochs():
    raw = _get_raw()
    events = _get_events()
    picks = _get_picks(raw)
    with warnings.catch_warnings(record=True):  # bad proj
        epochs = Epochs(raw, events[:10], event_id, tmin, tmax, picks=picks,
                        baseline=(None, 0))
    return epochs


@requires_sklearn
def test_plot_ica_components():
    """Test plotting of ICA solutions
    """
    import matplotlib.pyplot as plt
    raw = _get_raw()
    ica = ICA(noise_cov=read_cov(cov_fname), n_components=2,
              max_pca_components=3, n_pca_components=3)
    ica_picks = _get_picks(raw)
    with warnings.catch_warnings(record=True):
        ica.fit(raw, picks=ica_picks)
    warnings.simplefilter('always', UserWarning)
    with warnings.catch_warnings(record=True):
        for components in [0, [0], [0, 1], [0, 1] * 2, None]:
            ica.plot_components(components, image_interp='bilinear', res=16)
    ica.info = None
    assert_raises(ValueError, ica.plot_components, 1)
    assert_raises(RuntimeError, ica.plot_components, 1, ch_type='mag')
    plt.close('all')


@requires_sklearn
def test_plot_ica_sources():
    """Test plotting of ICA panel
    """
    import matplotlib.pyplot as plt
    raw = io.read_raw_fif(raw_fname,
                          preload=False).crop(0, 1, copy=False).load_data()
    picks = _get_picks(raw)
    epochs = _get_epochs()
    raw.pick_channels([raw.ch_names[k] for k in picks])
    ica_picks = pick_types(raw.info, meg=True, eeg=False, stim=False,
                           ecg=False, eog=False, exclude='bads')
    ica = ICA(n_components=2, max_pca_components=3, n_pca_components=3)
    ica.fit(raw, picks=ica_picks)
    raw.info['bads'] = ['MEG 0113']
    assert_raises(RuntimeError, ica.plot_sources, inst=raw)
    ica.plot_sources(epochs)
    epochs.info['bads'] = ['MEG 0113']
    assert_raises(RuntimeError, ica.plot_sources, inst=epochs)
    epochs.info['bads'] = []
    with warnings.catch_warnings(record=True):  # no labeled objects mpl
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
        ica.plot_sources(evoked, exclude=[0])
        ica.exclude = [0]
        ica.plot_sources(evoked)  # does the same thing
        ica.labels_ = dict(eog=[0])
        ica.labels_['eog/0/crazy-channel'] = [0]
        ica.plot_sources(evoked)  # now with labels
    assert_raises(ValueError, ica.plot_sources, 'meeow')
    plt.close('all')


@requires_sklearn
def test_plot_ica_overlay():
    """Test plotting of ICA cleaning
    """
    import matplotlib.pyplot as plt
    raw = _get_raw(preload=True)
    picks = _get_picks(raw)
    ica = ICA(noise_cov=read_cov(cov_fname), n_components=2,
              max_pca_components=3, n_pca_components=3)
    # can't use info.normalize_proj here because of how and when ICA and Epochs
    # objects do picking of Raw data
    with warnings.catch_warnings(record=True):  # bad proj
        ica.fit(raw, picks=picks)
    # don't test raw, needs preload ...
    with warnings.catch_warnings(record=True):  # bad proj
        ecg_epochs = create_ecg_epochs(raw, picks=picks)
    ica.plot_overlay(ecg_epochs.average())
    with warnings.catch_warnings(record=True):  # bad proj
        eog_epochs = create_eog_epochs(raw, picks=picks)
    ica.plot_overlay(eog_epochs.average())
    assert_raises(ValueError, ica.plot_overlay, raw[:2, :3][0])
    ica.plot_overlay(raw)
    plt.close('all')


@requires_sklearn
def test_plot_ica_scores():
    """Test plotting of ICA scores
    """
    import matplotlib.pyplot as plt
    raw = _get_raw()
    picks = _get_picks(raw)
    ica = ICA(noise_cov=read_cov(cov_fname), n_components=2,
              max_pca_components=3, n_pca_components=3)
    with warnings.catch_warnings(record=True):  # bad proj
        ica.fit(raw, picks=picks)
    ica.labels_ = dict()
    ica.labels_['eog/0/foo'] = 0
    ica.labels_['eog'] = 0
    ica.labels_['ecg'] = 1
    ica.plot_scores([0.3, 0.2], axhline=[0.1, -0.1])
    ica.plot_scores([0.3, 0.2], axhline=[0.1, -0.1], labels='foo')
    ica.plot_scores([0.3, 0.2], axhline=[0.1, -0.1], labels='eog')
    ica.plot_scores([0.3, 0.2], axhline=[0.1, -0.1], labels='ecg')
    assert_raises(
        ValueError,
        ica.plot_scores,
        [0.3, 0.2], axhline=[0.1, -0.1], labels=['one', 'one-too-many'])
    assert_raises(ValueError, ica.plot_scores, [0.2])
    plt.close('all')


@requires_sklearn
def test_plot_instance_components():
    """Test plotting of components as instances of raw and epochs."""
    import matplotlib.pyplot as plt
    raw = _get_raw()
    picks = _get_picks(raw)
    ica = ICA(noise_cov=read_cov(cov_fname), n_components=2,
              max_pca_components=3, n_pca_components=3)
    with warnings.catch_warnings(record=True):  # bad proj
        ica.fit(raw, picks=picks)
    fig = ica.plot_sources(raw, exclude=[0], title='Components')
    fig.canvas.key_press_event('down')
    fig.canvas.key_press_event('up')
    fig.canvas.key_press_event('right')
    fig.canvas.key_press_event('left')
    fig.canvas.key_press_event('o')
    fig.canvas.key_press_event('-')
    fig.canvas.key_press_event('+')
    fig.canvas.key_press_event('=')
    fig.canvas.key_press_event('pageup')
    fig.canvas.key_press_event('pagedown')
    fig.canvas.key_press_event('home')
    fig.canvas.key_press_event('end')
    fig.canvas.key_press_event('f11')
    ax = fig.get_axes()[0]
    line = ax.lines[0]
    _fake_click(fig, ax, [line.get_xdata()[0], line.get_ydata()[0]], 'data')
    _fake_click(fig, ax, [-0.1, 0.9])  # click on y-label
    fig.canvas.key_press_event('escape')
    plt.close('all')
    epochs = _get_epochs()
    fig = ica.plot_sources(epochs, exclude=[0], title='Components')
    fig.canvas.key_press_event('down')
    fig.canvas.key_press_event('up')
    fig.canvas.key_press_event('right')
    fig.canvas.key_press_event('left')
    fig.canvas.key_press_event('o')
    fig.canvas.key_press_event('-')
    fig.canvas.key_press_event('+')
    fig.canvas.key_press_event('=')
    fig.canvas.key_press_event('pageup')
    fig.canvas.key_press_event('pagedown')
    fig.canvas.key_press_event('home')
    fig.canvas.key_press_event('end')
    fig.canvas.key_press_event('f11')
    # Test a click
    ax = fig.get_axes()[0]
    line = ax.lines[0]
    _fake_click(fig, ax, [line.get_xdata()[0], line.get_ydata()[0]], 'data')
    _fake_click(fig, ax, [-0.1, 0.9])  # click on y-label
    fig.canvas.key_press_event('escape')
    plt.close('all')


run_tests_if_main()
