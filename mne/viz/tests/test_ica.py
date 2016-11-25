# Authors: Denis Engemann <denis.engemann@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: Simplified BSD

import os.path as op
import warnings

from numpy.testing import assert_raises, assert_equal, assert_array_equal
from nose.tools import assert_true

from mne import read_events, Epochs, read_cov, pick_types
from mne.io import read_raw_fif
from mne.preprocessing import ICA, create_ecg_epochs, create_eog_epochs
from mne.utils import run_tests_if_main, requires_sklearn
from mne.viz.ica import _create_properties_layout, plot_ica_properties
from mne.viz.utils import _fake_click

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
    with warnings.catch_warnings(record=True):  # bad proj
        epochs = Epochs(raw, events[:10], event_id, tmin, tmax, picks=picks)
    return epochs


@requires_sklearn
def test_plot_ica_components():
    """Test plotting of ICA solutions."""
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
            ica.plot_components(components, image_interp='bilinear', res=16,
                                colorbar=True)

        # test interactive mode (passing 'inst' arg)
        plt.close('all')
        ica.plot_components([0, 1], image_interp='bilinear', res=16, inst=raw)

        fig = plt.gcf()
        ax = [a for a in fig.get_children() if isinstance(a, plt.Axes)]
        lbl = ax[1].get_label()
        _fake_click(fig, ax[1], (0., 0.), xform='data')

        c_fig = plt.gcf()
        ax = [a for a in c_fig.get_children() if isinstance(a, plt.Axes)]
        labels = [a.get_label() for a in ax]

        for l in ['topomap', 'image', 'erp', 'spectrum', 'variance']:
            assert_true(l in labels)

        topomap_ax = ax[labels.index('topomap')]
        title = topomap_ax.get_title()
        assert_true(lbl == title)

    ica.info = None
    assert_raises(ValueError, ica.plot_components, 1)
    assert_raises(RuntimeError, ica.plot_components, 1, ch_type='mag')
    plt.close('all')


@requires_sklearn
def test_plot_ica_properties():
    """Test plotting of ICA properties."""
    import matplotlib.pyplot as plt

    raw = _get_raw(preload=True)
    raw.add_proj([], remove_existing=True)
    events = _get_events()
    picks = _get_picks(raw)[:6]
    pick_names = [raw.ch_names[k] for k in picks]
    raw.pick_channels(pick_names)

    with warnings.catch_warnings(record=True):  # bad proj
        epochs = Epochs(raw, events[:10], event_id, tmin, tmax,
                        baseline=(None, 0), preload=True)

    ica = ICA(noise_cov=read_cov(cov_fname), n_components=2,
              max_pca_components=2, n_pca_components=2)
    with warnings.catch_warnings(record=True):  # bad proj
        ica.fit(raw)

    # test _create_properties_layout
    fig, ax = _create_properties_layout()
    assert_equal(len(ax), 5)

    topoargs = dict(topomap_args={'res': 10})
    ica.plot_properties(raw, picks=0, **topoargs)
    ica.plot_properties(epochs, picks=1, dB=False, plot_std=1.5, **topoargs)
    ica.plot_properties(epochs, picks=1, image_args={'sigma': 1.5},
                        topomap_args={'res': 10, 'colorbar': True},
                        psd_args={'fmax': 65.}, plot_std=False,
                        figsize=[4.5, 4.5])
    plt.close('all')

    assert_raises(ValueError, ica.plot_properties, epochs, dB=list('abc'))
    assert_raises(ValueError, ica.plot_properties, epochs, plot_std=[])
    assert_raises(ValueError, ica.plot_properties, ica)
    assert_raises(ValueError, ica.plot_properties, [0.2])
    assert_raises(ValueError, plot_ica_properties, epochs, epochs)
    assert_raises(ValueError, ica.plot_properties, epochs,
                  psd_args='not dict')

    fig, ax = plt.subplots(2, 3)
    ax = ax.ravel()[:-1]
    ica.plot_properties(epochs, picks=1, axes=ax)
    fig = ica.plot_properties(raw, picks=[0, 1], **topoargs)
    assert_equal(len(fig), 2)
    assert_raises(ValueError, plot_ica_properties, epochs, ica, picks=[0, 1],
                  axes=ax)
    assert_raises(ValueError, ica.plot_properties, epochs, axes='not axes')
    plt.close('all')


@requires_sklearn
def test_plot_ica_sources():
    """Test plotting of ICA panel."""
    import matplotlib.pyplot as plt
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
    """Test plotting of ICA cleaning."""
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
    """Test plotting of ICA scores."""
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
