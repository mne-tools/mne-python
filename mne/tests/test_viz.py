import os.path as op
from functools import wraps
import numpy as np
from numpy.testing import assert_raises
from nose.tools import assert_true, assert_equal
import warnings

from mne import io, read_events, Epochs, SourceEstimate, read_cov, read_proj
from mne import make_field_map, pick_types
from mne.layouts import read_layout
from mne.pick import pick_channels_evoked
from mne.viz import (plot_topo, plot_topo_tfr, plot_topo_power,
                     plot_topo_phase_lock, plot_topo_image_epochs,
                     plot_evoked_topomap, plot_projs_topomap,
                     plot_sparse_source_estimates, plot_source_estimates,
                     plot_cov, mne_analyze_colormap, plot_image_epochs,
                     plot_connectivity_circle, circular_layout, plot_drop_log,
                     compare_fiff, plot_source_spectrogram, plot_events)
from mne.datasets import sample
from mne.source_space import read_source_spaces
from mne.preprocessing import ICA
from mne.constants import FIFF
from mne.utils import check_sklearn_version


warnings.simplefilter('always')  # enable b/c these tests throw warnings

# Set our plotters to test mode
import matplotlib
matplotlib.use('Agg')  # for testing don't use X server
import matplotlib.pyplot as plt

lacks_mayavi = False
try:
    from mayavi import mlab
except ImportError:
    try:
        from enthought.mayavi import mlab
    except ImportError:
        lacks_mayavi = True
requires_mayavi = np.testing.dec.skipif(lacks_mayavi, 'Requires mayavi')


def requires_sklearn(function):
    """Decorator to skip test if scikit-learn >= 0.12 is not available"""
    @wraps(function)
    def dec(*args, **kwargs):
        if not check_sklearn_version(min_version='0.12'):
            from nose.plugins.skip import SkipTest
            raise SkipTest('Test %s skipped, requires scikit-learn >= 0.12'
                           % function.__name__)
        ret = function(*args, **kwargs)
        return ret
    return dec

if not lacks_mayavi:
    mlab.options.backend = 'test'

data_dir = sample.data_path(download=False)
subjects_dir = op.join(data_dir, 'subjects')
ecg_fname = op.join(data_dir, 'MEG', 'sample', 'sample_audvis_ecg_proj.fif')

base_dir = op.join(op.dirname(__file__), '..', 'io', 'tests', 'data')
evoked_fname = op.join(base_dir, 'test-ave.fif')
fname = op.join(base_dir, 'test-ave.fif')
raw_fname = op.join(base_dir, 'test_raw.fif')
cov_fname = op.join(base_dir, 'test-cov.fif')
event_name = op.join(base_dir, 'test-eve.fif')
event_id, tmin, tmax = 1, -0.2, 0.5
n_chan = 15
layout = read_layout('Vectorview-all')


def _fake_click(fig, ax, point, xform='ax'):
    """Helper to fake a click at a relative point within axes"""
    if xform == 'ax':
        x, y = ax.transAxes.transform_point(point)
    elif xform == 'data':
        x, y = ax.transData.transform_point(point)
    else:
        raise ValueError('unknown transform')
    try:
        fig.canvas.button_press_event(x, y, 1, False, None)
    except:  # for old MPL
        fig.canvas.button_press_event(x, y, 1, False)


def _get_raw():
    return io.Raw(raw_fname, preload=False)


def _get_events():
    return read_events(event_name)


def _get_picks(raw):
    return pick_types(raw.info, meg=True, eeg=False, stim=False,
                           ecg=False, eog=False, exclude='bads')


def _get_epochs():
    raw = _get_raw()
    events = _get_events()
    picks = _get_picks(raw)
    # Use a subset of channels for plotting speed
    picks = np.round(np.linspace(0, len(picks) + 1, n_chan)).astype(int)
    epochs = Epochs(raw, events[:10], event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0))
    return epochs


def _get_epochs_delayed_ssp():
    raw = _get_raw()
    events = _get_events()
    picks = _get_picks(raw)
    reject = dict(mag=4e-12)
    epochs_delayed_ssp = Epochs(raw, events[:10], event_id, tmin, tmax,
                                picks=picks, baseline=(None, 0),
                                proj='delayed', reject=reject)
    return epochs_delayed_ssp


def test_plot_topo():
    """Test plotting of ERP topography
    """
    # Show topography
    evoked = _get_epochs().average()
    plot_topo(evoked, layout)
    warnings.simplefilter('always', UserWarning)
    picked_evoked = pick_channels_evoked(evoked, evoked.ch_names[:3])

    # test scaling
    with warnings.catch_warnings(record=True):
        for ylim in [dict(mag=[-600, 600]), None]:
            plot_topo([picked_evoked] * 2, layout, ylim=ylim)

        for evo in [evoked, [evoked, picked_evoked]]:
            assert_raises(ValueError, plot_topo, evo, layout, color=['y', 'b'])

        evoked_delayed_ssp = _get_epochs_delayed_ssp().average()
        ch_names = evoked_delayed_ssp.ch_names[:3]  # make it faster
        picked_evoked_delayed_ssp = pick_channels_evoked(evoked_delayed_ssp,
                                                         ch_names)
        plot_topo(picked_evoked_delayed_ssp, layout, proj='interactive')


def test_plot_topo_tfr():
    """Test plotting of TFR
    """
    # Make a fake dataset to plot
    epochs = _get_epochs()
    n_freqs = 11
    con = np.random.randn(n_chan, n_freqs, len(epochs.times))
    freqs = np.arange(n_freqs)
    # Show topography of connectivity from seed
    plot_topo_tfr(epochs, con, freqs, layout)
    plt.close('all')


def test_plot_topo_power():
    """Test plotting of power
    """
    epochs = _get_epochs()
    decim = 3
    frequencies = np.arange(7, 30, 3)  # define frequencies of interest
    power = np.abs(np.random.randn(n_chan, 7, 141))
    phase_lock = np.random.randn(n_chan, 7, 141)
    baseline = (None, 0)  # set the baseline for induced power
    title = 'Induced power - MNE sample data'
    plot_topo_power(epochs, power, frequencies, layout, baseline=baseline,
                    mode='ratio', decim=decim, vmin=0., vmax=14, title=title)
    title = 'Phase locking value - MNE sample data'
    plot_topo_phase_lock(epochs, phase_lock, frequencies, layout,
                         baseline=baseline, mode='mean', decim=decim,
                         title=title)
    plt.close('all')


def test_plot_topo_image_epochs():
    """Test plotting of epochs image topography
    """
    title = 'ERF images - MNE sample data'
    epochs = _get_epochs()
    plot_topo_image_epochs(epochs, layout, sigma=0.5, vmin=-200, vmax=200,
                           colorbar=True, title=title)
    plt.close('all')


def test_plot_evoked():
    """Test plotting of evoked
    """
    evoked = _get_epochs().average()
    with warnings.catch_warnings(record=True):
        evoked.plot(proj=True, hline=[1])
        # plot with bad channels excluded
        evoked.plot(exclude='bads')
        evoked.plot(exclude=evoked.info['bads'])  # does the same thing

        # test selective updating of dict keys is working.
        evoked.plot(hline=[1], units=dict(mag='femto foo'))
        evoked_delayed_ssp = _get_epochs_delayed_ssp().average()
        evoked_delayed_ssp.plot(proj='interactive')
        evoked_delayed_ssp.apply_proj()
        assert_raises(RuntimeError, evoked_delayed_ssp.plot,
                      proj='interactive')
        evoked_delayed_ssp.info['projs'] = []
        assert_raises(RuntimeError, evoked_delayed_ssp.plot,
                      proj='interactive')
        assert_raises(RuntimeError, evoked_delayed_ssp.plot,
                      proj='interactive', axes='foo')
        plt.close('all')


def test_plot_epochs():
    """ Test plotting epochs
    """
    epochs = _get_epochs()
    epochs.plot([0, 1], picks=[0, 2, 3], scalings=None, title_str='%s')
    epochs[0].plot(picks=[0, 2, 3], scalings=None, title_str='%s')
    # test clicking: should increase coverage on
    # 3200-3226, 3235, 3237, 3239-3242, 3245-3255, 3260-3280
    fig = plt.gcf()
    fig.canvas.button_press_event(10, 10, 'left')
    # now let's add a bad channel
    epochs.info['bads'] = [epochs.ch_names[0]]  # include a bad one
    epochs.plot([0, 1], picks=[0, 2, 3], scalings=None, title_str='%s')
    epochs[0].plot(picks=[0, 2, 3], scalings=None, title_str='%s')
    plt.close('all')


@sample.requires_sample_data
@requires_mayavi
def test_plot_sparse_source_estimates():
    """Test plotting of (sparse) source estimates
    """
    sample_src = read_source_spaces(op.join(data_dir, 'subjects', 'sample',
                                            'bem', 'sample-oct-6-src.fif'))

    # dense version
    vertices = [s['vertno'] for s in sample_src]
    n_time = 5
    n_verts = sum(len(v) for v in vertices)
    stc_data = np.zeros((n_verts * n_time))
    stc_data[(np.random.rand(20) * n_verts * n_time).astype(int)] = 1
    stc_data.shape = (n_verts, n_time)
    stc = SourceEstimate(stc_data, vertices, 1, 1)
    colormap = mne_analyze_colormap(format='matplotlib')
    # don't really need to test matplotlib method since it's not used now...
    colormap = mne_analyze_colormap()
    plot_source_estimates(stc, 'sample', colormap=colormap,
                          config_opts={'background': (1, 1, 0)},
                          subjects_dir=subjects_dir, colorbar=True)
    assert_raises(TypeError, plot_source_estimates, stc, 'sample',
                  figure='foo', hemi='both')

    # now do sparse version
    vertices = sample_src[0]['vertno']
    n_verts = len(vertices)
    stc_data = np.zeros((n_verts * n_time))
    stc_data[(np.random.rand(20) * n_verts * n_time).astype(int)] = 1
    stc_data.shape = (n_verts, n_time)
    inds = np.where(np.any(stc_data, axis=1))[0]
    stc_data = stc_data[inds]
    vertices = [vertices[inds], np.empty(0, dtype=np.int)]
    stc = SourceEstimate(stc_data, vertices, 1, 1)
    plot_sparse_source_estimates(sample_src, stc, bgcolor=(1, 1, 1),
                                 opacity=0.5, high_resolution=True)


def test_plot_cov():
    """Test plotting of covariances
    """
    raw = _get_raw()
    cov = read_cov(cov_fname)
    fig1, fig2 = plot_cov(cov, raw.info, proj=True)
    plt.close('all')


@requires_sklearn
def test_plot_ica_panel():
    """Test plotting of ICA panel
    """
    raw = _get_raw()
    ica_picks = pick_types(raw.info, meg=True, eeg=False, stim=False,
                                ecg=False, eog=False, exclude='bads')
    ica = ICA(noise_cov=read_cov(cov_fname), n_components=2,
              max_pca_components=3, n_pca_components=3)
    ica.decompose_raw(raw, picks=ica_picks)
    ica.plot_sources_raw(raw)
    plt.close('all')


def test_plot_image_epochs():
    """Test plotting of epochs image
    """
    epochs = _get_epochs()
    plot_image_epochs(epochs, picks=[1, 2])
    plt.close('all')


def test_plot_connectivity_circle():
    """Test plotting connectivity circle
    """
    node_order = ['frontalpole-lh', 'parsorbitalis-lh',
                  'lateralorbitofrontal-lh', 'rostralmiddlefrontal-lh',
                  'medialorbitofrontal-lh', 'parstriangularis-lh',
                  'rostralanteriorcingulate-lh', 'temporalpole-lh',
                  'parsopercularis-lh', 'caudalanteriorcingulate-lh',
                  'entorhinal-lh', 'superiorfrontal-lh', 'insula-lh',
                  'caudalmiddlefrontal-lh', 'superiortemporal-lh',
                  'parahippocampal-lh', 'middletemporal-lh',
                  'inferiortemporal-lh', 'precentral-lh',
                  'transversetemporal-lh', 'posteriorcingulate-lh',
                  'fusiform-lh', 'postcentral-lh', 'bankssts-lh',
                  'supramarginal-lh', 'isthmuscingulate-lh', 'paracentral-lh',
                  'lingual-lh', 'precuneus-lh', 'inferiorparietal-lh',
                  'superiorparietal-lh', 'pericalcarine-lh',
                  'lateraloccipital-lh', 'cuneus-lh', 'cuneus-rh',
                  'lateraloccipital-rh', 'pericalcarine-rh',
                  'superiorparietal-rh', 'inferiorparietal-rh', 'precuneus-rh',
                  'lingual-rh', 'paracentral-rh', 'isthmuscingulate-rh',
                  'supramarginal-rh', 'bankssts-rh', 'postcentral-rh',
                  'fusiform-rh', 'posteriorcingulate-rh',
                  'transversetemporal-rh', 'precentral-rh',
                  'inferiortemporal-rh', 'middletemporal-rh',
                  'parahippocampal-rh', 'superiortemporal-rh',
                  'caudalmiddlefrontal-rh', 'insula-rh', 'superiorfrontal-rh',
                  'entorhinal-rh', 'caudalanteriorcingulate-rh',
                  'parsopercularis-rh', 'temporalpole-rh',
                  'rostralanteriorcingulate-rh', 'parstriangularis-rh',
                  'medialorbitofrontal-rh', 'rostralmiddlefrontal-rh',
                  'lateralorbitofrontal-rh', 'parsorbitalis-rh',
                  'frontalpole-rh']
    label_names = ['bankssts-lh', 'bankssts-rh', 'caudalanteriorcingulate-lh',
                   'caudalanteriorcingulate-rh', 'caudalmiddlefrontal-lh',
                   'caudalmiddlefrontal-rh', 'cuneus-lh', 'cuneus-rh',
                   'entorhinal-lh', 'entorhinal-rh', 'frontalpole-lh',
                   'frontalpole-rh', 'fusiform-lh', 'fusiform-rh',
                   'inferiorparietal-lh', 'inferiorparietal-rh',
                   'inferiortemporal-lh', 'inferiortemporal-rh', 'insula-lh',
                   'insula-rh', 'isthmuscingulate-lh', 'isthmuscingulate-rh',
                   'lateraloccipital-lh', 'lateraloccipital-rh',
                   'lateralorbitofrontal-lh', 'lateralorbitofrontal-rh',
                   'lingual-lh', 'lingual-rh', 'medialorbitofrontal-lh',
                   'medialorbitofrontal-rh', 'middletemporal-lh',
                   'middletemporal-rh', 'paracentral-lh', 'paracentral-rh',
                   'parahippocampal-lh', 'parahippocampal-rh',
                   'parsopercularis-lh', 'parsopercularis-rh',
                   'parsorbitalis-lh', 'parsorbitalis-rh',
                   'parstriangularis-lh', 'parstriangularis-rh',
                   'pericalcarine-lh', 'pericalcarine-rh', 'postcentral-lh',
                   'postcentral-rh', 'posteriorcingulate-lh',
                   'posteriorcingulate-rh', 'precentral-lh', 'precentral-rh',
                   'precuneus-lh', 'precuneus-rh',
                   'rostralanteriorcingulate-lh',
                   'rostralanteriorcingulate-rh', 'rostralmiddlefrontal-lh',
                   'rostralmiddlefrontal-rh', 'superiorfrontal-lh',
                   'superiorfrontal-rh', 'superiorparietal-lh',
                   'superiorparietal-rh', 'superiortemporal-lh',
                   'superiortemporal-rh', 'supramarginal-lh',
                   'supramarginal-rh', 'temporalpole-lh', 'temporalpole-rh',
                   'transversetemporal-lh', 'transversetemporal-rh']

    group_boundaries = [0, len(label_names) / 2]
    node_angles = circular_layout(label_names, node_order, start_pos=90,
                                  group_boundaries=group_boundaries)
    con = np.random.randn(68, 68)
    plot_connectivity_circle(con, label_names, n_lines=300,
                             node_angles=node_angles, title='test',
                             )

    plt.close('all')
    assert_raises(ValueError, circular_layout, label_names, node_order,
                  group_boundaries=[-1])
    assert_raises(ValueError, circular_layout, label_names, node_order,
                  group_boundaries=[20, 0])


def test_plot_drop_log():
    """Test plotting a drop log
    """
    epochs = _get_epochs()
    epochs.drop_bad_epochs()
    epochs.plot_drop_log()

    plot_drop_log([['One'], [], []])
    plot_drop_log([['One'], ['Two'], []])
    plot_drop_log([['One'], ['One', 'Two'], []])
    plt.close('all')


def test_plot_raw():
    """Test plotting of raw data
    """
    raw = _get_raw()
    events = _get_events()
    plt.close('all')  # ensure all are closed
    with warnings.catch_warnings(record=True):
        fig = raw.plot(events=events, show_options=True)
        # test mouse clicks
        x = fig.get_axes()[0].lines[1].get_xdata().mean()
        y = fig.get_axes()[0].lines[1].get_ydata().mean()
        data_ax = fig.get_axes()[0]
        _fake_click(fig, data_ax, [x, y], xform='data')  # mark a bad channel
        _fake_click(fig, data_ax, [x, y], xform='data')  # unmark a bad channel
        _fake_click(fig, data_ax, [0.5, 0.999])  # click elsewhere in 1st axes
        _fake_click(fig, fig.get_axes()[1], [0.5, 0.5])  # change time
        _fake_click(fig, fig.get_axes()[2], [0.5, 0.5])  # change channels
        _fake_click(fig, fig.get_axes()[3], [0.5, 0.5])  # open SSP window
        fig.canvas.button_press_event(1, 1, 1)  # outside any axes
        # sadly these fail when no renderer is used (i.e., when using Agg):
        #ssp_fig = set(plt.get_fignums()) - set([fig.number])
        #assert_equal(len(ssp_fig), 1)
        #ssp_fig = plt.figure(list(ssp_fig)[0])
        #ax = ssp_fig.get_axes()[0]  # only one axis is used
        #t = [c for c in ax.get_children() if isinstance(c,
        #     matplotlib.text.Text)]
        #pos = np.array(t[0].get_position()) + 0.01
        #_fake_click(ssp_fig, ssp_fig.get_axes()[0], pos, xform='data')  # off
        #_fake_click(ssp_fig, ssp_fig.get_axes()[0], pos, xform='data')  # on
        # test keypresses
        fig.canvas.key_press_event('escape')
        fig.canvas.key_press_event('down')
        fig.canvas.key_press_event('up')
        fig.canvas.key_press_event('right')
        fig.canvas.key_press_event('left')
        fig.canvas.key_press_event('o')
        fig.canvas.key_press_event('escape')
        plt.close('all')


def test_plot_raw_psds():
    """Test plotting of raw psds
    """
    import matplotlib.pyplot as plt
    raw = _get_raw()
    # normal mode
    raw.plot_psds(tmax=2.0)
    # specific mode
    picks = pick_types(raw.info, meg='mag', eeg=False)[:4]
    raw.plot_psds(picks=picks, area_mode='range')
    ax = plt.axes()
    # if ax is supplied, picks must be, too:
    assert_raises(ValueError, raw.plot_psds, ax=ax)
    raw.plot_psds(picks=picks, ax=ax)
    plt.close('all')


@sample.requires_sample_data
def test_plot_topomap():
    """Test topomap plotting
    """
    # evoked
    warnings.simplefilter('always', UserWarning)
    with warnings.catch_warnings(record=True):
        evoked = io.read_evokeds(evoked_fname, 'Left Auditory',
                                  baseline=(None, 0))
        evoked.plot_topomap(0.1, 'mag', layout=layout)
        plot_evoked_topomap(evoked, None, ch_type='mag')
        times = [0.1, 0.2]
        plot_evoked_topomap(evoked, times, ch_type='eeg')
        plot_evoked_topomap(evoked, times, ch_type='grad')
        plot_evoked_topomap(evoked, times, ch_type='planar1')
        plot_evoked_topomap(evoked, times, ch_type='planar2')
        plot_evoked_topomap(evoked, times, ch_type='grad', show_names=True)

        p = plot_evoked_topomap(evoked, times, ch_type='grad',
                                show_names=lambda x: x.replace('MEG', ''))
        subplot = [x for x in p.get_children() if
                   isinstance(x, matplotlib.axes.Subplot)][0]
        assert_true(all('MEG' not in x.get_text()
                        for x in subplot.get_children()
                        if isinstance(x, matplotlib.text.Text)))

        # Test title
        def get_texts(p):
            return [x.get_text() for x in p.get_children() if
                    isinstance(x, matplotlib.text.Text)]

        p = plot_evoked_topomap(evoked, times, ch_type='eeg')
        assert_equal(len(get_texts(p)), 0)
        p = plot_evoked_topomap(evoked, times, ch_type='eeg', title='Custom')
        texts = get_texts(p)
        assert_equal(len(texts), 1)
        assert_equal(texts[0], 'Custom')

        # delaunay triangulation warning
        with warnings.catch_warnings(record=True):
            plot_evoked_topomap(evoked, times, ch_type='mag', layout='auto')
        assert_raises(RuntimeError, plot_evoked_topomap, evoked, 0.1, 'mag',
                      proj='interactive')  # projs have already been applied
        evoked.proj = False  # let's fake it like they haven't been applied
        plot_evoked_topomap(evoked, 0.1, 'mag', proj='interactive')
        assert_raises(RuntimeError, plot_evoked_topomap, evoked,
                      np.repeat(.1, 50))
        assert_raises(ValueError, plot_evoked_topomap, evoked, [-3e12, 15e6])

        projs = read_proj(ecg_fname)
        projs = [p for p in projs if p['desc'].lower().find('eeg') < 0]
        plot_projs_topomap(projs)
        plt.close('all')
        for ch in evoked.info['chs']:
            if ch['coil_type'] == FIFF.FIFFV_COIL_EEG:
                if ch['eeg_loc'] is not None:
                    ch['eeg_loc'].fill(0)
                ch['loc'].fill(0)
        assert_raises(RuntimeError, plot_evoked_topomap, evoked,
                      times, ch_type='eeg')


def test_compare_fiff():
    """Test comparing fiff files
    """
    compare_fiff(raw_fname, cov_fname, read_limit=0, show=False)
    plt.close('all')


@requires_sklearn
def test_plot_ica_topomap():
    """Test plotting of ICA solutions
    """
    raw = _get_raw()
    ica = ICA(noise_cov=read_cov(cov_fname), n_components=2,
              max_pca_components=3, n_pca_components=3)
    ica_picks = pick_types(raw.info, meg=True, eeg=False, stim=False,
                                ecg=False, eog=False, exclude='bads')
    ica.decompose_raw(raw, picks=ica_picks)
    warnings.simplefilter('always', UserWarning)
    with warnings.catch_warnings(record=True):
        for components in [0, [0], [0, 1], [0, 1] * 7]:
            ica.plot_topomap(components)
    ica.info = None
    assert_raises(RuntimeError, ica.plot_topomap, 1)
    plt.close('all')


@sample.requires_sample_data
def test_plot_source_spectrogram():
    """Test plotting of source spectrogram
    """
    sample_src = read_source_spaces(op.join(data_dir, 'subjects', 'sample',
                                            'bem', 'sample-oct-6-src.fif'))

    # dense version
    vertices = [s['vertno'] for s in sample_src]
    n_time = 5
    n_verts = sum(len(v) for v in vertices)
    stc_data = np.ones((n_verts, n_time))
    stc = SourceEstimate(stc_data, vertices, 1, 1)
    plot_source_spectrogram([stc, stc], [[1, 2], [3, 4]])
    assert_raises(ValueError, plot_source_spectrogram, [], [])


@requires_mayavi
@sample.requires_sample_data
def test_plot_evoked_field():
    trans_fname = op.join(data_dir, 'MEG', 'sample',
                          'sample_audvis_raw-trans.fif')
    evoked = io.read_evokeds(evoked_fname, condition='Left Auditory',
                               baseline=(-0.2, 0.0))
    evoked = pick_channels_evoked(evoked, evoked.ch_names[::10])  # speed
    for t in ['meg', None]:
        maps = make_field_map(evoked, trans_fname=trans_fname,
                              subject='sample', subjects_dir=subjects_dir,
                              n_jobs=1, ch_type=t)

        evoked.plot_field(maps, time=0.1)


def test_plot_events():
    raw = _get_raw()
    events = _get_events()
    plot_events(events, raw.info['sfreq'], raw.first_samp)
