import os.path as op
import numpy as np
from numpy.testing import assert_raises

from mne import fiff, read_events, Epochs, SourceEstimate, read_cov, read_proj
from mne.layouts import read_layout
from mne.fiff.pick import pick_channels_evoked
from mne.viz import plot_topo, plot_topo_tfr, plot_topo_power, \
                    plot_topo_phase_lock, plot_topo_image_epochs, \
                    plot_evoked_topomap, plot_projs_topomap, \
                    plot_sparse_source_estimates, plot_source_estimates, \
                    plot_cov, mne_analyze_colormap, plot_image_epochs, \
                    plot_connectivity_circle, circular_layout, plot_drop_log, \
                    compare_fiff
from mne.datasets.sample import data_path
from mne.source_space import read_source_spaces
from mne.preprocessing import ICA

# Set our plotters to test mode
import matplotlib
matplotlib.use('Agg')  # for testing don't use X server

lacks_mayavi = False
try:
    from mayavi import mlab
except ImportError:
    try:
        from enthought.mayavi import mlab
    except ImportError:
        lacks_mayavi = True
requires_mayavi = np.testing.dec.skipif(lacks_mayavi, 'Requires mayavi')

if not lacks_mayavi:
    mlab.options.backend = 'test'

data_dir = data_path()
subjects_dir = op.join(data_dir, 'subjects')
sample_src = read_source_spaces(op.join(data_dir, 'subjects', 'sample',
                                        'bem', 'sample-oct-6-src.fif'))
ecg_fname = op.join(data_dir, 'MEG', 'sample', 'sample_audvis_ecg_proj.fif')
evoked_fname = op.join(data_dir, 'MEG', 'sample', 'sample_audvis-ave.fif')
base_dir = op.join(op.dirname(__file__), '..', 'fiff', 'tests', 'data')
fname = op.join(base_dir, 'test-ave.fif')
raw_fname = op.join(base_dir, 'test_raw.fif')
cov_fname = op.join(base_dir, 'test-cov.fif')
event_name = op.join(base_dir, 'test-eve.fif')
event_id, tmin, tmax = 1, -0.2, 0.5
n_chan = 15

raw = fiff.Raw(raw_fname, preload=False)
events = read_events(event_name)
picks = fiff.pick_types(raw.info, meg=True, eeg=False, stim=False,
                        ecg=False, eog=False, exclude='bads')
# Use a subset of channels for plotting speed
picks = np.round(np.linspace(0, len(picks) + 1, n_chan)).astype(int)
epochs = Epochs(raw, events[:10], event_id, tmin, tmax, picks=picks,
                baseline=(None, 0))
evoked = epochs.average()
reject = dict(mag=4e-12)
epochs_delayed_ssp = Epochs(raw, events[:10], event_id, tmin, tmax,
                            picks=picks, baseline=(None, 0), proj='delayed',
                            reject=reject)
evoked_delayed_ssp = epochs_delayed_ssp.average()
layout = read_layout('Vectorview-all')


def test_plot_topo():
    """Test plotting of ERP topography
    """
    # Show topography
    plot_topo(evoked, layout)
    picked_evoked = pick_channels_evoked(evoked, evoked.ch_names[:3])

    # test scaling
    for ylim in [dict(mag=[-600, 600]), None]:
        plot_topo([picked_evoked] * 2, layout, ylim=ylim)

    for evo in [evoked, [evoked, picked_evoked]]:
        assert_raises(ValueError, plot_topo, evo, layout, color=['y', 'b'])

    plot_topo(evoked_delayed_ssp, layout, proj='interactive')


def test_plot_topo_tfr():
    """Test plotting of TFR
    """
    # Make a fake dataset to plot
    n_freqs = 11
    con = np.random.randn(n_chan, n_freqs, len(epochs.times))
    freqs = np.arange(n_freqs)
    # Show topography of connectivity from seed
    plot_topo_tfr(epochs, con, freqs, layout)


def test_plot_topo_power():
    """Test plotting of power
    """
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


def test_plot_topo_image_epochs():
    """Test plotting of epochs image topography
    """
    title = 'ERF images - MNE sample data'
    plot_topo_image_epochs(epochs, layout, sigma=0.5, vmin=-200, vmax=200,
                           colorbar=True, title=title)


def test_plot_evoked():
    """Test plotting of evoked
    """
    evoked.plot(proj=True, hline=[1])

    # plot with bad channels excluded
    evoked.plot(exclude='bads')
    evoked.plot(exclude=evoked.info['bads'])  # does the same thing

    # test selective updating of dict keys is working.
    evoked.plot(hline=[1], units=dict(mag='femto foo'))
    evoked_delayed_ssp.plot(proj='interactive')
    evoked_delayed_ssp.apply_proj()
    assert_raises(RuntimeError, evoked_delayed_ssp.plot, proj='interactive')
    evoked_delayed_ssp.info['projs'] = []
    assert_raises(RuntimeError, evoked_delayed_ssp.plot, proj='interactive')
    assert_raises(RuntimeError, evoked_delayed_ssp.plot, proj='interactive',
                  axes='foo')


@requires_mayavi
def test_plot_sparse_source_estimates():
    """Test plotting of (sparse) source estimates
    """
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
                          subjects_dir=subjects_dir)
    assert_raises(RuntimeError, plot_source_estimates, stc, 'sample',
                  figure='foo', hemi='both')

    # now do sparse version
    vertices = sample_src[0]['vertno']
    n_verts = len(vertices)
    stc_data = np.zeros((n_verts * n_time))
    stc_data[(np.random.rand(20) * n_verts * n_time).astype(int)] = 1
    stc_data.shape = (n_verts, n_time)
    inds = np.where(np.any(stc_data, axis=1))[0]
    stc_data = stc_data[inds]
    vertices = vertices[inds]
    stc = SourceEstimate(stc_data, vertices, 1, 1)
    plot_sparse_source_estimates(sample_src, stc, bgcolor=(1, 1, 1),
                                 opacity=0.5, high_resolution=True)


def test_plot_cov():
    """Test plotting of covariances
    """
    cov = read_cov(cov_fname)
    plot_cov(cov, raw.info, proj=True)


def test_plot_ica_panel():
    """Test plotting of ICA panel
    """
    ica_picks = fiff.pick_types(raw.info, meg=True, eeg=False, stim=False,
                                ecg=False, eog=False, exclude='bads')
    cov = read_cov(cov_fname)
    ica = ICA(noise_cov=cov, n_components=2, max_pca_components=3,
              n_pca_components=3)
    ica.decompose_raw(raw, picks=ica_picks)
    ica.plot_sources_raw(raw)


def test_plot_image_epochs():
    """Test plotting of epochs image
    """
    plot_image_epochs(epochs, picks=[1, 2])


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
    node_angles = circular_layout(label_names, node_order, start_pos=90)
    con = np.random.randn(68, 68)
    plot_connectivity_circle(con, label_names, n_lines=300,
                             node_angles=node_angles, title='test')


def test_plot_drop_log():
    """Test plotting a drop log
    """
    plot_drop_log(epochs.drop_log)
    plot_drop_log([['One'], [], []])
    plot_drop_log([['One'], ['Two'], []])
    plot_drop_log([['One'], ['One', 'Two'], []])


def test_plot_raw():
    """Test plotting of raw data
    """
    raw.plot(events=events, show_options=True)


def test_plot_topomap():
    """Testing topomap plotting
    """
    # evoked
    evoked = fiff.read_evoked(evoked_fname, 'Left Auditory',
                              baseline=(None, 0))
    evoked.plot_topomap(0.1, 'mag', layout=layout)
    plot_evoked_topomap(evoked, None, ch_type='mag')
    times = [0.1, 0.2]
    plot_evoked_topomap(evoked, times, ch_type='grad')
    plot_evoked_topomap(evoked, times, ch_type='planar1')
    plot_evoked_topomap(evoked, times, ch_type='mag', layout='auto')
    plot_evoked_topomap(evoked, 0.1, 'mag', proj='interactive')
    assert_raises(RuntimeError, plot_evoked_topomap, evoked, np.repeat(.1, 50))
    assert_raises(ValueError, plot_evoked_topomap, evoked, [-3e12, 15e6])

    # projs
    projs = read_proj(ecg_fname)[:7]
    plot_projs_topomap(projs)


def test_compare_fiff():
    """Test comparing fiff files
    """
    compare_fiff(raw_fname, cov_fname, read_limit=0, show=False)
