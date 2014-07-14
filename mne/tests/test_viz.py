import os.path as op
from functools import wraps
import numpy as np
from numpy.testing import assert_raises
from nose.tools import assert_true, assert_equal
import warnings

from mne import io, read_events, Epochs, SourceEstimate, read_cov, read_proj
from mne import make_field_map, pick_types, pick_channels_evoked, read_evokeds
from mne.layouts import read_layout
from mne.viz import (plot_topo, plot_topo_tfr, plot_topo_image_epochs,
                     plot_evoked_topomap, plot_projs_topomap,
                     plot_sparse_source_estimates, plot_source_estimates,
                     plot_cov, mne_analyze_colormap, plot_image_epochs,
                     plot_connectivity_circle, circular_layout, plot_drop_log,
                     compare_fiff, plot_source_spectrogram, plot_events,
                     plot_trans, plot_bem)
from mne.datasets import sample
from mne.source_space import read_source_spaces
from mne.io.constants import FIFF
from mne.preprocessing import ICA, create_ecg_epochs, create_eog_epochs
from mne.utils import check_sklearn_version
from mne.time_frequency.tfr import AverageTFR


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



def test_compare_fiff():
    """Test comparing fiff files
    """
    compare_fiff(raw_fname, cov_fname, read_limit=0, show=False)
    plt.close('all')

