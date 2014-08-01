# Authors: Denis Engemann <denis.engemann@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: Simplified BSD

import os.path as op
from functools import wraps
import warnings

from numpy.testing import assert_raises

from mne import io, read_events, Epochs, read_cov
from mne import pick_types
from mne.datasets import sample
from mne.utils import check_sklearn_version
from mne.preprocessing import ICA, create_ecg_epochs, create_eog_epochs


warnings.simplefilter('always')  # enable b/c these tests throw warnings

# Set our plotters to test mode
import matplotlib
matplotlib.use('Agg')  # for testing don't use X server
import matplotlib.pyplot as plt


data_dir = sample.data_path(download=False)
subjects_dir = op.join(data_dir, 'subjects')
ecg_fname = op.join(data_dir, 'MEG', 'sample', 'sample_audvis_ecg_proj.fif')

base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
evoked_fname = op.join(base_dir, 'test-ave.fif')
raw_fname = op.join(base_dir, 'test_raw.fif')
cov_fname = op.join(base_dir, 'test-cov.fif')
event_name = op.join(base_dir, 'test-eve.fif')
event_id, tmin, tmax = 1, -0.1, 0.2


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


def _get_raw():
    return io.Raw(raw_fname, preload=False)


def _get_events():
    return read_events(event_name)


def _get_picks(raw):
    return [0, 1, 2, 6, 7, 8, 12, 13, 14]  # take a only few channels


def _get_epochs():
    raw = _get_raw()
    events = _get_events()
    picks = _get_picks(raw)
    epochs = Epochs(raw, events[:10], event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0))
    return epochs


@requires_sklearn
def test_plot_ica_components():
    """Test plotting of ICA solutions
    """
    raw = _get_raw()
    ica = ICA(noise_cov=read_cov(cov_fname), n_components=2,
              max_pca_components=3, n_pca_components=3)
    ica_picks = _get_picks(raw)
    ica.fit(raw, picks=ica_picks)
    warnings.simplefilter('always', UserWarning)
    with warnings.catch_warnings(record=True):
        for components in [0, [0], [0, 1], [0, 1] * 2, None]:
            ica.plot_components(components, image_interp='bilinear', res=16)
    ica.info = None
    assert_raises(RuntimeError, ica.plot_components, 1)
    plt.close('all')


@requires_sklearn
def test_plot_ica_sources():
    """Test plotting of ICA panel
    """
    raw = io.Raw(raw_fname, preload=True)
    picks = _get_picks(raw)
    epochs = _get_epochs()
    raw.pick_channels([raw.ch_names[k] for k in picks])
    ica_picks = pick_types(raw.info, meg=True, eeg=False, stim=False,
                           ecg=False, eog=False, exclude='bads')
    ica = ICA(n_components=2, max_pca_components=3, n_pca_components=3)
    ica.fit(raw, picks=ica_picks)
    ica.plot_sources(raw)
    ica.plot_sources(epochs)
    with warnings.catch_warnings(record=True):  # no labeled objects mpl
        ica.plot_sources(epochs.average())
    assert_raises(ValueError, ica.plot_sources, 'meeow')
    plt.close('all')


@requires_sklearn
def test_plot_ica_overlay():
    """Test plotting of ICA cleaning
    """
    raw = _get_raw()
    picks = _get_picks(raw)
    ica = ICA(noise_cov=read_cov(cov_fname), n_components=2,
              max_pca_components=3, n_pca_components=3)
    ica.fit(raw, picks=picks)
    # don't test raw, needs preload ...
    ecg_epochs = create_ecg_epochs(raw, picks=picks)
    ica.plot_overlay(ecg_epochs.average())
    eog_epochs = create_eog_epochs(raw, picks=picks)
    ica.plot_overlay(eog_epochs.average())
    assert_raises(ValueError, ica.plot_overlay, raw[:2, :3][0])
    plt.close('all')


@requires_sklearn
def test_plot_ica_scores():
    """Test plotting of ICA scores
    """
    raw = _get_raw()
    picks = _get_picks(raw)
    ica = ICA(noise_cov=read_cov(cov_fname), n_components=2,
              max_pca_components=3, n_pca_components=3)
    ica.fit(raw, picks=picks)
    ica.plot_scores([0.3, 0.2], axhline=[0.1, -0.1])
    assert_raises(ValueError, ica.plot_scores, [0.2])
    plt.close('all')
