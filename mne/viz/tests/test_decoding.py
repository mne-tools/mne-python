# Authors: Denis Engemann <denis.engemann@gmail.com>
#          Jean-Remi King <jeanremi.king@gmail.com>
#
# License: Simplified BSD

import os.path as op
import warnings

from nose.tools import assert_raises, assert_equals

import numpy as np

from mne.epochs import equalize_epoch_counts, concatenate_epochs
from mne.decoding import GeneralizationAcrossTime
from mne import Epochs, read_events, pick_types
from mne.io import read_raw_fif
from mne.utils import requires_sklearn, run_tests_if_main
import matplotlib
matplotlib.use('Agg')  # for testing don't use X server


data_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(data_dir, 'test_raw.fif')
event_name = op.join(data_dir, 'test-eve.fif')


warnings.simplefilter('always')  # enable b/c these tests throw warnings


def _get_data(tmin=-0.2, tmax=0.5, event_id=dict(aud_l=1, vis_l=3),
              event_id_gen=dict(aud_l=2, vis_l=4), test_times=None):
    """Aux function for testing GAT viz."""
    gat = GeneralizationAcrossTime()
    raw = read_raw_fif(raw_fname)
    raw.add_proj([], remove_existing=True)
    events = read_events(event_name)
    picks = pick_types(raw.info, meg='mag', stim=False, ecg=False,
                       eog=False, exclude='bads')
    picks = picks[1:13:3]
    decim = 30
    # Test on time generalization within one condition
    with warnings.catch_warnings(record=True):
        epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                        preload=True, decim=decim)
    epochs_list = [epochs[k] for k in event_id]
    equalize_epoch_counts(epochs_list)
    epochs = concatenate_epochs(epochs_list)

    # Test default running
    gat = GeneralizationAcrossTime(test_times=test_times)
    gat.fit(epochs)
    gat.score(epochs)
    return gat


@requires_sklearn
def test_gat_plot_matrix():
    """Test GAT matrix plot."""
    gat = _get_data()
    gat.plot()
    del gat.scores_
    assert_raises(RuntimeError, gat.plot)


@requires_sklearn
def test_gat_plot_diagonal():
    """Test GAT diagonal plot."""
    gat = _get_data()
    gat.plot_diagonal()
    del gat.scores_
    assert_raises(RuntimeError, gat.plot)


@requires_sklearn
def test_gat_plot_times():
    """Test GAT times plot."""
    gat = _get_data()
    # test one line
    gat.plot_times(gat.train_times_['times'][0])
    # test multiple lines
    gat.plot_times(gat.train_times_['times'])
    # test multiple colors
    n_times = len(gat.train_times_['times'])
    colors = np.tile(['r', 'g', 'b'],
                     int(np.ceil(n_times / 3)))[:n_times]
    gat.plot_times(gat.train_times_['times'], color=colors)
    # test invalid time point
    assert_raises(ValueError, gat.plot_times, -1.)
    # test float type
    assert_raises(ValueError, gat.plot_times, 1)
    assert_raises(ValueError, gat.plot_times, 'diagonal')
    del gat.scores_
    assert_raises(RuntimeError, gat.plot)


def chance(ax):
    return ax.get_children()[1].get_lines()[0].get_ydata()[0]


@requires_sklearn
def test_gat_chance_level():
    """Test GAT plot_times chance level."""
    gat = _get_data()
    ax = gat.plot_diagonal(chance=False)
    ax = gat.plot_diagonal()
    assert_equals(chance(ax), .5)
    gat = _get_data(event_id=dict(aud_l=1, vis_l=3, aud_r=2, vis_r=4))
    ax = gat.plot_diagonal()
    assert_equals(chance(ax), .25)
    ax = gat.plot_diagonal(chance=1.234)
    assert_equals(chance(ax), 1.234)
    assert_raises(ValueError, gat.plot_diagonal, chance='foo')
    del gat.scores_
    assert_raises(RuntimeError, gat.plot)


@requires_sklearn
def test_gat_plot_nonsquared():
    """Test GAT diagonal plot."""
    gat = _get_data(test_times=dict(start=0.))
    gat.plot()
    ax = gat.plot_diagonal()
    scores = ax.get_children()[1].get_lines()[2].get_ydata()
    assert_equals(len(scores), len(gat.estimators_))

run_tests_if_main()
