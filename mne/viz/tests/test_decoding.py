# Authors: Denis Engemann <denis.engemann@gmail.com>
#
# License: Simplified BSD

import os.path as op
import warnings

from nose.tools import assert_raises

from mne.decoding import GeneralizationAcrossTime
from mne import io, Epochs, read_events, pick_types
from mne.utils import requires_sklearn, run_tests_if_main
import matplotlib
matplotlib.use('Agg')  # for testing don't use X server


data_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(data_dir, 'test_raw.fif')
event_name = op.join(data_dir, 'test-eve.fif')


warnings.simplefilter('always')  # enable b/c these tests throw warnings

# Set our plotters to test mode

tmin, tmax = -0.2, 0.5
event_id = dict(aud_l=1, vis_l=3)
event_id_gen = dict(aud_l=2, vis_l=4)


@requires_sklearn
def _get_data():
    """Aux function for testing GAT viz"""
    gat = GeneralizationAcrossTime()
    raw = io.Raw(raw_fname, preload=False)
    events = read_events(event_name)
    picks = pick_types(raw.info, meg='mag', stim=False, ecg=False,
                       eog=False, exclude='bads')
    picks = picks[1:13:3]
    decim = 30
    # Test on time generalization within one condition
    with warnings.catch_warnings(record=True):
        epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                        baseline=(None, 0), preload=True, decim=decim)

    # Test default running
    gat = GeneralizationAcrossTime()
    gat.fit(epochs)
    gat.score(epochs)
    return gat


def test_gat_plot_matrix():
    """Test GAT matrix plot"""
    gat = _get_data()
    gat.plot()
    del gat.scores_
    assert_raises(RuntimeError, gat.plot)


def test_gat_plot_diagonal():
    """Test GAT diagonal plot"""
    gat = _get_data()
    gat.plot_diagonal()
    del gat.scores_
    assert_raises(RuntimeError, gat.plot)

run_tests_if_main()
