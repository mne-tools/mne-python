# Author: Mainak Jas <mainak@neuro.hut.fi>
#         Romain Trachel <trachelr@gmail.com>
#
# License: BSD (3-clause)

import warnings
import os.path as op

from nose.tools import assert_raises, assert_true
from numpy.testing import assert_equal

from mne import io, read_events, Epochs, pick_types
from mne.decoding import LinearClassifier
from mne.utils import requires_sklearn

warnings.simplefilter('always')  # enable b/c these tests throw warnings

tmin, tmax = -0.2, 0.5
event_id = dict(aud_l=1, vis_l=3)

data_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(data_dir, 'test_raw.fif')
event_name = op.join(data_dir, 'test-eve.fif')


@requires_sklearn
def test_linear_classifier():
    """Test methods of LinearClassifier
    """
    from sklearn.svm import LinearSVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    raw = io.Raw(raw_fname, preload=False)
    events = read_events(event_name)
    picks = pick_types(raw.info, meg=True, stim=False, ecg=False,
                       eog=False, exclude='bads')
    picks = picks[1:13:3]

    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), decim=4, preload=True)

    labels = epochs.events[:, -1]
    epochs_data = epochs.get_data().reshape(len(labels), -1)

    clf = LinearClassifier()
    clf.fit(epochs_data, labels)

    # test patterns have been computed
    assert_true(clf.patterns_ is not None)
    # test filters have been computed
    assert_true(clf.filters_ is not None)

    # test classifier without a coef_ attribute
    clf = LinearClassifier(RandomForestClassifier())
    assert_raises(AssertionError, clf.fit, epochs_data, labels)

    # test get_params
    clf = LinearClassifier(LinearSVC(C=10))
    assert_equal(clf.get_params()['clf__C'], 10)

    # test set_params
    clf.set_params(clf__C=100)
    assert_equal(clf.get_params()['clf__C'], 100)

    # test it goes through a scikit-learn pipeline
    clf = LinearClassifier()
    sc = StandardScaler()
    test_pipe = Pipeline((('scaler', sc), ('clf', clf)))
    test_pipe.fit(epochs_data, labels)
    test_pipe.predict(epochs_data)
    test_pipe.score(epochs_data, labels)


@requires_sklearn
def test_plot_patterns():
    """Test plot_patterns of LinearModel
    """
    raw = io.Raw(raw_fname, preload=False)
    events = read_events(event_name)
    picks = pick_types(raw.info, meg=False, stim=False, ecg=False,
                       eog=False, eeg=True, exclude='bads')

    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=None, decim=8, preload=True)

    labels = epochs.events[:, -1]
    epochs_data = epochs.get_data().reshape(len(labels), -1)

    model = LinearModel()
    model.fit(epochs_data, labels)

    # test plot patterns
    model.plot_patterns(epochs.info)


@requires_sklearn
def test_plot_filters():
    """Test plot_filters of LinearModel
    """
    raw = io.Raw(raw_fname, preload=False)
    events = read_events(event_name)
    picks = pick_types(raw.info, meg=False, stim=False, ecg=False,
                       eog=False, eeg=True, exclude='bads')

    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=None, decim=8, preload=True)

    labels = epochs.events[:, -1]
    epochs_data = epochs.get_data().reshape(len(labels), -1)

    model = LinearModel()
    model.fit(epochs_data, labels)

    # test plot patterns
    model.plot_filters(epochs.info)
