# Author: Mainak Jas <mainak@neuro.hut.fi>
#         Romain Trachel <trachelr@gmail.com>
#
# License: BSD (3-clause)

import warnings
import os.path as op
import numpy as np

from nose.tools import assert_true, assert_raises, assert_is_not_none
from numpy.testing import assert_equal

from mne import io, read_events, Epochs, pick_types
from mne.decoding import LinearClassifier

from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.simplefilter('always')  # enable b/c these tests throw warnings

tmin, tmax = -0.2, 0.5
event_id = dict(aud_l=1, vis_l=3)
start, stop = 0, 8

data_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(data_dir, 'test_raw.fif')
event_name = op.join(data_dir, 'test-eve.fif')

def test_linear_classifier():
    """Test methods of LinearClassifier
    """
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
    assert_is_not_none(clf.patterns_)
    # test filters have been computed
    assert_is_not_none(clf.filters_)
    
    # test classifier without a coef_ attribute
    clf = LinearClassifier(RandomForestClassifier())
    assert_raises(AssertionError, clf.fit, epochs_data, labels)
    
    # test get_params
    clf = LinearClassifier(LinearSVC(C = 10))
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
