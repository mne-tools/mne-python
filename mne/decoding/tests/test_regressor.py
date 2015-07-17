# Author: Mainak Jas <mainak@neuro.hut.fi>
#         Romain Trachel <trachelr@gmail.com>
#
# License: BSD (3-clause)

import warnings
import os.path as op
import numpy as np

from nose.tools import assert_raises, assert_true
from numpy.testing import assert_equal

from mne import io, read_events, Epochs, pick_types
from mne.decoding import LinearRegressor
from mne.utils import requires_sklearn

warnings.simplefilter('always')  # enable b/c these tests throw warnings

tmin, tmax = -0.2, 0.5
event_id = dict(aud_l=1, vis_l=3)
start, stop = 0, 8

data_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(data_dir, 'test_raw.fif')
event_name = op.join(data_dir, 'test-eve.fif')


@requires_sklearn
def test_linear_regressor():
    """Test methods of LinearRegressor
    """
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    raw = io.Raw(raw_fname, preload=False)
    events = read_events(event_name)
    picks = pick_types(raw.info, meg=True, stim=False, ecg=False,
                       eog=False, exclude='bads')
    picks = picks[1:13:3]

    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), decim=4, preload=True)
    
    # get labels and type it float for regression
    labels = np.array(epochs.events[:, -1], dtype=np.float)
    epochs_data = epochs.get_data().reshape(len(labels), -1)
    
    reg = LinearRegressor()
    # test fit
    reg.fit(epochs_data, labels)
    
    # test patterns have been computed
    assert_true(reg.patterns_ is not None)
    # test filters have been computed
    assert_true(reg.filters_ is not None)

    # test predict
    y = reg.predict(epochs_data)
    
    # test classifier without a coef_ attribute
    reg = LinearRegressor(RandomForestRegressor())
    assert_raises(AssertionError, reg.fit, epochs_data, labels)
    
    # test get_params
    reg = LinearRegressor(Ridge(alpha = 10))
    assert_equal(reg.get_params()['reg__alpha'], 10)
    
    # test set_params
    reg.set_params(reg__alpha=100)
    assert_equal(reg.get_params()['reg__alpha'], 100)
    
    # test it goes through a scikit-learn pipeline
    reg = LinearRegressor()
    sc = StandardScaler()
    test_pipe = Pipeline((('scaler', sc), ('reg', reg)))
    test_pipe.fit(epochs_data, labels)
    test_pipe.predict(epochs_data)
    test_pipe.score(epochs_data, labels)
