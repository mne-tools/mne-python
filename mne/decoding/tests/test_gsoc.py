import os.path as op
from numpy.testing import assert_array_equal, assert_equal
from nose.tools import assert_raises
from mne import io, Epochs, read_events,  pick_types
from mne.decoding.gsoc import (_EpochsTransformerMixin,
                               UnsupervisedSpatialFilter)
from mne.decoding.transformer import EpochsVectorizer
from mne.utils import run_tests_if_main, requires_sklearn

base_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')
event_name = op.join(base_dir, 'test-eve.fif')

tmin, tmax = -0.1, 0.2
event_id = dict(cond2=2, cond3=3)


def _get_data():
    raw = io.read_raw_fif(raw_fname, add_eeg_ref=False, verbose=False,
                          preload=True)
    events = read_events(event_name)
    picks = pick_types(raw.info, meg=False, eeg=True, stim=False,
                       ecg=False, eog=False,
                       exclude='bads')[::8]
    return raw, events, picks


def test_EpochsTransformerMixin():
    raw, events, picks = _get_data()
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    preload=True, baseline=None, verbose=False)

    # Test _rehsape method wrong input
    etm = _EpochsTransformerMixin(n_chan=epochs.info['nchan'])
    assert_raises(ValueError, etm._reshape, raw)

    # Test _reshape correctness
    X = EpochsVectorizer().fit(epochs._data, None).transform(epochs._data)
    assert_array_equal(etm._reshape(X), epochs._data)
    assert_equal(etm._reshape(X).ndim, epochs._data.ndim)


@requires_sklearn
def test_UnsupervisedSpatialFilter():
    from sklearn.decomposition import PCA
    raw, events, picks = _get_data()
    epochs = Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    preload=True, baseline=None, verbose=False)

    # Test fit
    X = EpochsVectorizer().fit(epochs._data, None).transform(epochs._data)
    usf = UnsupervisedSpatialFilter(PCA(5), n_chan=epochs.info['nchan'])
    usf.fit(X)
    usf1 = UnsupervisedSpatialFilter(PCA(5), n_chan=epochs.info['nchan'])

    # test transform
    assert_equal(usf.transform(X).ndim, 3)

    # test fit_trasnform
    assert_array_equal(usf.transform(X), usf1.fit_transform(X))

    # assert shape
    assert_equal(usf.transform(X).shape[1], 5)

run_tests_if_main()
