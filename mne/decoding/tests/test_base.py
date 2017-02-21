import numpy as np
from numpy.testing import assert_array_equal, assert_raises, assert_equal

import mne
from mne.decoding import dummy_encoding


def test_dummy_encoding():
    """Test generating a target variable from epochs."""
    events = np.array([[0, 0, 1],
                       [1, 0, 2],
                       [2, 0, 1],
                       [3, 0, 1],
                       [4, 0, 2]])
    event_id = dict(a=1, b=2)
    epochs = mne.EpochsArray(np.random.randn(5, 2, 10),
                             mne.create_info(2, 1000.),
                             events, event_id=event_id)

    # Basic case of two classes
    assert_array_equal(dummy_encoding(epochs),
                       [0, 1, 0, 0, 1])
    assert_array_equal(dummy_encoding(epochs, ['a', 'b']),
                       [0, 1, 0, 0, 1])
    assert_array_equal(dummy_encoding(epochs, ['b', 'a']),
                       [1, 0, 1, 1, 0])
    assert_equal(dummy_encoding(epochs, ['b', 'a']).dtype, np.int)

    # Multiple classes
    events = np.array([[0, 0, 1],
                       [1, 0, 2],
                       [2, 0, 1],
                       [3, 0, 3],
                       [4, 0, 2]])
    event_id = dict(a=1, b=2, c=3, d=3)
    epochs = mne.EpochsArray(np.random.randn(5, 2, 10),
                             mne.create_info(2, 1000.),
                             events, event_id=event_id)
    assert_array_equal(dummy_encoding(epochs),
                       [[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 1, 1],
                        [0, 1, 0, 0]])
    assert_array_equal(dummy_encoding(epochs, ['b', 'a', 'd', 'c']),
                       [[0, 1, 0, 0],
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 1],
                        [1, 0, 0, 0]])

    # Case where some epochs don't belong to any class. Should generate a
    # matrix, not a list.
    assert_array_equal(dummy_encoding(epochs, ['a', 'b']),
                       [[1, 0],
                        [0, 1],
                        [1, 0],
                        [0, 0],
                        [0, 1]])

    # Case where some epochs belong to multiple classes. Should generate a
    # matrix, not a list.
    assert_array_equal(dummy_encoding(epochs, ['c', 'd']),
                       [[0, 0],
                        [0, 0],
                        [0, 0],
                        [1, 1],
                        [0, 0]])

    # Don't know why you would want to do this, but these should work
    assert_array_equal(dummy_encoding(epochs, ['a', 'a']),
                       [[1, 1],
                        [0, 0],
                        [1, 1],
                        [0, 0],
                        [0, 0]])

    # Invalid inputs
    assert_raises(ValueError, dummy_encoding, epochs, ['a', 'b', 'foo'])
    assert_raises(ValueError, dummy_encoding, epochs, ['a'])
    assert_raises(ValueError, dummy_encoding, epochs, [])
