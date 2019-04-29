import pickle
from mne.utils import BunchConstNamed


def test_pickle():
    """Test if BunchConstNamed object can be pickled."""
    b1 = BunchConstNamed()
    b1.x = 1
    b1.y = 2.12

    b2 = pickle.loads(pickle.dumps(b1))
    assert b1 == b2
