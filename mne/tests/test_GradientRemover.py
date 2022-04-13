import pytest
import numpy as np
from mne.preprocessing import GradientRemover

def n_trs():
    """Return a number of TRs for this test suite."""
    return 10

def tr_code():
    """Return a TR marker code for this test suite."""
    return 1

def samps_per_tr():
    """Return a number of samples per TR for this test suite."""
    return 100

def sample_trs():
    """Return a sample array of TR makers for this test suite."""
    return np.asarray([x * samps_per_tr() for x in range(n_trs())])

def sample_trs_longform():
    """Return a sample array of TR markers with the mne long event form."""
    return np.asarray(
        [[samps_per_tr(), 0, tr_code()] for x in range(n_trs())]
    )

def sample_data():
    """Return sample TR data (zeros) for use in this test suite."""
    return np.zeros((256, n_trs() * samps_per_tr()))


def test_window_validity():
    """Test window validity for a variety of windows."""
    data = sample_data()
    trs = sample_trs()
    # Odd numbers are not valid
    with pytest.raises(ValueError, match=r"Integer windows must be even"):
        GradientRemover(data, trs, 5)
    # Windows should have exactly 2 elements as a tuple 
    with pytest.raises(ValueError, match=r"Tuple windows must contain"):
        GradientRemover(data, trs, (2, 2, 2))
    # Windows should be a tuple of integers or an integer
    with pytest.raises(TypeError, match=r"Window must be a positive"):
        GradientRemover(data, trs, None)
    # Windows should only have positive values
    with pytest.raises(ValueError, match=r"Window must contain"):
        GradientRemover(data, trs, (-1, 1))
    with pytest.raises(ValueError, match=r"Window must contain"):
        GradientRemover(data, trs, (1, -1))
    # Windows should have at least one nonzero value
    with pytest.raises(ValueError, match=r"Window must contain"):
        GradientRemover(data, trs, (0, 0))

    # Passing a valid window should result in storing a valid window
    gr = GradientRemover(data, trs, (2, 2))
    assert gr.window == (2, 2)
    gr = GradientRemover(data, trs, 4)
    assert gr.window == (2, 2)


def test_tr_events_validity():
    """Test to ensure that tr_events is valid."""
    data = sample_data()
    trs = sample_trs()

    # We should get a sensible error message for incorrectly sized arrays
    with pytest.raises(ValueError, match=r"TRs must be a 1D array or"):
        GradientRemover(data, np.asarray([[1, 2], [1, 2]]))

    # Mangle the second TR of valid TRs
    trs[1] = trs[1] + 1 # Short form
    with pytest.raises(ValueError, match=r"TR spacings are not"):
        GradientRemover(data, trs)
    trs = sample_trs_longform()
    trs[1, 0] = trs[1, 0] + 1 # Long form
    with pytest.raises(ValueError, match=r"TR spacings are not"):
        GradientRemover(data, trs)
    # Make sure the correct spacing is found
    trs = sample_trs()
    gr = GradientRemover(data, trs)
    assert gr.tr_spacing == samps_per_tr()
    # Make sure the correct number of events are present
    # (Okay this is probably needless but I'd like to make sure)
    assert gr.n_tr == n_trs()


def test_get_tr():
    """Test for get_tr function to ensure valid indexing."""
    gr = GradientRemover(sample_data(), sample_trs())
    
    # Make sure negative index triggers error
    with pytest.raises(ValueError, match=r"Index -1"):
        gr.get_tr(-1)

    # Make sure too big of an index triggers error
    with pytest.raises(ValueError, match=r"Index"):
        gr.get_tr(len(sample_trs()) + 1)

    # Smoke test for early testing; consider removing
    assert gr.get_tr(0).shape[1] == gr.tr_spacing
