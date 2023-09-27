import numpy as np
import pytest

from mne.preprocessing import unify_bad_channels
from mne.time_frequency.tests.test_spectrum import _get_inst


@pytest.mark.parametrize(
    "instance", ("raw", "epochs", "evoked", "raw_spectrum", "epochs_spectrum")
)
def test_instance_support(instance, request, evoked):
    """Tests support of different classes."""
    # test unify_bads function on instance (single input, no bads scenario)
    inst = _get_inst(instance, request, evoked)
    inst_out = unify_bad_channels([inst])
    assert inst_out == [inst]


def test_error_raising(raw, epochs):
    """Tests input checking."""
    with pytest.raises(ValueError, match=r"empty list"):
        unify_bad_channels([])
    with pytest.raises(TypeError, match=r"must be an instance of"):
        unify_bad_channels(["bad_instance"])
    with pytest.raises(ValueError, match=r"same type"):
        unify_bad_channels([raw, epochs])
    with pytest.raises(ValueError):
        raw_alt = raw.copy()
        raw_alt.drop_channels(raw.info["ch_names"][0])
        unify_bad_channels([raw, raw_alt])


def test_bads_compilation(raw):
    """Tests that bads are compiled properly.

    Tests two cases: a) single instance passed to function with an existing
    bad, and b) multiple instances passed to function with varying compilation
    scenarios including empty bads, unique bads, and partially duplicated bads
    listed out-of-order.
    """
    assert raw.info["bads"] == []
    chns = raw.info["ch_names"][0:3]
    no_bad = raw.copy()
    one_bad = raw.copy()
    one_bad.info["bads"] = [chns[1]]
    three_bad = raw.copy()
    three_bad.info["bads"] = chns
    # scenario 1: single instance passed with actual bads
    s_out = unify_bad_channels([one_bad])
    assert len(s_out) == 1
    assert s_out[0].info["bads"] == [chns[1]], (s_out[0].info["bads"], chns[1])
    # scenario 2: multiple instances passed
    m_out = unify_bad_channels([one_bad, no_bad, three_bad])
    assert len(m_out) == 3
    correct_bads = [chns[1], chns[0], chns[2]]
    for i in np.arange(len(m_out)):
        assert m_out[i].info["bads"] == correct_bads
