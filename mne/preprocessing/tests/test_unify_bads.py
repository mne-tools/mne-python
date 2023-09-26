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


@pytest.mark.parametrize("instance", ([], ["bad_instance"], "mixed"))
def test_error_raising(instance, request, evoked):
    """Tests input checking."""
    if instance == "mixed":
        instance = [
            _get_inst("raw", request, evoked),
            _get_inst("epochs", request, evoked),
        ]
    with pytest.raises(TypeError):
        unify_bad_channels(instance)


def test_bads_compilation(raw):
    """Tests that bads are compiled properly.

    Tests two cases: a) single instance passed to function with an existing
    bad, and b) multiple instances passed to function with varying compilation
    scenarios including empty bads, unique bads, partially duplicated bads
    listed out-of-order, and nonsense channel names.
    """
    ## check unification scenarios
    chns = raw.info["ch_names"][0:3]
    # scenario 1: single instance passed with actual bads (already tested no bads)
    assert raw.info["bads"] == []
    raw.info["bads"] += [chns[0]]
    s_unified = unify_bad_channels([raw])
    assert len(s_unified) == 1
    assert s_unified[0].info["bads"] == [chns[0]], (s_unified[0].info["bads"], chns[0])
    # scenario 2: multiple instances passed, bads types as follows:
    # a) empty bads list, b) unique bads, c) overlapping out-of-order bads,
    # d) channel name not included in raw channels
    raws = [raw, raw.copy(), raw.copy(), raw.copy(), raw.copy()]
    assert raws[0].info["bads"] == [chns[0]]
    raws[1].info["bads"] = []
    raws[2].info["bads"] = [chns[2]]
    raws[3].info["bads"] = [chns[2], chns[1]]
    raws[4].info["bads"] = ["nonsense_ch_name"]
    # use unify_bads function
    m_unified = unify_bad_channels(raws)
    assert len(m_unified) == len(raws)
    # check results
    correct_bads = [chns[0], chns[2], chns[1], "nonsense_ch_name"]
    for i in np.arange(len(m_unified)):
        assert m_unified[i].info["bads"] == correct_bads
