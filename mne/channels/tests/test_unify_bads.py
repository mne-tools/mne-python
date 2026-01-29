# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import pytest

from mne.channels import unify_bad_channels


def test_error_raising(raw, epochs):
    """Tests input checking."""
    with pytest.raises(TypeError, match=r"must be an instance of list"):
        unify_bad_channels("bad input")
    with pytest.raises(ValueError, match=r"insts must not be empty"):
        unify_bad_channels([])
    with pytest.raises(TypeError, match=r"each object in insts must be an instance of"):
        unify_bad_channels(["bad_instance"])
    with pytest.raises(ValueError, match=r"same type"):
        unify_bad_channels([raw, epochs])
    with pytest.raises(ValueError, match=r"Channels do not match across"):
        raw_alt1 = raw.copy()
        raw_alt1.drop_channels(raw.info["ch_names"][-1])
        unify_bad_channels([raw, raw_alt1])  # ch diff preserving order
    with pytest.raises(ValueError, match=r"sorted differently"):
        raw_alt2 = raw.copy()
        new_order = [raw.ch_names[-1]] + raw.ch_names[:-1]
        raw_alt2.reorder_channels(new_order)
        unify_bad_channels([raw, raw_alt2])


def test_bads_compilation(raw):
    """Tests that bads are compiled properly.

    Tests two cases: a) single instance passed to function with an existing
    bad, and b) multiple instances passed to function with varying compilation
    scenarios including empty bads, unique bads, and partially duplicated bads
    listed out-of-order.

    Only the Raw instance type is tested, since bad channel implementation is
    controlled across instance types with a MixIn class.
    """
    assert raw.info["bads"] == []
    chns = raw.ch_names[:3]
    no_bad = raw.copy()
    one_bad = raw.copy()
    one_bad.info["bads"] = [chns[1]]
    three_bad = raw.copy()
    three_bad.info["bads"] = chns
    # scenario 1: single instance passed with actual bads
    s_out = unify_bad_channels([one_bad])
    assert len(s_out) == 1, len(s_out)
    assert s_out[0].info["bads"] == [chns[1]], (s_out[0].info["bads"], chns[1])
    # scenario 2: multiple instances passed
    m_out = unify_bad_channels([one_bad, no_bad, three_bad])
    assert len(m_out) == 3, len(m_out)
    expected_order = [chns[1], chns[0], chns[2]]
    for inst in m_out:
        assert inst.info["bads"] == expected_order
