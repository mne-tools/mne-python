from pathlib import Path

import numpy as np
import pytest

import mne
from mne.preprocessing import unify_bad_channels
from ..io import BaseRaw
from ..epochs import Epochs
from ..evoked import Evoked
from ..time_frequency.spectrum import BaseSpectrum


raw_fname = (
    Path(__file__).parent.parent.parent / "io" / "tests" / "data" / "test_raw.fif"
)
@pytest.mark.parametrize('instance', ('raw', 'epochs', 'evoked', 'spectrum'))
def test_instance_support(instance):
    # test unify_bads function on instance (single input, no bads scenario)
    unify_bad_channels(instance)


def test_bads_order(raw):

def test_unify_bads(raw, epochs):
    ## test error raising
    # err 1: no instance passed to function
    with pytest.raises(UserWarning): # FIX RAISE TYPE
        unify_bad_channels([])
    # err 2: bad instance passed to function
    bad_inst = 'bad_instance'
    with pytest.raises(UserWarning):  # FIX RAISE TYPE
        unify_bad_channels(bad_inst)
    # err 3: mixed instance types passed to function
    with pytest.raises(ValueError):
        unify_bad_channels([raw, epochs])
    ## check unification scenarios
    # scnario 1: single instance with actual bads (already tested no bads)
    raw.info['bads'] += raw.info['ch_names'][0]
    s_unified = unify_bad_channels(raw)
    assert len(s_unified) == 1
    assert s_unified[0].info['bads'] == raw.info['ch_names'][0]
    # scenario 2: multiple instances
    # a) empty bads list, b) unique bads, c) overlapping out-of-order bads,
    # d) channel name not included in raw channels
    chns = raw.info['ch_names'][0:3]
    raws = [raw, raw.copy(), raw.copy(), raw.copy(), raw.copy()]
    assert raws[0].info['bads'] == [chns[0]]
    raws[1].info['bads'] = []
    raws[2].info['bads'] = [chns[2]]
    raws[3].info['bads'] = [chns[2], chns[1]]
    raws[4].info['bads'] = ['nonsense_ch_name']
    # use unify_bads function
    m_unified = unify_bad_channels(raws)
    assert len(m_unified) == len(raws)
    # check results
    for i in np.arange(len(m_unified)):
        assert m_unified[i].info['bads'] ==
