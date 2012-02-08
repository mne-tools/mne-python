import os.path as op

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_equal

from ..datasets import sample
from ..fiff import Raw, Evoked, pick_types_forward
from ..minimum_norm.inverse import _make_stc
from .. import read_forward_solution, apply_forward, apply_forward_raw


examples_folder = op.join(op.dirname(__file__), '..', '..', 'examples')
data_path = sample.data_path(examples_folder)
fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis-meg-oct-6-fwd.fif')

fname_raw = op.join(op.dirname(__file__), '..', 'fiff', 'tests', 'data',
                    'test_raw.fif')

fname_evoked = op.join(op.dirname(__file__), '..', 'fiff', 'tests', 'data',
                       'test-ave.fif')


def test_io_forward():
    """Test IO for forward solutions
    """
    fwd = read_forward_solution(fname)
    fwd = read_forward_solution(fname, surf_ori=True)
    leadfield = fwd['sol']['data']
    assert_equal(leadfield.shape, (306, 22494))
    assert_equal(len(fwd['sol']['row_names']), 306)

    fwd = read_forward_solution(fname, force_fixed=True)
    leadfield = fwd['sol']['data']
    assert_equal(leadfield.shape, (306, 22494 / 3))
    assert_equal(len(fwd['sol']['row_names']), 306)


def test_apply_forward():
    """Test projection of source space data to sensor space
    """
    start = 0
    stop = 5
    n_times = stop - start - 1
    sfreq = 10.0
    t_start = 0.123

    fwd = read_forward_solution(fname, force_fixed=True)
    fwd = pick_types_forward(fwd, meg=True)

    vertno = [fwd['src'][0]['vertno'], fwd['src'][1]['vertno']]
    stc_data = np.ones((len(vertno[0]) + len(vertno[1]), n_times))
    stc = _make_stc(stc_data, t_start, 1.0 / sfreq, vertno)

    gain_sum = np.sum(fwd['sol']['data'], axis=1)

    # Evoked
    evoked = Evoked(fname_evoked, setno=0)
    evoked = apply_forward(fwd, stc, evoked, start=start, stop=stop)

    data = evoked.data
    times = evoked.times

    # do some tests
    assert_array_almost_equal(evoked.info['sfreq'], sfreq)
    assert_array_almost_equal(np.sum(data, axis=1), n_times * gain_sum)
    assert_array_almost_equal(times[0], t_start)
    assert_array_almost_equal(times[-1], t_start + (n_times - 1) / sfreq)

    # Raw
    raw = Raw(fname_raw)
    raw_proj = apply_forward_raw(fwd, stc, raw, start=start, stop=stop)
    data, times = raw_proj[:, :]

    # do some tests
    assert_array_almost_equal(raw_proj.info['sfreq'], sfreq)
    assert_array_almost_equal(np.sum(data, axis=1), n_times * gain_sum)
    assert_array_almost_equal(times[0], t_start)
    assert_array_almost_equal(times[-1], t_start + (n_times - 1) / sfreq)
