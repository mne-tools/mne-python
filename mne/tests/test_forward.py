import os.path as op

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_equal

from ..datasets import sample
from ..fiff import Raw, pick_channels
from ..minimum_norm.inverse import _make_stc
from .. import read_forward_solution, apply_forward_raw, SourceEstimate


examples_folder = op.join(op.dirname(__file__), '..', '..', 'examples')
data_path = sample.data_path(examples_folder)
fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis-meg-oct-6-fwd.fif')

fname_raw = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')


def test_io_forward():
    """Test IO for forward solutions
    """
    fwd = read_forward_solution(fname)
    fwd = read_forward_solution(fname, force_fixed=True)
    fwd = read_forward_solution(fname, surf_ori=True)
    leadfield = fwd['sol']['data']
    # XXX : test something


def test_apply_forward():
    """Test projection of source space data to sensor space
    """
    start = 0
    stop = 5
    n_times = stop - start - 1
    sfreq = 10.0
    t_start = 0.123

    fwd = read_forward_solution(fname, force_fixed=True)

    vertno = [fwd['src'][0]['vertno'], fwd['src'][1]['vertno']]
    stc_data = np.ones((len(vertno[0]) + len(vertno[1]), n_times))
    stc = _make_stc(stc_data, t_start, 1.0 / sfreq, vertno)

    raw = Raw(fname_raw)

    raw_proj = apply_forward_raw(fwd, stc, raw, start=start, stop=stop)

    proj_data, proj_times = raw_proj[:, :]

    sel = pick_channels(fwd['sol']['row_names'],
                        include=raw_proj.info['ch_names'])

    gain_sum = np.sum(fwd['sol']['data'][sel, :], axis=1)

    # do some tests
    assert_array_almost_equal(np.sum(proj_data, axis=1), n_times * gain_sum)
    assert_array_almost_equal(raw_proj.info['sfreq'], sfreq)
    assert_array_almost_equal(proj_times[0], t_start)
    assert_array_almost_equal(proj_times[-1], t_start + (n_times - 1) / sfreq)
