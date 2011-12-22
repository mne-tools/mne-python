import os.path as op

import numpy as np
from numpy.testing import assert_array_almost_equal

from ..datasets import sample
from ..fiff import pick_types

from ..fiff.raw import Raw

examples_folder = op.join(op.dirname(__file__), '..', '..', 'examples')
data_path = sample.data_path(examples_folder)
fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis_raw.fif')


def test_read_write():
    raw_orig = Raw(fname)
    data_orig, times_orig = raw_orig[:, :]

    tmp_fname = 'tmp.fif'
    raw_orig.save(tmp_fname)

    raw = Raw(tmp_fname)
    data, times = raw[:, :]

    assert_array_almost_equal(data_orig, data)
    assert_array_almost_equal(times_orig, times)


def test_modify_data():
    raw = Raw(fname)

    n_samp = raw.last_samp - raw.first_samp
    picks = pick_types(raw.info, meg='grad')

    data = np.random.randn(len(picks), n_samp / 2)

    raw[picks, :n_samp / 2] = data

    tmp_fname = 'tmp.fif'
    raw.save(tmp_fname)

    raw_new = Raw(tmp_fname)
    data_new, _ = raw_new[picks, :n_samp / 2]

    assert_array_almost_equal(data, data_new)