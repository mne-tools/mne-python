import os.path as op
import copy

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from nose.tools import assert_true

from mne.datasets import sample
from mne import read_label, read_forward_solution
from mne.simulation.source import generate_stc, generate_sparse_stc


examples_folder = op.join(op.dirname(__file__), '..', '..', '..' '/examples')
data_path = sample.data_path(examples_folder)
fname_fwd = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis-meg-oct-6-fwd.fif')
fwd = read_forward_solution(fname_fwd, force_fixed=True)
label_names = ['Aud-lh', 'Aud-rh']
labels = [read_label(op.join(data_path, 'MEG', 'sample', 'labels',
                    '%s.label' % label)) for label in label_names]


def test_generate_stc():
    """ Test generation of source estimate """
    mylabels = copy.deepcopy(labels)

    for i, label in enumerate(mylabels):
        label['values'] = 2 * i * np.ones(len(label['values']))

    n_times = 10
    tmin = 0
    tstep = 1e-3

    stc_data = np.ones((len(labels), n_times))
    stc = generate_stc(fwd['src'], mylabels, stc_data, tmin, tstep)

    assert_true(np.all(stc.data == 1.0))
    assert_true(stc.data.shape[1] == n_times)

    # test with function
    fun = lambda x: x ** 2
    stc = generate_stc(fwd['src'], mylabels, stc_data, tmin, tstep, fun)

    print stc.data
    # the first label has value 0, the second value 2
    assert_array_almost_equal(stc.data[0], np.zeros(n_times))
    assert_array_almost_equal(stc.data[-1], 4 * np.ones(n_times))


def test_generate_sparse_stc():
    """ Test generation of sparse source estimate """

    n_times = 10
    tmin = 0
    tstep = 1e-3

    stc_data = np.ones((len(labels), n_times))
    stc_1 = generate_sparse_stc(fwd['src'], labels, stc_data, tmin, tstep, 0)

    assert_true(np.all(stc_1.data == 1.0))
    assert_true(stc_1.data.shape[0] == len(labels))
    assert_true(stc_1.data.shape[1] == n_times)

    # make sure we get the same result when using the same seed
    stc_2 = generate_sparse_stc(fwd['src'], labels, stc_data, tmin, tstep, 0)

    assert_array_equal(stc_1.lh_vertno, stc_2.lh_vertno)
    assert_array_equal(stc_1.rh_vertno, stc_2.rh_vertno)
