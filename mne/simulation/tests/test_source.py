import os
import os.path as op
import copy

import numpy as np
from numpy.testing import assert_array_almost_equal

from ...datasets import sample
from ... import read_label
from ... import read_forward_solution

from ..source import circular_source_labels, generate_stc

examples_folder = op.join(op.dirname(__file__), '..', '..', '..' '/examples')
data_path = sample.data_path(examples_folder)
fname_fwd = op.join(data_path, 'MEG', 'sample', 'sample_audvis-meg-oct-6-fwd.fif')
fwd = read_forward_solution(fname_fwd, force_fixed=True)
label_names = ['Aud-lh', 'Aud-rh']
labels = [read_label(op.join(data_path, 'MEG', 'sample', 'labels', '%s.label' % label))
          for label in label_names]

def test_circular_source_labels():
    """ Test generation of circular source labels """

    seeds = [0, 50000]
    hemis = [0, 1]
    labels = circular_source_labels('sample', seeds, 3, hemis)

    for label, seed, hemi in zip(labels, seeds, hemis):
        assert(np.any(label['vertices'] == seed))
        if hemi == 0:
            assert(label['hemi'] == 'lh')
        else:
            assert(label['hemi'] == 'rh')


def test_generate_stc():
    """ Test generation of source estimate """
    mylabels = copy.deepcopy(labels)

    for i, label in enumerate(mylabels):
        label['values'] = 2 * i * np.ones(len(label['values']))

    n_times = 10
    tmin = 0
    tstep = 1e-3

    stc_data = np.ones((len(labels), n_times))
    stc = generate_stc(fwd, mylabels, stc_data, tmin, tstep)

    assert(np.all(stc.data == 1.0))
    assert(stc.data.shape[1] == n_times)

    # test with function
    fun = lambda x: x ** 2
    stc = generate_stc(fwd, mylabels, stc_data, tmin, tstep, fun)

    print stc.data
    # the first label has value 0, the second value 2
    assert_array_almost_equal(stc.data[0], np.zeros(n_times))
    assert_array_almost_equal(stc.data[-1], 4 * np.ones(n_times))

