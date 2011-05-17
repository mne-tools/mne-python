import os.path as op

import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy import linalg

import mne
from mne.datasets import sample

examples_folder = op.join(op.dirname(__file__), '..', '..', 'examples')
data_path = sample.data_path(examples_folder)
fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis-meg-lh.stc')


def test_io_stc():
    """Test IO for STC files
    """
    stc = mne.read_stc(fname)

    mne.write_stc("tmp.stc", stc['tmin'], stc['tstep'],
                             stc['vertices'], stc['data'])
    stc2 = mne.read_stc("tmp.stc")

    assert_array_almost_equal(stc['data'], stc2['data'])
    assert_array_almost_equal(stc['tmin'], stc2['tmin'])
    assert_array_almost_equal(stc['vertices'], stc2['vertices'])
    assert_array_almost_equal(stc['tstep'], stc2['tstep'])


def test_morph_data():
    """Test morphing of data
    """
    import mne
    from mne.datasets import sample

    subject_from = 'sample'
    subject_to = 'morph'

    fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis-meg')
    src_fname = op.join(data_path, 'MEG', 'sample',
                                            'sample_audvis-meg-oct-6-fwd.fif')

    stc_from = mne.SourceEstimate(fname)
    src_from = mne.read_source_spaces(src_fname)

    stc_to = mne.morph_data(subject_from, subject_to, src_from, stc_from, 3)

    stc_to.save('%s_audvis-meg' % subject_to)

    mean_from = stc_from.data.mean(axis=0)
    mean_to = stc_to.data.mean(axis=0)
    assert np.corrcoef(mean_to, mean_from).min() > 0.99
