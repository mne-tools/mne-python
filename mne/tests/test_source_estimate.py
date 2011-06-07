import os.path as op

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

import mne
from mne.datasets import sample
from mne import stats
from mne.source_estimate import spatio_temporal_tris_connectivity, \
                                spatio_temporal_src_connectivity


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
    subject_from = 'sample'
    subject_to = 'morph'
    fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis-meg')
    stc_from = mne.SourceEstimate(fname)
    stc_to = mne.morph_data(subject_from, subject_to, stc_from,
                            grade=3, smooth=12)

    stc_to.save('%s_audvis-meg' % subject_to)

    mean_from = stc_from.data.mean(axis=0)
    mean_to = stc_to.data.mean(axis=0)
    assert np.corrcoef(mean_to, mean_from).min() > 0.99


def test_spatio_temporal_tris_connectivity():
    """Test spatio-temporal connectivity"""
    tris = np.array([[0, 1, 2], [3, 4, 5]])
    connectivity = spatio_temporal_tris_connectivity(tris, 2)
    x = [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    components = stats.cluster_level._get_components(np.array(x), connectivity)
    assert_array_equal(components,
                       [0, 0, -2, -2, -2, -2, 0, -2, -2, -2, -2, 1])


def test_spatio_temporal_src_connectivity():
    """Test spatio-temporal connectivity"""
    tris = np.array([[0, 1, 2], [3, 4, 5]])
    src = [dict(), dict()]
    connectivity = spatio_temporal_tris_connectivity(tris, 2)
    src[0]['use_tris'] = np.array([[0, 1, 2]])
    src[1]['use_tris'] = np.array([[0, 1, 2]])
    connectivity2 = spatio_temporal_src_connectivity(src, 2)
    assert_array_equal(connectivity.todense(), connectivity2.todense())

