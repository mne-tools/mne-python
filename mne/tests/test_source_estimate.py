import os.path as op
from nose.tools import assert_true

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from ..datasets import sample
from .. import stats
from .. import read_stc, write_stc, SourceEstimate, morph_data
from ..source_estimate import spatio_temporal_tris_connectivity, \
                                spatio_temporal_src_connectivity


examples_folder = op.join(op.dirname(__file__), '..', '..', 'examples')
data_path = sample.data_path(examples_folder)
fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis-meg-lh.stc')


def test_io_stc():
    """Test IO for STC files
    """
    stc = read_stc(fname)

    write_stc("tmp.stc", stc['tmin'], stc['tstep'],
                             stc['vertices'], stc['data'])
    stc2 = read_stc("tmp.stc")

    assert_array_almost_equal(stc['data'], stc2['data'])
    assert_array_almost_equal(stc['tmin'], stc2['tmin'])
    assert_array_almost_equal(stc['vertices'], stc2['vertices'])
    assert_array_almost_equal(stc['tstep'], stc2['tstep'])


def test_io_w():
    """Test IO for w files
    """
    w_fname = op.join(data_path, 'MEG', 'sample',
                      'sample_audvis-meg-oct-6-fwd-sensmap')

    src = SourceEstimate(w_fname)

    src.save('tmp', ftype='w')

    src2 = SourceEstimate('tmp-lh.w')

    assert_array_almost_equal(src.data, src2.data)
    assert_array_almost_equal(src.lh_vertno, src2.lh_vertno)
    assert_array_almost_equal(src.rh_vertno, src2.rh_vertno)


def test_stc_arithmetic():
    """Test arithmetic for STC files
    """
    fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis-meg')
    stc = SourceEstimate(fname)
    data = stc.data.copy()

    out = list()
    for a in [data, stc]:
        a = a + a * 3 + 3 * a - a ** 2 / 2

        a += a
        a -= a
        a /= 2 * a
        a *= -a

        a += 2
        a -= 1
        a *= -1
        a /= 2
        a **= 3
        out.append(a)

    assert_array_equal(out[0], out[1].data)
    assert_array_equal(stc.sqrt().data, np.sqrt(stc.data))


def test_morph_data():
    """Test morphing of data
    """
    subject_from = 'sample'
    subject_to = 'fsaverage'
    fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis-meg')
    stc_from = SourceEstimate(fname)
    stc_to = morph_data(subject_from, subject_to, stc_from,
                            grade=3, smooth=12)

    stc_to.save('%s_audvis-meg' % subject_to)

    mean_from = stc_from.data.mean(axis=0)
    mean_to = stc_to.data.mean(axis=0)
    assert_true(np.corrcoef(mean_to, mean_from).min() > 0.99)


def test_spatio_temporal_tris_connectivity():
    """Test spatio-temporal connectivity from triangles"""
    tris = np.array([[0, 1, 2], [3, 4, 5]])
    connectivity = spatio_temporal_tris_connectivity(tris, 2)
    x = [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    components = stats.cluster_level._get_components(np.array(x), connectivity)
    assert_array_equal(components,
                       [0, 0, -2, -2, -2, -2, 0, -2, -2, -2, -2, 1])


def test_spatio_temporal_src_connectivity():
    """Test spatio-temporal connectivity from source spaaces"""
    tris = np.array([[0, 1, 2], [3, 4, 5]])
    src = [dict(), dict()]
    connectivity = spatio_temporal_tris_connectivity(tris, 2)
    src[0]['use_tris'] = np.array([[0, 1, 2]])
    src[1]['use_tris'] = np.array([[0, 1, 2]])
    connectivity2 = spatio_temporal_src_connectivity(src, 2)
    assert_array_equal(connectivity.todense(), connectivity2.todense())
