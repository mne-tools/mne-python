import os.path as op
from nose.tools import assert_true, assert_raises

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal

from mne.datasets import sample
from mne import stats, SourceEstimate
from mne import read_stc, write_stc, read_source_estimate, morph_data
from mne.source_estimate import spatio_temporal_tris_connectivity, \
                                spatio_temporal_src_connectivity, \
                                compute_morph_matrix, grade_to_vertices, \
                                morph_data_precomputed


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

    src = read_source_estimate(w_fname)

    src.save('tmp', ftype='w')

    src2 = read_source_estimate('tmp-lh.w')

    assert_array_almost_equal(src.data, src2.data)
    assert_array_almost_equal(src.lh_vertno, src2.lh_vertno)
    assert_array_almost_equal(src.rh_vertno, src2.rh_vertno)


def test_stc_arithmetic():
    """Test arithmetic for STC files
    """
    fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis-meg')
    stc = read_source_estimate(fname)
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


def test_stc_methods():
    """Test stc methods lh_data, rh_data, bin(), and center_of_mass()
    """
    fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis-meg')
    stc = read_source_estimate(fname)

    # lh_data / rh_data
    assert_array_equal(stc.lh_data, stc.data[:len(stc.lh_vertno)])
    assert_array_equal(stc.rh_data, stc.data[len(stc.lh_vertno):])

    # bin
    bin = stc.bin(.12)
    a = np.array((1,), dtype=stc.data.dtype)
    a[0] = np.mean(stc.data[0, stc.times < .12])
    assert a[0] == bin.data[0, 0]

    assert_raises(ValueError, stc.center_of_mass, 'sample')
    stc.lh_data[:] = 0
    vertex, hemi, t = stc.center_of_mass('sample')
    assert_true(hemi == 1)
    # XXX Should design a fool-proof test case, but here were the results:
    assert_true(vertex == 92717)
    assert_true(np.round(t, 3) == 0.123)


def test_morph_data():
    """Test morphing of data
    """
    subject_from = 'sample'
    subject_to = 'fsaverage'
    fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis-meg')
    stc_from = read_source_estimate(fname)
    stc_from.crop(0.09, 0.1)  # for faster computation
    # After running this:
    #    stc_from.save('%s_audvis-meg-cropped' % subject_from)
    # this was run from a command line:
    #    mne_make_movie --stcin sample_audvis-meg-cropped-lh.stc
    #        --subject sample --morph fsaverage --smooth 12 --morphgrade 3
    #        --stc fsaverage_audvis-meg-cropped
    # XXX These files should eventually be moved to the sample dataset and
    # removed from mne/fiff/tests/data/
    fname = op.join(op.dirname(__file__), '..', 'fiff', 'tests', 'data',
                    'fsaverage_audvis-meg-cropped')
    stc_to = read_source_estimate(fname)
    stc_to1 = morph_data(subject_from, subject_to, stc_from,
                            grade=3, smooth=12, buffer_size=1000)
    stc_to1.save('%s_audvis-meg' % subject_to)
    stc_to2 = morph_data(subject_from, subject_to, stc_from,
                            grade=3, smooth=12, buffer_size=3)
    # indexing silliness here due to mne_make_movie's indexing oddities
    assert_array_almost_equal(stc_to.data, stc_to1.data[:, 0][:, None], 5)
    assert_array_almost_equal(stc_to1.data, stc_to2.data)
    # make sure precomputed morph matrices work
    vertices_to = grade_to_vertices(subject_to, grade=3)
    morph_mat = compute_morph_matrix(subject_from, subject_to,
                                     stc_from.vertno, vertices_to,
                                     smooth=12)
    stc_to3 = morph_data_precomputed(subject_from, subject_to,
                                     stc_from, vertices_to, morph_mat)
    assert_array_almost_equal(stc_to1.data, stc_to3.data)

    mean_from = stc_from.data.mean(axis=0)
    mean_to = stc_to1.data.mean(axis=0)
    assert_true(np.corrcoef(mean_to, mean_from).min() > 0.999)

    # test two types of morphing:
    # 1) make sure we can fill by morphing
    stc_to5 = morph_data(subject_from, subject_to, stc_from,
                            grade=None, smooth=12, buffer_size=3)
    assert_true(stc_to5.data.shape[0] == 163842 + 163842)

    # 2) make sure we can specify vertices
    vertices_to = [np.arange(10242), np.arange(10242)]
    stc_to3 = morph_data(subject_from, subject_to, stc_from,
                            grade=vertices_to, smooth=12, buffer_size=3)
    stc_to4 = morph_data(subject_from, subject_to, stc_from,
                            grade=5, smooth=12, buffer_size=3)
    assert_array_almost_equal(stc_to3.data, stc_to4.data)


def test_spatio_temporal_tris_connectivity():
    """Test spatio-temporal connectivity from triangles"""
    tris = np.array([[0, 1, 2], [3, 4, 5]])
    connectivity = spatio_temporal_tris_connectivity(tris, 2)
    x = [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    components = stats.cluster_level._get_components(np.array(x), connectivity)
    # _get_components works differently now...
    old_fmt = [0, 0, -2, -2, -2, -2, 0, -2, -2, -2, -2, 1]
    new_fmt = np.array(old_fmt)
    new_fmt = [np.nonzero(new_fmt == v)[0]
               for v in np.unique(new_fmt[new_fmt >= 0])]
    assert_true(len(new_fmt), len(components))
    for c, n in zip(components, new_fmt):
        assert_array_equal(c, n)


def test_spatio_temporal_src_connectivity():
    """Test spatio-temporal connectivity from source spaces"""
    tris = np.array([[0, 1, 2], [3, 4, 5]])
    src = [dict(), dict()]
    connectivity = spatio_temporal_tris_connectivity(tris, 2)
    src[0]['use_tris'] = np.array([[0, 1, 2]])
    src[1]['use_tris'] = np.array([[0, 1, 2]])
    connectivity2 = spatio_temporal_src_connectivity(src, 2)
    assert_array_equal(connectivity.todense(), connectivity2.todense())
    # add test for dist connectivity
    src[0]['dist'] = np.ones((3, 3)) - np.eye(3)
    src[1]['dist'] = np.ones((3, 3)) - np.eye(3)
    src[0]['vertno'] = [0, 1, 2]
    src[1]['vertno'] = [0, 1, 2]
    connectivity3 = spatio_temporal_src_connectivity(src, 2, dist=2)
    assert_array_equal(connectivity.todense(), connectivity3.todense())