from __future__ import print_function
import os.path as op
from nose.tools import assert_true, assert_raises
import warnings
from copy import deepcopy

import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_allclose, assert_equal)

from scipy.fftpack import fft

from mne.datasets import testing
from mne import (stats, SourceEstimate, VolSourceEstimate, Label,
                 read_source_spaces, MixedSourceEstimate, read_source_estimate,
                 morph_data, extract_label_time_course,
                 spatio_temporal_tris_connectivity,
                 spatio_temporal_src_connectivity,
                 spatial_inter_hemi_connectivity)
from mne.source_estimate import (compute_morph_matrix, grade_to_vertices,
                                 grade_to_tris)

from mne.minimum_norm import read_inverse_operator
from mne.label import read_labels_from_annot, label_sign_flip
from mne.utils import (_TempDir, requires_pandas, requires_sklearn,
                       requires_h5py, run_tests_if_main, slow_test)

warnings.simplefilter('always')  # enable b/c these tests throw warnings

data_path = testing.data_path(download=False)
subjects_dir = op.join(data_path, 'subjects')
fname_inv = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-meg-eeg-oct-6-meg-inv.fif')
fname_t1 = op.join(data_path, 'subjects', 'sample', 'mri', 'T1.mgz')
fname_src = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-meg-eeg-oct-6-fwd.fif')
fname_src_3 = op.join(data_path, 'subjects', 'sample', 'bem',
                      'sample-oct-4-src.fif')
fname_stc = op.join(data_path, 'MEG', 'sample', 'sample_audvis_trunc-meg')
fname_smorph = op.join(data_path, 'MEG', 'sample',
                       'sample_audvis_trunc-meg')
fname_fmorph = op.join(data_path, 'MEG', 'sample',
                       'fsaverage_audvis_trunc-meg')
fname_vol = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-grad-vol-7-fwd-sensmap-vol.w')
fname_vsrc = op.join(data_path, 'MEG', 'sample',
                     'sample_audvis_trunc-meg-vol-7-fwd.fif')
rng = np.random.RandomState(0)


@testing.requires_testing_data
def test_spatial_inter_hemi_connectivity():
    """Test spatial connectivity between hemispheres"""
    # trivial cases
    conn = spatial_inter_hemi_connectivity(fname_src_3, 5e-6)
    assert_equal(conn.data.size, 0)
    conn = spatial_inter_hemi_connectivity(fname_src_3, 5e6)
    assert_equal(conn.data.size, np.prod(conn.shape) // 2)
    # actually interesting case (1cm), should be between 2 and 10% of verts
    src = read_source_spaces(fname_src_3)
    conn = spatial_inter_hemi_connectivity(src, 10e-3)
    conn = conn.tocsr()
    n_src = conn.shape[0]
    assert_true(n_src * 0.02 < conn.data.size < n_src * 0.10)
    assert_equal(conn[:src[0]['nuse'], :src[0]['nuse']].data.size, 0)
    assert_equal(conn[-src[1]['nuse']:, -src[1]['nuse']:].data.size, 0)
    c = (conn.T + conn) / 2. - conn
    c.eliminate_zeros()
    assert_equal(c.data.size, 0)
    # check locations
    upper_right = conn[:src[0]['nuse'], src[0]['nuse']:].toarray()
    assert_equal(upper_right.sum(), conn.sum() // 2)
    good_labels = ['S_pericallosal', 'Unknown', 'G_and_S_cingul-Mid-Post',
                   'G_cuneus']
    for hi, hemi in enumerate(('lh', 'rh')):
        has_neighbors = src[hi]['vertno'][np.where(np.any(upper_right,
                                                          axis=1 - hi))[0]]
        labels = read_labels_from_annot('sample', 'aparc.a2009s', hemi,
                                        subjects_dir=subjects_dir)
        use_labels = [l.name[:-3] for l in labels
                      if np.in1d(l.vertices, has_neighbors).any()]
        assert_true(set(use_labels) - set(good_labels) == set())


@slow_test
@testing.requires_testing_data
def test_volume_stc():
    """Test volume STCs
    """
    tempdir = _TempDir()
    N = 100
    data = np.arange(N)[:, np.newaxis]
    datas = [data, data, np.arange(2)[:, np.newaxis]]
    vertno = np.arange(N)
    vertnos = [vertno, vertno[:, np.newaxis], np.arange(2)[:, np.newaxis]]
    vertno_reads = [vertno, vertno, np.arange(2)]
    for data, vertno, vertno_read in zip(datas, vertnos, vertno_reads):
        stc = VolSourceEstimate(data, vertno, 0, 1)
        fname_temp = op.join(tempdir, 'temp-vl.stc')
        stc_new = stc
        for _ in range(2):
            stc_new.save(fname_temp)
            stc_new = read_source_estimate(fname_temp)
            assert_true(isinstance(stc_new, VolSourceEstimate))
            assert_array_equal(vertno_read, stc_new.vertices)
            assert_array_almost_equal(stc.data, stc_new.data)

    # now let's actually read a MNE-C processed file
    stc = read_source_estimate(fname_vol, 'sample')
    assert_true(isinstance(stc, VolSourceEstimate))

    assert_true('sample' in repr(stc))
    stc_new = stc
    assert_raises(ValueError, stc.save, fname_vol, ftype='whatever')
    for _ in range(2):
        fname_temp = op.join(tempdir, 'temp-vol.w')
        stc_new.save(fname_temp, ftype='w')
        stc_new = read_source_estimate(fname_temp)
        assert_true(isinstance(stc_new, VolSourceEstimate))
        assert_array_equal(stc.vertices, stc_new.vertices)
        assert_array_almost_equal(stc.data, stc_new.data)

    # save the stc as a nifti file and export
    try:
        import nibabel as nib
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            src = read_source_spaces(fname_vsrc)
        vol_fname = op.join(tempdir, 'stc.nii.gz')
        stc.save_as_volume(vol_fname, src,
                           dest='surf', mri_resolution=False)
        with warnings.catch_warnings(record=True):  # nib<->numpy
            img = nib.load(vol_fname)
        assert_true(img.shape == src[0]['shape'] + (len(stc.times),))

        with warnings.catch_warnings(record=True):  # nib<->numpy
            t1_img = nib.load(fname_t1)
        stc.save_as_volume(op.join(tempdir, 'stc.nii.gz'), src,
                           dest='mri', mri_resolution=True)
        with warnings.catch_warnings(record=True):  # nib<->numpy
            img = nib.load(vol_fname)
        assert_true(img.shape == t1_img.shape + (len(stc.times),))
        assert_array_almost_equal(img.affine, t1_img.affine, decimal=5)

        # export without saving
        img = stc.as_volume(src, dest='mri', mri_resolution=True)
        assert_true(img.shape == t1_img.shape + (len(stc.times),))
        assert_array_almost_equal(img.affine, t1_img.affine, decimal=5)

    except ImportError:
        print('Save as nifti test skipped, needs NiBabel')


@testing.requires_testing_data
def test_expand():
    """Test stc expansion
    """
    stc = read_source_estimate(fname_stc, 'sample')
    assert_true('sample' in repr(stc))
    labels_lh = read_labels_from_annot('sample', 'aparc', 'lh',
                                       subjects_dir=subjects_dir)
    new_label = labels_lh[0] + labels_lh[1]
    stc_limited = stc.in_label(new_label)
    stc_new = stc_limited.copy()
    stc_new.data.fill(0)
    for label in labels_lh[:2]:
        stc_new += stc.in_label(label).expand(stc_limited.vertices)
    assert_raises(TypeError, stc_new.expand, stc_limited.vertices[0])
    assert_raises(ValueError, stc_new.expand, [stc_limited.vertices[0]])
    # make sure we can't add unless vertno agree
    assert_raises(ValueError, stc.__add__, stc.in_label(labels_lh[0]))


def _fake_stc(n_time=10):
    verts = [np.arange(10), np.arange(90)]
    return SourceEstimate(np.random.rand(100, n_time), verts, 0, 1e-1, 'foo')


def test_io_stc():
    """Test IO for STC files
    """
    tempdir = _TempDir()
    stc = _fake_stc()
    stc.save(op.join(tempdir, "tmp.stc"))
    stc2 = read_source_estimate(op.join(tempdir, "tmp.stc"))

    assert_array_almost_equal(stc.data, stc2.data)
    assert_array_almost_equal(stc.tmin, stc2.tmin)
    assert_equal(len(stc.vertices), len(stc2.vertices))
    for v1, v2 in zip(stc.vertices, stc2.vertices):
        assert_array_almost_equal(v1, v2)
    assert_array_almost_equal(stc.tstep, stc2.tstep)


@requires_h5py
def test_io_stc_h5():
    """Test IO for STC files using HDF5
    """
    tempdir = _TempDir()
    stc = _fake_stc()
    assert_raises(ValueError, stc.save, op.join(tempdir, 'tmp'), ftype='foo')
    out_name = op.join(tempdir, 'tmp')
    stc.save(out_name, ftype='h5')
    stc.save(out_name, ftype='h5')  # test overwrite
    stc3 = read_source_estimate(out_name)
    stc4 = read_source_estimate(out_name + '-stc.h5')
    assert_raises(RuntimeError, read_source_estimate, out_name, subject='bar')
    for stc_new in stc3, stc4:
        assert_equal(stc_new.subject, stc.subject)
        assert_array_equal(stc_new.data, stc.data)
        assert_array_equal(stc_new.tmin, stc.tmin)
        assert_array_equal(stc_new.tstep, stc.tstep)
        assert_equal(len(stc_new.vertices), len(stc.vertices))
        for v1, v2 in zip(stc_new.vertices, stc.vertices):
            assert_array_equal(v1, v2)


def test_io_w():
    """Test IO for w files
    """
    tempdir = _TempDir()
    stc = _fake_stc(n_time=1)
    w_fname = op.join(tempdir, 'fake')
    stc.save(w_fname, ftype='w')
    src = read_source_estimate(w_fname)
    src.save(op.join(tempdir, 'tmp'), ftype='w')
    src2 = read_source_estimate(op.join(tempdir, 'tmp-lh.w'))
    assert_array_almost_equal(src.data, src2.data)
    assert_array_almost_equal(src.lh_vertno, src2.lh_vertno)
    assert_array_almost_equal(src.rh_vertno, src2.rh_vertno)


def test_stc_arithmetic():
    """Test arithmetic for STC files
    """
    stc = _fake_stc()
    data = stc.data.copy()

    out = list()
    for a in [data, stc]:
        a = a + a * 3 + 3 * a - a ** 2 / 2

        a += a
        a -= a
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            a /= 2 * a
        a *= -a

        a += 2
        a -= 1
        a *= -1
        a /= 2
        b = 2 + a
        b = 2 - a
        b = +a
        assert_array_equal(b.data, a.data)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            a **= 3
        out.append(a)

    assert_array_equal(out[0], out[1].data)
    assert_array_equal(stc.sqrt().data, np.sqrt(stc.data))

    stc_mean = stc.mean()
    assert_array_equal(stc_mean.data, np.mean(stc.data, 1)[:, None])


@slow_test
@testing.requires_testing_data
def test_stc_methods():
    """Test stc methods lh_data, rh_data, bin, center_of_mass, resample"""
    stc = read_source_estimate(fname_stc)

    # lh_data / rh_data
    assert_array_equal(stc.lh_data, stc.data[:len(stc.lh_vertno)])
    assert_array_equal(stc.rh_data, stc.data[len(stc.lh_vertno):])

    # bin
    bin = stc.bin(.12)
    a = np.array((1,), dtype=stc.data.dtype)
    a[0] = np.mean(stc.data[0, stc.times < .12])
    assert a[0] == bin.data[0, 0]

    assert_raises(ValueError, stc.center_of_mass, 'sample')
    assert_raises(TypeError, stc.center_of_mass, 'sample',
                  subjects_dir=subjects_dir, surf=1)
    stc.lh_data[:] = 0
    vertex, hemi, t = stc.center_of_mass('sample', subjects_dir=subjects_dir)
    assert_true(hemi == 1)
    # XXX Should design a fool-proof test case, but here were the results:
    assert_equal(vertex, 124791)
    assert_equal(np.round(t, 2), 0.12)

    stc = read_source_estimate(fname_stc)
    stc.subject = 'sample'
    label_lh = read_labels_from_annot('sample', 'aparc', 'lh',
                                      subjects_dir=subjects_dir)[0]
    label_rh = read_labels_from_annot('sample', 'aparc', 'rh',
                                      subjects_dir=subjects_dir)[0]
    label_both = label_lh + label_rh
    for label in (label_lh, label_rh, label_both):
        assert_true(isinstance(stc.shape, tuple) and len(stc.shape) == 2)
        stc_label = stc.in_label(label)
        if label.hemi != 'both':
            if label.hemi == 'lh':
                verts = stc_label.vertices[0]
            else:  # label.hemi == 'rh':
                verts = stc_label.vertices[1]
            n_vertices_used = len(label.get_vertices_used(verts))
            assert_equal(len(stc_label.data), n_vertices_used)
    stc_lh = stc.in_label(label_lh)
    assert_raises(ValueError, stc_lh.in_label, label_rh)
    label_lh.subject = 'foo'
    assert_raises(RuntimeError, stc.in_label, label_lh)

    stc_new = deepcopy(stc)
    o_sfreq = 1.0 / stc.tstep
    # note that using no padding for this STC reduces edge ringing...
    stc_new.resample(2 * o_sfreq, npad=0, n_jobs=2)
    assert_true(stc_new.data.shape[1] == 2 * stc.data.shape[1])
    assert_true(stc_new.tstep == stc.tstep / 2)
    stc_new.resample(o_sfreq, npad=0)
    assert_true(stc_new.data.shape[1] == stc.data.shape[1])
    assert_true(stc_new.tstep == stc.tstep)
    assert_array_almost_equal(stc_new.data, stc.data, 5)


@testing.requires_testing_data
def test_extract_label_time_course():
    """Test extraction of label time courses from stc
    """
    n_stcs = 3
    n_times = 50

    src = read_inverse_operator(fname_inv)['src']
    vertices = [src[0]['vertno'], src[1]['vertno']]
    n_verts = len(vertices[0]) + len(vertices[1])

    # get some labels
    labels_lh = read_labels_from_annot('sample', hemi='lh',
                                       subjects_dir=subjects_dir)
    labels_rh = read_labels_from_annot('sample', hemi='rh',
                                       subjects_dir=subjects_dir)
    labels = list()
    labels.extend(labels_lh[:5])
    labels.extend(labels_rh[:4])

    n_labels = len(labels)

    label_means = np.arange(n_labels)[:, None] * np.ones((n_labels, n_times))
    label_maxs = np.arange(n_labels)[:, None] * np.ones((n_labels, n_times))

    # compute the mean with sign flip
    label_means_flipped = np.zeros_like(label_means)
    for i, label in enumerate(labels):
        label_means_flipped[i] = i * np.mean(label_sign_flip(label, src))

    # generate some stc's with known data
    stcs = list()
    for i in range(n_stcs):
        data = np.zeros((n_verts, n_times))
        # set the value of the stc within each label
        for j, label in enumerate(labels):
            if label.hemi == 'lh':
                idx = np.intersect1d(vertices[0], label.vertices)
                idx = np.searchsorted(vertices[0], idx)
            elif label.hemi == 'rh':
                idx = np.intersect1d(vertices[1], label.vertices)
                idx = len(vertices[0]) + np.searchsorted(vertices[1], idx)
            data[idx] = label_means[j]

        this_stc = SourceEstimate(data, vertices, 0, 1)
        stcs.append(this_stc)

    # test some invalid inputs
    assert_raises(ValueError, extract_label_time_course, stcs, labels,
                  src, mode='notamode')

    # have an empty label
    empty_label = labels[0].copy()
    empty_label.vertices += 1000000
    assert_raises(ValueError, extract_label_time_course, stcs, empty_label,
                  src, mode='mean')

    # but this works:
    with warnings.catch_warnings(record=True):  # empty label
        tc = extract_label_time_course(stcs, empty_label, src, mode='mean',
                                       allow_empty=True)
    for arr in tc:
        assert_true(arr.shape == (1, n_times))
        assert_array_equal(arr, np.zeros((1, n_times)))

    # test the different modes
    modes = ['mean', 'mean_flip', 'pca_flip', 'max']

    for mode in modes:
        label_tc = extract_label_time_course(stcs, labels, src, mode=mode)
        label_tc_method = [stc.extract_label_time_course(labels, src,
                           mode=mode) for stc in stcs]
        assert_true(len(label_tc) == n_stcs)
        assert_true(len(label_tc_method) == n_stcs)
        for tc1, tc2 in zip(label_tc, label_tc_method):
            assert_true(tc1.shape == (n_labels, n_times))
            assert_true(tc2.shape == (n_labels, n_times))
            assert_true(np.allclose(tc1, tc2, rtol=1e-8, atol=1e-16))
            if mode == 'mean':
                assert_array_almost_equal(tc1, label_means)
            if mode == 'mean_flip':
                assert_array_almost_equal(tc1, label_means_flipped)
            if mode == 'max':
                assert_array_almost_equal(tc1, label_maxs)

    # test label with very few vertices (check SVD conditionals)
    label = Label(vertices=src[0]['vertno'][:2], hemi='lh')
    x = label_sign_flip(label, src)
    assert_true(len(x) == 2)
    label = Label(vertices=[], hemi='lh')
    x = label_sign_flip(label, src)
    assert_true(x.size == 0)


@slow_test
@testing.requires_testing_data
def test_morph_data():
    """Test morphing of data
    """
    tempdir = _TempDir()
    subject_from = 'sample'
    subject_to = 'fsaverage'
    stc_from = read_source_estimate(fname_smorph, subject='sample')
    stc_to = read_source_estimate(fname_fmorph)
    # make sure we can specify grade
    stc_from.crop(0.09, 0.1)  # for faster computation
    stc_to.crop(0.09, 0.1)  # for faster computation
    assert_array_equal(stc_to.time_as_index([0.09, 0.1], use_rounding=True),
                       [0, len(stc_to.times) - 1])
    assert_raises(ValueError, stc_from.morph, subject_to, grade=3, smooth=-1,
                  subjects_dir=subjects_dir)
    stc_to1 = stc_from.morph(subject_to, grade=3, smooth=12, buffer_size=1000,
                             subjects_dir=subjects_dir)
    stc_to1.save(op.join(tempdir, '%s_audvis-meg' % subject_to))
    # Morphing to a density that is too high should raise an informative error
    # (here we need to push to grade=6, but for some subjects even grade=5
    # will break)
    assert_raises(ValueError, stc_to1.morph, subject_from, grade=6,
                  subjects_dir=subjects_dir)
    # make sure we can specify vertices
    vertices_to = grade_to_vertices(subject_to, grade=3,
                                    subjects_dir=subjects_dir)
    stc_to2 = morph_data(subject_from, subject_to, stc_from,
                         grade=vertices_to, smooth=12, buffer_size=1000,
                         subjects_dir=subjects_dir)
    # make sure we can use different buffer_size
    stc_to3 = morph_data(subject_from, subject_to, stc_from,
                         grade=vertices_to, smooth=12, buffer_size=3,
                         subjects_dir=subjects_dir)
    # make sure we get a warning about # of steps
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        morph_data(subject_from, subject_to, stc_from,
                   grade=vertices_to, smooth=1, buffer_size=3,
                   subjects_dir=subjects_dir)
    assert_equal(len(w), 2)

    assert_array_almost_equal(stc_to.data, stc_to1.data, 5)
    assert_array_almost_equal(stc_to1.data, stc_to2.data)
    assert_array_almost_equal(stc_to1.data, stc_to3.data)
    # make sure precomputed morph matrices work
    morph_mat = compute_morph_matrix(subject_from, subject_to,
                                     stc_from.vertices, vertices_to,
                                     smooth=12, subjects_dir=subjects_dir)
    stc_to3 = stc_from.morph_precomputed(subject_to, vertices_to, morph_mat)
    assert_array_almost_equal(stc_to1.data, stc_to3.data)
    assert_raises(ValueError, stc_from.morph_precomputed,
                  subject_to, vertices_to, 'foo')
    assert_raises(ValueError, stc_from.morph_precomputed,
                  subject_to, [vertices_to[0]], morph_mat)
    assert_raises(ValueError, stc_from.morph_precomputed,
                  subject_to, [vertices_to[0][:-1], vertices_to[1]], morph_mat)
    assert_raises(ValueError, stc_from.morph_precomputed, subject_to,
                  vertices_to, morph_mat, subject_from='foo')

    # steps warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        compute_morph_matrix(subject_from, subject_to,
                             stc_from.vertices, vertices_to,
                             smooth=1, subjects_dir=subjects_dir)
    assert_equal(len(w), 2)

    mean_from = stc_from.data.mean(axis=0)
    mean_to = stc_to1.data.mean(axis=0)
    assert_true(np.corrcoef(mean_to, mean_from).min() > 0.999)

    # make sure we can fill by morphing
    stc_to5 = morph_data(subject_from, subject_to, stc_from, grade=None,
                         smooth=12, buffer_size=3, subjects_dir=subjects_dir)
    assert_true(stc_to5.data.shape[0] == 163842 + 163842)

    # Morph sparse data
    # Make a sparse stc
    stc_from.vertices[0] = stc_from.vertices[0][[100, 500]]
    stc_from.vertices[1] = stc_from.vertices[1][[200]]
    stc_from._data = stc_from._data[:3]

    assert_raises(RuntimeError, stc_from.morph, subject_to, sparse=True,
                  grade=5, subjects_dir=subjects_dir)

    stc_to_sparse = stc_from.morph(subject_to, grade=None, sparse=True,
                                   subjects_dir=subjects_dir)
    assert_array_almost_equal(np.sort(stc_from.data.sum(axis=1)),
                              np.sort(stc_to_sparse.data.sum(axis=1)))
    assert_equal(len(stc_from.rh_vertno), len(stc_to_sparse.rh_vertno))
    assert_equal(len(stc_from.lh_vertno), len(stc_to_sparse.lh_vertno))
    assert_equal(stc_to_sparse.subject, subject_to)
    assert_equal(stc_from.tmin, stc_from.tmin)
    assert_equal(stc_from.tstep, stc_from.tstep)

    stc_from.vertices[0] = np.array([], dtype=np.int64)
    stc_from._data = stc_from._data[:1]

    stc_to_sparse = stc_from.morph(subject_to, grade=None, sparse=True,
                                   subjects_dir=subjects_dir)
    assert_array_almost_equal(np.sort(stc_from.data.sum(axis=1)),
                              np.sort(stc_to_sparse.data.sum(axis=1)))
    assert_equal(len(stc_from.rh_vertno), len(stc_to_sparse.rh_vertno))
    assert_equal(len(stc_from.lh_vertno), len(stc_to_sparse.lh_vertno))
    assert_equal(stc_to_sparse.subject, subject_to)
    assert_equal(stc_from.tmin, stc_from.tmin)
    assert_equal(stc_from.tstep, stc_from.tstep)


def _my_trans(data):
    """FFT that adds an additional dimension by repeating result"""
    data_t = fft(data)
    data_t = np.concatenate([data_t[:, :, None], data_t[:, :, None]], axis=2)
    return data_t, None


def test_transform_data():
    """Test applying linear (time) transform to data"""
    # make up some data
    n_sensors, n_vertices, n_times = 10, 20, 4
    kernel = rng.randn(n_vertices, n_sensors)
    sens_data = rng.randn(n_sensors, n_times)

    vertices = np.arange(n_vertices)
    data = np.dot(kernel, sens_data)

    for idx, tmin_idx, tmax_idx in\
            zip([None, np.arange(n_vertices // 2, n_vertices)],
                [None, 1], [None, 3]):

        if idx is None:
            idx_use = slice(None, None)
        else:
            idx_use = idx

        data_f, _ = _my_trans(data[idx_use, tmin_idx:tmax_idx])

        for stc_data in (data, (kernel, sens_data)):
            stc = VolSourceEstimate(stc_data, vertices=vertices,
                                    tmin=0., tstep=1.)
            stc_data_t = stc.transform_data(_my_trans, idx=idx,
                                            tmin_idx=tmin_idx,
                                            tmax_idx=tmax_idx)
            assert_allclose(data_f, stc_data_t)


def test_transform():
    """Test applying linear (time) transform to data"""
    # make up some data
    n_verts_lh, n_verts_rh, n_times = 10, 10, 10
    vertices = [np.arange(n_verts_lh), n_verts_lh + np.arange(n_verts_rh)]
    data = rng.randn(n_verts_lh + n_verts_rh, n_times)
    stc = SourceEstimate(data, vertices=vertices, tmin=-0.1, tstep=0.1)

    # data_t.ndim > 2 & copy is True
    stcs_t = stc.transform(_my_trans, copy=True)
    assert_true(isinstance(stcs_t, list))
    assert_array_equal(stc.times, stcs_t[0].times)
    assert_equal(stc.vertices, stcs_t[0].vertices)

    data = np.concatenate((stcs_t[0].data[:, :, None],
                           stcs_t[1].data[:, :, None]), axis=2)
    data_t = stc.transform_data(_my_trans)
    assert_array_equal(data, data_t)  # check against stc.transform_data()

    # data_t.ndim > 2 & copy is False
    assert_raises(ValueError, stc.transform, _my_trans, copy=False)

    # data_t.ndim = 2 & copy is True
    tmp = deepcopy(stc)
    stc_t = stc.transform(np.abs, copy=True)
    assert_true(isinstance(stc_t, SourceEstimate))
    assert_array_equal(stc.data, tmp.data)  # xfrm doesn't modify original?

    # data_t.ndim = 2 & copy is False
    times = np.round(1000 * stc.times)
    verts = np.arange(len(stc.lh_vertno),
                      len(stc.lh_vertno) + len(stc.rh_vertno), 1)
    verts_rh = stc.rh_vertno
    t_idx = [np.where(times >= -50)[0][0], np.where(times <= 500)[0][-1]]
    data_t = stc.transform_data(np.abs, idx=verts, tmin_idx=t_idx[0],
                                tmax_idx=t_idx[-1])
    stc.transform(np.abs, idx=verts, tmin=-50, tmax=500, copy=False)
    assert_true(isinstance(stc, SourceEstimate))
    assert_true((stc.tmin == 0.) & (stc.times[-1] == 0.5))
    assert_true(len(stc.vertices[0]) == 0)
    assert_equal(stc.vertices[1], verts_rh)
    assert_array_equal(stc.data, data_t)

    times = np.round(1000 * stc.times)
    t_idx = [np.where(times >= 0)[0][0], np.where(times <= 250)[0][-1]]
    data_t = stc.transform_data(np.abs, tmin_idx=t_idx[0], tmax_idx=t_idx[-1])
    stc.transform(np.abs, tmin=0, tmax=250, copy=False)
    assert_true((stc.tmin == 0.) & (stc.times[-1] == 0.2))
    assert_array_equal(stc.data, data_t)


@requires_sklearn
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


@testing.requires_testing_data
def test_spatio_temporal_src_connectivity():
    """Test spatio-temporal connectivity from source spaces"""
    tris = np.array([[0, 1, 2], [3, 4, 5]])
    src = [dict(), dict()]
    connectivity = spatio_temporal_tris_connectivity(tris, 2)
    src[0]['use_tris'] = np.array([[0, 1, 2]])
    src[1]['use_tris'] = np.array([[0, 1, 2]])
    src[0]['vertno'] = np.array([0, 1, 2])
    src[1]['vertno'] = np.array([0, 1, 2])
    connectivity2 = spatio_temporal_src_connectivity(src, 2)
    assert_array_equal(connectivity.todense(), connectivity2.todense())
    # add test for dist connectivity
    src[0]['dist'] = np.ones((3, 3)) - np.eye(3)
    src[1]['dist'] = np.ones((3, 3)) - np.eye(3)
    src[0]['vertno'] = [0, 1, 2]
    src[1]['vertno'] = [0, 1, 2]
    connectivity3 = spatio_temporal_src_connectivity(src, 2, dist=2)
    assert_array_equal(connectivity.todense(), connectivity3.todense())
    # add test for source space connectivity with omitted vertices
    inverse_operator = read_inverse_operator(fname_inv)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        src_ = inverse_operator['src']
        connectivity = spatio_temporal_src_connectivity(src_, n_times=2)
    assert_equal(len(w), 1)
    a = connectivity.shape[0] / 2
    b = sum([s['nuse'] for s in inverse_operator['src']])
    assert_true(a == b)

    assert_equal(grade_to_tris(5).shape, [40960, 3])


@requires_pandas
def test_to_data_frame():
    """Test stc Pandas exporter"""
    n_vert, n_times = 10, 5
    vertices = [np.arange(n_vert, dtype=np.int), np.empty(0, dtype=np.int)]
    data = rng.randn(n_vert, n_times)
    stc_surf = SourceEstimate(data, vertices=vertices, tmin=0, tstep=1,
                              subject='sample')
    stc_vol = VolSourceEstimate(data, vertices=vertices[0], tmin=0, tstep=1,
                                subject='sample')
    for stc in [stc_surf, stc_vol]:
        assert_raises(ValueError, stc.to_data_frame, index=['foo', 'bar'])
        for ncat, ind in zip([1, 0], ['time', ['subject', 'time']]):
            df = stc.to_data_frame(index=ind)
            assert_true(df.index.names == ind
                        if isinstance(ind, list) else [ind])
            assert_array_equal(df.values.T[ncat:], stc.data)
            # test that non-indexed data were present as categorial variables
            assert_true(all([c in ['time', 'subject'] for c in
                             df.reset_index().columns][:2]))


def test_get_peak():
    """Test peak getter
    """
    n_vert, n_times = 10, 5
    vertices = [np.arange(n_vert, dtype=np.int), np.empty(0, dtype=np.int)]
    data = rng.randn(n_vert, n_times)
    stc_surf = SourceEstimate(data, vertices=vertices, tmin=0, tstep=1,
                              subject='sample')

    stc_vol = VolSourceEstimate(data, vertices=vertices[0], tmin=0, tstep=1,
                                subject='sample')

    for ii, stc in enumerate([stc_surf, stc_vol]):
        assert_raises(ValueError, stc.get_peak, tmin=-100)
        assert_raises(ValueError, stc.get_peak, tmax=90)
        assert_raises(ValueError, stc.get_peak, tmin=0.002, tmax=0.001)

        vert_idx, time_idx = stc.get_peak()
        vertno = np.concatenate(stc.vertices) if ii == 0 else stc.vertices
        assert_true(vert_idx in vertno)
        assert_true(time_idx in stc.times)

        ch_idx, time_idx = stc.get_peak(vert_as_index=True,
                                        time_as_index=True)
        assert_true(vert_idx < stc.data.shape[0])
        assert_true(time_idx < len(stc.times))


@testing.requires_testing_data
def test_mixed_stc():
    """Test source estimate from mixed source space
    """
    N = 90  # number of sources
    T = 2  # number of time points
    S = 3  # number of source spaces

    data = rng.randn(N, T)
    vertno = S * [np.arange(N // S)]

    # make sure error is raised if vertices are not a list of length >= 2
    assert_raises(ValueError, MixedSourceEstimate, data=data,
                  vertices=[np.arange(N)])

    stc = MixedSourceEstimate(data, vertno, 0, 1)

    vol = read_source_spaces(fname_vsrc)

    # make sure error is raised for plotting surface with volume source
    assert_raises(ValueError, stc.plot_surface, src=vol)


run_tests_if_main()
