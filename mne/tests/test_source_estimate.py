# -*- coding: utf-8 -*-
#
# License: BSD (3-clause)

from copy import deepcopy
import os.path as op

import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_allclose, assert_equal)
import pytest
from scipy import sparse

from mne import (stats, SourceEstimate, VectorSourceEstimate,
                 VolSourceEstimate, Label, read_source_spaces,
                 read_evokeds, MixedSourceEstimate, find_events, Epochs,
                 read_source_estimate, extract_label_time_course,
                 spatio_temporal_tris_connectivity,
                 spatio_temporal_src_connectivity, read_cov,
                 spatial_inter_hemi_connectivity, read_forward_solution,
                 spatial_src_connectivity, spatial_tris_connectivity,
                 SourceSpaces, VolVectorSourceEstimate,
                 convert_forward_solution, pick_types_forward)
from mne.datasets import testing
from mne.fixes import fft, _get_img_fdata
from mne.source_estimate import grade_to_tris, _get_vol_mask
from mne.source_space import _get_src_nn
from mne.minimum_norm import (read_inverse_operator, apply_inverse,
                              apply_inverse_epochs, make_inverse_operator)
from mne.label import read_labels_from_annot, label_sign_flip
from mne.utils import (requires_pandas, requires_sklearn,
                       requires_h5py, run_tests_if_main, requires_nibabel)
from mne.io import read_raw_fif

data_path = testing.data_path(download=False)
subjects_dir = op.join(data_path, 'subjects')
fname_inv = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-meg-eeg-oct-6-meg-inv.fif')
fname_inv_fixed = op.join(
    data_path, 'MEG', 'sample',
    'sample_audvis_trunc-meg-eeg-oct-4-meg-fixed-inv.fif')
fname_fwd = op.join(
    data_path, 'MEG', 'sample', 'sample_audvis_trunc-meg-eeg-oct-4-fwd.fif')
fname_cov = op.join(
    data_path, 'MEG', 'sample', 'sample_audvis_trunc-cov.fif')
fname_evoked = op.join(data_path, 'MEG', 'sample',
                       'sample_audvis_trunc-ave.fif')
fname_raw = op.join(data_path, 'MEG', 'sample', 'sample_audvis_trunc_raw.fif')
fname_t1 = op.join(data_path, 'subjects', 'sample', 'mri', 'T1.mgz')
fname_src = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-meg-eeg-oct-6-fwd.fif')
fname_src_fs = op.join(data_path, 'subjects', 'fsaverage', 'bem',
                       'fsaverage-ico-5-src.fif')
fname_src_3 = op.join(data_path, 'subjects', 'sample', 'bem',
                      'sample-oct-4-src.fif')
fname_stc = op.join(data_path, 'MEG', 'sample', 'sample_audvis_trunc-meg')
fname_vol = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-grad-vol-7-fwd-sensmap-vol.w')
fname_vsrc = op.join(data_path, 'MEG', 'sample',
                     'sample_audvis_trunc-meg-vol-7-fwd.fif')
fname_inv_vol = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc-meg-vol-7-meg-inv.fif')
rng = np.random.RandomState(0)


@testing.requires_testing_data
def test_spatial_inter_hemi_connectivity():
    """Test spatial connectivity between hemispheres."""
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
    assert (n_src * 0.02 < conn.data.size < n_src * 0.10)
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
        assert (set(use_labels) - set(good_labels) == set())


@pytest.mark.slowtest
@testing.requires_testing_data
@requires_h5py
def test_volume_stc(tmpdir):
    """Test volume STCs."""
    N = 100
    data = np.arange(N)[:, np.newaxis]
    datas = [data,
             data,
             np.arange(2)[:, np.newaxis],
             np.arange(6).reshape(2, 3, 1)]
    vertno = np.arange(N)
    vertnos = [vertno,
               vertno[:, np.newaxis],
               np.arange(2)[:, np.newaxis],
               np.arange(2)]
    vertno_reads = [vertno, vertno, np.arange(2), np.arange(2)]
    for data, vertno, vertno_read in zip(datas, vertnos, vertno_reads):
        if data.ndim in (1, 2):
            stc = VolSourceEstimate(data, vertno, 0, 1)
            ext = 'stc'
            klass = VolSourceEstimate
        else:
            assert data.ndim == 3
            stc = VolVectorSourceEstimate(data, vertno, 0, 1)
            ext = 'h5'
            klass = VolVectorSourceEstimate
        fname_temp = tmpdir.join('temp-vl.' + ext)
        stc_new = stc
        for _ in range(2):
            stc_new.save(fname_temp)
            stc_new = read_source_estimate(fname_temp)
            assert isinstance(stc_new, klass)
            assert_array_equal(vertno_read, stc_new.vertices)
            assert_array_almost_equal(stc.data, stc_new.data)

    # now let's actually read a MNE-C processed file
    stc = read_source_estimate(fname_vol, 'sample')
    assert (isinstance(stc, VolSourceEstimate))

    assert ('sample' in repr(stc))
    stc_new = stc
    pytest.raises(ValueError, stc.save, fname_vol, ftype='whatever')
    for ftype in ['w', 'h5']:
        for _ in range(2):
            fname_temp = tmpdir.join('temp-vol.%s' % ftype)
            stc_new.save(fname_temp, ftype=ftype)
            stc_new = read_source_estimate(fname_temp)
            assert (isinstance(stc_new, VolSourceEstimate))
            assert_array_equal(stc.vertices, stc_new.vertices)
            assert_array_almost_equal(stc.data, stc_new.data)


@requires_nibabel()
@testing.requires_testing_data
def test_stc_as_volume():
    """Test previous volume source estimate morph."""
    import nibabel as nib
    inverse_operator_vol = read_inverse_operator(fname_inv_vol)

    # Apply inverse operator
    stc_vol = read_source_estimate(fname_vol, 'sample')

    img = stc_vol.as_volume(inverse_operator_vol['src'], mri_resolution=True,
                            dest='42')
    t1_img = nib.load(fname_t1)
    # always assure nifti and dimensionality
    assert isinstance(img, nib.Nifti1Image)
    assert img.header.get_zooms()[:3] == t1_img.header.get_zooms()[:3]

    img = stc_vol.as_volume(inverse_operator_vol['src'], mri_resolution=False)

    assert isinstance(img, nib.Nifti1Image)
    assert img.shape[:3] == inverse_operator_vol['src'][0]['shape'][:3]

    with pytest.raises(ValueError, match='Invalid value.*output.*'):
        stc_vol.as_volume(inverse_operator_vol['src'], format='42')


@testing.requires_testing_data
@requires_nibabel()
def test_save_vol_stc_as_nifti(tmpdir):
    """Save the stc as a nifti file and export."""
    import nibabel as nib
    src = read_source_spaces(fname_vsrc)
    vol_fname = tmpdir.join('stc.nii.gz')

    # now let's actually read a MNE-C processed file
    stc = read_source_estimate(fname_vol, 'sample')
    assert (isinstance(stc, VolSourceEstimate))

    stc.save_as_volume(vol_fname, src,
                       dest='surf', mri_resolution=False)
    with pytest.warns(None):  # nib<->numpy
        img = nib.load(str(vol_fname))
    assert (img.shape == src[0]['shape'] + (len(stc.times),))

    with pytest.warns(None):  # nib<->numpy
        t1_img = nib.load(fname_t1)
    stc.save_as_volume(tmpdir.join('stc.nii.gz'), src,
                       dest='mri', mri_resolution=True)
    with pytest.warns(None):  # nib<->numpy
        img = nib.load(str(vol_fname))
    assert (img.shape == t1_img.shape + (len(stc.times),))
    assert_allclose(img.affine, t1_img.affine, atol=1e-5)

    # export without saving
    img = stc.as_volume(src, dest='mri', mri_resolution=True)
    assert (img.shape == t1_img.shape + (len(stc.times),))
    assert_allclose(img.affine, t1_img.affine, atol=1e-5)

    src = SourceSpaces([src[0], src[0]])
    stc = VolSourceEstimate(np.r_[stc.data, stc.data],
                            [stc.vertices, stc.vertices],
                            tmin=stc.tmin, tstep=stc.tstep, subject='sample')
    img = stc.as_volume(src, dest='mri', mri_resolution=False)
    assert (img.shape == src[0]['shape'] + (len(stc.times),))


@testing.requires_testing_data
def test_expand():
    """Test stc expansion."""
    stc_ = read_source_estimate(fname_stc, 'sample')
    vec_stc_ = VectorSourceEstimate(np.zeros((stc_.data.shape[0], 3,
                                              stc_.data.shape[1])),
                                    stc_.vertices, stc_.tmin, stc_.tstep,
                                    stc_.subject)

    for stc in [stc_, vec_stc_]:
        assert ('sample' in repr(stc))
        labels_lh = read_labels_from_annot('sample', 'aparc', 'lh',
                                           subjects_dir=subjects_dir)
        new_label = labels_lh[0] + labels_lh[1]
        stc_limited = stc.in_label(new_label)
        stc_new = stc_limited.copy()
        stc_new.data.fill(0)
        for label in labels_lh[:2]:
            stc_new += stc.in_label(label).expand(stc_limited.vertices)
        pytest.raises(TypeError, stc_new.expand, stc_limited.vertices[0])
        pytest.raises(ValueError, stc_new.expand, [stc_limited.vertices[0]])
        # make sure we can't add unless vertno agree
        pytest.raises(ValueError, stc.__add__, stc.in_label(labels_lh[0]))


def _fake_stc(n_time=10):
    verts = [np.arange(10), np.arange(90)]
    return SourceEstimate(np.random.rand(100, n_time), verts, 0, 1e-1, 'foo')


def _fake_vec_stc(n_time=10):
    verts = [np.arange(10), np.arange(90)]
    return VectorSourceEstimate(np.random.rand(100, 3, n_time), verts, 0, 1e-1,
                                'foo')


@testing.requires_testing_data
def test_stc_snr():
    """Test computing SNR from a STC."""
    inv = read_inverse_operator(fname_inv_fixed)
    fwd = read_forward_solution(fname_fwd)
    cov = read_cov(fname_cov)
    evoked = read_evokeds(fname_evoked, baseline=(None, 0))[0].crop(0, 0.01)
    stc = apply_inverse(evoked, inv)
    assert (stc.data < 0).any()
    with pytest.warns(RuntimeWarning, match='nAm'):
        stc.estimate_snr(evoked.info, fwd, cov)  # dSPM
    with pytest.warns(RuntimeWarning, match='free ori'):
        abs(stc).estimate_snr(evoked.info, fwd, cov)
    stc = apply_inverse(evoked, inv, method='MNE')
    snr = stc.estimate_snr(evoked.info, fwd, cov)
    assert_allclose(snr.times, evoked.times)
    snr = snr.data
    assert snr.max() < -10
    assert snr.min() > -120


def test_stc_attributes():
    """Test STC attributes."""
    stc = _fake_stc(n_time=10)
    vec_stc = _fake_vec_stc(n_time=10)

    n_times = len(stc.times)
    assert_equal(stc._data.shape[-1], n_times)
    assert_array_equal(stc.times, stc.tmin + np.arange(n_times) * stc.tstep)
    assert_array_almost_equal(
        stc.times, [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    def attempt_times_mutation(stc):
        stc.times -= 1

    def attempt_assignment(stc, attr, val):
        setattr(stc, attr, val)

    # .times is read-only
    pytest.raises(ValueError, attempt_times_mutation, stc)
    pytest.raises(ValueError, attempt_assignment, stc, 'times', [1])

    # Changing .tmin or .tstep re-computes .times
    stc.tmin = 1
    assert (type(stc.tmin) == float)
    assert_array_almost_equal(
        stc.times, [1., 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9])

    stc.tstep = 1
    assert (type(stc.tstep) == float)
    assert_array_almost_equal(
        stc.times, [1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])

    # tstep <= 0 is not allowed
    pytest.raises(ValueError, attempt_assignment, stc, 'tstep', 0)
    pytest.raises(ValueError, attempt_assignment, stc, 'tstep', -1)

    # Changing .data re-computes .times
    stc.data = np.random.rand(100, 5)
    assert_array_almost_equal(
        stc.times, [1., 2., 3., 4., 5.])

    # .data must match the number of vertices
    pytest.raises(ValueError, attempt_assignment, stc, 'data', [[1]])
    pytest.raises(ValueError, attempt_assignment, stc, 'data', None)

    # .data much match number of dimensions
    pytest.raises(ValueError, attempt_assignment, stc, 'data', np.arange(100))
    pytest.raises(ValueError, attempt_assignment, vec_stc, 'data',
                  [np.arange(100)])
    pytest.raises(ValueError, attempt_assignment, vec_stc, 'data',
                  [[[np.arange(100)]]])

    # .shape attribute must also work when ._data is None
    stc._kernel = np.zeros((2, 2))
    stc._sens_data = np.zeros((2, 3))
    stc._data = None
    assert_equal(stc.shape, (2, 3))

    # bad size of data
    stc = _fake_stc()
    data = stc.data[:, np.newaxis, :]
    with pytest.raises(ValueError, match='2 dimensions for SourceEstimate'):
        SourceEstimate(data, stc.vertices)
    stc = SourceEstimate(data[:, 0, 0], stc.vertices, 0, 1)
    assert stc.data.shape == (len(data), 1)


def test_io_stc(tmpdir):
    """Test IO for STC files."""
    stc = _fake_stc()
    stc.save(tmpdir.join("tmp.stc"))
    stc2 = read_source_estimate(tmpdir.join("tmp.stc"))

    assert_array_almost_equal(stc.data, stc2.data)
    assert_array_almost_equal(stc.tmin, stc2.tmin)
    assert_equal(len(stc.vertices), len(stc2.vertices))
    for v1, v2 in zip(stc.vertices, stc2.vertices):
        assert_array_almost_equal(v1, v2)
    assert_array_almost_equal(stc.tstep, stc2.tstep)


@requires_h5py
def test_io_stc_h5(tmpdir):
    """Test IO for STC files using HDF5."""
    for stc in [_fake_stc(), _fake_vec_stc()]:
        pytest.raises(ValueError, stc.save, tmpdir.join('tmp'),
                      ftype='foo')
        out_name = tmpdir.join('tmp')
        stc.save(out_name, ftype='h5')
        stc.save(out_name, ftype='h5')  # test overwrite
        stc3 = read_source_estimate(out_name)
        stc4 = read_source_estimate(out_name + '-stc')
        stc5 = read_source_estimate(out_name + '-stc.h5')
        pytest.raises(RuntimeError, read_source_estimate, out_name,
                      subject='bar')
        for stc_new in stc3, stc4, stc5:
            assert_equal(stc_new.subject, stc.subject)
            assert_array_equal(stc_new.data, stc.data)
            assert_array_equal(stc_new.tmin, stc.tmin)
            assert_array_equal(stc_new.tstep, stc.tstep)
            assert_equal(len(stc_new.vertices), len(stc.vertices))
            for v1, v2 in zip(stc_new.vertices, stc.vertices):
                assert_array_equal(v1, v2)


def test_io_w(tmpdir):
    """Test IO for w files."""
    stc = _fake_stc(n_time=1)
    w_fname = tmpdir.join('fake')
    stc.save(w_fname, ftype='w')
    src = read_source_estimate(w_fname)
    src.save(tmpdir.join('tmp'), ftype='w')
    src2 = read_source_estimate(tmpdir.join('tmp-lh.w'))
    assert_array_almost_equal(src.data, src2.data)
    assert_array_almost_equal(src.lh_vertno, src2.lh_vertno)
    assert_array_almost_equal(src.rh_vertno, src2.rh_vertno)


def test_stc_arithmetic():
    """Test arithmetic for STC files."""
    stc = _fake_stc()
    data = stc.data.copy()
    vec_stc = _fake_vec_stc()
    vec_data = vec_stc.data.copy()

    out = list()
    for a in [data, stc, vec_data, vec_stc]:
        a = a + a * 3 + 3 * a - a ** 2 / 2

        a += a
        a -= a
        with np.errstate(invalid='ignore'):
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
        with np.errstate(invalid='ignore'):
            a **= 3
        out.append(a)

    assert_array_equal(out[0], out[1].data)
    assert_array_equal(out[2], out[3].data)
    assert_array_equal(stc.sqrt().data, np.sqrt(stc.data))
    assert_array_equal(vec_stc.sqrt().data, np.sqrt(vec_stc.data))
    assert_array_equal(abs(stc).data, abs(stc.data))
    assert_array_equal(abs(vec_stc).data, abs(vec_stc.data))

    stc_sum = stc.sum()
    assert_array_equal(stc_sum.data, stc.data.sum(1, keepdims=True))
    stc_mean = stc.mean()
    assert_array_equal(stc_mean.data, stc.data.mean(1, keepdims=True))
    vec_stc_mean = vec_stc.mean()
    assert_array_equal(vec_stc_mean.data, vec_stc.data.mean(2, keepdims=True))


@pytest.mark.slowtest
@testing.requires_testing_data
def test_stc_methods():
    """Test stc methods lh_data, rh_data, bin(), resample()."""
    stc_ = read_source_estimate(fname_stc)

    # Make a vector version of the above source estimate
    x = stc_.data[:, np.newaxis, :]
    yz = np.zeros((x.shape[0], 2, x.shape[2]))
    vec_stc_ = VectorSourceEstimate(
        np.concatenate((x, yz), 1),
        stc_.vertices, stc_.tmin, stc_.tstep, stc_.subject
    )

    for stc in [stc_, vec_stc_]:
        # lh_data / rh_data
        assert_array_equal(stc.lh_data, stc.data[:len(stc.lh_vertno)])
        assert_array_equal(stc.rh_data, stc.data[len(stc.lh_vertno):])

        # bin
        binned = stc.bin(.12)
        a = np.mean(stc.data[..., :np.searchsorted(stc.times, .12)], axis=-1)
        assert_array_equal(a, binned.data[..., 0])

        stc = read_source_estimate(fname_stc)
        stc.subject = 'sample'
        label_lh = read_labels_from_annot('sample', 'aparc', 'lh',
                                          subjects_dir=subjects_dir)[0]
        label_rh = read_labels_from_annot('sample', 'aparc', 'rh',
                                          subjects_dir=subjects_dir)[0]
        label_both = label_lh + label_rh
        for label in (label_lh, label_rh, label_both):
            assert (isinstance(stc.shape, tuple) and len(stc.shape) == 2)
            stc_label = stc.in_label(label)
            if label.hemi != 'both':
                if label.hemi == 'lh':
                    verts = stc_label.vertices[0]
                else:  # label.hemi == 'rh':
                    verts = stc_label.vertices[1]
                n_vertices_used = len(label.get_vertices_used(verts))
                assert_equal(len(stc_label.data), n_vertices_used)
        stc_lh = stc.in_label(label_lh)
        pytest.raises(ValueError, stc_lh.in_label, label_rh)
        label_lh.subject = 'foo'
        pytest.raises(RuntimeError, stc.in_label, label_lh)

        stc_new = deepcopy(stc)
        o_sfreq = 1.0 / stc.tstep
        # note that using no padding for this STC reduces edge ringing...
        stc_new.resample(2 * o_sfreq, npad=0)
        assert (stc_new.data.shape[1] == 2 * stc.data.shape[1])
        assert (stc_new.tstep == stc.tstep / 2)
        stc_new.resample(o_sfreq, npad=0)
        assert (stc_new.data.shape[1] == stc.data.shape[1])
        assert (stc_new.tstep == stc.tstep)
        assert_array_almost_equal(stc_new.data, stc.data, 5)


@testing.requires_testing_data
def test_center_of_mass():
    """Test computing the center of mass on an stc."""
    stc = read_source_estimate(fname_stc)
    pytest.raises(ValueError, stc.center_of_mass, 'sample')
    stc.lh_data[:] = 0
    vertex, hemi, t = stc.center_of_mass('sample', subjects_dir=subjects_dir)
    assert (hemi == 1)
    # XXX Should design a fool-proof test case, but here were the
    # results:
    assert_equal(vertex, 124791)
    assert_equal(np.round(t, 2), 0.12)


@testing.requires_testing_data
def test_extract_label_time_course():
    """Test extraction of label time courses from stc."""
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
    pytest.raises(ValueError, extract_label_time_course, stcs, labels,
                  src, mode='notamode')

    # have an empty label
    empty_label = labels[0].copy()
    empty_label.vertices += 1000000
    pytest.raises(ValueError, extract_label_time_course, stcs, empty_label,
                  src, mode='mean')

    # but this works:
    with pytest.warns(RuntimeWarning, match='does not contain any vertices'):
        tc = extract_label_time_course(stcs, empty_label, src, mode='mean',
                                       allow_empty=True)
    for arr in tc:
        assert (arr.shape == (1, n_times))
        assert_array_equal(arr, np.zeros((1, n_times)))

    # test the different modes
    modes = ['mean', 'mean_flip', 'pca_flip', 'max']

    for mode in modes:
        label_tc = extract_label_time_course(stcs, labels, src, mode=mode)
        label_tc_method = [stc.extract_label_time_course(labels, src,
                                                         mode=mode)
                           for stc in stcs]
        assert (len(label_tc) == n_stcs)
        assert (len(label_tc_method) == n_stcs)
        for tc1, tc2 in zip(label_tc, label_tc_method):
            assert (tc1.shape == (n_labels, n_times))
            assert (tc2.shape == (n_labels, n_times))
            assert (np.allclose(tc1, tc2, rtol=1e-8, atol=1e-16))
            if mode == 'mean':
                assert_array_almost_equal(tc1, label_means)
            if mode == 'mean_flip':
                assert_array_almost_equal(tc1, label_means_flipped)
            if mode == 'max':
                assert_array_almost_equal(tc1, label_maxs)

    # test label with very few vertices (check SVD conditionals)
    label = Label(vertices=src[0]['vertno'][:2], hemi='lh')
    x = label_sign_flip(label, src)
    assert (len(x) == 2)
    label = Label(vertices=[], hemi='lh')
    x = label_sign_flip(label, src)
    assert (x.size == 0)


@testing.requires_testing_data
def test_extract_label_time_course_equiv():
    """Test extraction of label time courses from stc equivalences."""
    label = read_labels_from_annot('sample', 'aparc', 'lh', regexp='transv',
                                   subjects_dir=subjects_dir)
    assert len(label) == 1
    label = label[0]
    inv = read_inverse_operator(fname_inv)
    evoked = read_evokeds(fname_evoked, baseline=(None, 0))[0].crop(0, 0.01)
    stc = apply_inverse(evoked, inv, pick_ori='normal', label=label)
    stc_full = apply_inverse(evoked, inv, pick_ori='normal')
    stc_in_label = stc_full.in_label(label)
    mean = stc.extract_label_time_course(label, inv['src'])
    mean_2 = stc_in_label.extract_label_time_course(label, inv['src'])
    assert_allclose(mean, mean_2)
    inv['src'][0]['vertno'] = np.array([], int)
    assert len(stc_in_label.vertices[0]) == 22
    with pytest.raises(ValueError, match='22/22 left hemisphere.*missing'):
        stc_in_label.extract_label_time_course(label, inv['src'])


def _my_trans(data):
    """FFT that adds an additional dimension by repeating result."""
    data_t = fft(data)
    data_t = np.concatenate([data_t[:, :, None], data_t[:, :, None]], axis=2)
    return data_t, None


def test_transform_data():
    """Test applying linear (time) transform to data."""
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
    # bad sens_data
    sens_data = sens_data[..., np.newaxis]
    with pytest.raises(ValueError, match='sensor data must have 2'):
        VolSourceEstimate((kernel, sens_data), vertices)


def test_transform():
    """Test applying linear (time) transform to data."""
    # make up some data
    n_verts_lh, n_verts_rh, n_times = 10, 10, 10
    vertices = [np.arange(n_verts_lh), n_verts_lh + np.arange(n_verts_rh)]
    data = rng.randn(n_verts_lh + n_verts_rh, n_times)
    stc = SourceEstimate(data, vertices=vertices, tmin=-0.1, tstep=0.1)

    # data_t.ndim > 2 & copy is True
    stcs_t = stc.transform(_my_trans, copy=True)
    assert (isinstance(stcs_t, list))
    assert_array_equal(stc.times, stcs_t[0].times)
    assert_equal(stc.vertices, stcs_t[0].vertices)

    data = np.concatenate((stcs_t[0].data[:, :, None],
                           stcs_t[1].data[:, :, None]), axis=2)
    data_t = stc.transform_data(_my_trans)
    assert_array_equal(data, data_t)  # check against stc.transform_data()

    # data_t.ndim > 2 & copy is False
    pytest.raises(ValueError, stc.transform, _my_trans, copy=False)

    # data_t.ndim = 2 & copy is True
    tmp = deepcopy(stc)
    stc_t = stc.transform(np.abs, copy=True)
    assert (isinstance(stc_t, SourceEstimate))
    assert_array_equal(stc.data, tmp.data)  # xfrm doesn't modify original?

    # data_t.ndim = 2 & copy is False
    times = np.round(1000 * stc.times)
    verts = np.arange(len(stc.lh_vertno),
                      len(stc.lh_vertno) + len(stc.rh_vertno), 1)
    verts_rh = stc.rh_vertno
    tmin_idx = np.searchsorted(times, 0)
    tmax_idx = np.searchsorted(times, 501)  # Include 500ms in the range
    data_t = stc.transform_data(np.abs, idx=verts, tmin_idx=tmin_idx,
                                tmax_idx=tmax_idx)
    stc.transform(np.abs, idx=verts, tmin=-50, tmax=500, copy=False)
    assert (isinstance(stc, SourceEstimate))
    assert_equal(stc.tmin, 0.)
    assert_equal(stc.times[-1], 0.5)
    assert_equal(len(stc.vertices[0]), 0)
    assert_equal(stc.vertices[1], verts_rh)
    assert_array_equal(stc.data, data_t)

    times = np.round(1000 * stc.times)
    tmin_idx, tmax_idx = np.searchsorted(times, 0), np.searchsorted(times, 250)
    data_t = stc.transform_data(np.abs, tmin_idx=tmin_idx, tmax_idx=tmax_idx)
    stc.transform(np.abs, tmin=0, tmax=250, copy=False)
    assert_equal(stc.tmin, 0.)
    assert_equal(stc.times[-1], 0.2)
    assert_array_equal(stc.data, data_t)


@requires_sklearn
def test_spatio_temporal_tris_connectivity():
    """Test spatio-temporal connectivity from triangles."""
    tris = np.array([[0, 1, 2], [3, 4, 5]])
    connectivity = spatio_temporal_tris_connectivity(tris, 2)
    x = [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    components = stats.cluster_level._get_components(np.array(x), connectivity)
    # _get_components works differently now...
    old_fmt = [0, 0, -2, -2, -2, -2, 0, -2, -2, -2, -2, 1]
    new_fmt = np.array(old_fmt)
    new_fmt = [np.nonzero(new_fmt == v)[0]
               for v in np.unique(new_fmt[new_fmt >= 0])]
    assert len(new_fmt) == len(components)
    for c, n in zip(components, new_fmt):
        assert_array_equal(c, n)


@testing.requires_testing_data
def test_spatio_temporal_src_connectivity():
    """Test spatio-temporal connectivity from source spaces."""
    tris = np.array([[0, 1, 2], [3, 4, 5]])
    src = [dict(), dict()]
    connectivity = spatio_temporal_tris_connectivity(tris, 2)
    src[0]['use_tris'] = np.array([[0, 1, 2]])
    src[1]['use_tris'] = np.array([[0, 1, 2]])
    src[0]['vertno'] = np.array([0, 1, 2])
    src[1]['vertno'] = np.array([0, 1, 2])
    src[0]['type'] = 'surf'
    src[1]['type'] = 'surf'
    connectivity2 = spatio_temporal_src_connectivity(src, 2)
    assert_array_equal(connectivity.todense(), connectivity2.todense())
    # add test for dist connectivity
    src[0]['dist'] = np.ones((3, 3)) - np.eye(3)
    src[1]['dist'] = np.ones((3, 3)) - np.eye(3)
    src[0]['vertno'] = [0, 1, 2]
    src[1]['vertno'] = [0, 1, 2]
    src[0]['type'] = 'surf'
    src[1]['type'] = 'surf'
    connectivity3 = spatio_temporal_src_connectivity(src, 2, dist=2)
    assert_array_equal(connectivity.todense(), connectivity3.todense())
    # add test for source space connectivity with omitted vertices
    inverse_operator = read_inverse_operator(fname_inv)
    src_ = inverse_operator['src']
    with pytest.warns(RuntimeWarning, match='will have holes'):
        connectivity = spatio_temporal_src_connectivity(src_, n_times=2)
    a = connectivity.shape[0] / 2
    b = sum([s['nuse'] for s in inverse_operator['src']])
    assert (a == b)

    assert_equal(grade_to_tris(5).shape, [40960, 3])


@requires_pandas
def test_to_data_frame():
    """Test stc Pandas exporter."""
    n_vert, n_times = 10, 5
    vertices = [np.arange(n_vert, dtype=np.int), np.empty(0, dtype=np.int)]
    data = rng.randn(n_vert, n_times)
    stc_surf = SourceEstimate(data, vertices=vertices, tmin=0, tstep=1,
                              subject='sample')
    stc_vol = VolSourceEstimate(data, vertices=vertices[0], tmin=0, tstep=1,
                                subject='sample')
    for stc in [stc_surf, stc_vol]:
        df = stc.to_data_frame()
        # test data preservation (first 2 dataframe elements are subj & time)
        assert_array_equal(df.values.T[2:], stc.data)
        # test long format
        df_long = stc.to_data_frame(long_format=True)
        assert(len(df_long) == stc.data.size)
        expected = ('subject', 'time', 'source', 'value')
        assert set(expected) == set(df_long.columns)


@requires_pandas
@pytest.mark.parametrize('index', ('time', ['time', 'subject'], None))
def test_to_data_frame_index(index):
    """Test index creation in stc Pandas exporter."""
    n_vert, n_times = 10, 5
    vertices = [np.arange(n_vert, dtype=np.int), np.empty(0, dtype=np.int)]
    data = rng.randn(n_vert, n_times)
    stc = SourceEstimate(data, vertices=vertices, tmin=0, tstep=1,
                         subject='sample')
    df = stc.to_data_frame(index=index)
    # test index setting
    if not isinstance(index, list):
        index = [index]
    assert (df.index.names == index)
    # test that non-indexed data were present as columns
    non_index = list(set(['time', 'subject']) - set(index))
    if len(non_index):
        assert all(np.in1d(non_index, df.columns))


def test_get_peak():
    """Test peak getter."""
    n_vert, n_times = 10, 5
    vertices = [np.arange(n_vert, dtype=np.int), np.empty(0, dtype=np.int)]
    data = rng.randn(n_vert, n_times)
    stc_surf = SourceEstimate(data, vertices=vertices, tmin=0, tstep=1,
                              subject='sample')
    stc_vol = VolSourceEstimate(data, vertices=vertices[0], tmin=0, tstep=1,
                                subject='sample')

    # Versions with only one time point
    stc_surf_1 = SourceEstimate(data[:, :1], vertices=vertices, tmin=0,
                                tstep=1, subject='sample')
    stc_vol_1 = VolSourceEstimate(data[:, :1], vertices=vertices[0], tmin=0,
                                  tstep=1, subject='sample')

    for ii, stc in enumerate([stc_surf, stc_vol, stc_surf_1, stc_vol_1]):
        pytest.raises(ValueError, stc.get_peak, tmin=-100)
        pytest.raises(ValueError, stc.get_peak, tmax=90)
        pytest.raises(ValueError, stc.get_peak, tmin=0.002, tmax=0.001)

        vert_idx, time_idx = stc.get_peak()
        vertno = np.concatenate(stc.vertices) if ii in [0, 2] else stc.vertices
        assert (vert_idx in vertno)
        assert (time_idx in stc.times)

        data_idx, time_idx = stc.get_peak(vert_as_index=True,
                                          time_as_index=True)
        assert_equal(data_idx, np.argmax(np.abs(stc.data[:, time_idx])))
        assert_equal(time_idx, np.argmax(np.abs(stc.data[data_idx, :])))


@requires_h5py
@testing.requires_testing_data
def test_mixed_stc(tmpdir):
    """Test source estimate from mixed source space."""
    N = 90  # number of sources
    T = 2  # number of time points
    S = 3  # number of source spaces

    data = rng.randn(N, T)
    vertno = S * [np.arange(N // S)]

    # make sure error is raised if vertices are not a list of length >= 2
    pytest.raises(ValueError, MixedSourceEstimate, data=data,
                  vertices=[np.arange(N)])

    stc = MixedSourceEstimate(data, vertno, 0, 1)

    vol = read_source_spaces(fname_vsrc)

    # make sure error is raised for plotting surface with volume source
    pytest.raises(ValueError, stc.plot_surface, src=vol)

    fname = tmpdir.join('mixed-stc.h5')
    stc.save(fname)
    stc_out = read_source_estimate(fname)
    assert_array_equal(stc_out.vertices, vertno)
    assert_array_equal(stc_out.data, data)
    assert stc_out.tmin == 0
    assert stc_out.tstep == 1
    assert isinstance(stc_out, MixedSourceEstimate)


@pytest.mark.parametrize('klass, kind',
                         ((VectorSourceEstimate, 'surf'),
                          (VolVectorSourceEstimate, 'vol'),
                          (VolVectorSourceEstimate, 'discrete')))
def test_vec_stc_basic(klass, kind):
    """Test (vol)vector source estimate."""
    nn = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [np.sqrt(1 / 3.)] * 3
    ])

    data = np.array([
        [1, 0, 0],
        [0, 2, 0],
        [3, 0, 0],
        [1, 1, 1],
    ])[:, :, np.newaxis]
    if klass is VolVectorSourceEstimate:
        src = SourceSpaces([dict(nn=nn, type=kind)])
        verts = np.arange(4)
    else:
        src = SourceSpaces([dict(nn=nn[:2], type=kind),
                            dict(nn=nn[2:], type=kind)])
        verts = [np.array([0, 1]), np.array([0, 1])]
    stc = klass(data, verts, 0, 1, 'foo')

    # Magnitude of the vectors
    assert_array_equal(stc.magnitude().data[:, 0], [1, 2, 3, np.sqrt(3)])

    # Vector components projected onto the vertex normals
    if kind == 'vol':
        with pytest.raises(RuntimeError, match='surface or discrete'):
            stc.normal(src)
        return
    normal = stc.normal(src)
    assert_array_equal(normal.data[:, 0], [1, 2, 0, np.sqrt(3)])

    stc = klass(data[:, :, 0], verts, 0, 1)  # upbroadcast
    assert stc.data.shape == (len(data), 3, 1)
    # Bad data
    with pytest.raises(ValueError, match='of length 3'):
        klass(data[:, :2], verts, 0, 1)
    data = data[:, :, np.newaxis]
    with pytest.raises(ValueError, match='3 dimensions for .*VectorSource'):
        klass(data, verts, 0, 1)


@pytest.fixture(scope='module', params=[testing._pytest_param()])
def invs():
    """Inverses of various amounts of loose."""
    fwd = read_forward_solution(fname_fwd)
    fwd = pick_types_forward(fwd, meg=True, eeg=False)
    fwd_surf = convert_forward_solution(fwd, surf_ori=True)
    evoked = read_evokeds(fname_evoked, baseline=(None, 0))[0]
    noise_cov = read_cov(fname_cov)
    free = make_inverse_operator(
        evoked.info, fwd, noise_cov, loose=1.)
    free_surf = make_inverse_operator(
        evoked.info, fwd_surf, noise_cov, loose=1.)
    freeish = make_inverse_operator(
        evoked.info, fwd, noise_cov, loose=0.9999)
    fixed = make_inverse_operator(
        evoked.info, fwd, noise_cov, loose=0.)
    fixedish = make_inverse_operator(
        evoked.info, fwd, noise_cov, loose=0.0001)
    assert_allclose(free['source_nn'],
                    np.kron(np.ones(fwd['nsource']), np.eye(3)).T,
                    atol=1e-7)
    # This is the one exception:
    assert not np.allclose(free['source_nn'], free_surf['source_nn'])
    assert_allclose(free['source_nn'],
                    np.tile(np.eye(3), (free['nsource'], 1)), atol=1e-7)
    # All others are similar:
    for other in (freeish, fixedish):
        assert_allclose(free_surf['source_nn'], other['source_nn'], atol=1e-7)
    assert_allclose(
        free_surf['source_nn'][2::3], fixed['source_nn'], atol=1e-7)
    expected_nn = np.concatenate([_get_src_nn(s) for s in fwd['src']])
    assert_allclose(fixed['source_nn'], expected_nn, atol=1e-7)
    return evoked, free, free_surf, freeish, fixed, fixedish


bad_normal = pytest.param(
    'normal', marks=pytest.mark.xfail(raises=AssertionError))


@pytest.mark.parametrize('pick_ori', [None, 'normal', 'vector'])
def test_vec_stc_inv_free(invs, pick_ori):
    """Test vector STC behavior with two free-orientation inverses."""
    evoked, free, free_surf, _, _, _ = invs
    stc_free = apply_inverse(evoked, free, pick_ori=pick_ori)
    stc_free_surf = apply_inverse(evoked, free_surf, pick_ori=pick_ori)
    assert_allclose(stc_free.data, stc_free_surf.data, atol=1e-5)


@pytest.mark.parametrize('pick_ori', [None, 'normal', 'vector'])
def test_vec_stc_inv_free_surf(invs, pick_ori):
    """Test vector STC behavior with free and free-ish orientation invs."""
    evoked, _, free_surf, freeish, _, _ = invs
    stc_free = apply_inverse(evoked, free_surf, pick_ori=pick_ori)
    stc_freeish = apply_inverse(evoked, freeish, pick_ori=pick_ori)
    assert_allclose(stc_free.data, stc_freeish.data, atol=1e-3)


@pytest.mark.parametrize('pick_ori', (None, 'normal', 'vector'))
def test_vec_stc_inv_fixed(invs, pick_ori):
    """Test vector STC behavior with fixed-orientation inverses."""
    evoked, _, _, _, fixed, fixedish = invs
    stc_fixed = apply_inverse(evoked, fixed)
    stc_fixedish = apply_inverse(evoked, fixedish, pick_ori=pick_ori)
    if pick_ori == 'vector':
        # two ways here: with magnitude...
        assert_allclose(
            abs(stc_fixed).data, stc_fixedish.magnitude().data, atol=1e-2)
        # ... and when picking the normal (signed)
        stc_fixedish = stc_fixedish.normal(fixedish['src'])
    elif pick_ori is None:
        stc_fixed = abs(stc_fixed)
    else:
        assert pick_ori == 'normal'  # no need to modify
    assert_allclose(stc_fixed.data, stc_fixedish.data, atol=1e-2)


@testing.requires_testing_data
def test_epochs_vector_inverse():
    """Test vector inverse consistency between evoked and epochs."""
    raw = read_raw_fif(fname_raw)
    events = find_events(raw, stim_channel='STI 014')[:2]
    reject = dict(grad=2000e-13, mag=4e-12, eog=150e-6)

    epochs = Epochs(raw, events, None, 0, 0.01, baseline=None,
                    reject=reject, preload=True)

    assert_equal(len(epochs), 2)

    evoked = epochs.average(picks=range(len(epochs.ch_names)))

    inv = read_inverse_operator(fname_inv)

    method = "MNE"
    snr = 3.
    lambda2 = 1. / snr ** 2

    stcs_epo = apply_inverse_epochs(epochs, inv, lambda2, method=method,
                                    pick_ori='vector', return_generator=False)
    stc_epo = np.mean(stcs_epo)

    stc_evo = apply_inverse(evoked, inv, lambda2, method=method,
                            pick_ori='vector')

    assert_allclose(stc_epo.data, stc_evo.data, rtol=1e-9, atol=0)


@requires_sklearn
@testing.requires_testing_data
def test_vol_connectivity():
    """Test volume connectivity."""
    vol = read_source_spaces(fname_vsrc)

    pytest.raises(ValueError, spatial_src_connectivity, vol, dist=1.)

    connectivity = spatial_src_connectivity(vol)
    n_vertices = vol[0]['inuse'].sum()
    assert_equal(connectivity.shape, (n_vertices, n_vertices))
    assert (np.all(connectivity.data == 1))
    assert (isinstance(connectivity, sparse.coo_matrix))

    connectivity2 = spatio_temporal_src_connectivity(vol, n_times=2)
    assert_equal(connectivity2.shape, (2 * n_vertices, 2 * n_vertices))
    assert (np.all(connectivity2.data == 1))


@testing.requires_testing_data
def test_spatial_src_connectivity():
    """Test spatial connectivity functionality."""
    # oct
    src = read_source_spaces(fname_src)
    assert src[0]['dist'] is not None  # distance info
    with pytest.warns(RuntimeWarning, match='will have holes'):
        con = spatial_src_connectivity(src).toarray()
    con_dist = spatial_src_connectivity(src, dist=0.01).toarray()
    assert (con == con_dist).mean() > 0.75
    # ico
    src = read_source_spaces(fname_src_fs)
    con = spatial_src_connectivity(src).tocsr()
    con_tris = spatial_tris_connectivity(grade_to_tris(5)).tocsr()
    assert con.shape == con_tris.shape
    assert_array_equal(con.data, con_tris.data)
    assert_array_equal(con.indptr, con_tris.indptr)
    assert_array_equal(con.indices, con_tris.indices)
    # one hemi
    con_lh = spatial_src_connectivity(src[:1]).tocsr()
    con_lh_tris = spatial_tris_connectivity(grade_to_tris(5)).tocsr()
    con_lh_tris = con_lh_tris[:10242, :10242].tocsr()
    assert_array_equal(con_lh.data, con_lh_tris.data)
    assert_array_equal(con_lh.indptr, con_lh_tris.indptr)
    assert_array_equal(con_lh.indices, con_lh_tris.indices)


@requires_sklearn
@requires_nibabel()
@testing.requires_testing_data
def test_vol_mask():
    """Test extraction of volume mask."""
    src = read_source_spaces(fname_vsrc)
    mask = _get_vol_mask(src)
    # Let's use an alternative way that should be equivalent
    vertices = src[0]['vertno']
    n_vertices = len(vertices)
    data = (1 + np.arange(n_vertices))[:, np.newaxis]
    stc_tmp = VolSourceEstimate(data, vertices, tmin=0., tstep=1.)
    img = stc_tmp.as_volume(src, mri_resolution=False)
    img_data = _get_img_fdata(img)[:, :, :, 0].T
    mask_nib = (img_data != 0)
    assert_array_equal(img_data[mask_nib], data[:, 0])
    assert_array_equal(np.where(mask_nib.ravel())[0], src[0]['vertno'])
    assert_array_equal(mask, mask_nib)
    assert_array_equal(img_data.shape, mask.shape)


run_tests_if_main()
