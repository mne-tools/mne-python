# -*- coding: utf-8 -*-
#
# License: BSD-3-Clause

from contextlib import nullcontext
from copy import deepcopy
import os
import os.path as op
import re
from shutil import copyfile

import numpy as np
from numpy.fft import fft
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_allclose, assert_equal, assert_array_less)
import pytest
from scipy import sparse
from scipy.optimize import fmin_cobyla
from scipy.spatial.distance import cdist

import mne
from mne import (stats, SourceEstimate, VectorSourceEstimate,
                 VolSourceEstimate, Label, read_source_spaces,
                 read_evokeds, MixedSourceEstimate, find_events, Epochs,
                 read_source_estimate, extract_label_time_course,
                 spatio_temporal_tris_adjacency, stc_near_sensors,
                 spatio_temporal_src_adjacency, read_cov, EvokedArray,
                 spatial_inter_hemi_adjacency, read_forward_solution,
                 spatial_src_adjacency, spatial_tris_adjacency, pick_info,
                 SourceSpaces, VolVectorSourceEstimate, read_trans, pick_types,
                 MixedVectorSourceEstimate, setup_volume_source_space,
                 convert_forward_solution, pick_types_forward,
                 compute_source_morph, labels_to_stc, scale_mri,
                 write_source_spaces)
from mne.datasets import testing
from mne.fixes import _get_img_fdata
from mne.io import read_info
from mne.io.constants import FIFF
from mne.morph_map import _make_morph_map_hemi
from mne.source_estimate import grade_to_tris, _get_vol_mask
from mne.source_space import _get_src_nn
from mne.transforms import apply_trans, invert_transform, transform_surface_to
from mne.minimum_norm import (read_inverse_operator, apply_inverse,
                              apply_inverse_epochs, make_inverse_operator)
from mne.label import read_labels_from_annot, label_sign_flip
from mne.utils import (requires_pandas, requires_sklearn, catch_logging,
                       requires_nibabel, requires_version, _record_warnings)
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
fname_fs_t1 = op.join(data_path, 'subjects', 'fsaverage', 'mri', 'T1.mgz')
fname_aseg = op.join(data_path, 'subjects', 'sample', 'mri', 'aseg.mgz')
fname_src = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-meg-eeg-oct-6-fwd.fif')
fname_src_fs = op.join(data_path, 'subjects', 'fsaverage', 'bem',
                       'fsaverage-ico-5-src.fif')
bem_path = op.join(data_path, 'subjects', 'sample', 'bem')
fname_src_3 = op.join(bem_path, 'sample-oct-4-src.fif')
fname_src_vol = op.join(bem_path, 'sample-volume-7mm-src.fif')
fname_stc = op.join(data_path, 'MEG', 'sample', 'sample_audvis_trunc-meg')
fname_vol = op.join(data_path, 'MEG', 'sample',
                    'sample_audvis_trunc-grad-vol-7-fwd-sensmap-vol.w')
fname_vsrc = op.join(data_path, 'MEG', 'sample',
                     'sample_audvis_trunc-meg-vol-7-fwd.fif')
fname_inv_vol = op.join(data_path, 'MEG', 'sample',
                        'sample_audvis_trunc-meg-vol-7-meg-inv.fif')
fname_nirx = op.join(data_path, 'NIRx', 'nirscout', 'nirx_15_0_recording')
rng = np.random.RandomState(0)


@testing.requires_testing_data
def test_stc_baseline_correction():
    """Test baseline correction for source estimate objects."""
    # test on different source estimates
    stcs = [read_source_estimate(fname_stc),
            read_source_estimate(fname_vol, 'sample')]
    # test on different "baseline" intervals
    baselines = [(0., 0.1), (None, None)]

    for stc in stcs:
        times = stc.times

        for (start, stop) in baselines:
            # apply baseline correction, then check if it worked
            stc = stc.apply_baseline(baseline=(start, stop))

            t0 = start or stc.times[0]
            t1 = stop or stc.times[-1]
            # index for baseline interval (include boundary latencies)
            imin = np.abs(times - t0).argmin()
            imax = np.abs(times - t1).argmin() + 1
            # data matrix from baseline interval
            data_base = stc.data[:, imin:imax]
            mean_base = data_base.mean(axis=1)
            zero_array = np.zeros(mean_base.shape[0])
            # test if baseline properly subtracted (mean=zero for all sources)
            assert_array_almost_equal(mean_base, zero_array)


@testing.requires_testing_data
def test_spatial_inter_hemi_adjacency():
    """Test spatial adjacency between hemispheres."""
    # trivial cases
    conn = spatial_inter_hemi_adjacency(fname_src_3, 5e-6)
    assert_equal(conn.data.size, 0)
    conn = spatial_inter_hemi_adjacency(fname_src_3, 5e6)
    assert_equal(conn.data.size, np.prod(conn.shape) // 2)
    # actually interesting case (1cm), should be between 2 and 10% of verts
    src = read_source_spaces(fname_src_3)
    conn = spatial_inter_hemi_adjacency(src, 10e-3)
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
        use_labels = [label.name[:-3] for label in labels
                      if np.in1d(label.vertices, has_neighbors).any()]
        assert (set(use_labels) - set(good_labels) == set())


@pytest.mark.slowtest
@testing.requires_testing_data
@requires_version('h5io')
def test_volume_stc(tmp_path):
    """Test volume STCs."""
    from h5io import write_hdf5
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
            stc = VolSourceEstimate(data, [vertno], 0, 1)
            ext = 'stc'
            klass = VolSourceEstimate
        else:
            assert data.ndim == 3
            stc = VolVectorSourceEstimate(data, [vertno], 0, 1)
            ext = 'h5'
            klass = VolVectorSourceEstimate
        fname_temp = tmp_path / ('temp-vl.' + ext)
        stc_new = stc
        n = 3 if ext == 'h5' else 2
        for ii in range(n):
            if ii < 2:
                stc_new.save(fname_temp, overwrite=True)
            else:
                # Pass stc.vertices[0], an ndarray, to ensure support for
                # the way we used to write volume STCs
                write_hdf5(
                    str(fname_temp), dict(
                        vertices=stc.vertices[0], data=stc.data,
                        tmin=stc.tmin, tstep=stc.tstep,
                        subject=stc.subject, src_type=stc._src_type),
                    title='mnepython', overwrite=True)
            stc_new = read_source_estimate(fname_temp)
            assert isinstance(stc_new, klass)
            assert_array_equal(vertno_read, stc_new.vertices[0])
            assert_array_almost_equal(stc.data, stc_new.data)

    # now let's actually read a MNE-C processed file
    stc = read_source_estimate(fname_vol, 'sample')
    assert isinstance(stc, VolSourceEstimate)

    assert 'sample' in repr(stc)
    assert ' kB' in repr(stc)

    stc_new = stc
    fname_temp = tmp_path / ('temp-vl.stc')
    with pytest.raises(ValueError, match="'ftype' parameter"):
        stc.save(fname_vol, ftype='whatever', overwrite=True)
    for ftype in ['w', 'h5']:
        for _ in range(2):
            fname_temp = tmp_path / ('temp-vol.%s' % ftype)
            stc_new.save(fname_temp, ftype=ftype, overwrite=True)
            stc_new = read_source_estimate(fname_temp)
            assert (isinstance(stc_new, VolSourceEstimate))
            assert_array_equal(stc.vertices[0], stc_new.vertices[0])
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
def test_save_vol_stc_as_nifti(tmp_path):
    """Save the stc as a nifti file and export."""
    import nibabel as nib
    src = read_source_spaces(fname_vsrc)
    vol_fname = tmp_path / 'stc.nii.gz'

    # now let's actually read a MNE-C processed file
    stc = read_source_estimate(fname_vol, 'sample')
    assert (isinstance(stc, VolSourceEstimate))

    stc.save_as_volume(vol_fname, src,
                       dest='surf', mri_resolution=False)
    with _record_warnings():  # nib<->numpy
        img = nib.load(str(vol_fname))
    assert (img.shape == src[0]['shape'] + (len(stc.times),))

    with _record_warnings():  # nib<->numpy
        t1_img = nib.load(fname_t1)
    stc.save_as_volume(vol_fname, src, dest='mri', mri_resolution=True,
                       overwrite=True)
    with _record_warnings():  # nib<->numpy
        img = nib.load(str(vol_fname))
    assert (img.shape == t1_img.shape + (len(stc.times),))
    assert_allclose(img.affine, t1_img.affine, atol=1e-5)

    # export without saving
    img = stc.as_volume(src, dest='mri', mri_resolution=True)
    assert (img.shape == t1_img.shape + (len(stc.times),))
    assert_allclose(img.affine, t1_img.affine, atol=1e-5)

    src = SourceSpaces([src[0], src[0]])
    stc = VolSourceEstimate(np.r_[stc.data, stc.data],
                            [stc.vertices[0], stc.vertices[0]],
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


def _fake_stc(n_time=10, is_complex=False):
    np.random.seed(7)
    verts = [np.arange(10), np.arange(90)]
    data = np.random.rand(100, n_time)
    if is_complex:
        data.astype(complex)
    return SourceEstimate(data, verts, 0, 1e-1, 'foo')


def _fake_vec_stc(n_time=10, is_complex=False):
    np.random.seed(7)
    verts = [np.arange(10), np.arange(90)]
    data = np.random.rand(100, 3, n_time)
    if is_complex:
        data.astype(complex)
    return VectorSourceEstimate(data, verts, 0, 1e-1,
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
        SourceEstimate(data, stc.vertices, 0, 1)
    stc = SourceEstimate(data[:, 0, 0], stc.vertices, 0, 1)
    assert stc.data.shape == (len(data), 1)


def test_io_stc(tmp_path):
    """Test IO for STC files."""
    stc = _fake_stc()
    stc.save(tmp_path / "tmp.stc")
    stc2 = read_source_estimate(tmp_path / "tmp.stc")

    assert_array_almost_equal(stc.data, stc2.data)
    assert_array_almost_equal(stc.tmin, stc2.tmin)
    assert_equal(len(stc.vertices), len(stc2.vertices))
    for v1, v2 in zip(stc.vertices, stc2.vertices):
        assert_array_almost_equal(v1, v2)
    assert_array_almost_equal(stc.tstep, stc2.tstep)
    # test warning for complex data
    stc2.data = stc2.data.astype(np.complex128)
    with pytest.raises(ValueError, match='Cannot save complex-valued STC'):
        stc2.save(tmp_path / 'complex.stc')


@requires_version('h5io')
@pytest.mark.parametrize('is_complex', (True, False))
@pytest.mark.parametrize('vector', (True, False))
def test_io_stc_h5(tmp_path, is_complex, vector):
    """Test IO for STC files using HDF5."""
    if vector:
        stc = _fake_vec_stc(is_complex=is_complex)
    else:
        stc = _fake_stc(is_complex=is_complex)
    match = 'can only be written' if vector else "Invalid value for the 'ftype"
    with pytest.raises(ValueError, match=match):
        stc.save(tmp_path / 'tmp.h5', ftype='foo')
    out_name = str(tmp_path / 'tmp')
    stc.save(out_name, ftype='h5')
    # test overwrite
    assert op.isfile(out_name + '-stc.h5')
    with pytest.raises(FileExistsError, match='Destination file exists'):
        stc.save(out_name, ftype='h5')
    stc.save(out_name, ftype='h5', overwrite=True)
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


def test_io_w(tmp_path):
    """Test IO for w files."""
    stc = _fake_stc(n_time=1)
    w_fname = tmp_path / 'fake'
    stc.save(w_fname, ftype='w')
    src = read_source_estimate(w_fname)
    src.save(tmp_path / 'tmp', ftype='w')
    src2 = read_source_estimate(tmp_path / 'tmp-lh.w')
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
@pytest.mark.parametrize('kind', ('surface', 'mixed'))
@pytest.mark.parametrize('vector', (False, True))
def test_extract_label_time_course(kind, vector):
    """Test extraction of label time courses from (Mixed)SourceEstimate."""
    n_stcs = 3
    n_times = 50

    src = read_inverse_operator(fname_inv)['src']
    if kind == 'mixed':
        pytest.importorskip('nibabel')
        label_names = ('Left-Cerebellum-Cortex',
                       'Right-Cerebellum-Cortex')
        src += setup_volume_source_space(
            'sample', pos=20., volume_label=label_names,
            subjects_dir=subjects_dir, add_interpolator=False)
        klass = MixedVectorSourceEstimate
    else:
        klass = VectorSourceEstimate
    if not vector:
        klass = klass._scalar_class
    vertices = [s['vertno'] for s in src]
    n_verts = np.array([len(v) for v in vertices])
    vol_means = np.arange(-1, 1 - len(src), -1)
    vol_means_t = np.repeat(vol_means[:, np.newaxis], n_times, axis=1)

    # get some labels
    labels_lh = read_labels_from_annot('sample', hemi='lh',
                                       subjects_dir=subjects_dir)
    labels_rh = read_labels_from_annot('sample', hemi='rh',
                                       subjects_dir=subjects_dir)
    labels = list()
    labels.extend(labels_lh[:5])
    labels.extend(labels_rh[:4])

    n_labels = len(labels)

    label_tcs = dict(
        mean=np.arange(n_labels)[:, None] * np.ones((n_labels, n_times)))
    label_tcs['max'] = label_tcs['mean']

    # compute the mean with sign flip
    label_tcs['mean_flip'] = np.zeros_like(label_tcs['mean'])
    for i, label in enumerate(labels):
        label_tcs['mean_flip'][i] = i * np.mean(
            label_sign_flip(label, src[:2]))

    # generate some stc's with known data
    stcs = list()
    pad = (((0, 0), (2, 0), (0, 0)), 'constant')
    for i in range(n_stcs):
        data = np.zeros((n_verts.sum(), n_times))
        # set the value of the stc within each label
        for j, label in enumerate(labels):
            if label.hemi == 'lh':
                idx = np.intersect1d(vertices[0], label.vertices)
                idx = np.searchsorted(vertices[0], idx)
            elif label.hemi == 'rh':
                idx = np.intersect1d(vertices[1], label.vertices)
                idx = len(vertices[0]) + np.searchsorted(vertices[1], idx)
            data[idx] = label_tcs['mean'][j]
        for j in range(len(vol_means)):
            offset = n_verts[:2 + j].sum()
            data[offset:offset + n_verts[j]] = vol_means[j]

        if vector:
            # the values it on the Z axis
            data = np.pad(data[:, np.newaxis], *pad)
        this_stc = klass(data, vertices, 0, 1)
        stcs.append(this_stc)

    if vector:
        for key in label_tcs:
            label_tcs[key] = np.pad(label_tcs[key][:, np.newaxis], *pad)
        vol_means_t = np.pad(vol_means_t[:, np.newaxis], *pad)

    # test some invalid inputs
    with pytest.raises(ValueError, match="Invalid value for the 'mode'"):
        extract_label_time_course(stcs, labels, src, mode='notamode')

    # have an empty label
    empty_label = labels[0].copy()
    empty_label.vertices += 1000000
    with pytest.raises(ValueError, match='does not contain any vertices'):
        extract_label_time_course(stcs, empty_label, src)

    # but this works:
    with pytest.warns(RuntimeWarning, match='does not contain any vertices'):
        tc = extract_label_time_course(stcs, empty_label, src,
                                       allow_empty=True)
    end_shape = (3, n_times) if vector else (n_times,)
    for arr in tc:
        assert arr.shape == (1 + len(vol_means),) + end_shape
        assert_array_equal(arr[:1], np.zeros((1,) + end_shape))
        if len(vol_means):
            assert_array_equal(arr[1:], vol_means_t)

    # test the different modes
    modes = ['mean', 'mean_flip', 'pca_flip', 'max', 'auto']

    for mode in modes:
        if vector and mode not in ('mean', 'max', 'auto'):
            with pytest.raises(ValueError, match='when using a vector'):
                extract_label_time_course(stcs, labels, src, mode=mode)
            continue
        with _record_warnings():  # SVD convergence on arm64
            label_tc = extract_label_time_course(stcs, labels, src, mode=mode)
        label_tc_method = [stc.extract_label_time_course(labels, src,
                                                         mode=mode)
                           for stc in stcs]
        assert (len(label_tc) == n_stcs)
        assert (len(label_tc_method) == n_stcs)
        for tc1, tc2 in zip(label_tc, label_tc_method):
            assert tc1.shape == (n_labels + len(vol_means),) + end_shape
            assert tc2.shape == (n_labels + len(vol_means),) + end_shape
            assert_allclose(tc1, tc2, rtol=1e-8, atol=1e-16)
            if mode == 'auto':
                use_mode = 'mean' if vector else 'mean_flip'
            else:
                use_mode = mode
            # XXX we don't check pca_flip, probably should someday...
            if use_mode in ('mean', 'max', 'mean_flip'):
                assert_array_almost_equal(tc1[:n_labels], label_tcs[use_mode])
            assert_array_almost_equal(tc1[n_labels:], vol_means_t)

    # test label with very few vertices (check SVD conditionals)
    label = Label(vertices=src[0]['vertno'][:2], hemi='lh')
    x = label_sign_flip(label, src[:2])
    assert (len(x) == 2)
    label = Label(vertices=[], hemi='lh')
    x = label_sign_flip(label, src[:2])
    assert (x.size == 0)


@testing.requires_testing_data
@pytest.mark.parametrize('label_type, mri_res, vector, test_label, cf, call', [
    (str, False, False, False, 'head', 'meth'),  # head frame
    (str, False, False, str, 'mri', 'func'),  # fastest, default for testing
    (str, False, True, int, 'mri', 'func'),  # vector
    (str, True, False, False, 'mri', 'func'),  # mri_resolution
    (list, True, False, False, 'mri', 'func'),  # volume label as list
    (dict, True, False, False, 'mri', 'func'),  # volume label as dict
])
def test_extract_label_time_course_volume(
        src_volume_labels, label_type, mri_res, vector, test_label, cf, call):
    """Test extraction of label time courses from Vol(Vector)SourceEstimate."""
    src_labels, volume_labels, lut = src_volume_labels
    n_tot = 46
    assert n_tot == len(src_labels)
    inv = read_inverse_operator(fname_inv_vol)
    if cf == 'head':
        src = inv['src']
        assert src[0]['coord_frame'] == FIFF.FIFFV_COORD_HEAD
        rr = apply_trans(invert_transform(inv['mri_head_t']), src[0]['rr'])
    else:
        assert cf == 'mri'
        src = read_source_spaces(fname_src_vol)
        assert src[0]['coord_frame'] == FIFF.FIFFV_COORD_MRI
        rr = src[0]['rr']
    for s in src_labels:
        assert_allclose(s['rr'], rr, atol=1e-7)
    assert len(src) == 1 and src.kind == 'volume'
    klass = VolVectorSourceEstimate
    if not vector:
        klass = klass._scalar_class
    vertices = [src[0]['vertno']]
    n_verts = len(src[0]['vertno'])
    n_times = 50
    data = vertex_values = np.arange(1, n_verts + 1)
    end_shape = (n_times,)
    if vector:
        end_shape = (3,) + end_shape
        data = np.pad(data[:, np.newaxis], ((0, 0), (2, 0)), 'constant')
    data = np.repeat(data[..., np.newaxis], n_times, -1)
    stcs = [klass(data.astype(float), vertices, 0, 1)]

    def eltc(*args, **kwargs):
        if call == 'func':
            return extract_label_time_course(stcs, *args, **kwargs)
        else:
            assert call == 'meth'
            return [stcs[0].extract_label_time_course(*args, **kwargs)]

    with pytest.raises(RuntimeError, match='atlas vox_mri_t does not match'):
        eltc(fname_fs_t1, src, mri_resolution=mri_res)
    assert len(src_labels) == 46  # includes unknown
    assert_array_equal(
        src[0]['vertno'],  # src includes some in "unknown" space
        np.sort(np.concatenate([s['vertno'] for s in src_labels])))
    # spot check
    assert src_labels[-1]['seg_name'] == 'CC_Anterior'
    assert src[0]['nuse'] == 4157
    assert len(src[0]['vertno']) == 4157
    assert sum(s['nuse'] for s in src_labels) == 4157
    assert_array_equal(src_labels[-1]['vertno'], [8011, 8032, 8557])
    assert_array_equal(
        np.where(np.in1d(src[0]['vertno'], [8011, 8032, 8557]))[0],
        [2672, 2688, 2995])
    # triage "labels" argument
    if mri_res:
        # All should be there
        missing = []
    else:
        # Nearest misses these
        missing = ['Left-vessel', 'Right-vessel', '5th-Ventricle',
                   'non-WM-hypointensities']
    n_want = len(src_labels)
    if label_type is str:
        labels = fname_aseg
    elif label_type is list:
        labels = (fname_aseg, volume_labels)
    else:
        assert label_type is dict
        labels = (fname_aseg, {k: lut[k] for k in volume_labels})
        assert mri_res
        assert len(missing) == 0
        # we're going to add one that won't exist
        missing = ['intentionally_bad']
        labels[1][missing[0]] = 10000
        n_want += 1
        n_tot += 1
    n_want -= len(missing)

    # actually do the testing
    if cf == 'head' and not mri_res:  # some missing
        with pytest.warns(RuntimeWarning, match='any vertices'):
            eltc(labels, src, allow_empty=True, mri_resolution=mri_res)
    for mode in ('mean', 'max'):
        with catch_logging() as log:
            label_tc = eltc(labels, src, mode=mode, allow_empty='ignore',
                            mri_resolution=mri_res, verbose=True)
        log = log.getvalue()
        assert re.search('^Reading atlas.*aseg\\.mgz\n', log) is not None
        if len(missing):
            # assert that the missing ones get logged
            assert 'does not contain' in log
            assert repr(missing) in log
        else:
            assert 'does not contain' not in log
        assert '\n%d/%d atlas regions had at least' % (n_want, n_tot) in log
        assert len(label_tc) == 1
        label_tc = label_tc[0]
        assert label_tc.shape == (n_tot,) + end_shape
        if vector:
            assert_array_equal(label_tc[:, :2], 0.)
            label_tc = label_tc[:, 2]
        assert label_tc.shape == (n_tot, n_times)
        # let's test some actual values by trusting the masks provided by
        # setup_volume_source_space. mri_resolution=True does some
        # interpolation so we should not expect equivalence, False does
        # nearest so we should.
        if mri_res:
            rtol = 0.2 if mode == 'mean' else 0.8  # max much more sensitive
        else:
            rtol = 0.
        for si, s in enumerate(src_labels):
            func = dict(mean=np.mean, max=np.max)[mode]
            these = vertex_values[np.in1d(src[0]['vertno'], s['vertno'])]
            assert len(these) == s['nuse']
            if si == 0 and s['seg_name'] == 'Unknown':
                continue  # unknown is crappy
            if s['nuse'] == 0:
                want = 0.
                if mri_res:
                    # this one is totally due to interpolation, so no easy
                    # test here
                    continue
            else:
                want = func(these)
            assert_allclose(label_tc[si], want, atol=1e-6, rtol=rtol)
            # compare with in_label, only on every fourth for speed
            if test_label is not False and si % 4 == 0:
                label = s['seg_name']
                if test_label is int:
                    label = lut[label]
                in_label = stcs[0].in_label(
                    label, fname_aseg, src).data
                assert in_label.shape == (s['nuse'],) + end_shape
                if vector:
                    assert_array_equal(in_label[:, :2], 0.)
                    in_label = in_label[:, 2]
                if want == 0:
                    assert in_label.shape[0] == 0
                else:
                    in_label = func(in_label)
                    assert_allclose(in_label, want, atol=1e-6, rtol=rtol)
        if mode == 'mean' and not vector:  # check the reverse
            if label_type is dict:
                ctx = pytest.warns(RuntimeWarning, match='does not contain')
            else:
                ctx = nullcontext()
            with ctx:
                stc_back = labels_to_stc(labels, label_tc, src=src)
            assert stc_back.data.shape == stcs[0].data.shape
            corr = np.corrcoef(stc_back.data.ravel(),
                               stcs[0].data.ravel())[0, 1]
            assert 0.6 < corr < 0.63
            assert_allclose(_varexp(label_tc, label_tc), 1.)
            ve = _varexp(stc_back.data, stcs[0].data)
            assert 0.83 < ve < 0.85
            with _record_warnings():  # ignore no output
                label_tc_rt = extract_label_time_course(
                    stc_back, labels, src=src, mri_resolution=mri_res,
                    allow_empty=True)
            assert label_tc_rt.shape == label_tc.shape
            corr = np.corrcoef(label_tc.ravel(), label_tc_rt.ravel())[0, 1]
            lower, upper = (0.99, 0.999) if mri_res else (0.95, 0.97)
            assert lower < corr < upper
            ve = _varexp(label_tc_rt, label_tc)
            lower, upper = (0.99, 0.999) if mri_res else (0.97, 0.99)
            assert lower < ve < upper


def _varexp(got, want):
    return max(
        1 - np.linalg.norm(got.ravel() - want.ravel()) ** 2 /
        np.linalg.norm(want) ** 2, 0.)


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

    vertices = [np.arange(n_vertices)]
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
        VolSourceEstimate((kernel, sens_data), vertices, 0, 1)


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
def test_spatio_temporal_tris_adjacency():
    """Test spatio-temporal adjacency from triangles."""
    tris = np.array([[0, 1, 2], [3, 4, 5]])
    adjacency = spatio_temporal_tris_adjacency(tris, 2)
    x = [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    components = stats.cluster_level._get_components(np.array(x), adjacency)
    # _get_components works differently now...
    old_fmt = [0, 0, -2, -2, -2, -2, 0, -2, -2, -2, -2, 1]
    new_fmt = np.array(old_fmt)
    new_fmt = [np.nonzero(new_fmt == v)[0]
               for v in np.unique(new_fmt[new_fmt >= 0])]
    assert len(new_fmt) == len(components)
    for c, n in zip(components, new_fmt):
        assert_array_equal(c, n)


@testing.requires_testing_data
def test_spatio_temporal_src_adjacency():
    """Test spatio-temporal adjacency from source spaces."""
    tris = np.array([[0, 1, 2], [3, 4, 5]])
    src = [dict(), dict()]
    adjacency = spatio_temporal_tris_adjacency(tris, 2).todense()
    assert_allclose(np.diag(adjacency), 1.)
    src[0]['use_tris'] = np.array([[0, 1, 2]])
    src[1]['use_tris'] = np.array([[0, 1, 2]])
    src[0]['vertno'] = np.array([0, 1, 2])
    src[1]['vertno'] = np.array([0, 1, 2])
    src[0]['type'] = 'surf'
    src[1]['type'] = 'surf'
    adjacency2 = spatio_temporal_src_adjacency(src, 2)
    assert_array_equal(adjacency2.todense(), adjacency)
    # add test for dist adjacency
    src[0]['dist'] = np.ones((3, 3)) - np.eye(3)
    src[1]['dist'] = np.ones((3, 3)) - np.eye(3)
    src[0]['vertno'] = [0, 1, 2]
    src[1]['vertno'] = [0, 1, 2]
    src[0]['type'] = 'surf'
    src[1]['type'] = 'surf'
    adjacency3 = spatio_temporal_src_adjacency(src, 2, dist=2)
    assert_array_equal(adjacency3.todense(), adjacency)
    # add test for source space adjacency with omitted vertices
    inverse_operator = read_inverse_operator(fname_inv)
    src_ = inverse_operator['src']
    with pytest.warns(RuntimeWarning, match='will have holes'):
        adjacency = spatio_temporal_src_adjacency(src_, n_times=2)
    a = adjacency.shape[0] / 2
    b = sum([s['nuse'] for s in inverse_operator['src']])
    assert (a == b)

    assert_equal(grade_to_tris(5).shape, [40960, 3])


@requires_pandas
def test_to_data_frame():
    """Test stc Pandas exporter."""
    n_vert, n_times = 10, 5
    vertices = [np.arange(n_vert, dtype=np.int64), np.empty(0, dtype=np.int64)]
    data = rng.randn(n_vert, n_times)
    stc_surf = SourceEstimate(data, vertices=vertices, tmin=0, tstep=1,
                              subject='sample')
    stc_vol = VolSourceEstimate(data, vertices=vertices[:1], tmin=0, tstep=1,
                                subject='sample')
    for stc in [stc_surf, stc_vol]:
        df = stc.to_data_frame()
        # test data preservation (first 2 dataframe elements are subj & time)
        assert_array_equal(df.values.T[2:], stc.data)
        # test long format
        df_long = stc.to_data_frame(long_format=True)
        assert len(df_long) == stc.data.size
        expected = ('subject', 'time', 'source', 'value')
        assert set(expected) == set(df_long.columns)


@requires_pandas
@pytest.mark.parametrize('index', ('time', ['time', 'subject'], None))
def test_to_data_frame_index(index):
    """Test index creation in stc Pandas exporter."""
    n_vert, n_times = 10, 5
    vertices = [np.arange(n_vert, dtype=np.int64), np.empty(0, dtype=np.int64)]
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


@pytest.mark.parametrize('kind', ('surface', 'mixed', 'volume'))
@pytest.mark.parametrize('vector', (False, True))
@pytest.mark.parametrize('n_times', (5, 1))
def test_get_peak(kind, vector, n_times):
    """Test peak getter."""
    n_vert = 10
    vertices = [np.arange(n_vert)]
    if kind == 'surface':
        klass = VectorSourceEstimate
        vertices += [np.empty(0, int)]
    elif kind == 'mixed':
        klass = MixedVectorSourceEstimate
        vertices += [np.empty(0, int), np.empty(0, int)]
    else:
        assert kind == 'volume'
        klass = VolVectorSourceEstimate
    data = np.zeros((n_vert, n_times))
    data[1, -1] = 1
    if vector:
        data = np.repeat(data[:, np.newaxis], 3, 1)
    else:
        klass = klass._scalar_class
    stc = klass(data, vertices, 0, 1)

    with pytest.raises(ValueError, match='out of bounds'):
        stc.get_peak(tmin=-100)
    with pytest.raises(ValueError, match='out of bounds'):
        stc.get_peak(tmax=90)
    with pytest.raises(ValueError,
                       match='must be <=' if n_times > 1 else 'out of'):
        stc.get_peak(tmin=0.002, tmax=0.001)

    vert_idx, time_idx = stc.get_peak()
    vertno = np.concatenate(stc.vertices)
    assert vert_idx in vertno
    assert time_idx in stc.times
    data_idx, time_idx = stc.get_peak(vert_as_index=True, time_as_index=True)
    if vector:
        use_data = stc.magnitude().data
    else:
        use_data = stc.data
    assert data_idx == 1
    assert time_idx == n_times - 1
    assert data_idx == np.argmax(np.abs(use_data[:, time_idx]))
    assert time_idx == np.argmax(np.abs(use_data[data_idx, :]))
    if kind == 'surface':
        data_idx_2, time_idx_2 = stc.get_peak(
            vert_as_index=True, time_as_index=True, hemi='lh')
        assert data_idx_2 == data_idx
        assert time_idx_2 == time_idx
        with pytest.raises(RuntimeError, match='no vertices'):
            stc.get_peak(hemi='rh')


@requires_version('h5io')
@testing.requires_testing_data
def test_mixed_stc(tmp_path):
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

    # make sure error is raised for plotting surface with volume source
    fname = tmp_path / 'mixed-stc.h5'
    stc.save(fname)
    stc_out = read_source_estimate(fname)
    assert_array_equal(stc_out.vertices, vertno)
    assert_array_equal(stc_out.data, data)
    assert stc_out.tmin == 0
    assert stc_out.tstep == 1
    assert isinstance(stc_out, MixedSourceEstimate)


@requires_version('h5io')
@pytest.mark.parametrize('klass, kind', [
    (VectorSourceEstimate, 'surf'),
    (VolVectorSourceEstimate, 'vol'),
    (VolVectorSourceEstimate, 'discrete'),
    (MixedVectorSourceEstimate, 'mixed'),
])
@pytest.mark.parametrize('dtype', [
    np.float32, np.float64, np.complex64, np.complex128])
def test_vec_stc_basic(tmp_path, klass, kind, dtype):
    """Test (vol)vector source estimate."""
    nn = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [np.sqrt(1. / 2.), 0, np.sqrt(1. / 2.)],
        [np.sqrt(1 / 3.)] * 3
    ], np.float32)

    data = np.array([
        [1, 0, 0],
        [0, 2, 0],
        [-3, 0, 0],
        [1, 1, 1],
    ], dtype)[:, :, np.newaxis]
    amplitudes = np.array([1, 2, 3, np.sqrt(3)], dtype)
    magnitudes = amplitudes.copy()
    normals = np.array([1, 2, -3. / np.sqrt(2), np.sqrt(3)], dtype)
    if dtype in (np.complex64, np.complex128):
        data *= 1j
        amplitudes *= 1j
        normals *= 1j
    directions = np.array(
        [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [1. / np.sqrt(3)] * 3])
    vol_kind = kind if kind in ('discrete', 'vol') else 'vol'
    vol_src = SourceSpaces([dict(nn=nn, type=vol_kind)])
    assert vol_src.kind == dict(vol='volume').get(vol_kind, vol_kind)
    vol_verts = [np.arange(4)]
    surf_src = SourceSpaces([dict(nn=nn[:2], type='surf'),
                             dict(nn=nn[2:], type='surf')])
    assert surf_src.kind == 'surface'
    surf_verts = [np.array([0, 1]), np.array([0, 1])]
    if klass is VolVectorSourceEstimate:
        src = vol_src
        verts = vol_verts
    elif klass is VectorSourceEstimate:
        src = surf_src
        verts = surf_verts
    if klass is MixedVectorSourceEstimate:
        src = surf_src + vol_src
        verts = surf_verts + vol_verts
        assert src.kind == 'mixed'
        data = np.tile(data, (2, 1, 1))
        amplitudes = np.tile(amplitudes, 2)
        magnitudes = np.tile(magnitudes, 2)
        normals = np.tile(normals, 2)
        directions = np.tile(directions, (2, 1))
    stc = klass(data, verts, 0, 1, 'foo')
    amplitudes = amplitudes[:, np.newaxis]
    magnitudes = magnitudes[:, np.newaxis]

    # Magnitude of the vectors
    assert_array_equal(stc.magnitude().data, magnitudes)

    # Vector components projected onto the vertex normals
    if kind in ('vol', 'mixed'):
        with pytest.raises(RuntimeError, match='surface or discrete'):
            stc.project('normal', src)[0]
    else:
        normal = stc.project('normal', src)[0]
        assert_allclose(normal.data[:, 0], normals)

    # Maximal-variance component, either to keep amps pos or to align to src-nn
    projected, got_directions = stc.project('pca')
    assert_allclose(got_directions, directions)
    assert_allclose(projected.data, amplitudes)
    projected, got_directions = stc.project('pca', src)
    flips = np.array([[1], [1], [-1.], [1]])
    if klass is MixedVectorSourceEstimate:
        flips = np.tile(flips, (2, 1))
    assert_allclose(got_directions, directions * flips)
    assert_allclose(projected.data, amplitudes * flips)

    out_name = tmp_path / 'temp.h5'
    stc.save(out_name)
    stc_read = read_source_estimate(out_name)
    assert_allclose(stc.data, stc_read.data)
    assert len(stc.vertices) == len(stc_read.vertices)
    for v1, v2 in zip(stc.vertices, stc_read.vertices):
        assert_array_equal(v1, v2)

    stc = klass(data[:, :, 0], verts, 0, 1)  # upbroadcast
    assert stc.data.shape == (len(data), 3, 1)
    # Bad data
    with pytest.raises(ValueError, match='must have shape.*3'):
        klass(data[:, :2], verts, 0, 1)
    data = data[:, :, np.newaxis]
    with pytest.raises(ValueError, match='3 dimensions for .*VectorSource'):
        klass(data, verts, 0, 1)


@pytest.mark.parametrize('real', (True, False))
def test_source_estime_project(real):
    """Test projecting a source estimate onto direction of max power."""
    n_src, n_times = 4, 100
    rng = np.random.RandomState(0)
    data = rng.randn(n_src, 3, n_times)
    if not real:
        data = data + 1j * rng.randn(n_src, 3, n_times)
        assert data.dtype == np.complex128
    else:
        assert data.dtype == np.float64

    # Make sure that the normal we get maximizes the power
    # (i.e., minimizes the negative power)
    want_nn = np.empty((n_src, 3))
    for ii in range(n_src):
        x0 = np.ones(3)

        def objective(x):
            x = x / np.linalg.norm(x)
            return -np.linalg.norm(np.dot(x, data[ii]))
        want_nn[ii] = fmin_cobyla(objective, x0, (), rhobeg=0.1, rhoend=1e-6)
    want_nn /= np.linalg.norm(want_nn, axis=1, keepdims=True)

    stc = VolVectorSourceEstimate(data, [np.arange(n_src)], 0, 1)
    stc_max, directions = stc.project('pca')
    flips = np.sign(np.sum(directions * want_nn, axis=1, keepdims=True))
    directions *= flips
    assert_allclose(directions, want_nn, atol=2e-6)


@testing.requires_testing_data
def test_source_estime_project_label():
    """Test projecting a source estimate onto direction of max power."""
    fwd = read_forward_solution(fname_fwd)
    fwd = pick_types_forward(fwd, meg=True, eeg=False)

    evoked = read_evokeds(fname_evoked, baseline=(None, 0))[0]
    noise_cov = read_cov(fname_cov)
    free = make_inverse_operator(
        evoked.info, fwd, noise_cov, loose=1.)
    stc_free = apply_inverse(evoked, free, pick_ori='vector')

    stc_pca = stc_free.project('pca', fwd['src'])[0]

    labels_lh = read_labels_from_annot('sample', 'aparc', 'lh',
                                       subjects_dir=subjects_dir)
    new_label = labels_lh[0] + labels_lh[1]

    stc_in_label = stc_free.in_label(new_label)
    stc_pca_in_label = stc_pca.in_label(new_label)

    stc_in_label_pca = stc_in_label.project('pca', fwd['src'])[0]
    assert_array_equal(stc_pca_in_label.data, stc_in_label_pca.data)


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
    stc_fixed_vector = apply_inverse(evoked, fixed, pick_ori='vector')
    assert_allclose(stc_fixed.data,
                    stc_fixed_vector.project('normal', fixed['src'])[0].data)
    stc_fixedish = apply_inverse(evoked, fixedish, pick_ori=pick_ori)
    if pick_ori == 'vector':
        assert_allclose(stc_fixed_vector.data, stc_fixedish.data, atol=1e-2)
        # two ways here: with magnitude...
        assert_allclose(
            abs(stc_fixed).data, stc_fixedish.magnitude().data, atol=1e-2)
        # ... and when picking the normal (signed)
        stc_fixedish = stc_fixedish.project('normal', fixedish['src'])[0]
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
def test_vol_adjacency():
    """Test volume adjacency."""
    vol = read_source_spaces(fname_vsrc)

    pytest.raises(ValueError, spatial_src_adjacency, vol, dist=1.)

    adjacency = spatial_src_adjacency(vol)
    n_vertices = vol[0]['inuse'].sum()
    assert_equal(adjacency.shape, (n_vertices, n_vertices))
    assert (np.all(adjacency.data == 1))
    assert (isinstance(adjacency, sparse.coo_matrix))

    adjacency2 = spatio_temporal_src_adjacency(vol, n_times=2)
    assert_equal(adjacency2.shape, (2 * n_vertices, 2 * n_vertices))
    assert (np.all(adjacency2.data == 1))


@testing.requires_testing_data
def test_spatial_src_adjacency():
    """Test spatial adjacency functionality."""
    # oct
    src = read_source_spaces(fname_src)
    assert src[0]['dist'] is not None  # distance info
    with pytest.warns(RuntimeWarning, match='will have holes'):
        con = spatial_src_adjacency(src).toarray()
    con_dist = spatial_src_adjacency(src, dist=0.01).toarray()
    assert (con == con_dist).mean() > 0.75
    # ico
    src = read_source_spaces(fname_src_fs)
    con = spatial_src_adjacency(src).tocsr()
    con_tris = spatial_tris_adjacency(grade_to_tris(5)).tocsr()
    assert con.shape == con_tris.shape
    assert_array_equal(con.data, con_tris.data)
    assert_array_equal(con.indptr, con_tris.indptr)
    assert_array_equal(con.indices, con_tris.indices)
    # one hemi
    con_lh = spatial_src_adjacency(src[:1]).tocsr()
    con_lh_tris = spatial_tris_adjacency(grade_to_tris(5)).tocsr()
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
    vertices = [src[0]['vertno']]
    n_vertices = len(vertices[0])
    data = (1 + np.arange(n_vertices))[:, np.newaxis]
    stc_tmp = VolSourceEstimate(data, vertices, tmin=0., tstep=1.)
    img = stc_tmp.as_volume(src, mri_resolution=False)
    img_data = _get_img_fdata(img)[:, :, :, 0].T
    mask_nib = (img_data != 0)
    assert_array_equal(img_data[mask_nib], data[:, 0])
    assert_array_equal(np.where(mask_nib.ravel())[0], src[0]['vertno'])
    assert_array_equal(mask, mask_nib)
    assert_array_equal(img_data.shape, mask.shape)


@testing.requires_testing_data
def test_stc_near_sensors(tmp_path):
    """Test stc_near_sensors."""
    info = read_info(fname_evoked)
    # pick the left EEG sensors
    picks = pick_types(info, meg=False, eeg=True, exclude=())
    picks = [pick for pick in picks if info['chs'][pick]['loc'][0] < 0]
    pick_info(info, picks, copy=False)
    with info._unlock():
        info['projs'] = []
    info['bads'] = []
    assert info['nchan'] == 33
    evoked = EvokedArray(np.eye(info['nchan']), info)
    trans = read_trans(fname_fwd)
    assert trans['to'] == FIFF.FIFFV_COORD_HEAD
    this_dir = str(tmp_path)
    # testing does not have pial, so fake it
    os.makedirs(op.join(this_dir, 'sample', 'surf'))
    for hemi in ('lh', 'rh'):
        copyfile(op.join(subjects_dir, 'sample', 'surf', f'{hemi}.white'),
                 op.join(this_dir, 'sample', 'surf', f'{hemi}.pial'))
    # here we use a distance is smaller than the inter-sensor distance
    kwargs = dict(subject='sample', trans=trans, subjects_dir=this_dir,
                  verbose=True, distance=0.005)
    with pytest.raises(ValueError, match='No appropriate channels'):
        stc_near_sensors(evoked, **kwargs)
    evoked.set_channel_types({ch_name: 'ecog' for ch_name in evoked.ch_names})
    with catch_logging() as log:
        stc = stc_near_sensors(evoked, **kwargs)
    log = log.getvalue()
    assert 'Minimum projected intra-sensor distance: 7.' in log  # 7.4
    # this should be left-hemisphere dominant
    assert 5000 > len(stc.vertices[0]) > 4000
    assert 200 > len(stc.vertices[1]) > 100
    # and at least one vertex should have the channel values
    dists = cdist(stc.data, evoked.data)
    assert np.isclose(dists, 0., atol=1e-6).any(0).all()

    src = read_source_spaces(fname_src)  # uses "white" but should be okay
    for s in src:
        transform_surface_to(s, 'head', trans, copy=False)
    assert src[0]['coord_frame'] == FIFF.FIFFV_COORD_HEAD
    stc_src = stc_near_sensors(evoked, src=src, **kwargs)
    assert len(stc_src.data) == 7928
    with pytest.warns(RuntimeWarning, match='not included'):  # some removed
        stc_src_full = compute_source_morph(
            stc_src, 'sample', 'sample', smooth=5, spacing=None,
            subjects_dir=subjects_dir).apply(stc_src)
    lh_idx = np.searchsorted(stc_src_full.vertices[0], stc.vertices[0])
    rh_idx = np.searchsorted(stc_src_full.vertices[1], stc.vertices[1])
    rh_idx += len(stc_src_full.vertices[0])
    sub_data = stc_src_full.data[np.concatenate([lh_idx, rh_idx])]
    assert sub_data.shape == stc.data.shape
    corr = np.corrcoef(stc.data.ravel(), sub_data.ravel())[0, 1]
    assert 0.6 < corr < 0.7

    # now single-weighting mode
    stc_w = stc_near_sensors(evoked, mode='single', **kwargs)
    assert_array_less(stc_w.data, stc.data + 1e-3)  # some tol
    assert len(stc_w.data) == len(stc.data)
    # at least one for each sensor should have projected right on it
    dists = cdist(stc_w.data, evoked.data)
    assert np.isclose(dists, 0., atol=1e-6).any(0).all()

    # finally, nearest mode: all should match
    stc_n = stc_near_sensors(evoked, mode='nearest', **kwargs)
    assert len(stc_n.data) == len(stc.data)
    # at least one for each sensor should have projected right on it
    dists = cdist(stc_n.data, evoked.data)
    assert np.isclose(dists, 0., atol=1e-6).any(1).all()  # all vert eq some ch

    # these are EEG electrodes, so the distance 0.01 is too small for the
    # scalp+skull. Even at a distance of 33 mm EEG 060 is too far:
    with pytest.warns(RuntimeWarning, match='Channel missing in STC: EEG 060'):
        stc = stc_near_sensors(evoked, trans, 'sample', subjects_dir=this_dir,
                               project=False, distance=0.033)
    assert stc.data.any(0).sum() == len(evoked.ch_names) - 1

    # and now with volumetric projection
    src = read_source_spaces(fname_vsrc)
    with catch_logging() as log:
        stc_vol = stc_near_sensors(
            evoked, trans, 'sample', src=src, surface=None,
            subjects_dir=subjects_dir, distance=0.033, verbose=True)
    assert isinstance(stc_vol, VolSourceEstimate)
    log = log.getvalue()
    assert '4157 volume vertices' in log


@requires_version('pymatreader')
@testing.requires_testing_data
def test_stc_near_sensors_picks():
    """Test using picks with stc_near_sensors."""
    info = mne.io.read_raw_nirx(fname_nirx).info
    evoked = mne.EvokedArray(np.ones((len(info['ch_names']), 1)), info)
    src = mne.read_source_spaces(fname_src_fs)
    kwargs = dict(
        evoked=evoked, subject='fsaverage', trans='fsaverage',
        subjects_dir=subjects_dir, src=src, surface=None, project=True)
    with pytest.raises(ValueError, match='No appropriate channels'):
        stc_near_sensors(**kwargs)
    picks = np.arange(len(info['ch_names']))
    data = stc_near_sensors(picks=picks, **kwargs).data
    assert len(data) == 20484
    assert (data >= 0).all()
    data = data[data > 0]
    n_pts = len(data)
    assert 500 < n_pts < 600
    lo, hi = np.percentile(data, (5, 95))
    assert 0.01 < lo < 0.1
    assert 1.3 < hi < 1.7  # > 1
    data = stc_near_sensors(picks=picks, mode='weighted', **kwargs).data
    assert (data >= 0).all()
    data = data[data > 0]
    assert len(data) == n_pts
    assert_array_equal(data, 1.)  # values preserved


def _make_morph_map_hemi_same(subject_from, subject_to, subjects_dir,
                              reg_from, reg_to):
    return _make_morph_map_hemi(subject_from, subject_from, subjects_dir,
                                reg_from, reg_from)


@requires_nibabel()
@testing.requires_testing_data
@pytest.mark.parametrize('kind', (
    pytest.param('volume', marks=[requires_version('dipy'),
                                  pytest.mark.slowtest]),
    'surface',
))
@pytest.mark.parametrize('scale', ((1.0, 0.8, 1.2), 1., 0.9))
def test_scale_morph_labels(kind, scale, monkeypatch, tmp_path):
    """Test label extraction, morphing, and MRI scaling relationships."""
    tempdir = str(tmp_path)
    subject_from = 'sample'
    subject_to = 'small'
    testing_dir = op.join(subjects_dir, subject_from)
    from_dir = op.join(tempdir, subject_from)
    for root in ('mri', 'surf', 'label', 'bem'):
        os.makedirs(op.join(from_dir, root), exist_ok=True)
    for hemi in ('lh', 'rh'):
        for root, fname in (('surf', 'sphere'), ('surf', 'white'),
                            ('surf', 'sphere.reg'),
                            ('label', 'aparc.annot')):
            use_fname = op.join(root, f'{hemi}.{fname}')
            copyfile(op.join(testing_dir, use_fname),
                     op.join(from_dir, use_fname))
    for root, fname in (('mri', 'aseg.mgz'), ('mri', 'brain.mgz')):
        use_fname = op.join(root, fname)
        copyfile(op.join(testing_dir, use_fname),
                 op.join(from_dir, use_fname))
    del testing_dir
    if kind == 'surface':
        src_from = read_source_spaces(fname_src_3)
        assert src_from[0]['dist'] is None
        assert src_from[0]['nearest'] is not None
        # avoid patch calc
        src_from[0]['nearest'] = src_from[1]['nearest'] = None
        assert len(src_from) == 2
        assert src_from[0]['nuse'] == src_from[1]['nuse'] == 258
        klass = SourceEstimate
        labels_from = read_labels_from_annot(
            subject_from, subjects_dir=tempdir)
        n_labels = len(labels_from)
        write_source_spaces(op.join(tempdir, subject_from, 'bem',
                                    f'{subject_from}-oct-4-src.fif'), src_from)
    else:
        assert kind == 'volume'
        pytest.importorskip('dipy')
        src_from = read_source_spaces(fname_src_vol)
        src_from[0]['subject_his_id'] = subject_from
        labels_from = op.join(
            tempdir, subject_from, 'mri', 'aseg.mgz')
        n_labels = 46
        assert op.isfile(labels_from)
        klass = VolSourceEstimate
        assert len(src_from) == 1
        assert src_from[0]['nuse'] == 4157
        write_source_spaces(
            op.join(from_dir, 'bem', 'sample-vol20-src.fif'), src_from)
    scale_mri(subject_from, subject_to, scale, subjects_dir=tempdir,
              annot=True, skip_fiducials=True, verbose=True,
              overwrite=True)
    if kind == 'surface':
        src_to = read_source_spaces(
            op.join(tempdir, subject_to, 'bem',
                    f'{subject_to}-oct-4-src.fif'))
        labels_to = read_labels_from_annot(
            subject_to, subjects_dir=tempdir)
        # Save time since we know these subjects are identical
        monkeypatch.setattr(mne.morph_map, '_make_morph_map_hemi',
                            _make_morph_map_hemi_same)
    else:
        src_to = read_source_spaces(
            op.join(tempdir, subject_to, 'bem',
                    f'{subject_to}-vol20-src.fif'))
        labels_to = op.join(
            tempdir, subject_to, 'mri', 'aseg.mgz')
    # 1. Label->STC->Label for the given subject should be identity
    #    (for surfaces at least; for volumes it's not as clean as this
    #     due to interpolation)
    n_times = 50
    rng = np.random.RandomState(0)
    label_tc = rng.randn(n_labels, n_times)
    # check that a random permutation of our labels yields a terrible
    # correlation
    corr = np.corrcoef(label_tc.ravel(),
                       rng.permutation(label_tc).ravel())[0, 1]
    assert -0.06 < corr < 0.06
    # project label activations to full source space
    with pytest.raises(ValueError, match='subject'):
        labels_to_stc(labels_from, label_tc, src=src_from, subject='foo')
    stc = labels_to_stc(labels_from, label_tc, src=src_from)
    assert stc.subject == 'sample'
    assert isinstance(stc, klass)
    label_tc_from = extract_label_time_course(
        stc, labels_from, src_from, mode='mean')
    if kind == 'surface':
        assert_allclose(label_tc, label_tc_from, rtol=1e-12, atol=1e-12)
    else:
        corr = np.corrcoef(label_tc.ravel(), label_tc_from.ravel())[0, 1]
        assert 0.93 < corr < 0.95

    #
    # 2. Changing STC subject to the surrogate and then extracting
    #
    stc.subject = subject_to
    label_tc_to = extract_label_time_course(
        stc, labels_to, src_to, mode='mean')
    assert_allclose(label_tc_from, label_tc_to, rtol=1e-12, atol=1e-12)
    stc.subject = subject_from

    #
    # 3. Morphing STC to new subject then extracting
    #
    if isinstance(scale, tuple) and kind == 'volume':
        ctx = nullcontext()
        test_morph = True
    elif kind == 'surface':
        ctx = pytest.warns(RuntimeWarning, match='not included')
        test_morph = True
    else:
        ctx = nullcontext()
        test_morph = True
    with ctx:  # vertices not included
        morph = compute_source_morph(
            src_from, subject_to=subject_to, src_to=src_to,
            subjects_dir=tempdir, niter_sdr=(), smooth=1,
            zooms=14., verbose=True)  # speed up with higher zooms
    if kind == 'volume':
        got_affine = morph.pre_affine.affine
        want_affine = np.eye(4)
        want_affine.ravel()[::5][:3] = 1. / np.array(scale, float)
        # just a scaling (to within 1% if zooms=None, 20% with zooms=10)
        assert_allclose(want_affine[:, :3], got_affine[:, :3], atol=0.4)
        assert got_affine[3, 3] == 1.
        # little translation (to within `limit` mm)
        move = np.linalg.norm(got_affine[:3, 3])
        limit = 2. if scale == 1. else 12
        assert move < limit, scale
    if test_morph:
        stc_to = morph.apply(stc)
        label_tc_to_morph = extract_label_time_course(
            stc_to, labels_to, src_to, mode='mean')
        if kind == 'volume':
            corr = np.corrcoef(
                label_tc.ravel(), label_tc_to_morph.ravel())[0, 1]
            if isinstance(scale, tuple):
                # some other fixed constant
                # min_, max_ = 0.84, 0.855  # zooms='auto' values
                min_, max_ = 0.57, 0.67
            elif scale == 1:
                # min_, max_ = 0.85, 0.875  # zooms='auto' values
                min_, max_ = 0.72, 0.76
            else:
                # min_, max_ = 0.84, 0.855  # zooms='auto' values
                min_, max_ = 0.46, 0.63
            assert min_ < corr <= max_, scale
        else:
            assert_allclose(
                label_tc, label_tc_to_morph, atol=1e-12, rtol=1e-12)

    #
    # 4. The same round trip from (1) but in the warped space
    #
    stc = labels_to_stc(labels_to, label_tc, src=src_to)
    assert isinstance(stc, klass)
    label_tc_to = extract_label_time_course(
        stc, labels_to, src_to, mode='mean')
    if kind == 'surface':
        assert_allclose(label_tc, label_tc_to, rtol=1e-12, atol=1e-12)
    else:
        corr = np.corrcoef(label_tc.ravel(), label_tc_to.ravel())[0, 1]
        assert 0.93 < corr < 0.96, scale


@testing.requires_testing_data
@pytest.mark.parametrize('kind', [
    'surface',
    pytest.param('volume', marks=[pytest.mark.slowtest,
                                  requires_version('nibabel')]),
])
def test_label_extraction_subject(kind):
    """Test that label extraction subject is treated properly."""
    if kind == 'surface':
        inv = read_inverse_operator(fname_inv)
        labels = read_labels_from_annot(
            'sample', subjects_dir=subjects_dir)
        labels_fs = read_labels_from_annot(
            'fsaverage', subjects_dir=subjects_dir)
        labels_fs = [label for label in labels_fs
                     if not label.name.startswith('unknown')]
        assert all(label.subject == 'sample' for label in labels)
        assert all(label.subject == 'fsaverage' for label in labels_fs)
        assert len(labels) == len(labels_fs) == 68
        n_labels = 68
    else:
        assert kind == 'volume'
        inv = read_inverse_operator(fname_inv_vol)
        inv['src'][0]['subject_his_id'] = 'sample'  # modernize
        labels = op.join(subjects_dir, 'sample', 'mri', 'aseg.mgz')
        labels_fs = op.join(subjects_dir, 'fsaverage', 'mri', 'aseg.mgz')
        n_labels = 46
    src = inv['src']
    assert src.kind == kind
    assert src._subject == 'sample'
    ave = read_evokeds(fname_evoked)[0].apply_baseline((None, 0)).crop(0, 0.01)
    assert len(ave.times) == 4
    stc = apply_inverse(ave, inv)
    assert stc.subject == 'sample'
    ltc = extract_label_time_course(stc, labels, src)
    stc.subject = 'fsaverage'
    with pytest.raises(ValueError, match=r'source spac.*not match.* stc\.sub'):
        extract_label_time_course(stc, labels, src)
    stc.subject = 'sample'
    assert ltc.shape == (n_labels, 4)
    if kind == 'volume':
        with pytest.raises(RuntimeError, match='atlas.*not match.*source spa'):
            extract_label_time_course(stc, labels_fs, src)
    else:
        with pytest.raises(ValueError, match=r'label\.sub.*not match.* stc\.'):
            extract_label_time_course(stc, labels_fs, src)
        stc.subject = None
        with pytest.raises(ValueError, match=r'label\.sub.*not match.* sourc'):
            extract_label_time_course(stc, labels_fs, src)
