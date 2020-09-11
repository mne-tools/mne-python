# -*- coding: utf-8 -*-
# Author: Tommy Clausner <Tommy.Clausner@gmail.com>
#
# License: BSD (3-clause)
import os.path as op

import pytest
import numpy as np
from numpy.testing import (assert_array_less, assert_allclose,
                           assert_array_equal)
from scipy.spatial.distance import cdist

import mne
from mne import (SourceEstimate, VolSourceEstimate, VectorSourceEstimate,
                 read_evokeds, SourceMorph, compute_source_morph,
                 read_source_morph, read_source_estimate,
                 read_forward_solution, grade_to_vertices,
                 setup_volume_source_space, make_forward_solution,
                 make_sphere_model, make_ad_hoc_cov, VolVectorSourceEstimate,
                 read_freesurfer_lut)
from mne.datasets import testing
from mne.fixes import _get_img_fdata
from mne.minimum_norm import (apply_inverse, read_inverse_operator,
                              make_inverse_operator)
from mne.source_space import (get_volume_labels_from_aseg, _get_mri_info_data,
                              _get_atlas_values, _add_interpolator)
from mne.utils import (run_tests_if_main, requires_nibabel, check_version,
                       requires_dipy, requires_h5py)
from mne.fixes import _get_args

# Setup paths

data_path = testing.data_path(download=False)
sample_dir = op.join(data_path, 'MEG', 'sample')
subjects_dir = op.join(data_path, 'subjects')
fname_evoked = op.join(sample_dir, 'sample_audvis-ave.fif')
fname_trans = op.join(sample_dir, 'sample_audvis_trunc-trans.fif')
fname_inv_vol = op.join(sample_dir,
                        'sample_audvis_trunc-meg-vol-7-meg-inv.fif')
fname_fwd_vol = op.join(sample_dir,
                        'sample_audvis_trunc-meg-vol-7-fwd.fif')
fname_vol_w = op.join(sample_dir,
                      'sample_audvis_trunc-grad-vol-7-fwd-sensmap-vol.w')
fname_inv_surf = op.join(sample_dir,
                         'sample_audvis_trunc-meg-eeg-oct-6-meg-inv.fif')
fname_fmorph = op.join(data_path, 'MEG', 'sample',
                       'fsaverage_audvis_trunc-meg')
fname_smorph = op.join(sample_dir, 'sample_audvis_trunc-meg')
fname_t1 = op.join(subjects_dir, 'sample', 'mri', 'T1.mgz')
fname_vol = op.join(subjects_dir, 'sample', 'bem', 'sample-volume-7mm-src.fif')
fname_brain = op.join(subjects_dir, 'sample', 'mri', 'brain.mgz')
fname_aseg = op.join(subjects_dir, 'sample', 'mri', 'aseg.mgz')
fname_fs_vol = op.join(subjects_dir, 'fsaverage', 'bem',
                       'fsaverage-vol7-nointerp-src.fif.gz')
fname_aseg_fs = op.join(subjects_dir, 'fsaverage', 'mri', 'aseg.mgz')
fname_stc = op.join(sample_dir, 'fsaverage_audvis_trunc-meg')


def _real_vec_stc():
    inv = read_inverse_operator(fname_inv_surf)
    with pytest.warns(RuntimeWarning, match='Cropping removes baseline'):
        evoked = (read_evokeds(fname_evoked, baseline=(None, 0))[0]
                  .crop(0, 0.01))
    return apply_inverse(evoked, inv, pick_ori='vector')


def test_sourcemorph_consistency():
    """Test SourceMorph class consistency."""
    assert _get_args(SourceMorph.__init__)[1:] == \
        mne.morph._SOURCE_MORPH_ATTRIBUTES


@testing.requires_testing_data
def test_sparse_morph():
    """Test sparse morphing."""
    rng = np.random.RandomState(0)
    vertices_fs = [np.sort(rng.permutation(np.arange(10242))[:4]),
                   np.sort(rng.permutation(np.arange(10242))[:6])]
    data = rng.randn(10, 1)
    stc_fs = SourceEstimate(data, vertices_fs, 1, 1, 'fsaverage')
    spheres_fs = [mne.read_surface(op.join(
        subjects_dir, 'fsaverage', 'surf', '%s.sphere.reg' % hemi))[0]
        for hemi in ('lh', 'rh')]
    spheres_sample = [mne.read_surface(op.join(
        subjects_dir, 'sample', 'surf', '%s.sphere.reg' % hemi))[0]
        for hemi in ('lh', 'rh')]
    morph_fs_sample = compute_source_morph(
        stc_fs, 'fsaverage', 'sample', sparse=True, spacing=None,
        subjects_dir=subjects_dir)
    stc_sample = morph_fs_sample.apply(stc_fs)
    offset = 0
    orders = list()
    for v1, s1, v2, s2 in zip(stc_fs.vertices, spheres_fs,
                              stc_sample.vertices, spheres_sample):
        dists = cdist(s1[v1], s2[v2])
        order = np.argmin(dists, axis=-1)
        assert_array_less(dists[np.arange(len(order)), order], 1.5)  # mm
        orders.append(order + offset)
        offset += len(order)
    assert_allclose(stc_fs.data, stc_sample.data[np.concatenate(orders)])
    # Return
    morph_sample_fs = compute_source_morph(
        stc_sample, 'sample', 'fsaverage', sparse=True, spacing=None,
        subjects_dir=subjects_dir)
    stc_fs_return = morph_sample_fs.apply(stc_sample)
    offset = 0
    orders = list()
    for v1, s, v2 in zip(stc_fs.vertices, spheres_fs, stc_fs_return.vertices):
        dists = cdist(s[v1], s[v2])
        order = np.argmin(dists, axis=-1)
        assert_array_less(dists[np.arange(len(order)), order], 1.5)  # mm
        orders.append(order + offset)
        offset += len(order)
    assert_allclose(stc_fs.data, stc_fs_return.data[np.concatenate(orders)])


@testing.requires_testing_data
def test_xhemi_morph():
    """Test cross-hemisphere morphing."""
    stc = read_source_estimate(fname_stc, subject='sample')
    # smooth 1 for speed where possible
    smooth = 4
    spacing = 4
    n_grade_verts = 2562
    stc = compute_source_morph(
        stc, 'sample', 'fsaverage_sym', smooth=smooth, warn=False,
        spacing=spacing, subjects_dir=subjects_dir).apply(stc)
    morph = compute_source_morph(
        stc, 'fsaverage_sym', 'fsaverage_sym', smooth=1, xhemi=True,
        warn=False, spacing=[stc.vertices[0], []],
        subjects_dir=subjects_dir)
    stc_xhemi = morph.apply(stc)
    assert stc_xhemi.data.shape[0] == n_grade_verts
    assert stc_xhemi.rh_data.shape[0] == 0
    assert len(stc_xhemi.vertices[1]) == 0
    assert stc_xhemi.lh_data.shape[0] == n_grade_verts
    assert len(stc_xhemi.vertices[0]) == n_grade_verts
    # complete reversal mapping
    morph = compute_source_morph(
        stc, 'fsaverage_sym', 'fsaverage_sym', smooth=smooth, xhemi=True,
        warn=False, spacing=stc.vertices, subjects_dir=subjects_dir)
    mm = morph.morph_mat
    assert mm.shape == (n_grade_verts * 2,) * 2
    assert mm.size > n_grade_verts * 2
    assert mm[:n_grade_verts, :n_grade_verts].size == 0  # L to L
    assert mm[n_grade_verts:, n_grade_verts:].size == 0  # R to L
    assert mm[n_grade_verts:, :n_grade_verts].size > n_grade_verts  # L to R
    assert mm[:n_grade_verts, n_grade_verts:].size > n_grade_verts  # R to L
    # more complicated reversal mapping
    vertices_use = [stc.vertices[0], np.arange(10242)]
    n_src_verts = len(vertices_use[1])
    assert vertices_use[0].shape == (n_grade_verts,)
    assert vertices_use[1].shape == (n_src_verts,)
    # ensure it's sufficiently diffirent to manifest round-trip errors
    assert np.in1d(vertices_use[1], stc.vertices[1]).mean() < 0.3
    morph = compute_source_morph(
        stc, 'fsaverage_sym', 'fsaverage_sym', smooth=smooth, xhemi=True,
        warn=False, spacing=vertices_use, subjects_dir=subjects_dir)
    mm = morph.morph_mat
    assert mm.shape == (n_grade_verts + n_src_verts, n_grade_verts * 2)
    assert mm[:n_grade_verts, :n_grade_verts].size == 0
    assert mm[n_grade_verts:, n_grade_verts:].size == 0
    assert mm[:n_grade_verts, n_grade_verts:].size > n_grade_verts
    assert mm[n_grade_verts:, :n_grade_verts].size > n_src_verts
    # morph forward then back
    stc_xhemi = morph.apply(stc)
    morph = compute_source_morph(
        stc_xhemi, 'fsaverage_sym', 'fsaverage_sym', smooth=smooth,
        xhemi=True, warn=False, spacing=stc.vertices,
        subjects_dir=subjects_dir)
    stc_return = morph.apply(stc_xhemi)
    for hi in range(2):
        assert_array_equal(stc_return.vertices[hi], stc.vertices[hi])
    correlation = np.corrcoef(stc.data.ravel(), stc_return.data.ravel())[0, 1]
    assert correlation > 0.9  # not great b/c of sparse grade + small smooth


@testing.requires_testing_data
@pytest.mark.parametrize('smooth, lower, upper, n_warn, dtype', [
    (None, 0.959, 0.963, 0, float),
    (3, 0.968, 0.971, 2, complex),
    ('nearest', 0.98, 0.99, 0, float),
])
def test_surface_source_morph_round_trip(smooth, lower, upper, n_warn, dtype):
    """Test round-trip morphing yields similar STCs."""
    kwargs = dict(smooth=smooth, warn=True, subjects_dir=subjects_dir)
    stc = mne.read_source_estimate(fname_smorph)
    if dtype is complex:
        stc.data = 1j * stc.data
        assert_array_equal(stc.data.real, 0.)
    if smooth == 'nearest' and not check_version('scipy', '1.3'):
        with pytest.raises(ValueError, match='required to use nearest'):
            morph = compute_source_morph(stc, 'sample', 'fsaverage', **kwargs)
        return
    with pytest.warns(None) as w:
        morph = compute_source_morph(stc, 'sample', 'fsaverage', **kwargs)
    w = [ww for ww in w if 'vertices not included' in str(ww.message)]
    assert len(w) == n_warn
    assert morph.morph_mat.shape == (20484, len(stc.data))
    stc_fs = morph.apply(stc)
    morph_back = compute_source_morph(
        stc_fs, 'fsaverage', 'sample', spacing=stc.vertices, **kwargs)
    assert morph_back.morph_mat.shape == (len(stc.data), 20484)
    stc_back = morph_back.apply(stc_fs)
    corr = np.corrcoef(stc.data.ravel(), stc_back.data.ravel())[0, 1]
    assert lower <= corr <= upper
    # check the round-trip power
    assert_power_preserved(stc, stc_back)


def assert_power_preserved(orig, new, limits=(1., 1.05)):
    """Assert that the power is preserved during a round-trip morph."""
    __tracebackhide__ = True
    for kind in ('real', 'imag'):
        numer = np.linalg.norm(getattr(orig.data, kind))
        denom = np.linalg.norm(getattr(new.data, kind))
        if numer == denom == 0.:  # no data of this type
            continue
        power_ratio = numer / denom
        min_, max_ = limits
        assert min_ < power_ratio < max_, f'Power ratio {kind}'


@requires_h5py
@testing.requires_testing_data
def test_surface_vector_source_morph(tmpdir):
    """Test surface and vector source estimate morph."""
    inverse_operator_surf = read_inverse_operator(fname_inv_surf)

    stc_surf = read_source_estimate(fname_smorph, subject='sample')
    stc_surf.crop(0.09, 0.1)  # for faster computation

    stc_vec = _real_vec_stc()

    source_morph_surf = compute_source_morph(
        inverse_operator_surf['src'], subjects_dir=subjects_dir,
        smooth=1, warn=False)  # smooth 1 for speed
    assert source_morph_surf.subject_from == 'sample'
    assert source_morph_surf.subject_to == 'fsaverage'
    assert source_morph_surf.kind == 'surface'
    assert isinstance(source_morph_surf.src_data, dict)
    assert isinstance(source_morph_surf.src_data['vertices_from'], list)
    assert isinstance(source_morph_surf, SourceMorph)
    stc_surf_morphed = source_morph_surf.apply(stc_surf)
    assert isinstance(stc_surf_morphed, SourceEstimate)
    stc_vec_morphed = source_morph_surf.apply(stc_vec)
    with pytest.raises(ValueError, match="Invalid value for the 'output'"):
        source_morph_surf.apply(stc_surf, output='nifti1')

    # check if correct class after morphing
    assert isinstance(stc_surf_morphed, SourceEstimate)
    assert isinstance(stc_vec_morphed, VectorSourceEstimate)

    # check __repr__
    assert 'surface' in repr(source_morph_surf)

    # check loading and saving for surf
    source_morph_surf.save(tmpdir.join('42.h5'))

    source_morph_surf_r = read_source_morph(tmpdir.join('42.h5'))

    assert (all([read == saved for read, saved in
                 zip(sorted(source_morph_surf_r.__dict__),
                     sorted(source_morph_surf.__dict__))]))

    # check wrong subject correction
    stc_surf.subject = None
    assert isinstance(source_morph_surf.apply(stc_surf), SourceEstimate)

    # degenerate
    stc_vol = read_source_estimate(fname_vol_w, 'sample')
    with pytest.raises(TypeError, match='stc_from must be an instance'):
        source_morph_surf.apply(stc_vol)


@requires_h5py
@requires_nibabel()
@requires_dipy()
@pytest.mark.slowtest
@testing.requires_testing_data
def test_volume_source_morph(tmpdir):
    """Test volume source estimate morph, special cases and exceptions."""
    import nibabel as nib
    inverse_operator_vol = read_inverse_operator(fname_inv_vol)
    stc_vol = read_source_estimate(fname_vol_w, 'sample')

    # check for invalid input type
    with pytest.raises(TypeError, match='src must be'):
        compute_source_morph(src=42)

    # check for raising an error if neither
    # inverse_operator_vol['src'][0]['subject_his_id'] nor subject_from is set,
    # but attempting to perform a volume morph
    src = inverse_operator_vol['src']
    assert src._subject is None  # already None on disk (old!)

    with pytest.raises(ValueError, match='subject_from could not be inferred'):
        with pytest.warns(RuntimeWarning, match='recommend regenerating'):
            compute_source_morph(src=src, subjects_dir=subjects_dir)

    # check infer subject_from from src[0]['subject_his_id']
    src[0]['subject_his_id'] = 'sample'

    with pytest.raises(ValueError, match='Inter-hemispheric morphing'):
        compute_source_morph(src=src, subjects_dir=subjects_dir, xhemi=True)

    with pytest.raises(ValueError, match='Only surface.*sparse morph'):
        compute_source_morph(src=src, sparse=True, subjects_dir=subjects_dir)

    # terrible quality but fast
    zooms = 20
    kwargs = dict(zooms=zooms, niter_sdr=(1,), niter_affine=(1,))
    source_morph_vol = compute_source_morph(
        subjects_dir=subjects_dir, src=fname_inv_vol,
        subject_from='sample', **kwargs)
    shape = (13,) * 3  # for the given zooms

    assert source_morph_vol.subject_from == 'sample'

    # the brain used in sample data has shape (255, 255, 255)
    assert tuple(source_morph_vol.sdr_morph.domain_shape) == shape

    assert tuple(source_morph_vol.pre_affine.domain_shape) == shape

    # proofs the above
    assert_array_equal(source_morph_vol.zooms, (zooms,) * 3)

    # assure proper src shape
    mri_size = (src[0]['mri_height'], src[0]['mri_depth'], src[0]['mri_width'])
    assert source_morph_vol.src_data['src_shape_full'] == mri_size

    fwd = read_forward_solution(fname_fwd_vol)
    fwd['src'][0]['subject_his_id'] = 'sample'  # avoid further warnings
    source_morph_vol = compute_source_morph(
        fwd['src'], 'sample', 'sample', subjects_dir=subjects_dir,
        **kwargs)

    # check wrong subject_to
    with pytest.raises(IOError, match='cannot read file'):
        compute_source_morph(fwd['src'], 'sample', '42',
                             subjects_dir=subjects_dir)

    # two different ways of saving
    source_morph_vol.save(tmpdir.join('vol'))

    # check loading
    source_morph_vol_r = read_source_morph(tmpdir.join('vol-morph.h5'))

    # check for invalid file name handling ()
    with pytest.raises(IOError, match='not found'):
        read_source_morph(tmpdir.join('42'))

    # check morph
    stc_vol_morphed = source_morph_vol.apply(stc_vol)
    # old way, verts do not match
    assert not np.array_equal(stc_vol_morphed.vertices[0], stc_vol.vertices[0])

    # vector
    stc_vol_vec = VolVectorSourceEstimate(
        np.tile(stc_vol.data[:, np.newaxis], (1, 3, 1)),
        stc_vol.vertices, 0, 1)
    stc_vol_vec_morphed = source_morph_vol.apply(stc_vol_vec)
    assert isinstance(stc_vol_vec_morphed, VolVectorSourceEstimate)
    for ii in range(3):
        assert_allclose(stc_vol_vec_morphed.data[:, ii], stc_vol_morphed.data)

    # check output as NIfTI
    assert isinstance(source_morph_vol.apply(stc_vol_vec, output='nifti2'),
                      nib.Nifti2Image)

    # check for subject_from mismatch
    source_morph_vol_r.subject_from = '42'
    with pytest.raises(ValueError, match='subject_from must match'):
        source_morph_vol_r.apply(stc_vol_morphed)

    # check if nifti is in grid morph space with voxel_size == spacing
    img_morph_res = source_morph_vol.apply(stc_vol, output='nifti1')

    # assure morph spacing
    assert isinstance(img_morph_res, nib.Nifti1Image)
    assert img_morph_res.header.get_zooms()[:3] == (zooms,) * 3

    # assure src shape
    img_mri_res = source_morph_vol.apply(stc_vol, output='nifti1',
                                         mri_resolution=True)
    assert isinstance(img_mri_res, nib.Nifti1Image)
    assert (img_mri_res.shape == (src[0]['mri_height'], src[0]['mri_depth'],
                                  src[0]['mri_width']) +
            (img_mri_res.shape[3],))

    # check if nifti is defined resolution with voxel_size == (5., 5., 5.)
    img_any_res = source_morph_vol.apply(stc_vol, output='nifti1',
                                         mri_resolution=(5., 5., 5.))
    assert isinstance(img_any_res, nib.Nifti1Image)
    assert img_any_res.header.get_zooms()[:3] == (5., 5., 5.)

    # check if morph outputs correct data
    assert isinstance(stc_vol_morphed, VolSourceEstimate)

    # check if loaded and saved objects contain the same
    assert (all([read == saved for read, saved in
                 zip(sorted(source_morph_vol_r.__dict__),
                     sorted(source_morph_vol.__dict__))]))

    # check __repr__
    assert 'volume' in repr(source_morph_vol)

    # check Nifti2Image
    assert isinstance(
        source_morph_vol.apply(stc_vol, mri_resolution=True,
                               mri_space=True, output='nifti2'),
        nib.Nifti2Image)

    # Degenerate conditions
    with pytest.raises(TypeError, match='output must be'):
        source_morph_vol.apply(stc_vol, output=1)
    with pytest.raises(ValueError, match='subject_from does not match'):
        compute_source_morph(src=src, subject_from='42')
    with pytest.raises(ValueError, match='output'):
        source_morph_vol.apply(stc_vol, output='42')
    with pytest.raises(ValueError, match='subject_to cannot be None'):
        compute_source_morph(src, 'sample', None,
                             subjects_dir=subjects_dir)
    # Check if not morphed, but voxel size not boolean, raise ValueError.
    # Note that this check requires dipy to not raise the dipy ImportError
    # before checking if the actual voxel size error will raise.
    with pytest.raises(ValueError, match='Cannot infer original voxel size'):
        stc_vol.as_volume(inverse_operator_vol['src'], mri_resolution=4)

    stc_surf = read_source_estimate(fname_stc, 'sample')
    with pytest.raises(TypeError, match='stc_from must be an instance'):
        source_morph_vol.apply(stc_surf)

    # src_to
    # zooms=20 does not match src_to zooms (7)
    with pytest.raises(ValueError, match='If src_to is provided, zooms shoul'):
        source_morph_vol = compute_source_morph(
            fwd['src'], subject_from='sample', src_to=fwd['src'],
            subject_to='sample', subjects_dir=subjects_dir, **kwargs)
    # hack the src_to "zooms" to make it seem like a pos=20. source space
    fwd['src'][0]['src_mri_t']['trans'][:3, :3] = 0.02 * np.eye(3)
    source_morph_vol = compute_source_morph(
        fwd['src'], subject_from='sample', src_to=fwd['src'],
        subject_to='sample', subjects_dir=subjects_dir, **kwargs)
    stc_vol_2 = source_morph_vol.apply(stc_vol)
    # new way, verts match
    assert_array_equal(stc_vol.vertices[0], stc_vol_2.vertices[0])
    stc_vol_bad = VolSourceEstimate(
        stc_vol.data[:-1], [stc_vol.vertices[0][:-1]],
        stc_vol.tmin, stc_vol.tstep)
    match = (
        'vertices do not match between morph \\(4157\\) and stc \\(4156\\).*'
        '\n.*\n.*\n.*Vertices were likely excluded during forward computatio.*'
    )
    with pytest.raises(ValueError, match=match):
        source_morph_vol.apply(stc_vol_bad)


@requires_h5py
@requires_nibabel()
@requires_dipy()
@pytest.mark.slowtest
@testing.requires_testing_data
@pytest.mark.parametrize('subject_from, subject_to, lower, upper, dtype', [
    ('sample', 'fsaverage', 8.5, 9, float),
    ('fsaverage', 'fsaverage', 7, 7.5, float),
    ('sample', 'sample', 6, 7, complex),
])
def test_volume_source_morph_round_trip(
        tmpdir, subject_from, subject_to, lower, upper, dtype):
    """Test volume source estimate morph round-trips well."""
    import nibabel as nib
    from nibabel.processing import resample_from_to
    src = dict()
    if 'sample' in (subject_from, subject_to):
        src['sample'] = mne.read_source_spaces(fname_vol)
        src['sample'][0]['subject_his_id'] = 'sample'
        assert src['sample'][0]['nuse'] == 4157
    if 'fsaverage' in (subject_from, subject_to):
        # Created to save space with:
        #
        # bem = op.join(op.dirname(mne.__file__), 'data', 'fsaverage',
        #               'fsaverage-inner_skull-bem.fif')
        # src_fsaverage = mne.setup_volume_source_space(
        #     'fsaverage', pos=7., bem=bem, mindist=0,
        #     subjects_dir=subjects_dir, add_interpolator=False)
        # mne.write_source_spaces(fname_fs_vol, src_fsaverage, overwrite=True)
        #
        # For speed we do it without the interpolator because it's huge.
        src['fsaverage'] = mne.read_source_spaces(fname_fs_vol)
        src['fsaverage'][0].update(
            vol_dims=np.array([23, 29, 25]), seg_name='brain')
        _add_interpolator(src['fsaverage'], True)
        assert src['fsaverage'][0]['nuse'] == 6379
    src_to, src_from = src[subject_to], src[subject_from]
    del src
    # No SDR just for speed once everything works
    kwargs = dict(niter_sdr=(), niter_affine=(1,),
                  subjects_dir=subjects_dir, verbose=True)
    morph_from_to = compute_source_morph(
        src=src_from, src_to=src_to, subject_to=subject_to, **kwargs)
    morph_to_from = compute_source_morph(
        src=src_to, src_to=src_from, subject_to=subject_from, **kwargs)
    use = np.linspace(0, src_from[0]['nuse'] - 1, 10).round().astype(int)
    data = np.eye(src_from[0]['nuse'])[:, use]
    if dtype is complex:
        data = data * 1j
    stc_from = VolSourceEstimate(data, [src_from[0]['vertno']], 0, 1)
    stc_from_rt = morph_to_from.apply(morph_from_to.apply(stc_from))
    maxs = np.argmax(stc_from_rt.data, axis=0)
    src_rr = src_from[0]['rr'][src_from[0]['vertno']]
    dists = 1000 * np.linalg.norm(src_rr[use] - src_rr[maxs], axis=1)
    mu = np.mean(dists)
    assert lower <= mu < upper  # fsaverage=7.97; 25.4 without src_ras_t fix
    # check that pre_affine is close to identity when subject_to==subject_from
    if subject_to == subject_from:
        for morph in (morph_to_from, morph_from_to):
            assert_allclose(
                morph.pre_affine.affine, np.eye(4), atol=1e-2)
    # check that power is more or less preserved
    ratio = stc_from.data.size / stc_from_rt.data.size
    limits = ratio * np.array([1, 1.2])
    stc_from.crop(0, 0)._data.fill(1.)
    stc_from_rt = morph_to_from.apply(morph_from_to.apply(stc_from))
    assert_power_preserved(stc_from, stc_from_rt, limits=limits)
    # before and after morph, check the proportion of vertices
    # that are inside and outside the brainmask.mgz
    brain = nib.load(op.join(subjects_dir, subject_from, 'mri', 'brain.mgz'))
    mask = _get_img_fdata(brain) > 0
    if subject_from == subject_to == 'sample':
        for stc in [stc_from, stc_from_rt]:
            img = stc.as_volume(src_from, mri_resolution=True)
            img = nib.Nifti1Image(  # abs to convert complex
                np.abs(_get_img_fdata(img)[:, :, :, 0]), img.affine)
            img = _get_img_fdata(resample_from_to(img, brain, order=1))
            assert img.shape == mask.shape
            in_ = img[mask].astype(bool).mean()
            out = img[~mask].astype(bool).mean()
            assert 0.97 < in_ < 0.98
            assert out < 0.02


@pytest.mark.slowtest
@testing.requires_testing_data
def test_morph_stc_dense():
    """Test morphing stc."""
    subject_from = 'sample'
    subject_to = 'fsaverage'
    stc_from = read_source_estimate(fname_smorph, subject='sample')
    stc_to = read_source_estimate(fname_fmorph)
    # make sure we can specify grade
    stc_from.crop(0.09, 0.1)  # for faster computation
    stc_to.crop(0.09, 0.1)  # for faster computation
    assert_array_equal(stc_to.time_as_index([0.09, 0.1], use_rounding=True),
                       [0, len(stc_to.times) - 1])

    # After dep change this to:
    stc_to1 = compute_source_morph(
        subject_to=subject_to, spacing=3, smooth=12, src=stc_from,
        subjects_dir=subjects_dir).apply(stc_from)
    assert_allclose(stc_to.data, stc_to1.data, atol=1e-5)

    mean_from = stc_from.data.mean(axis=0)
    mean_to = stc_to1.data.mean(axis=0)
    assert np.corrcoef(mean_to, mean_from).min() > 0.999

    vertices_to = grade_to_vertices(subject_to, grade=3,
                                    subjects_dir=subjects_dir)

    # make sure we can fill by morphing
    with pytest.warns(RuntimeWarning, match='consider increasing'):
        morph = compute_source_morph(
            stc_from, subject_from, subject_to, spacing=None, smooth=1,
            subjects_dir=subjects_dir)
    stc_to5 = morph.apply(stc_from)
    assert stc_to5.data.shape[0] == 163842 + 163842

    # Morph vector data
    stc_vec = _real_vec_stc()
    stc_vec_to1 = compute_source_morph(
        stc_vec, subject_from, subject_to, subjects_dir=subjects_dir,
        spacing=vertices_to, smooth=1, warn=False).apply(stc_vec)
    assert stc_vec_to1.subject == subject_to
    assert stc_vec_to1.tmin == stc_vec.tmin
    assert stc_vec_to1.tstep == stc_vec.tstep
    assert len(stc_vec_to1.lh_vertno) == 642
    assert len(stc_vec_to1.rh_vertno) == 642

    # Degenerate conditions

    # Morphing to a density that is too high should raise an informative error
    # (here we need to push to grade=6, but for some subjects even grade=5
    # will break)
    with pytest.raises(ValueError, match='Cannot use icosahedral grade 6 '):
        compute_source_morph(
            stc_to1, subject_from=subject_to, subject_to=subject_from,
            spacing=6, subjects_dir=subjects_dir)
    del stc_to1

    with pytest.raises(ValueError, match='smooth.* has to be at least 1'):
        compute_source_morph(
            stc_from, subject_from, subject_to, spacing=5, smooth=-1,
            subjects_dir=subjects_dir)

    # subject from mismatch
    with pytest.raises(ValueError, match="subject_from does not match"):
        compute_source_morph(stc_from, subject_from='foo',
                             subjects_dir=subjects_dir)

    # only one set of vertices
    with pytest.raises(ValueError, match="grade.*list must have two elements"):
        compute_source_morph(
            stc_from, subject_from=subject_from, spacing=[vertices_to[0]],
            subjects_dir=subjects_dir)


@testing.requires_testing_data
def test_morph_stc_sparse():
    """Test morphing stc with sparse=True."""
    subject_from = 'sample'
    subject_to = 'fsaverage'
    # Morph sparse data
    # Make a sparse stc
    stc_from = read_source_estimate(fname_smorph, subject='sample')
    stc_from.vertices[0] = stc_from.vertices[0][[100, 500]]
    stc_from.vertices[1] = stc_from.vertices[1][[200]]
    stc_from._data = stc_from._data[:3]

    stc_to_sparse = compute_source_morph(
        stc_from, subject_from=subject_from, subject_to=subject_to,
        spacing=None, sparse=True, subjects_dir=subjects_dir).apply(stc_from)

    assert_allclose(np.sort(stc_from.data.sum(axis=1)),
                    np.sort(stc_to_sparse.data.sum(axis=1)))
    assert len(stc_from.rh_vertno) == len(stc_to_sparse.rh_vertno)
    assert len(stc_from.lh_vertno) == len(stc_to_sparse.lh_vertno)
    assert stc_to_sparse.subject == subject_to
    assert stc_from.tmin == stc_from.tmin
    assert stc_from.tstep == stc_from.tstep

    stc_from.vertices[0] = np.array([], dtype=np.int64)
    stc_from._data = stc_from._data[:1]

    stc_to_sparse = compute_source_morph(
        stc_from, subject_from, subject_to, spacing=None, sparse=True,
        subjects_dir=subjects_dir).apply(stc_from)

    assert_allclose(np.sort(stc_from.data.sum(axis=1)),
                    np.sort(stc_to_sparse.data.sum(axis=1)))
    assert len(stc_from.rh_vertno) == len(stc_to_sparse.rh_vertno)
    assert len(stc_from.lh_vertno) == len(stc_to_sparse.lh_vertno)
    assert stc_to_sparse.subject == subject_to
    assert stc_from.tmin == stc_from.tmin
    assert stc_from.tstep == stc_from.tstep

    # Degenerate cases
    with pytest.raises(ValueError, match='spacing must be set to None'):
        compute_source_morph(
            stc_from, subject_from=subject_from, subject_to=subject_to,
            spacing=5, sparse=True, subjects_dir=subjects_dir)
    with pytest.raises(ValueError, match='xhemi=True can only be used with'):
        compute_source_morph(
            stc_from, subject_from=subject_from, subject_to=subject_to,
            spacing=None, sparse=True, xhemi=True, subjects_dir=subjects_dir)


@requires_nibabel()
@testing.requires_testing_data
@pytest.mark.parametrize('sl, n_real, n_mri, n_orig', [
    # First and last should add up, middle can have overlap should be <= sum
    (slice(0, 1), 37, 123, 8),
    (slice(1, 2), 51, 225, 12),
    (slice(0, 2), 88, 330, 20),
])
def test_volume_labels_morph(tmpdir, sl, n_real, n_mri, n_orig):
    """Test generating a source space from volume label."""
    import nibabel as nib
    n_use = (sl.stop - sl.start) // (sl.step or 1)
    # see gh-5224
    evoked = mne.read_evokeds(fname_evoked)[0].crop(0, 0)
    evoked.pick_channels(evoked.ch_names[:306:8])
    evoked.info.normalize_proj()
    n_ch = len(evoked.ch_names)
    aseg_fname = op.join(subjects_dir, 'sample', 'mri', 'aseg.mgz')
    lut, _ = read_freesurfer_lut()
    label_names = sorted(get_volume_labels_from_aseg(aseg_fname))
    use_label_names = label_names[sl]
    src = setup_volume_source_space(
        'sample', subjects_dir=subjects_dir, volume_label=use_label_names,
        mri=aseg_fname)
    assert len(src) == n_use
    assert src.kind == 'volume'
    n_src = sum(s['nuse'] for s in src)
    sphere = make_sphere_model('auto', 'auto', evoked.info)
    fwd = make_forward_solution(evoked.info, fname_trans, src, sphere)
    assert fwd['sol']['data'].shape == (n_ch, n_src * 3)
    inv = make_inverse_operator(evoked.info, fwd, make_ad_hoc_cov(evoked.info),
                                loose=1.)
    stc = apply_inverse(evoked, inv)
    assert stc.data.shape == (n_src, 1)
    img = stc.as_volume(src, mri_resolution=True)
    assert img.shape == (86, 86, 86, 1)
    n_on = np.array(img.dataobj).astype(bool).sum()
    aseg_img = _get_img_fdata(nib.load(fname_aseg))
    n_got_real = np.in1d(
        aseg_img.ravel(), [lut[name] for name in use_label_names]).sum()
    assert n_got_real == n_real
    # - This was 291 on `master` before gh-5590
    # - Refactoring transforms it became 279 with a < 1e-8 change in vox_mri_t
    # - Dropped to 123 once nearest-voxel was used in gh-7653
    # - Jumped back up to 330 with morphing fixes actually correctly
    #   interpolating across all volumes
    assert aseg_img.shape == img.shape[:3]
    assert n_on == n_mri
    for ii in range(2):
        # should work with (ii=0) or without (ii=1) the interpolator
        if ii:
            src[0]['interpolator'] = None
        img = stc.as_volume(src, mri_resolution=False)
        n_on = np.array(img.dataobj).astype(bool).sum()
        # was 20 on `master` before gh-5590
        # then 44 before gh-7653, which took it back to 20
        assert n_on == n_orig
    # without the interpolator, this should fail
    assert src[0]['interpolator'] is None
    with pytest.raises(RuntimeError, match=r'.*src\[0\], .* mri_resolution'):
        stc.as_volume(src, mri_resolution=True)


@pytest.fixture(scope='session', params=[testing._pytest_param()])
def _mixed_morph_srcs():
    # create a mixed source space
    labels_vol = ['Left-Cerebellum-Cortex', 'Right-Cerebellum-Cortex']
    src = mne.setup_source_space('sample', spacing='oct3',
                                 add_dist=False, subjects_dir=subjects_dir)
    src += mne.setup_volume_source_space(
        'sample', mri=fname_aseg, pos=10.0,
        volume_label=labels_vol, subjects_dir=subjects_dir,
        add_interpolator=True, verbose=True)
    # create the destination space
    src_fs = mne.read_source_spaces(
        op.join(subjects_dir, 'fsaverage', 'bem', 'fsaverage-ico-5-src.fif'))
    src_fs += mne.setup_volume_source_space(
        'fsaverage', pos=7., volume_label=labels_vol,
        subjects_dir=subjects_dir, add_interpolator=False, verbose=True)
    del labels_vol

    with pytest.raises(ValueError, match='src_to must be provided .* mixed'):
        mne.compute_source_morph(
            src=src, subject_from='sample', subject_to='fsaverage',
            subjects_dir=subjects_dir)

    with pytest.warns(RuntimeWarning, match='not included in smoothing'):
        morph = mne.compute_source_morph(
            src=src, subject_from='sample', subject_to='fsaverage',
            subjects_dir=subjects_dir, niter_affine=[1, 0, 0],
            niter_sdr=[1, 0, 0], src_to=src_fs, smooth=5, verbose=True)
    return morph, src, src_fs


@requires_nibabel()
@requires_dipy()
@pytest.mark.parametrize('vector', (False, True))
def test_mixed_source_morph(_mixed_morph_srcs, vector):
    """Test mixed source space morphing."""
    import nibabel as nib
    morph, src, src_fs = _mixed_morph_srcs
    # Test some basic properties in the subject's own space
    lut, _ = read_freesurfer_lut()
    ids = [lut[s['seg_name']] for s in src[2:]]
    del lut
    vertices = [s['vertno'] for s in src]
    n_vertices = sum(len(v) for v in vertices)
    data = np.zeros((n_vertices, 3, 1))
    data[:, 1] = 1.
    klass = mne.MixedVectorSourceEstimate
    if not vector:
        data = data[:, 1]
        klass = klass._scalar_class
    stc = klass(data, vertices, 0, 1, 'sample')
    vol_info = _get_mri_info_data(fname_aseg, data=True)
    rrs = np.concatenate([src[2]['rr'][sp['vertno']] for sp in src[2:]])
    n_want = np.in1d(_get_atlas_values(vol_info, rrs), ids).sum()
    img = _get_img_fdata(stc.volume().as_volume(src, mri_resolution=False))
    assert img.astype(bool).sum() == n_want
    img_res = nib.load(fname_aseg)
    n_want = np.in1d(_get_img_fdata(img_res), ids).sum()
    img = _get_img_fdata(stc.volume().as_volume(src, mri_resolution=True))
    assert img.astype(bool).sum() > n_want  # way more get interpolated into

    with pytest.raises(TypeError, match='stc_from must be an instance'):
        morph.apply(1.)

    # Now actually morph
    stc_fs = morph.apply(stc)
    img = stc_fs.volume().as_volume(src_fs, mri_resolution=False)
    vol_info = _get_mri_info_data(fname_aseg_fs, data=True)
    rrs = np.concatenate([src_fs[2]['rr'][sp['vertno']] for sp in src_fs[2:]])
    n_want = np.in1d(_get_atlas_values(vol_info, rrs), ids).sum()
    with pytest.raises(ValueError, match=r'stc\.subject does not match src s'):
        stc_fs.volume().as_volume(src, mri_resolution=False)
    img = _get_img_fdata(
        stc_fs.volume().as_volume(src_fs, mri_resolution=False))
    assert img.astype(bool).sum() == n_want  # correct number of voxels

    # Morph separate parts and compare to morphing the entire one
    stc_fs_surf = morph.apply(stc.surface())
    stc_fs_vol = morph.apply(stc.volume())
    stc_fs_2 = stc_fs.__class__(
        np.concatenate([stc_fs_surf.data, stc_fs_vol.data]),
        stc_fs_surf.vertices + stc_fs_vol.vertices, stc_fs.tmin, stc_fs.tstep,
        stc_fs.subject)
    assert_allclose(stc_fs.data, stc_fs_2.data)


run_tests_if_main()
