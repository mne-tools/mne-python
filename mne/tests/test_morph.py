# -*- coding: utf-8 -*-
# Author: Tommy Clausner <Tommy.Clausner@gmail.com>
#
# License: BSD (3-clause)
import os.path as op
import warnings

import pytest
import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_allclose)
from mne import (SourceEstimate, VolSourceEstimate, VectorSourceEstimate,
                 read_evokeds, SourceMorph, read_source_morph,
                 compute_morph_matrix, morph_data, read_source_estimate,
                 grade_to_vertices, read_source_spaces)
from mne.datasets import sample, testing
from mne.source_space import SourceSpaces
from mne.minimum_norm import apply_inverse, read_inverse_operator
from mne.utils import (run_tests_if_main, requires_nibabel, _TempDir,
                       string_types)

# Setup paths
tempdir = _TempDir()

data_path = testing.data_path(download=False)

sample_dir = op.join(data_path, 'MEG', 'sample')

subjects_dir = op.join(data_path, 'subjects')

fname_evoked = op.join(sample_dir, 'sample_audvis-ave.fif')

fname_fwd_vol = op.join(sample_dir,
                        'sample_audvis_trunc-meg-vol-7-fwd.fif')
fname_inv_vol = op.join(sample_dir,
                        'sample_audvis_trunc-meg-vol-7-meg-inv.fif')
fname_vol = op.join(sample_dir,
                    'sample_audvis_trunc-grad-vol-7-fwd-sensmap-vol.w')
fname_inv_surf = op.join(sample_dir,
                         'sample_audvis_trunc-meg-eeg-oct-6-meg-inv.fif')

fname_smorph = op.join(sample_dir, 'sample_audvis_trunc-meg')
fname_fmorph = op.join(sample_dir, 'fsaverage_audvis_trunc-meg')

fname_t1 = op.join(subjects_dir, 'sample', 'mri', 'T1.mgz')


def _real_vec_stc():
    inv = read_inverse_operator(fname_inv_surf)
    evoked = read_evokeds(fname_evoked, baseline=(None, 0))[0].crop(0, 0.01)
    return apply_inverse(evoked, inv, pick_ori='vector')


@requires_nibabel()
@testing.requires_testing_data
def test_save_vol_stc_as_nifti():
    """Save the stc as a nifti file and export."""
    import nibabel as nib
    tempdir = _TempDir()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        src = read_source_spaces(fname_fwd_vol)
    vol_fname = op.join(tempdir, 'stc.nii.gz')

    # now let's actually read a MNE-C processed file
    stc = read_source_estimate(fname_vol, 'sample')
    assert (isinstance(stc, VolSourceEstimate))

    stc.save_as_volume(vol_fname, src,
                       dest='surf', mri_resolution=False)
    with warnings.catch_warnings(record=True):  # nib<->numpy
        img = nib.load(vol_fname)
    assert (img.shape == src[0]['shape'] + (len(stc.times),))

    with warnings.catch_warnings(record=True):  # nib<->numpy
        t1_img = nib.load(fname_t1)
    stc.save_as_volume(op.join(tempdir, 'stc.nii.gz'), src,
                       dest='mri', mri_resolution=True)
    with warnings.catch_warnings(record=True):  # nib<->numpy
        img = nib.load(vol_fname)
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

    pytest.raises(ValueError, stc.as_volume, src, mri_resolution=(4., 4., 4.))


@pytest.mark.slowtest
@testing.requires_testing_data
def test_morph_data():
    """Test morphing of data."""
    subject_from = 'sample'
    subject_to = 'fsaverage'

    stc_from = read_source_estimate(fname_smorph, subject='sample')
    stc_to = read_source_estimate(fname_fmorph)

    # make sure we can specify grade
    stc_from.crop(0.09, 0.1)  # for faster computation
    stc_to.crop(0.09, 0.1)  # for faster computation
    assert_array_equal(stc_to.time_as_index([0.09, 0.1], use_rounding=True),
                       [0, len(stc_to.times) - 1])
    with warnings.catch_warnings(record=True):
        pytest.raises(ValueError, stc_from.morph, subject_to, grade=5,
                      smooth=-1,
                      subjects_dir=subjects_dir)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        stc_to1 = stc_from.morph(subject_to, grade=3, smooth=12,
                                 subjects_dir=subjects_dir)

    stc_to1.save(op.join(tempdir, '%s_audvis-meg' % subject_to))
    # Morphing to a density that is too high should raise an informative error
    # (here we need to push to grade=6, but for some subjects even grade=5
    # will break)
    with warnings.catch_warnings(record=True):
        pytest.raises(ValueError, stc_to1.morph, subject_from, grade=6,
                      subjects_dir=subjects_dir)
    # make sure we can specify vertices
    vertices_to = grade_to_vertices(subject_to, grade=3,
                                    subjects_dir=subjects_dir)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        stc_to2 = morph_data(subject_from, subject_to, stc_from,
                             grade=vertices_to, smooth=12,
                             subjects_dir=subjects_dir)

    # make sure we get a warning about # of steps
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        morph_data(subject_from, subject_to, stc_from,
                   grade=vertices_to, smooth=1,
                   subjects_dir=subjects_dir)
    assert sum('consider increasing' in str(ww.message) for ww in w) == 2

    assert_array_almost_equal(stc_to.data, stc_to1.data, 5)
    assert_array_almost_equal(stc_to1.data, stc_to2.data)
    # make sure precomputed morph matrices work
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        morph_mat = compute_morph_matrix(subject_from, subject_to,
                                         stc_from.vertices, vertices_to,
                                         smooth=12, subjects_dir=subjects_dir)
    # compute_morph_matrix is deprecated
    assert sum('deprecated' in str(ww.message) for ww in w) == 1

    with warnings.catch_warnings(record=True):
        stc_to2 = stc_from.morph_precomputed(subject_to, vertices_to,
                                             morph_mat)
    assert_array_almost_equal(stc_to1.data, stc_to2.data)

    with warnings.catch_warnings(record=True):
        pytest.raises(ValueError, stc_from.morph_precomputed,
                      subject_to, vertices_to, 'foo')
        pytest.raises(ValueError, stc_from.morph_precomputed,
                      subject_to, [vertices_to[0]], morph_mat)
        pytest.raises(ValueError, stc_from.morph_precomputed,
                      subject_to, [vertices_to[0][:-1], vertices_to[1]],
                      morph_mat)
        pytest.raises(ValueError, stc_from.morph_precomputed, subject_to,
                      vertices_to, morph_mat, subject_from='foo')

    # steps warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        compute_morph_matrix(subject_from, subject_to,
                             stc_from.vertices, vertices_to,
                             smooth=1, subjects_dir=subjects_dir)

    mean_from = stc_from.data.mean(axis=0)
    mean_to = stc_to1.data.mean(axis=0)
    assert (np.corrcoef(mean_to, mean_from).min() > 0.999)

    # make sure we can fill by morphing
    with warnings.catch_warnings(record=True):
        stc_to5 = morph_data(subject_from, subject_to, stc_from, grade=None,
                             smooth=12, subjects_dir=subjects_dir)
    assert (stc_to5.data.shape[0] == 10242 + 10242)

    # Morph sparse data
    # Make a sparse stc
    stc_from.vertices[0] = stc_from.vertices[0][[100, 500]]
    stc_from.vertices[1] = stc_from.vertices[1][[200]]
    stc_from._data = stc_from._data[:3]

    with warnings.catch_warnings(record=True):
        pytest.raises(RuntimeError, stc_from.morph, subject_to, sparse=True,
                      grade=5, subjects_dir=subjects_dir)

    with warnings.catch_warnings(record=True):
        stc_to_sparse = stc_from.morph(subject_to, grade=None, sparse=True,
                                       subjects_dir=subjects_dir)
    assert_array_almost_equal(np.sort(stc_from.data.sum(axis=1)),
                              np.sort(stc_to_sparse.data.sum(axis=1)))
    assert len(stc_from.rh_vertno) == len(stc_to_sparse.rh_vertno)
    assert len(stc_from.lh_vertno) == len(stc_to_sparse.lh_vertno)
    assert stc_to_sparse.subject == subject_to
    assert stc_from.tmin == stc_from.tmin
    assert stc_from.tstep == stc_from.tstep

    stc_from.vertices[0] = np.array([], dtype=np.int64)
    stc_from._data = stc_from._data[:1]

    with warnings.catch_warnings(record=True):
        stc_to_sparse = stc_from.morph(subject_to, grade=None, sparse=True,
                                       subjects_dir=subjects_dir)
    assert_array_almost_equal(np.sort(stc_from.data.sum(axis=1)),
                              np.sort(stc_to_sparse.data.sum(axis=1)))

    assert len(stc_from.rh_vertno) == len(stc_to_sparse.rh_vertno)
    assert len(stc_from.lh_vertno) == len(stc_to_sparse.lh_vertno)
    assert stc_to_sparse.subject == subject_to
    assert stc_from.tmin == stc_from.tmin
    assert stc_from.tstep == stc_from.tstep

    # Morph vector data
    stc_vec = _real_vec_stc()

    # Ignore warnings about number of steps
    with warnings.catch_warnings(record=True) as w:
        stc_vec_to1 = stc_vec.morph(subject_to, grade=3, smooth=12,
                                    subjects_dir=subjects_dir)
        stc_vec_to2 = stc_vec.morph_precomputed(subject_to, vertices_to,
                                                morph_mat)
    assert_array_almost_equal(stc_vec_to1.data, stc_vec_to2.data)


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
    with warnings.catch_warnings(record=True):  # nib<->numpy
        t1_img = nib.load(fname_t1)
    # always assure nifti and dimensionality
    assert isinstance(img, nib.Nifti1Image)
    assert img.header.get_zooms()[:3] == t1_img.header.get_zooms()[:3]

    img = stc_vol.as_volume(inverse_operator_vol['src'], mri_resolution=False)

    assert isinstance(img, nib.Nifti1Image)
    assert img.shape[:3] == inverse_operator_vol['src'][0]['shape'][:3]

    # Check if not morphed, but voxel size not boolean, raise ValueError
    pytest.raises(ValueError, stc_vol.as_volume, inverse_operator_vol['src'],
                  mri_resolution=(4., 4., 4.))


@testing.requires_testing_data
def test_surface_vector_source_morph():
    """Test surface and vector source estimate morph."""
    inverse_operator_surf = read_inverse_operator(fname_inv_surf)

    stc_surf = read_source_estimate(fname_smorph, subject='sample')
    stc_surf.crop(0.09, 0.1)  # for faster computation

    stc_vec = _real_vec_stc()

    source_morph_surf = SourceMorph(inverse_operator_surf['src'],
                                    subjects_dir=subjects_dir)

    assert isinstance(source_morph_surf, SourceMorph)

    stc_surf_morphed = source_morph_surf(stc_surf)
    stc_vec_morphed = source_morph_surf(stc_vec)

    pytest.raises(ValueError, source_morph_surf.as_volume, stc_surf_morphed)

    # check if correct class after morphing
    assert isinstance(stc_surf_morphed, SourceEstimate)
    assert isinstance(stc_vec_morphed, VectorSourceEstimate)

    # check __repr__
    assert isinstance(source_morph_surf.__repr__(), string_types)

    # check laoding and saving for surf
    source_morph_surf.save(op.join(tempdir, '42.h5'))

    source_morph_surf_r = read_source_morph(op.join(tempdir, '42.h5'))

    assert (all([read == saved for read, saved in
                 zip(sorted(source_morph_surf_r.__dict__),
                     sorted(source_morph_surf.__dict__))]))


@requires_nibabel()
@pytest.mark.slowtest
@sample.requires_sample_data
def test_volume_source_morph():
    """Test volume source estimate morph, special cases and exceptions."""
    import nibabel as nib
    data_path = sample.data_path(
        download=False)  # because testing data has no brain.mgz
    subjects_dir = op.join(data_path, 'subjects')
    sample_dir = op.join(data_path, 'MEG', 'sample')
    fname_inv_vol = op.join(sample_dir,
                            'sample_audvis-meg-vol-7-meg-inv.fif')

    inverse_operator_vol = read_inverse_operator(fname_inv_vol)

    stc_vol = read_source_estimate(fname_vol, 'sample')

    # check for invalid input type
    pytest.raises(ValueError, SourceMorph, 42)

    # check for raising an error if neither
    # inverse_operator_vol['src'][0]['subject_his_id'] nor subject_from is set,
    # but attempting to perform a volume morph
    src = inverse_operator_vol['src']
    src[0]['subject_his_id'] = None
    pytest.raises(ValueError, SourceMorph, src, subjects_dir=subjects_dir)

    # check path to src provided, but invalid
    pytest.raises(IOError, SourceMorph, '42', subjects_dir=subjects_dir)

    # check infer subject_from from src[0]['subject_his_id']
    src[0]['subject_his_id'] = 'sample'
    source_morph_vol = SourceMorph(inverse_operator_vol['src'],
                                   subjects_dir=subjects_dir,
                                   niter_affine=(10, 10, 10),
                                   niter_sdr=(3, 3, 3), spacing=7)

    # the brain used in sample data has shape (255, 255, 255)
    assert (tuple(source_morph_vol.data['DiffeomorphicMap']['domain_shape']) ==
            (37, 37, 37))

    assert (tuple(source_morph_vol.data['AffineMap']['domain_shape']) ==
            (37, 37, 37))

    # proofs the above
    assert source_morph_vol.spacing == 7

    # assure proper src shape
    assert source_morph_vol.data['src_shape_full'] == (
        src[0]['mri_height'], src[0]['mri_depth'], src[0]['mri_width'])

    # check full mri resolution registration (takes very long)
    # source_morph_vol = SourceMorph(
    #     op.join(sample_dir, 'sample_audvis-meg-vol-7-fwd.fif'),
    #     subject_from='sample', subjects_dir=op.join(data_path, 'subjects'),
    #     grid_spacing=None)

    # check input via path to src and path to subject_to
    source_morph_vol = SourceMorph(
        op.join(sample_dir, 'sample_audvis-meg-vol-7-fwd.fif'),
        subject_from='sample',
        subject_to=op.join(data_path, 'subjects', 'fsaverage', 'mri',
                           'brain.mgz'),
        subjects_dir=subjects_dir, niter_affine=(10, 10, 10),
        niter_sdr=(3, 3, 3), spacing=7)

    # check wrong subject_to
    pytest.raises(IOError, SourceMorph,
                  op.join(sample_dir, 'sample_audvis-meg-vol-7-fwd.fif'),
                  subject_from='sample', subject_to='42',
                  subjects_dir=subjects_dir)

    # two different ways of saving
    source_morph_vol.save(op.join(tempdir, 'vol'))
    source_morph_vol.save(op.join(tempdir, 'vol.h5'))

    # check loading
    source_morph_vol_r = read_source_morph(op.join(tempdir, 'vol.h5'))

    # check for invalid file name handling
    pytest.raises(IOError, read_source_morph, op.join(tempdir, '42'))

    # check morph
    stc_vol_morphed = source_morph_vol(stc_vol)

    # check for subject_from mismatch
    source_morph_vol_r.subject_from = '42'
    pytest.raises(ValueError, source_morph_vol_r, stc_vol_morphed)

    # check if nifti is in grid morph space with voxel_size == spacing
    img_morph_res = source_morph_vol.as_volume(stc_vol_morphed,
                                               mri_resolution=False)

    # assure morph spacing
    assert isinstance(img_morph_res, nib.Nifti1Image)
    assert img_morph_res.header.get_zooms()[:3] == (7., 7., 7.)

    # assure src shape
    img_mri_res = source_morph_vol.as_volume(stc_vol_morphed,
                                             mri_resolution=True)
    assert isinstance(img_mri_res, nib.Nifti1Image)
    assert (img_mri_res.shape == (src[0]['mri_height'], src[0]['mri_depth'],
                                  src[0]['mri_width']) +
            (img_mri_res.shape[3],))

    # check if nifti is defined resolution with voxel_size == (5., 5., 5.)
    img_any_res = source_morph_vol.as_volume(stc_vol_morphed,
                                             mri_resolution=(5., 5., 5.),
                                             fname=op.join(tempdir, '42'))
    assert isinstance(img_any_res, nib.Nifti1Image)
    assert img_any_res.header.get_zooms()[:3] == (5., 5., 5.)

    # check if morph outputs correct data
    assert isinstance(stc_vol_morphed, VolSourceEstimate)

    # check if loaded and saved objects contain the same
    assert (all([read == saved for read, saved in
                 zip(sorted(source_morph_vol_r.__dict__),
                     sorted(source_morph_vol.__dict__))]))

    # check __repr__
    assert isinstance(source_morph_vol.__repr__(), string_types)
    assert isinstance(SourceMorph(None).__repr__(), string_types)


run_tests_if_main()
