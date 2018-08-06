# -*- coding: utf-8 -*-
# Author: Tommy Clausner <Tommy.Clausner@gmail.com>
#
# License: BSD (3-clause)
import os
import warnings

import pytest
import numpy as np
from mne import (SourceEstimate, VolSourceEstimate, VectorSourceEstimate,
                 read_evokeds, SourceMorph, read_source_morph,
                 read_source_estimate, read_forward_solution)
from mne.datasets import sample, testing
from mne.minimum_norm import apply_inverse, read_inverse_operator
from mne.utils import (run_tests_if_main, requires_nibabel, _TempDir,
                       requires_dipy, string_types)

# Setup paths
tempdir = _TempDir()

data_path = testing.data_path(download=False)

sample_dir = os.path.join(data_path, 'MEG', 'sample')

subjects_dir = os.path.join(data_path, 'subjects')

fname_evoked = os.path.join(sample_dir, 'sample_audvis-ave.fif')

fname_inv_vol = os.path.join(sample_dir,
                             'sample_audvis_trunc-meg-vol-7-meg-inv.fif')
fname_vol = os.path.join(sample_dir,
                         'sample_audvis_trunc-grad-vol-7-fwd-sensmap-vol.w')
fname_inv_surf = os.path.join(sample_dir,
                              'sample_audvis_trunc-meg-eeg-oct-6-meg-inv.fif')

fname_smorph = os.path.join(sample_dir, 'sample_audvis_trunc-meg')

fname_t1 = os.path.join(subjects_dir, 'sample', 'mri', 'T1.mgz')


def _real_vec_stc():
    inv = read_inverse_operator(fname_inv_surf)
    evoked = read_evokeds(fname_evoked, baseline=(None, 0))[0].crop(0, 0.01)
    return apply_inverse(evoked, inv, pick_ori='vector')


@requires_nibabel()
@requires_dipy()
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
    with pytest.raises(ValueError, match='Cannot infer original voxel size.*'):
        stc_vol.as_volume(inverse_operator_vol['src'], mri_resolution=4)


@requires_nibabel()
@requires_dipy()
@testing.requires_testing_data
def test_surface_vector_source_morph():
    """Test surface and vector source estimate morph."""
    inverse_operator_surf = read_inverse_operator(fname_inv_surf)

    stc_surf = read_source_estimate(fname_smorph, subject='sample')
    stc_surf.crop(0.09, 0.1)  # for faster computation

    stc_vec = _real_vec_stc()

    source_morph_surf = SourceMorph(subjects_dir=subjects_dir,
                                    spacing=[np.arange(10242)] * 2,
                                    src=inverse_operator_surf['src'])

    assert isinstance(source_morph_surf, SourceMorph)

    assert isinstance(SourceMorph(subjects_dir=subjects_dir)(stc_surf),
                      SourceEstimate)

    stc_surf_morphed = source_morph_surf(stc_surf)
    stc_vec_morphed = source_morph_surf(stc_vec)
    with pytest.raises(ValueError, match='Only volume source estimates.*'):
        source_morph_surf.as_volume(stc_surf_morphed)

    # check if correct class after morphing
    assert isinstance(stc_surf_morphed, SourceEstimate)
    assert isinstance(stc_vec_morphed, VectorSourceEstimate)

    # check __repr__
    assert isinstance(source_morph_surf.__repr__(), string_types)

    # check loading and saving for surf
    source_morph_surf.save(os.path.join(tempdir, '42.h5'))

    source_morph_surf_r = read_source_morph(os.path.join(tempdir, '42.h5'))

    assert (all([read == saved for read, saved in
                 zip(sorted(source_morph_surf_r.__dict__),
                     sorted(source_morph_surf.__dict__))]))

    # check wrong subject correction
    stc_surf.subject = None
    assert isinstance(source_morph_surf(stc_surf), SourceEstimate)


@requires_nibabel()
@requires_dipy()
@pytest.mark.slowtest
@sample.requires_sample_data
def test_volume_source_morph():
    """Test volume source estimate morph, special cases and exceptions."""
    import nibabel as nib
    data_path = sample.data_path(
        download=False)  # because testing data has no brain.mgz
    subjects_dir = os.path.join(data_path, 'subjects')
    sample_dir = os.path.join(data_path, 'MEG', 'sample')
    fname_inv_vol = os.path.join(sample_dir,
                                 'sample_audvis-meg-vol-7-meg-inv.fif')

    inverse_operator_vol = read_inverse_operator(fname_inv_vol)

    stc_vol = read_source_estimate(fname_vol, 'sample')

    # check for invalid input type
    with pytest.raises(ValueError, match='src must be an instance of .*'):
        SourceMorph(src=42)

    # check for raising an error if neither
    # inverse_operator_vol['src'][0]['subject_his_id'] nor subject_from is set,
    # but attempting to perform a volume morph
    src = inverse_operator_vol['src']
    src[0]['subject_his_id'] = None

    with pytest.raises(ValueError, match='subject_from is None. Please .*'):
        SourceMorph(src=src, subjects_dir=subjects_dir)

    # check infer subject_from from src[0]['subject_his_id']
    src[0]['subject_his_id'] = 'sample'
    source_morph_vol = SourceMorph(subjects_dir=subjects_dir,
                                   src=inverse_operator_vol['src'],
                                   niter_affine=(1,),
                                   niter_sdr=(1,), spacing=7)

    assert source_morph_vol.subject_from == 'sample'

    # the brain used in sample data has shape (255, 255, 255)
    assert (tuple(
        source_morph_vol.params['DiffeomorphicMap']['domain_shape']) ==
        (37, 37, 37))

    assert (tuple(source_morph_vol.params['AffineMap']['domain_shape']) ==
            (37, 37, 37))

    # proofs the above
    assert source_morph_vol.spacing == 7

    # assure proper src shape
    assert source_morph_vol.params['src_shape_full'] == (
        src[0]['mri_height'], src[0]['mri_depth'], src[0]['mri_width'])

    # check full mri resolution registration (takes very long)
    # source_morph_vol = SourceMorph(
    #     os.path.join(sample_dir, 'sample_audvis-meg-vol-7-fwd.fif'),
    #     subject_from='sample',
    #     subjects_dir=os.path.join(data_path, 'subjects'),
    #     grid_spacing=None)

    fwd = read_forward_solution(
        os.path.join(sample_dir, 'sample_audvis-meg-vol-7-fwd.fif'))
    # check input via path to src and path to subject_to
    source_morph_vol = SourceMorph(
        subject_from='sample',
        subject_to=os.path.join(data_path, 'subjects', 'fsaverage', 'mri',
                                'brain.mgz'),
        subjects_dir=subjects_dir, niter_affine=(1,),
        src=fwd['src'],
        niter_sdr=(1,), spacing=7)

    # check wrong subject_to
    with pytest.raises(IOError, match='cannot read file.*'):
        SourceMorph(subject_from='sample', subject_to='42',
                    subjects_dir=subjects_dir, src=fwd['src'])

    # two different ways of saving
    source_morph_vol.save(os.path.join(tempdir, 'vol'))
    source_morph_vol.save(os.path.join(tempdir, 'vol.h5'))

    # check loading
    source_morph_vol_r = read_source_morph(os.path.join(tempdir, 'vol.h5'))

    # check for invalid file name handling ()
    with pytest.raises(IOError, match='file .* not found'):
        read_source_morph(os.path.join(tempdir, '42'))

    # check morph
    stc_vol_morphed = source_morph_vol(stc_vol)

    # check as_volume=True
    assert isinstance(source_morph_vol(stc_vol, as_volume=True),
                      nib.Nifti1Image)

    # check for subject_from mismatch
    source_morph_vol_r.subject_from = '42'
    with pytest.raises(ValueError, match='.*subject_from must match.*'):
        source_morph_vol_r(stc_vol_morphed)

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
                                             fname=os.path.join(tempdir, '42'))
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

    # check Nifti2Image
    assert isinstance(
        source_morph_vol.as_volume(stc_vol_morphed, mri_resolution=True,
                                   mri_space=True, format='nifti2'),
        nib.Nifti2Image)

    with pytest.raises(ValueError, match='subject_from does not match.*'):
        SourceMorph(src=src, subject_from='42')

    # check wrong format
    with pytest.raises(ValueError, match='invalid format specifier.*'):
        source_morph_vol.as_volume(stc_vol_morphed, format='42')

    with pytest.raises(ValueError, match='invalid format specifier.*'):
        stc_vol_morphed.as_volume(inverse_operator_vol['src'], format='42')


run_tests_if_main()
