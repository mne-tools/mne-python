# -*- coding: utf-8 -*-
# Author: Tommy Clausner <Tommy.Clausner@gmail.com>
#
# License: BSD (3-clause)
import os.path as op

import nibabel as nib
from mne import SourceEstimate, VolSourceEstimate, VectorSourceEstimate
from mne import read_evokeds, SourceMorph, read_source_morph
from mne.datasets import sample
from mne.minimum_norm import apply_inverse, read_inverse_operator
from mne.utils import (run_tests_if_main, requires_nibabel, _TempDir,
                       string_types)
from nose.tools import assert_raises

# Setup paths
tempdir = _TempDir()
data_path = sample.data_path()
sample_dir = op.join(data_path, 'MEG', 'sample')

fname_evoked = op.join(sample_dir, 'sample_audvis-ave.fif')
fname_inv_vol = op.join(sample_dir, 'sample_audvis-meg-vol-7-meg-inv.fif')
fname_inv_surf = op.join(sample_dir, 'sample_audvis-meg-oct-6-meg-inv.fif')


def test_surface_vector_source_morph():
    """Test surface and vector source estimate morph."""
    evoked = read_evokeds(fname_evoked, condition=0, baseline=(None, 0))
    inverse_operator_surf = read_inverse_operator(fname_inv_surf)

    # Apply inverse operator
    stc_surf = apply_inverse(evoked, inverse_operator_surf, 1.0 / 3.0 ** 2,
                             "dSPM")

    stc_vec = apply_inverse(evoked, inverse_operator_surf, 1.0 / 3.0 ** 2,
                            "dSPM", pick_ori="vector")

    stc_surf.crop(0.087, 0.087)
    stc_vec.crop(0.087, 0.087)

    source_morph_surf = SourceMorph(inverse_operator_surf['src'],
                                    subjects_dir=data_path + '/subjects')

    stc_surf_morphed = source_morph_surf(stc_surf)
    stc_vec_morphed = source_morph_surf(stc_vec)

    # check if correct class after morphing
    assert isinstance(stc_surf_morphed, SourceEstimate)
    assert isinstance(stc_vec_morphed, VectorSourceEstimate)

    # check __repr__
    assert isinstance(source_morph_surf.__repr__(), string_types)


@requires_nibabel()
def test_volume_source_morph():
    """Test volume source estimate morph, special cases and exceptions."""
    evoked = read_evokeds(fname_evoked, condition=0, baseline=(None, 0))
    inverse_operator_vol = read_inverse_operator(fname_inv_vol)

    # Apply inverse operator
    stc_vol = apply_inverse(evoked, inverse_operator_vol, 1.0 / 3.0 ** 2,
                            "dSPM")

    stc_vol.crop(0.087, 0.087)

    # check for invalid input type
    assert_raises(ValueError, SourceMorph, 42)

    # check for raising an error if neither
    # inverse_operator_vol['src'][0]['subject_his_id'] nor subject_from is set
    src = inverse_operator_vol['src']
    src[0]['subject_his_id'] = None
    assert_raises(KeyError, SourceMorph, src,
                  subjects_dir=data_path + '/subjects')

    # check infer subject_from from src[0]['subject_his_id']
    src[0]['subject_his_id'] = 'sample'
    source_morph_vol = SourceMorph(inverse_operator_vol['src'],
                                   subjects_dir=data_path + '/subjects')

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
        subjects_dir=op.join(data_path, 'subjects'))

    # check wrong subject_to
    assert_raises(IOError, SourceMorph,
                  op.join(sample_dir, 'sample_audvis-meg-vol-7-fwd.fif'),
                  subject_from='sample', subject_to='42',
                  subjects_dir=op.join(data_path, 'subjects'))

    # two different ways of saving
    source_morph_vol.save(op.join(tempdir, 'vol'))
    source_morph_vol.save(op.join(tempdir, 'vol.h5'))

    # check loading
    source_morph_vol_r = read_source_morph(op.join(tempdir, 'vol.h5'))

    # check for invalid file name handling
    assert_raises(IOError, read_source_morph, op.join(tempdir, '42'))

    # check morph
    stc_vol_morphed = source_morph_vol(stc_vol)

    # check for subject_from mismatch
    source_morph_vol_r.subject_from = '42'
    assert_raises(ValueError, source_morph_vol_r, stc_vol_morphed)

    # check if nifti is in grid morph space with voxel_size == grid_spacing
    img_morph_res = source_morph_vol.as_volume(stc_vol_morphed,
                                               mri_resolution=False)
    assert isinstance(img_morph_res, nib.Nifti1Image)
    grid_spacing = source_morph_vol.grid_spacing
    assert img_morph_res.header.get_zooms()[:3] == grid_spacing

    # check if nifti is mri resolution with voxel_size == (1., 1., 1.)
    img_mri_res = source_morph_vol.as_volume(stc_vol_morphed,
                                             mri_resolution=True)
    assert isinstance(img_mri_res, nib.Nifti1Image)
    assert img_mri_res.header.get_zooms()[:3] == (1., 1., 1.)

    # check if nifti is defined resolution with voxel_size == (7., 7., 7.)
    img_any_res = source_morph_vol.as_volume(stc_vol_morphed,
                                             mri_resolution=(7., 7., 7.),
                                             fname=op.join(tempdir, '42'))
    assert isinstance(img_any_res, nib.Nifti1Image)
    assert img_any_res.header.get_zooms()[:3] == (7., 7., 7.)

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
