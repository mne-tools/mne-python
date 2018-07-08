# -*- coding: utf-8 -*-
# Author: Tommy Clausner <Tommy.Clausner@gmail.com>
#
# License: BSD (3-clause)
import os

import nibabel as nib
from mne import SourceEstimate, VolSourceEstimate
from mne import read_evokeds, SourceMorph, read_source_morph
from mne.datasets import sample
from mne.minimum_norm import apply_inverse, read_inverse_operator
from mne.utils import run_tests_if_main, requires_nibabel

print(__doc__)

# Setup paths
data_path = sample.data_path()
sample_dir = data_path + '/MEG/sample'

fname_evoked = sample_dir + '/sample_audvis-ave.fif'
fname_inv_vol = sample_dir + '/sample_audvis-meg-vol-7-meg-inv.fif'
fname_inv_surf = sample_dir + '/sample_audvis-meg-oct-6-meg-inv.fif'


@requires_nibabel()
def test_surface_source_morph():
    """Test surface source morph."""
    evoked = read_evokeds(fname_evoked, condition=0, baseline=(None, 0))
    inverse_operator_surf = read_inverse_operator(fname_inv_surf)

    # Apply inverse operator
    stc_surf = apply_inverse(evoked, inverse_operator_surf, 1.0 / 3.0 ** 2,
                             "dSPM")

    stc_surf.crop(0.087, 0.087)

    source_morph_surf = SourceMorph(inverse_operator_surf['src'],
                                    subjects_dir=data_path + '/subjects')

    source_morph_surf.save('surf')
    source_morph_surf.save('surf.h5')

    source_morph_surf = read_source_morph('surf.h5')
    stc_surf_morphed = source_morph_surf(stc_surf)
    os.remove('surf-morph.h5')
    os.remove('surf.h5')

    assert isinstance(stc_surf_morphed, SourceEstimate)


@requires_nibabel()
def test_volume_source_morph():
    """Test volume source morph."""
    evoked = read_evokeds(fname_evoked, condition=0, baseline=(None, 0))
    inverse_operator_vol = read_inverse_operator(fname_inv_vol)

    # Apply inverse operator
    stc_vol = apply_inverse(evoked, inverse_operator_vol, 1.0 / 3.0 ** 2,
                            "dSPM")

    stc_vol.crop(0.087, 0.087)

    source_morph_vol = SourceMorph(
        data_path + '/MEG/sample' + '/sample_audvis-meg-vol-7-fwd.fif',
        subject_from='sample',
        subject_to=data_path + '/subjects/fsaverage/mri/brain.mgz',
        subjects_dir=data_path + '/subjects')
    source_morph_vol.save('vol')
    source_morph_vol.save('vol.h5')

    source_morph_vol = read_source_morph('vol.h5')
    stc_vol_morphed = source_morph_vol(stc_vol)
    os.remove('vol-morph.h5')
    os.remove('vol.h5')

    img_morph_res = source_morph_vol.as_volume(stc_vol_morphed,
                                               mri_resolution=False)

    assert isinstance(img_morph_res, nib.Nifti1Image)

    img_mri_res = source_morph_vol.as_volume(stc_vol_morphed,
                                             mri_resolution=True)

    assert isinstance(img_mri_res, nib.Nifti1Image)

    img_any_res = source_morph_vol.as_volume(stc_vol_morphed,
                                             mri_resolution=(3., 3., 3.))

    assert isinstance(img_any_res, nib.Nifti1Image)
    assert isinstance(stc_vol_morphed, VolSourceEstimate)


run_tests_if_main()
