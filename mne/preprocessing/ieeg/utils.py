# Authors: Alex Rockhill <aprockhill@mailbox.org>
#
# License: BSD-3-Clause

from itertools import combinations
import numpy as np

from ...channels import make_dig_montage
from ...io.pick import _picks_to_idx
from ...surface import (_read_mri_surface, fast_cross_3d, read_surface,
                        _read_patch, _compute_nearest)
from ...transforms import (apply_trans, invert_transform, _cart_to_sph,
                           _ensure_trans)
from ...utils import verbose, get_subjects_dir, _validate_type, _ensure_int


def make_montage_volume(montage, base_image, thresh=0.5, max_peak_dist=1,
                        voxels_max=100, use_min=False, verbose=None):
    """Make a volume from intracranial electrode contact locations.

    Find areas of the input volume with intensity greater than
    a threshold surrounding local extrema near the channel location.
    Monotonicity from the peak is enforced to prevent channels
    bleeding into each other.

    Parameters
    ----------
    montage : instance of mne.channels.DigMontage
        The montage object containing the channels.
    base_image : path-like | nibabel.spatialimages.SpatialImage
        Path to a volumetric scan (e.g. CT) of the subject. Can be in any
        format readable by nibabel. Can also be a nibabel image object.
        Local extrema (max or min) should be nearby montage channel locations.
    thresh : float
        The threshold relative to the peak to determine the size
        of the sensors on the volume.
    max_peak_dist : int
        The number of voxels away from the channel location to
        look in the ``image``. This will depend on the accuracy of
        the channel locations, the default (one voxel in all directions)
        will work only with localizations that are that accurate.
    voxels_max : int
        The maximum number of voxels for each channel.
    use_min : bool
        Whether to hypointensities in the volume as channel locations.
        Default False uses hyperintensities.
    %(verbose)s

    Returns
    -------
    elec_image : nibabel.spatialimages.SpatialImage
        An image in Freesurfer surface RAS space with voxel values
        corresponding to the index of the channel. The background
        is 0s and this index starts at 1.
    """
    _require_version('nibabel', 'SDR morph', '2.1.0')
    import nibabel as nib

    _validate_type(montage, DigMontage, 'montage')
    _validate_type(base_image, nib.spatialimages.SpatialImage, 'base_image')
    _validate_type(thresh, float, 'thresh')
    if thresh < 0 or thresh >= 1:
        raise ValueError(f'`thresh` must be between 0 and 1, got {thresh}')
    _validate_type(max_peak_dist, int, 'max_peak_dist')
    _validate_type(voxels_max, int, 'voxels_max')
    _validate_type(use_min, bool, 'use_min')

    # load image and make sure it's in surface RAS
    if not isinstance(base_image, nib.spatialimages.SpatialImage):
        base_image = nib.load(base_image)

    # get montage channel coordinates
    ch_dict = montage.get_positions()
    if ch_dict['coord_frame'] != 'mri':
        bad_coord_frames = np.unique([d['coord_frame'] for d in montage.dig])
        bad_coord_frames = ', '.join([
            _frame_to_str[cf] if cf in _frame_to_str else str(cf)
            for cf in bad_coord_frames])
        raise RuntimeError('Coordinate frame not supported, expected '
                           f'"mri", got {bad_coord_frames}')

    ch_names = list(ch_dict['ch_pos'].keys())
    ch_coords = np.array([ch_dict['ch_pos'][name] for name in ch_names])

    # convert to freesurfer voxel space
    ch_coords = apply_trans(
        np.linalg.inv(fs_from_img.header.get_vox2ras_tkr()), ch_coords * 1000)

    # take channel coordinates and use the image to transform them
    # into a volume where all the voxels over a threshold nearby
    # are labeled with an index
    image_data = np.array(base_image.dataobj)
    if use_min:
        image_data *= -1
    image_from = np.zeros(base_image.shape, dtype=int)
    for i, ch_coord in enumerate(ch_coords):
        if np.isnan(ch_coord).any():
            continue
        # this looks up to a voxel away, it may be marked imperfectly
        volume = _voxel_neighbors(ch_coord, image_data, thresh=thresh,
                                  max_peak_dist=max_peak_dist,
                                  voxels_max=voxels_max)
        for voxel in volume:
            if image_from[voxel] != 0:
                # some voxels ambiguous because the contacts are bridged on
                # the image so assign the voxel to the nearest contact location
                dist_old = np.sqrt(
                    (ch_coords[image_from[voxel] - 1] - voxel)**2).sum()
                dist_new = np.sqrt((ch_coord - voxel)**2).sum()
                if dist_new < dist_old:
                    image_from[voxel] = i + 1
            else:
                image_from[voxel] = i + 1

    # apply the mapping
    image_from = nib.spatialimages.SpatialImage(image_from, fs_from_img.affine)
    _warn_missing_chs(montage, image_from, after_warp=False)

    template_brain = nib.load(
        op.join(subjects_dir_to, subject_to, 'mri', 'brain.mgz'))

    image_to = apply_volume_registration(
        image_from, template_brain, reg_affine, sdr_morph,
        interpolation='nearest')

    after_warp = \
        'SDR warp' if sdr_morph is not None else 'affine transformation'
    _warn_missing_chs(montage, image_to, after_warp=after_warp)

    # recover the contact positions as the center of mass
    warped_data = np.asanyarray(image_to.dataobj)
    for val, ch_coord in enumerate(ch_coords, 1):
        ch_coord[:] = np.mean(np.where(warped_data == val), axis=1)

    # convert back to surface RAS of the template
    fs_to_img = nib.load(
        op.join(subjects_dir_to, subject_to, 'mri', 'brain.mgz'))
    ch_coords = apply_trans(
        fs_to_img.header.get_vox2ras_tkr(), ch_coords) / 1000

    # make warped montage
    montage_warped = make_dig_montage(
        dict(zip(ch_names, ch_coords)), coord_frame='mri')
    return montage_warped, image_from, image_to
