# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np

from ...channels import DigMontage, make_dig_montage
from ...surface import _voxel_neighbors
from ...transforms import Transform, _frame_to_str, apply_trans
from ...utils import _check_option, _pl, _require_version, _validate_type, verbose, warn


@verbose
def warp_montage(montage, moving, static, reg_affine, sdr_morph, verbose=None):
    """Warp a montage to a template with image volumes using SDR.

    .. note:: This is likely only applicable for channels inside the brain
              (intracranial electrodes).

    Parameters
    ----------
    montage : instance of mne.channels.DigMontage
        The montage object containing the channels.
    %(moving)s
    %(static)s
    %(reg_affine)s
    %(sdr_morph)s
    %(verbose)s

    Returns
    -------
    montage_warped : mne.channels.DigMontage
        The modified montage object containing the channels.
    """
    _require_version("nibabel", "warp montage", "2.1.0")
    _require_version("dipy", "warping points using SDR", "1.6.0")

    from dipy.align.imwarp import DiffeomorphicMap
    from nibabel import MGHImage
    from nibabel.spatialimages import SpatialImage

    _validate_type(moving, SpatialImage, "moving")
    _validate_type(static, SpatialImage, "static")
    _validate_type(reg_affine, np.ndarray, "reg_affine")
    _check_option("reg_affine.shape", reg_affine.shape, ((4, 4),))
    _validate_type(sdr_morph, (DiffeomorphicMap, None), "sdr_morph")
    _validate_type(montage, DigMontage, "montage")

    moving_mgh = MGHImage(np.array(moving.dataobj).astype(np.float32), moving.affine)
    static_mgh = MGHImage(np.array(static.dataobj).astype(np.float32), static.affine)
    del moving, static

    # get montage channel coordinates
    ch_dict = montage.get_positions()
    if ch_dict["coord_frame"] != "mri":
        bad_coord_frames = np.unique([d["coord_frame"] for d in montage.dig])
        bad_coord_frames = ", ".join(
            [
                _frame_to_str[cf] if cf in _frame_to_str else str(cf)
                for cf in bad_coord_frames
            ]
        )
        raise RuntimeError(
            f'Coordinate frame not supported, expected "mri", got {bad_coord_frames}'
        )
    ch_names = list(ch_dict["ch_pos"].keys())
    ch_coords = np.array([ch_dict["ch_pos"][name] for name in ch_names])

    ch_coords = apply_trans(  # convert to moving voxel space
        np.linalg.inv(moving_mgh.header.get_vox2ras_tkr()), ch_coords * 1000
    )
    # next, to moving scanner RAS
    ch_coords = apply_trans(moving_mgh.header.get_vox2ras(), ch_coords)

    # now, apply reg_affine
    ch_coords = apply_trans(
        Transform(  # to static ras
            fro="ras", to="ras", trans=np.linalg.inv(reg_affine)
        ),
        ch_coords,
    )

    # now, apply SDR morph
    if sdr_morph is not None:
        ch_coords = sdr_morph.transform_points(
            ch_coords,
            coord2world=sdr_morph.domain_grid2world,
            world2coord=sdr_morph.domain_world2grid,
        )

    # back to voxels but now for the static image
    ch_coords = apply_trans(np.linalg.inv(static_mgh.header.get_vox2ras()), ch_coords)

    # finally, back to surface RAS
    ch_coords = apply_trans(static_mgh.header.get_vox2ras_tkr(), ch_coords) / 1000

    # make warped montage
    montage_warped = make_dig_montage(dict(zip(ch_names, ch_coords)), coord_frame="mri")
    return montage_warped


def _warn_missing_chs(info, dig_image, after_warp=False):
    """Warn that channels are missing."""
    # ensure that each electrode contact was marked in at least one voxel
    missing = set(np.arange(1, len(info.ch_names) + 1)).difference(
        set(np.unique(np.array(dig_image.dataobj)))
    )
    missing_ch = [info.ch_names[idx - 1] for idx in missing]
    if missing_ch:
        warn(
            f"Channel{_pl(missing_ch)} "
            f'{", ".join(repr(ch) for ch in missing_ch)} not assigned '
            "voxels " + (f" after applying {after_warp}" if after_warp else "")
        )


@verbose
def make_montage_volume(
    montage,
    base_image,
    thresh=0.5,
    max_peak_dist=1,
    voxels_max=100,
    use_min=False,
    verbose=None,
):
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
    _require_version("nibabel", "montage volume", "2.1.0")
    import nibabel as nib

    _validate_type(montage, DigMontage, "montage")
    _validate_type(base_image, nib.spatialimages.SpatialImage, "base_image")
    _validate_type(thresh, float, "thresh")
    if thresh < 0 or thresh >= 1:
        raise ValueError(f"`thresh` must be between 0 and 1, got {thresh}")
    _validate_type(max_peak_dist, int, "max_peak_dist")
    _validate_type(voxels_max, int, "voxels_max")
    _validate_type(use_min, bool, "use_min")

    # load image and make sure it's in surface RAS
    if not isinstance(base_image, nib.spatialimages.SpatialImage):
        base_image = nib.load(base_image)

    base_image_mgh = nib.MGHImage(
        np.array(base_image.dataobj).astype(np.float32), base_image.affine
    )
    del base_image

    # get montage channel coordinates
    ch_dict = montage.get_positions()
    if ch_dict["coord_frame"] != "mri":
        bad_coord_frames = np.unique([d["coord_frame"] for d in montage.dig])
        bad_coord_frames = ", ".join(
            [
                _frame_to_str[cf] if cf in _frame_to_str else str(cf)
                for cf in bad_coord_frames
            ]
        )
        raise RuntimeError(
            f'Coordinate frame not supported, expected "mri", got {bad_coord_frames}'
        )

    ch_names = list(ch_dict["ch_pos"].keys())
    ch_coords = np.array([ch_dict["ch_pos"][name] for name in ch_names])

    # convert to voxel space
    ch_coords = apply_trans(
        np.linalg.inv(base_image_mgh.header.get_vox2ras_tkr()), ch_coords * 1000
    )

    # take channel coordinates and use the image to transform them
    # into a volume where all the voxels over a threshold nearby
    # are labeled with an index
    image_data = np.array(base_image_mgh.dataobj)
    if use_min:
        image_data *= -1
    elec_image = np.zeros(base_image_mgh.shape, dtype=int)
    for i, ch_coord in enumerate(ch_coords):
        if np.isnan(ch_coord).any():
            continue
        # this looks up to a voxel away, it may be marked imperfectly
        volume = _voxel_neighbors(
            ch_coord,
            image_data,
            thresh=thresh,
            max_peak_dist=max_peak_dist,
            voxels_max=voxels_max,
        )
        for voxel in volume:
            if elec_image[voxel] != 0:
                # some voxels ambiguous because the contacts are bridged on
                # the image so assign the voxel to the nearest contact location
                dist_old = np.sqrt(
                    (ch_coords[elec_image[voxel] - 1] - voxel) ** 2
                ).sum()
                dist_new = np.sqrt((ch_coord - voxel) ** 2).sum()
                if dist_new < dist_old:
                    elec_image[voxel] = i + 1
            else:
                elec_image[voxel] = i + 1

    # assemble the volume
    elec_image = nib.spatialimages.SpatialImage(elec_image, base_image_mgh.affine)
    _warn_missing_chs(montage, elec_image, after_warp=False)

    return elec_image
