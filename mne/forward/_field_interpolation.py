import numpy as np
from scipy import linalg

from ..fiff import FIFF
from ..fiff.pick import pick_types, pick_info
from ..fiff.proj import _has_eeg_average_ref_proj
from ..transforms import (invert_transform, combine_transforms, apply_trans,
                          _coord_frame_name)
from ._make_forward import _create_coils
from ._lead_dots import (_do_self_dots, _do_surf_map_dots,
                         _do_eeg_self_dots, _do_eeg_surf_map_dots)
from ..utils import logger


def _is_axial_coil(coil):
    is_ax = coil['class'] in (FIFF.FWD_COILC_MAG, FIFF.FWD_COILC_AXIAL_GRAD,
                              FIFF.FWD_COILC_AXIAL_GRAD2)
    return is_ax


def _ad_hoc_noise(coils, ctype='meg'):
    v = np.empty(len(coils))
    if ctype == 'meg':
        axs = [_is_axial_coil(coil) for coil in coils]
        v[axs] = 4e-28  # 20e-15 ** 2
        v[np.logical_not(axs)] = 2.5e-25  # 5e-13 ** 2
    else:
        v.fill(1e-12)  # 1e-6 ** 2
    v = np.diag(v)
    cov = dict(diag=True, data=v, eig=None, eigvec=None)
    return cov


def _prepare_field_mapping(info, surf, ctype='meg',
                           origin_frame=FIFF.FIFFV_COORD_HEAD,
                           sphere_origin=(0.0, 0.0, 0.04),
                           int_rad=0.06, subjects_dir=None):
    """Do the dot products"""
    #
    # Step 1. Prepare the coil definitions
    #
    if ctype not in ('meg', 'eeg'):
        raise ValueError('unknown coil type "%s"' % ctype)
    if ctype in ('meg_helmet', 'meg_head'):
        picks = pick_types(info, meg=True, eeg=False, ref_meg=False)
        logger.info('Prepare MEG mapping...')
    else:
        picks = pick_types(info, meg=False, eeg=True, ref_meg=False)
        logger.info('Prepare EEG mapping...')
    if len(picks) == 0:
        raise RuntimeError('cannot map, no channels found')
    chs = pick_info(info, picks)

    # create coil defs
    head_mri_t = invert_transform(info['mri_head_t'])
    if ctype == 'meg':
        meg_mri_t = combine_transforms(info['meg_head_t'], head_mri_t,
                                       FIFF.FIFFV_COORD_DEVICE,
                                       FIFF.FIFFV_COORD_MRI)
        coils = _create_coils(chs, FIFF.FWD_COIL_ACCURACY_NORMAL,
                              meg_mri_t, coil_type='meg')
        logger.info('Coil definitions in MRI coordinates created.')
    else:  # EEG
        coils = _create_coils(chs, coil_type='eeg')

    #
    # Step 2. Calculate the dot products
    #
    my_origin = sphere_origin
    if origin_frame == FIFF.FIFFV_COORD_HEAD:
        my_origin = apply_trans(head_mri_t, my_origin)
    elif origin_frame != FIFF.FIFFV_COORD_MRI:
        raise RuntimeError('Origin cannot be specified in %s coordinates'
                           % _coord_frame_name(origin_frame))
    if ctype == 'meg':
        # MEG...
        self_dots = _do_self_dots(int_rad, False, coils, my_origin)
        surface_dots = _do_surf_map_dots(int_rad, False, coils, surf,
                                         my_origin)
        noise = _ad_hoc_noise(coils, ctype='meg')
        miss = 1e-4  # Smoothing criterion for MEG
    else:
         # EEG...
        self_dots = _do_eeg_self_dots(0.7 * int_rad, False, coils, my_origin)
        surface_dots = _do_eeg_surf_map_dots(0.7 * int_rad, False, coils,
                                             surf, None, my_origin)
        noise = _ad_hoc_noise(coils, ctype='eeg')
        miss = 1e-3  # Smoothing criterion for EEG

    #
    # Step 4. Return the result
    #
    res = dict(kind=ctype, surf=surf, picks=picks, coils=coils,
               origin=my_origin, noise=noise, self_dots=self_dots,
               surface_dots=surface_dots, int_rad=int_rad, miss=miss)
    logger.info('Field mapping data ready')
    return res


def _compute_mapping_matrix(fmd, proj=None):
    """Do the hairy computations"""
    if 'mapping_mat' in fmd:
        raise RuntimeError('mapping matrix recomputation attempted')
    if proj is not None:
        raise NotImplementedError('projection not supported yet')
    do_proj = False

    # XXX Deal with bad channels and add SSP?

    logger.info('preparing the mapping matrix...')

    # Pick the correct channels from the dot products
    mat = fmd['self_dots']  # no need to sub-select b/c no bads ignored

    # Pick the correct channels for the noise covariance
    noise_cov = fmd['noise']  # still don't need to sub-select

    # ...then whitening
    if not noise_cov['diag']:
        whitener = np.zeros((noise_cov['dim'], noise_cov['dim']))
        eig = noise_cov['eig']
        nzero = (eig > 0)
        whitener[nzero, nzero] = 1.0 / np.sqrt(eig[nzero])
        whitener = np.dot(whitener, noise_cov['eigvec'])
    else:
        whitener = np.diag(1.0 / np.sqrt(noise_cov['data'].ravel()))
    mat = np.dot(whitener.T, np.dot(mat, whitener))

    # SVD is numerically better than the eigenvalue composition even if
    # mat is supposed to be symmetric and positive definite
    logger.info('SVD...')
    uu, sing, vv = linalg.svd(mat, overwrite_a=True)

    # Eigenvalue truncation
    sumk = np.cumsum(sing)
    sumk /= sumk[-1]
    fmd['nest'] = np.where(sumk > (1.0 - fmd['miss']))[0][-1]
    logger.info('Truncate at %d missing %g' % (fmd['nest'], fmd['miss']))
    sing = 1.0 / sing
    sing[fmd['nest'] + 1:] = 0.0

    # Put the inverse together
    logger.info('Put the inverse together...')
    mat = np.dot(vv, np.dot(sing, uu))

    # Sandwich with the whitener
    mat = np.dot(whitener.T, np.dot(mat, whitener))

    # Finally sandwhich in the selection matrix
    # This one picks up the correct lead field project
    mapping_mat = np.dot(fmd['surface_dots'], mat)

    # Optionally apply the average electrode reference to the final field map
    if fmd['kind'] == FIFF.FIELD_MAP_EEG and do_proj:
        if _has_eeg_average_ref_proj(proj):
            logger.info('The map will have average electrode reference')
            mapping_mat -= np.mean(mapping_mat, axis=0)[np.newaxis, :]
    return mapping_mat


def make_surface_mapping(info, surf, ctype='meg'):
    """Re-map M/EEG data to a surface

    Parameters
    ----------
    info : instance of fiff.meas_info.Info
        Measurement info.
    surf : dict
        The surface to map the data to. The required fields are `'rr'` and
        `'nn'`.
    ctype : str
        Must be either `'meg'` or `'eeg'`, determines the type of field.

    Returns
    -------
    mapping : array
        A n_vertices x n_sensors array that remaps the MEG or EEG data,
        as `new_data = np.dot(mapping, data)`.
    """
    if not all([key in surf for key in ['rr', 'nn']]):
        raise KeyError('surf must have both "rr" and "nn"')
    fmd = _prepare_field_mapping(info, surf, ctype)
    mapping_mat = _compute_mapping_matrix(fmd, None)
    return mapping_mat
