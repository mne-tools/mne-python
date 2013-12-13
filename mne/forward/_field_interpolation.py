import numpy as np
from os import path as op
from scipy import linalg

from ..fiff import FIFF
from ..fiff.pick import pick_types, pick_info
from ..fiff.proj import _has_eeg_average_ref_proj
from ..utils import logger
from ..surface import read_bem_surfaces
from ..transforms import (invert_transform, combine_transforms, apply_trans,
                          transform_source_space_to, _coord_frame_name)
from ._make_forward import _read_coil_defs, _create_coils
from ._lead_dots import (_do_self_dots, _do_surf_map_dots,
                         _do_eeg_self_dots, _do_eeg_surf_map_dots)


def _is_axial_coil(coil):
    if coil['class'] in (FIFF.FWD_COILC_MAG, FIFF.FWD_COILC_AXIAL_GRAD,
                         FIFF.FWD_COILC_AXIAL_GRAD2):
        return True
    else:
        return False


def _get_smooth_weights(surf, alpha):
    """Get weights to do a distance dependent smoothing"""
    logger.info('Adding smoothing weights to surface ...')
    smooth_weights = [1.0 / (surf['vert_dist'][k] + alpha)
                      for k in range(surf['np'])]
    return smooth_weights


def _ad_hoc_noise(coils, ctype='meg'):
    grad_std = 5e-13
    mag_std = 20e-15
    eeg_std = 1e-6
    if ctype == 'meg':
        v = np.array([(mag_std ** 2 if _is_axial_coil(coil) else grad_std ** 2)
                      for coil in coils])
    else:
        v = np.empty(len(coils))
        v.fill(eeg_std ** 2)
    v = np.diag(v)
    cov = dict(data=v)
    return cov


def _select_helmet(chs):
    # Educated guess for the helmet type based on channels
    system = '306m'
    for ch in chs:
        if ch['kind'] == FIFF.FIFFV_MEG_CH:
            coil_type = ch['coil_type'] & 0xFFFF
            if coil_type == FIFF.FIFFV_COIL_NM_122:
                system = '122m'
                break
            elif coil_type // 1000 == 3:  # All Vectorview coils are 30xx
                system = '306m'
                break
            elif (coil_type == FIFF.FIFFV_COIL_MAGNES_MAG or
                  coil_type == FIFF.FIFFV_COIL_MAGNES_GRAD):
                nmag = np.sum([c['kind'] == FIFF.FIFF_MEG_CH for c in chs])
                system = 'Magnes_3600wh' if nmag > 150 else 'Magnes_2500wh'
                break
            elif coil_type == FIFF.FIFFV_COIL_CTF_GRAD:
                system = 'CTF_275'
                break
            elif coil_type == FIFF.FIFFV_COIL_KIT_GRAD:
                system = 'KIT'
                break
            elif coil_type == FIFF.FIFFV_COIL_BABY_GRAD:
                system = 'BabySQUID'
                break
    helmet_file = op.join(op.split(__file__)[0], '..', 'data', 'helmets',
                          system + '.fif.gz')
    return helmet_file


def _load_and_transform_meg_helmet(info):
    """This routine loads the helmet for field interpolation"""
    fname = _select_helmet(info['chs'])
    if 'mri_head_t' not in info or not info['mri_head_t']:
        raise RuntimeError('MRI <-> head coordinate transformation missing')

    surf = read_bem_surfaces(fname, False, FIFF.FIFFV_MNE_SURF_MEG_HELMET)

    # Ignore what the file says
    surf['coord_frame'] = FIFF.FIFFV_COORD_DEVICE
    transform_source_space_to(surf, FIFF.FIFFV_COORD_HEAD, info['meg_head_t'])
    transform_source_space_to(surf, FIFF.FIFFV_COORD_MRI, info['mri_head_t'])
    logger.info('Interpolation surface is now in MRI coordinates')
    return surf


def _prepare_field_mapping(subject, info, which, head_surface_map,
                           sphere_origin, origin_frame,
                           intrad, subjects_dir):
    """Do the dot products"""
    default_origin = np.array([0.0, 0.0, 0.04])

    #
    # Step 1. Prepare the coil definitions
    #
    if which not in (FIFF.FIELD_MAP_MEG, FIFF.FIELD_MAP_EEG):
        raise ValueError('which must be MEG or EEG')
    if which == FIFF.FIELD_MAP_MEG:
        picks = pick_types(info, meg=True, eeg=False, reg_meg=False)
        which_str = 'MEG'
    else:
        picks = pick_types(info, meg=False, eeg=True, reg_meg=False)
        which_str = 'EEG'
    if len(picks) == 0:
        raise RuntimeError('cannot map, no channels found')
    chs = pick_info(info, picks)

    logger.info('Prepare %s mapping...' % which_str)

    if which == FIFF.FIELD_MAP_EEG or head_surface_map:
        # Load the head surface from the BEM
        this_bem = op.join(subjects_dir, subject, 'bem',
                           '%s-bem.fif' % subject)
        surf = read_bem_surfaces(this_bem, True, FIFF.FIFFV_BEM_SURF_ID_HEAD)
        logger.info('Using all nodes of the head surface for mapping.')
        need_smooth_weights = True
    else:  # MEG map on the helmet
        surf = _load_and_transform_meg_helmet(info)
        need_smooth_weights = False

    # create coil defs
    head_mri_t = invert_transform(info['mri_head_t'])
    if which == FIFF.FIELD_MAP_MEG:
        meg_mri_t = combine_transforms(info['meg_head_t'], head_mri_t,
                                       FIFF.FIFFV_COORD_DEVICE,
                                       FIFF.FIFFV_COORD_MRI)
        templates = _read_coil_defs()
        coils = _create_coils(templates, chs, FIFF.FWD_COIL_ACCURACY_NORMAL,
                              meg_mri_t, coil_type='meg')
        logger.info('Coil definitions in MRI coordinates created.')
    else:  # EEG
        eeg_xform = {'trans': np.eye(4), 'to': FIFF.FIFFV_COORD_HEAD,
                     'from': FIFF.FIFFV_COORD_HEAD}
        coils = _create_coils(templates, chs, None, eeg_xform, coil_type='eeg')

    #
    # Step 2. Calculate the dot products
    #
    if origin_frame == FIFF.FIFFV_COORD_UNKNOWN:
        my_origin = default_origin
        origin_frame = FIFF.FIFFV_COORD_HEAD
    else:
        my_origin = sphere_origin
    if origin_frame == FIFF.FIFFV_COORD_HEAD:
        apply_trans(head_mri_t, my_origin)
    elif origin_frame != FIFF.FIFFV_COORD_MRI:
        raise RuntimeError('Origin cannot be specified in %s coordinates'
                           % _coord_frame_name(origin_frame))
    if which == FIFF.FIELD_MAP_MEG:
        # MEG...
        self_dots = _do_self_dots(intrad, False, coils, my_origin)
        surface_dots = _do_surf_map_dots(intrad, False, coils, surf,
                                         my_origin)
        noise = _ad_hoc_noise(coils, ctype='meg')
    else:
         # EEG...
        self_dots = _do_eeg_self_dots(0.7 * intrad, False, coils, my_origin)
        surface_dots = _do_eeg_surf_map_dots(0.7 * intrad, False, coils,
                                             surf, None, my_origin)
        noise = _ad_hoc_noise(coils, ctype='eeg')

    #
    # Step 4. Return the result
    #
    res = dict(kind=which, surf=surf, picks=picks, coils=coils,
               origin=my_origin, noise=noise, self_dots=self_dots,
               surface_dots=surface_dots, intrad=intrad, smooth_weights=None)
    if need_smooth_weights:
        logger.info('Adding smoothing weights...')
        res['smooth_weights'] = _get_smooth_weights(surf, 0.5)
    logger.info('Field mapping data ready')
    return res


def _compute_mapping_matrix(fmd, proj=None):
    """Do the hairy computations"""
    if 'mapping_mat' in fmd:
        raise RuntimeError('mapping matrix recomputation attempted')
    if proj is not None:
        raise NotImplementedError('projection not supported yet')
    do_proj = False

    # XXX Deal with bad channels?
    nsel = len(fmd['picks'])
    uu = np.zeros((nsel, nsel))
    vv = np.zeros((nsel, nsel))

    logger.info('preparing the mapping matrix...')

    # Pick the correct channels from the dot products
    mat = fmd['self_dots']  # no need to sub-select b/c no bads ignored

    # Pick the correct channels for the noise covariance
    noise_cov = fmd['noise']  # still don't need to sub-select

    # XXX Add SSP?

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
    fmd['mapping_mat'] = np.dot(fmd['surface_dots'], mat)

    # Optionally apply the average electrode reference to the final field map
    if fmd['kind'] == FIFF.FIELD_MAP_EEG and do_proj:
        if _has_eeg_average_ref_proj(proj):
            logger.info('The map will have average electrode reference')
            m = fmd['mapping_mat']
            m -= np.mean(m, axis=0)[np.newaxis, :]


def _interpolate_extrapolate(fmd, meas):
    """Apply the MNE interpolation/extrapolation"""
    # Dense map, ignore smoothing
    return np.dot(fmd['mapping_mat'], meas)
