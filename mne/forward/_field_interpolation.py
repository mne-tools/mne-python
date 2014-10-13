import numpy as np
from scipy import linalg
from copy import deepcopy

from ..io.constants import FIFF
from ..io.pick import pick_types, pick_info
from ..surface import get_head_surf, get_meg_helmet_surf

from ..io.proj import _has_eeg_average_ref_proj, make_projector
from ..transforms import transform_surface_to, read_trans, _find_trans
from ._make_forward import _create_coils
from ._lead_dots import (_do_self_dots, _do_surface_dots, _get_legen_table,
                         _get_legen_lut_fast, _get_legen_lut_accurate)
from ..parallel import check_n_jobs
from ..utils import logger, verbose
from ..fixes import partial


def _is_axial_coil(coil):
    is_ax = coil['coil_class'] in (FIFF.FWD_COILC_MAG,
                                   FIFF.FWD_COILC_AXIAL_GRAD,
                                   FIFF.FWD_COILC_AXIAL_GRAD2)
    return is_ax


def _ad_hoc_noise(coils, ch_type='meg'):
    v = np.empty(len(coils))
    if ch_type == 'meg':
        axs = np.array([_is_axial_coil(coil) for coil in coils], dtype=bool)
        v[axs] = 4e-28  # 20e-15 ** 2
        v[np.logical_not(axs)] = 2.5e-25  # 5e-13 ** 2
    else:
        v.fill(1e-12)  # 1e-6 ** 2
    cov = dict(diag=True, data=v, eig=None, eigvec=None)
    return cov


def _compute_mapping_matrix(fmd, info):
    """Do the hairy computations"""
    logger.info('preparing the mapping matrix...')
    # assemble a projector and apply it to the data
    ch_names = fmd['ch_names']
    projs = info.get('projs', list())
    proj_op = make_projector(projs, ch_names)[0]
    proj_dots = np.dot(proj_op.T, np.dot(fmd['self_dots'], proj_op))

    noise_cov = fmd['noise']
    # Whiten
    if not noise_cov['diag']:
        raise NotImplementedError  # this shouldn't happen
    whitener = np.diag(1.0 / np.sqrt(noise_cov['data'].ravel()))
    whitened_dots = np.dot(whitener.T, np.dot(proj_dots, whitener))

    # SVD is numerically better than the eigenvalue composition even if
    # mat is supposed to be symmetric and positive definite
    uu, sing, vv = linalg.svd(whitened_dots, full_matrices=False,
                              overwrite_a=True)

    # Eigenvalue truncation
    sumk = np.cumsum(sing)
    sumk /= sumk[-1]
    fmd['nest'] = np.where(sumk > (1.0 - fmd['miss']))[0][0]
    logger.info('Truncate at %d missing %g' % (fmd['nest'], fmd['miss']))
    sing = 1.0 / sing[:fmd['nest']]

    # Put the inverse together
    logger.info('Put the inverse together...')
    inv = np.dot(uu[:, :fmd['nest']] * sing, vv[:fmd['nest']]).T

    # Sandwich with the whitener
    inv_whitened = np.dot(whitener.T, np.dot(inv, whitener))

    # Take into account that the lead fields used to compute
    # d->surface_dots were unprojected
    inv_whitened_proj = (np.dot(inv_whitened.T, proj_op)).T

    # Finally sandwich in the selection matrix
    # This one picks up the correct lead field projection
    mapping_mat = np.dot(fmd['surface_dots'], inv_whitened_proj)

    # Optionally apply the average electrode reference to the final field map
    if fmd['kind'] == 'eeg':
        if _has_eeg_average_ref_proj(projs):
            logger.info('The map will have average electrode reference')
            mapping_mat -= np.mean(mapping_mat, axis=0)[np.newaxis, :]
    return mapping_mat


@verbose
def _make_surface_mapping(info, surf, ch_type='meg', trans=None, mode='fast',
                          n_jobs=1, verbose=None):
    """Re-map M/EEG data to a surface

    Parameters
    ----------
    info : instance of io.meas_info.Info
        Measurement info.
    surf : dict
        The surface to map the data to. The required fields are `'rr'`,
        `'nn'`, and `'coord_frame'`. Must be in head coordinates.
    ch_type : str
        Must be either `'meg'` or `'eeg'`, determines the type of field.
    trans : None | dict
        If None, no transformation applied. Should be a Head<->MRI
        transformation.
    mode : str
        Either `'accurate'` or `'fast'`, determines the quality of the
        Legendre polynomial expansion used. `'fast'` should be sufficient
        for most applications.
    n_jobs : int
        Number of permutations to run in parallel (requires joblib package).
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    mapping : array
        A n_vertices x n_sensors array that remaps the MEG or EEG data,
        as `new_data = np.dot(mapping, data)`.
    """
    if not all([key in surf for key in ['rr', 'nn']]):
        raise KeyError('surf must have both "rr" and "nn"')
    if 'coord_frame' not in surf:
        raise KeyError('The surface coordinate frame must be specified '
                       'in surf["coord_frame"]')
    if mode not in ['accurate', 'fast']:
        raise ValueError('mode must be "accurate" or "fast", not "%s"' % mode)

    # deal with coordinate frames here -- always go to "head" (easiest)
    if surf['coord_frame'] == FIFF.FIFFV_COORD_MRI:
        if trans is None or FIFF.FIFFV_COORD_MRI not in [trans['to'],
                                                         trans['from']]:
            raise ValueError('trans must be a Head<->MRI transform if the '
                             'surface is not in head coordinates.')
        surf = transform_surface_to(deepcopy(surf), 'head', trans)

    n_jobs = check_n_jobs(n_jobs)

    #
    # Step 1. Prepare the coil definitions
    # Do the dot products, assume surf in head coords
    #
    if ch_type not in ('meg', 'eeg'):
        raise ValueError('unknown coil type "%s"' % ch_type)
    if ch_type == 'meg':
        picks = pick_types(info, meg=True, eeg=False, ref_meg=False)
        logger.info('Prepare MEG mapping...')
    else:
        picks = pick_types(info, meg=False, eeg=True, ref_meg=False)
        logger.info('Prepare EEG mapping...')
    if len(picks) == 0:
        raise RuntimeError('cannot map, no channels found')
    chs = pick_info(info, picks)['chs']

    # create coil defs in head coordinates
    if ch_type == 'meg':
        # Put them in head coordinates
        coils = _create_coils(chs, FIFF.FWD_COIL_ACCURACY_NORMAL,
                              info['dev_head_t'], coil_type='meg')[0]
        type_str = 'coils'
        miss = 1e-4  # Smoothing criterion for MEG
    else:  # EEG
        coils = _create_coils(chs, coil_type='eeg')[0]
        type_str = 'electrodes'
        miss = 1e-3  # Smoothing criterion for EEG

    #
    # Step 2. Calculate the dot products
    #
    my_origin = np.array([0.0, 0.0, 0.04])
    int_rad = 0.06
    noise = _ad_hoc_noise(coils, ch_type)
    if mode == 'fast':
        # Use 50 coefficients with nearest-neighbor interpolation
        lut, n_fact = _get_legen_table(ch_type, False, 50)
        lut_fun = partial(_get_legen_lut_fast, lut=lut)
    else:  # 'accurate'
        # Use 100 coefficients with linear interpolation
        lut, n_fact = _get_legen_table(ch_type, False, 100)
        lut_fun = partial(_get_legen_lut_accurate, lut=lut)
    logger.info('Computing dot products for %i %s...' % (len(coils), type_str))
    self_dots = _do_self_dots(int_rad, False, coils, my_origin, ch_type,
                              lut_fun, n_fact, n_jobs)
    sel = np.arange(len(surf['rr']))  # eventually we should do sub-selection
    logger.info('Computing dot products for %i surface locations...'
                % len(sel))
    surface_dots = _do_surface_dots(int_rad, False, coils, surf, sel,
                                    my_origin, ch_type, lut_fun, n_fact,
                                    n_jobs)

    #
    # Step 4. Return the result
    #
    ch_names = [c['ch_name'] for c in chs]
    fmd = dict(kind=ch_type, surf=surf, ch_names=ch_names, coils=coils,
               origin=my_origin, noise=noise, self_dots=self_dots,
               surface_dots=surface_dots, int_rad=int_rad, miss=miss)
    logger.info('Field mapping data ready')

    fmd['data'] = _compute_mapping_matrix(fmd, info)

    # Remove some unecessary fields
    del fmd['self_dots']
    del fmd['surface_dots']
    del fmd['int_rad']
    del fmd['miss']
    return fmd


def make_field_map(evoked, trans_fname='auto', subject=None, subjects_dir=None,
                   ch_type=None, mode='fast', n_jobs=1):
    """Compute surface maps used for field display in 3D

    Parameters
    ----------
    evoked : Evoked | Epochs | Raw
        The measurement file. Need to have info attribute.
    trans_fname : str | 'auto' | None
        The full path to the `*-trans.fif` file produced during
        coregistration. If present or found using 'auto'
        the maps will be in MRI coordinates.
        If None, map for EEG data will not be available.
    subject : str | None
        The subject name corresponding to FreeSurfer environment
        variable SUBJECT. If None, map for EEG data will not be available.
    subjects_dir : str
        The path to the freesurfer subjects reconstructions.
        It corresponds to Freesurfer environment variable SUBJECTS_DIR.
    ch_type : None | 'eeg' | 'meg'
        If None, a map for each available channel type will be returned.
        Else only the specified type will be used.
    mode : str
        Either `'accurate'` or `'fast'`, determines the quality of the
        Legendre polynomial expansion used. `'fast'` should be sufficient
        for most applications.
    n_jobs : int
        The number of jobs to run in parallel.

    Returns
    -------
    surf_maps : list
        The surface maps to be used for field plots. The list contains
        separate ones for MEG and EEG (if both MEG and EEG are present).
    """
    info = evoked.info

    if ch_type is None:
        types = [t for t in ['eeg', 'meg'] if t in evoked]
    else:
        if ch_type not in ['eeg', 'meg']:
            raise ValueError("ch_type should be 'eeg' or 'meg' (got %s)"
                             % ch_type)
        types = [ch_type]

    if trans_fname == 'auto':
        # let's try to do this in MRI coordinates so they're easy to plot
        trans_fname = _find_trans(subject, subjects_dir)

    if 'eeg' in types and trans_fname is None:
        logger.info('No trans file available. EEG data ignored.')
        types.remove('eeg')

    if len(types) == 0:
        raise RuntimeError('No data available for mapping.')

    trans = None
    if trans_fname is not None:
        trans = read_trans(trans_fname)

    surfs = []
    for this_type in types:
        if this_type == 'meg':
            surf = get_meg_helmet_surf(info, trans)
        else:
            surf = get_head_surf(subject, subjects_dir=subjects_dir)
        surfs.append(surf)

    surf_maps = list()

    for this_type, this_surf in zip(types, surfs):
        this_map = _make_surface_mapping(evoked.info, this_surf, this_type,
                                         trans, n_jobs=n_jobs)
        this_map['surf'] = this_surf  # XXX : a bit weird...
        surf_maps.append(this_map)

    return surf_maps
