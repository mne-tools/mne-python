# -*- coding: utf-8 -*-
# Authors: Matti Hämäläinen <msh@nmr.mgh.harvard.edu>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Eric Larson <larsoner@uw.edu>

# The computations in this code were primarily derived from Matti Hämäläinen's
# C code.

from copy import deepcopy

import numpy as np
from scipy import linalg

from ..io.constants import FWD
from ..bem import _check_origin
from ..io.pick import pick_types, pick_info
from ..surface import get_head_surf, get_meg_helmet_surf

from ..io.proj import _has_eeg_average_ref_proj, make_projector
from ..transforms import (transform_surface_to, read_trans, _find_trans,
                          _ensure_trans)
from ._make_forward import _create_meg_coils, _create_eeg_els, _read_coil_defs
from ._lead_dots import (_do_self_dots, _do_surface_dots, _get_legen_table,
                         _do_cross_dots)
from ..parallel import check_n_jobs
from ..utils import logger, verbose, _check_option
from ..epochs import EpochsArray, BaseEpochs
from ..evoked import Evoked, EvokedArray


def _is_axial_coil(coil):
    """Determine if the coil is axial."""
    is_ax = coil['coil_class'] in (
        FWD.COILC_MAG, FWD.COILC_AXIAL_GRAD, FWD.COILC_AXIAL_GRAD2)
    return is_ax


def _ad_hoc_noise(coils, ch_type='meg'):
    """Create ad-hoc noise covariance."""
    # XXX should de-duplicate with make_ad_hoc_cov
    v = np.empty(len(coils))
    if ch_type == 'meg':
        axs = np.array([_is_axial_coil(coil) for coil in coils], dtype=bool)
        v[axs] = 4e-28  # 20e-15 ** 2
        v[np.logical_not(axs)] = 2.5e-25  # 5e-13 ** 2
    else:
        v.fill(1e-12)  # 1e-6 ** 2
    cov = dict(diag=True, data=v, eig=None, eigvec=None)
    return cov


def _setup_dots(mode, coils, ch_type):
    """Set up dot products."""
    from scipy.interpolate import interp1d
    int_rad = 0.06
    noise = _ad_hoc_noise(coils, ch_type)
    n_coeff, interp = (50, 'nearest') if mode == 'fast' else (100, 'linear')
    lut, n_fact = _get_legen_table(ch_type, False, n_coeff, verbose=False)
    lut_fun = interp1d(np.linspace(-1, 1, lut.shape[0]), lut, interp, axis=0)
    return int_rad, noise, lut_fun, n_fact


def _compute_mapping_matrix(fmd, info):
    """Do the hairy computations."""
    logger.info('    Preparing the mapping matrix...')
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
    fmd['nest'] = np.where(sumk > (1.0 - fmd['miss']))[0][0] + 1
    logger.info('    Truncating at %d/%d components to omit less than %g '
                '(%0.2g)' % (fmd['nest'], len(sing), fmd['miss'],
                             1. - sumk[fmd['nest'] - 1]))
    sing = 1.0 / sing[:fmd['nest']]

    # Put the inverse together
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
            logger.info('    The map will have average electrode reference')
            mapping_mat -= np.mean(mapping_mat, axis=0)[np.newaxis, :]
    return mapping_mat


def _map_meg_channels(info_from, info_to, mode='fast', origin=(0., 0., 0.04)):
    """Find mapping from one set of channels to another.

    Parameters
    ----------
    info_from : instance of Info
        The measurement data to interpolate from.
    info_to : instance of Info
        The measurement info to interpolate to.
    mode : str
        Either `'accurate'` or `'fast'`, determines the quality of the
        Legendre polynomial expansion used. `'fast'` should be sufficient
        for most applications.
    origin : array-like, shape (3,) | str
        Origin of the sphere in the head coordinate frame and in meters.
        Can be ``'auto'``, which means a head-digitization-based origin
        fit. Default is ``(0., 0., 0.04)``.

    Returns
    -------
    mapping : array
        A mapping matrix of shape len(pick_to) x len(pick_from).
    """
    # no need to apply trans because both from and to coils are in device
    # coordinates
    templates = _read_coil_defs(verbose=False)
    coils_from = _create_meg_coils(info_from['chs'], 'normal',
                                   info_from['dev_head_t'], templates)
    coils_to = _create_meg_coils(info_to['chs'], 'normal',
                                 info_to['dev_head_t'], templates)
    miss = 1e-4  # Smoothing criterion for MEG
    origin = _check_origin(origin, info_from)
    #
    # Step 2. Calculate the dot products
    #
    int_rad, noise, lut_fun, n_fact = _setup_dots(mode, coils_from, 'meg')
    logger.info('    Computing dot products for %i coils...'
                % (len(coils_from)))
    self_dots = _do_self_dots(int_rad, False, coils_from, origin, 'meg',
                              lut_fun, n_fact, n_jobs=1)
    logger.info('    Computing cross products for coils %i x %i coils...'
                % (len(coils_from), len(coils_to)))
    cross_dots = _do_cross_dots(int_rad, False, coils_from, coils_to,
                                origin, 'meg', lut_fun, n_fact).T

    ch_names = [c['ch_name'] for c in info_from['chs']]
    fmd = dict(kind='meg', ch_names=ch_names,
               origin=origin, noise=noise, self_dots=self_dots,
               surface_dots=cross_dots, int_rad=int_rad, miss=miss)

    #
    # Step 3. Compute the mapping matrix
    #
    mapping = _compute_mapping_matrix(fmd, info_from)
    return mapping


def _as_meg_type_inst(inst, ch_type='grad', mode='fast'):
    """Compute virtual evoked using interpolated fields in mag/grad channels.

    Parameters
    ----------
    inst : instance of mne.Evoked or mne.Epochs
        The evoked or epochs object.
    ch_type : str
        The destination channel type. It can be 'mag' or 'grad'.
    mode : str
        Either `'accurate'` or `'fast'`, determines the quality of the
        Legendre polynomial expansion used. `'fast'` should be sufficient
        for most applications.

    Returns
    -------
    inst : instance of mne.EvokedArray or mne.EpochsArray
        The transformed evoked object containing only virtual channels.
    """
    _check_option('ch_type', ch_type, ['mag', 'grad'])

    # pick the original and destination channels
    pick_from = pick_types(inst.info, meg=True, eeg=False,
                           ref_meg=False)
    pick_to = pick_types(inst.info, meg=ch_type, eeg=False,
                         ref_meg=False)

    if len(pick_to) == 0:
        raise ValueError('No channels matching the destination channel type'
                         ' found in info. Please pass an evoked containing'
                         'both the original and destination channels. Only the'
                         ' locations of the destination channels will be used'
                         ' for interpolation.')

    info_from = pick_info(inst.info, pick_from)
    info_to = pick_info(inst.info, pick_to)
    mapping = _map_meg_channels(info_from, info_to, mode=mode)

    # compute data by multiplying by the 'gain matrix' from
    # original sensors to virtual sensors
    if hasattr(inst, 'get_data'):
        data = inst.get_data()
    else:
        data = inst.data

    ndim = data.ndim
    if ndim == 2:
        data = data[np.newaxis, :, :]

    data_ = np.empty((data.shape[0], len(mapping), data.shape[2]),
                     dtype=data.dtype)
    for d, d_ in zip(data, data_):
        d_[:] = np.dot(mapping, d[pick_from])

    # keep only the destination channel types
    info = pick_info(inst.info, sel=pick_to, copy=True)

    # change channel names to emphasize they contain interpolated data
    for ch in info['chs']:
        ch['ch_name'] += '_v'
    info._update_redundant()
    info._check_consistency()
    if isinstance(inst, Evoked):
        assert ndim == 2
        data_ = data_[0]  # undo new axis
        inst_ = EvokedArray(data_, info, tmin=inst.times[0],
                            comment=inst.comment, nave=inst.nave)
    else:
        assert isinstance(inst, BaseEpochs)
        inst_ = EpochsArray(data_, info, tmin=inst.tmin,
                            events=inst.events,
                            event_id=inst.event_id,
                            metadata=inst.metadata)

    return inst_


@verbose
def _make_surface_mapping(info, surf, ch_type='meg', trans=None, mode='fast',
                          n_jobs=1, origin=(0., 0., 0.04), verbose=None):
    """Re-map M/EEG data to a surface.

    Parameters
    ----------
    info : instance of Info
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
    %(n_jobs)s
    origin : array-like, shape (3,) | str
        Origin of the sphere in the head coordinate frame and in meters.
        The default is ``'auto'``, which means a head-digitization-based
        origin fit.
    %(verbose)s

    Returns
    -------
    mapping : array
        A n_vertices x n_sensors array that remaps the MEG or EEG data,
        as `new_data = np.dot(mapping, data)`.
    """
    if not all(key in surf for key in ['rr', 'nn']):
        raise KeyError('surf must have both "rr" and "nn"')
    if 'coord_frame' not in surf:
        raise KeyError('The surface coordinate frame must be specified '
                       'in surf["coord_frame"]')
    _check_option('mode', mode, ['accurate', 'fast'])

    # deal with coordinate frames here -- always go to "head" (easiest)
    orig_surf = surf
    surf = transform_surface_to(deepcopy(surf), 'head', trans)
    n_jobs = check_n_jobs(n_jobs)
    origin = _check_origin(origin, info)

    #
    # Step 1. Prepare the coil definitions
    # Do the dot products, assume surf in head coords
    #
    _check_option('ch_type', ch_type, ['meg', 'eeg'])
    if ch_type == 'meg':
        picks = pick_types(info, meg=True, eeg=False, ref_meg=False)
        logger.info('Prepare MEG mapping...')
    else:
        picks = pick_types(info, meg=False, eeg=True, ref_meg=False)
        logger.info('Prepare EEG mapping...')
    if len(picks) == 0:
        raise RuntimeError('cannot map, no channels found')
    # XXX this code does not do any checking for compensation channels,
    # but it seems like this must be intentional from the ref_meg=False
    # (presumably from the C code)
    chs = [info['chs'][pick] for pick in picks]

    # create coil defs in head coordinates
    if ch_type == 'meg':
        # Put them in head coordinates
        coils = _create_meg_coils(chs, 'normal', info['dev_head_t'])
        type_str = 'coils'
        miss = 1e-4  # Smoothing criterion for MEG
    else:  # EEG
        coils = _create_eeg_els(chs)
        type_str = 'electrodes'
        miss = 1e-3  # Smoothing criterion for EEG

    #
    # Step 2. Calculate the dot products
    #
    int_rad, noise, lut_fun, n_fact = _setup_dots(mode, coils, ch_type)
    logger.info('Computing dot products for %i %s...' % (len(coils), type_str))
    self_dots = _do_self_dots(int_rad, False, coils, origin, ch_type,
                              lut_fun, n_fact, n_jobs)
    sel = np.arange(len(surf['rr']))  # eventually we should do sub-selection
    logger.info('Computing dot products for %i surface locations...'
                % len(sel))
    surface_dots = _do_surface_dots(int_rad, False, coils, surf, sel,
                                    origin, ch_type, lut_fun, n_fact,
                                    n_jobs)

    #
    # Step 4. Return the result
    #
    ch_names = [c['ch_name'] for c in chs]
    fmd = dict(kind=ch_type, surf=surf, ch_names=ch_names, coils=coils,
               origin=origin, noise=noise, self_dots=self_dots,
               surface_dots=surface_dots, int_rad=int_rad, miss=miss)
    logger.info('Field mapping data ready')

    fmd['data'] = _compute_mapping_matrix(fmd, info)
    # bring the original back, whatever coord frame it was in
    fmd['surf'] = orig_surf

    # Remove some unnecessary fields
    del fmd['self_dots']
    del fmd['surface_dots']
    del fmd['int_rad']
    del fmd['miss']
    return fmd


@verbose
def make_field_map(evoked, trans='auto', subject=None, subjects_dir=None,
                   ch_type=None, mode='fast', meg_surf='helmet',
                   origin=(0., 0., 0.04), n_jobs=1, verbose=None):
    """Compute surface maps used for field display in 3D.

    Parameters
    ----------
    evoked : Evoked | Epochs | Raw
        The measurement file. Need to have info attribute.
    trans : str | 'auto' | None
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
    mode : 'accurate' | 'fast'
        Either `'accurate'` or `'fast'`, determines the quality of the
        Legendre polynomial expansion used. `'fast'` should be sufficient
        for most applications.
    meg_surf : 'helmet' | 'head'
        Should be ``'helmet'`` or ``'head'`` to specify in which surface
        to compute the MEG field map. The default value is ``'helmet'``.
    origin : array-like, shape (3,) | 'auto'
        Origin of the sphere in the head coordinate frame and in meters.
        Can be ``'auto'``, which means a head-digitization-based origin
        fit. Default is ``(0., 0., 0.04)``.

        .. versionadded:: 0.11
    %(n_jobs)s
    %(verbose)s

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
        _check_option('ch_type', ch_type, ['eeg', 'meg'])
        types = [ch_type]

    if trans == 'auto':
        # let's try to do this in MRI coordinates so they're easy to plot
        trans = _find_trans(subject, subjects_dir)

    if 'eeg' in types and trans is None:
        logger.info('No trans file available. EEG data ignored.')
        types.remove('eeg')

    if len(types) == 0:
        raise RuntimeError('No data available for mapping.')

    if trans is not None:
        if isinstance(trans, str):
            trans = read_trans(trans)
        trans = _ensure_trans(trans, 'head', 'mri')

    _check_option('meg_surf', meg_surf, ['helmet', 'head'])

    surfs = []
    for this_type in types:
        if this_type == 'meg' and meg_surf == 'helmet':
            surf = get_meg_helmet_surf(info, trans)
        else:
            surf = get_head_surf(subject, subjects_dir=subjects_dir)
        surfs.append(surf)

    surf_maps = list()

    for this_type, this_surf in zip(types, surfs):
        this_map = _make_surface_mapping(evoked.info, this_surf, this_type,
                                         trans, n_jobs=n_jobs, origin=origin,
                                         mode=mode)
        surf_maps.append(this_map)

    return surf_maps
