# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larsoner@uw.edu>
#
# License: BSD (3-clause)

import os
from os import path as op
import numpy as np

from ..io import read_info, _loc_to_coil_trans, _loc_to_eeg_loc, Info
from ..io.pick import _has_kit_refs, pick_types, pick_info
from ..io.constants import FIFF
from ..transforms import (_ensure_trans, transform_surface_to, apply_trans,
                          _get_trans, _print_coord_trans, _coord_frame_name,
                          Transform)
from ..utils import logger, verbose, warn
from ..source_space import (_ensure_src, _filter_source_spaces,
                            _make_discrete_source_space, SourceSpaces)
from ..source_estimate import VolSourceEstimate
from ..surface import _normalize_vectors
from ..bem import read_bem_solution, _bem_find_surface, ConductorModel
from ..externals.six import string_types

from .forward import (Forward, write_forward_solution, _merge_meg_eeg_fwds,
                      convert_forward_solution)
from ._compute_forward import _compute_forwards


_accuracy_dict = dict(normal=FIFF.FWD_COIL_ACCURACY_NORMAL,
                      accurate=FIFF.FWD_COIL_ACCURACY_ACCURATE)


@verbose
def _read_coil_defs(elekta_defs=False, verbose=None):
    """Read a coil definition file.

    Parameters
    ----------
    elekta_defs : bool
        If true, prepend Elekta's coil definitions for numerical
        integration (from Abramowitz and Stegun section 25.4.62).
        Note that this will likely cause duplicate coil definitions,
        so the first matching coil should be selected for optimal
        integration parameters.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    res : list of dict
        The coils. It is a dictionary with valid keys:
        'cosmag' | 'coil_class' | 'coord_frame' | 'rmag' | 'type' |
        'chname' | 'accuracy'.
        cosmag contains the direction of the coils and rmag contains the
        position vector.
    """
    coil_dir = op.join(op.split(__file__)[0], '..', 'data')
    coils = list()
    if elekta_defs:
        coils += _read_coil_def_file(op.join(coil_dir, 'coil_def_Elekta.dat'))
    coils += _read_coil_def_file(op.join(coil_dir, 'coil_def.dat'))
    return coils


def _read_coil_def_file(fname):
    """Read a coil def file."""
    big_val = 0.5
    coils = list()
    with open(fname, 'r') as fid:
        lines = fid.readlines()
    lines = lines[::-1]
    while len(lines) > 0:
        line = lines.pop()
        if line[0] != '#':
            vals = np.fromstring(line, sep=' ')
            assert len(vals) in (6, 7)  # newer numpy can truncate comment
            start = line.find('"')
            end = len(line.strip()) - 1
            assert line.strip()[end] == '"'
            desc = line[start:end]
            npts = int(vals[3])
            coil = dict(coil_type=vals[1], coil_class=vals[0], desc=desc,
                        accuracy=vals[2], size=vals[4], base=vals[5])
            # get parameters of each component
            rmag = list()
            cosmag = list()
            w = list()
            for p in range(npts):
                # get next non-comment line
                line = lines.pop()
                while(line[0] == '#'):
                    line = lines.pop()
                vals = np.fromstring(line, sep=' ')
                assert len(vals) == 7
                # Read and verify data for each integration point
                w.append(vals[0])
                rmag.append(vals[[1, 2, 3]])
                cosmag.append(vals[[4, 5, 6]])
            w = np.array(w)
            rmag = np.array(rmag)
            cosmag = np.array(cosmag)
            size = np.sqrt(np.sum(cosmag ** 2, axis=1))
            if np.any(np.sqrt(np.sum(rmag ** 2, axis=1)) > big_val):
                raise RuntimeError('Unreasonable integration point')
            if np.any(size <= 0):
                raise RuntimeError('Unreasonable normal')
            cosmag /= size[:, np.newaxis]
            coil.update(dict(w=w, cosmag=cosmag, rmag=rmag))
            coils.append(coil)
    logger.info('%d coil definitions read', len(coils))
    return coils


def _create_meg_coil(coilset, ch, acc, do_es):
    """Create a coil definition using templates, transform if necessary."""
    # Also change the coordinate frame if so desired
    if ch['kind'] not in [FIFF.FIFFV_MEG_CH, FIFF.FIFFV_REF_MEG_CH]:
        raise RuntimeError('%s is not a MEG channel' % ch['ch_name'])

    # Simple linear search from the coil definitions
    for coil in coilset:
        if coil['coil_type'] == (ch['coil_type'] & 0xFFFF) and \
                coil['accuracy'] == acc:
            break
    else:
        raise RuntimeError('Desired coil definition not found '
                           '(type = %d acc = %d)' % (ch['coil_type'], acc))

    # Apply a coordinate transformation if so desired
    coil_trans = _loc_to_coil_trans(ch['loc'])

    # Create the result
    res = dict(chname=ch['ch_name'], coil_class=coil['coil_class'],
               accuracy=coil['accuracy'], base=coil['base'], size=coil['size'],
               type=ch['coil_type'], w=coil['w'], desc=coil['desc'],
               coord_frame=FIFF.FIFFV_COORD_DEVICE, rmag_orig=coil['rmag'],
               cosmag_orig=coil['cosmag'], coil_trans_orig=coil_trans,
               r0=coil_trans[:3, 3],
               rmag=apply_trans(coil_trans, coil['rmag']),
               cosmag=apply_trans(coil_trans, coil['cosmag'], False))
    if do_es:
        r0_exey = (np.dot(coil['rmag'][:, :2], coil_trans[:3, :2].T) +
                   coil_trans[:3, 3])
        res.update(ex=coil_trans[:3, 0], ey=coil_trans[:3, 1],
                   ez=coil_trans[:3, 2], r0_exey=r0_exey)
    return res


def _create_eeg_el(ch, t=None):
    """Create an electrode definition, transform coords if necessary."""
    if ch['kind'] != FIFF.FIFFV_EEG_CH:
        raise RuntimeError('%s is not an EEG channel. Cannot create an '
                           'electrode definition.' % ch['ch_name'])
    if t is None:
        t = Transform('head', 'head')  # identity, no change
    if t.from_str != 'head':
        raise RuntimeError('Inappropriate coordinate transformation')

    r0ex = _loc_to_eeg_loc(ch['loc'])
    if r0ex.shape[1] == 1:  # no reference
        w = np.array([1.])
    else:  # has reference
        w = np.array([1., -1.])

    # Optional coordinate transformation
    r0ex = apply_trans(t['trans'], r0ex.T)

    # The electrode location
    cosmag = r0ex.copy()
    _normalize_vectors(cosmag)
    res = dict(chname=ch['ch_name'], coil_class=FIFF.FWD_COILC_EEG, w=w,
               accuracy=_accuracy_dict['normal'], type=ch['coil_type'],
               coord_frame=t['to'], rmag=r0ex, cosmag=cosmag)
    return res


def _create_meg_coils(chs, acc, t=None, coilset=None, do_es=False):
    """Create a set of MEG coils in the head coordinate frame."""
    acc = _accuracy_dict[acc] if isinstance(acc, string_types) else acc
    coilset = _read_coil_defs(verbose=False) if coilset is None else coilset
    coils = [_create_meg_coil(coilset, ch, acc, do_es) for ch in chs]
    _transform_orig_meg_coils(coils, t, do_es=do_es)
    return coils


def _transform_orig_meg_coils(coils, t, do_es=True):
    """Transform original (device) MEG coil positions."""
    if t is None:
        return
    for coil in coils:
        coil_trans = np.dot(t['trans'], coil['coil_trans_orig'])
        coil.update(
            coord_frame=t['to'], r0=coil_trans[:3, 3],
            rmag=apply_trans(coil_trans, coil['rmag_orig']),
            cosmag=apply_trans(coil_trans, coil['cosmag_orig'], False))
        if do_es:
            r0_exey = (np.dot(coil['rmag_orig'][:, :2],
                              coil_trans[:3, :2].T) + coil_trans[:3, 3])
            coil.update(ex=coil_trans[:3, 0], ey=coil_trans[:3, 1],
                        ez=coil_trans[:3, 2], r0_exey=r0_exey)


def _create_eeg_els(chs):
    """Create a set of EEG electrodes in the head coordinate frame."""
    return [_create_eeg_el(ch) for ch in chs]


@verbose
def _setup_bem(bem, bem_extra, neeg, mri_head_t, verbose=None):
    """Set up a BEM for forward computation."""
    logger.info('')
    if isinstance(bem, string_types):
        logger.info('Setting up the BEM model using %s...\n' % bem_extra)
        bem = read_bem_solution(bem)
    if not isinstance(bem, ConductorModel):
        raise TypeError('bem must be a string or ConductorModel')
    if bem['is_sphere']:
        logger.info('Using the sphere model.\n')
        if len(bem['layers']) == 0 and neeg > 0:
            raise RuntimeError('Spherical model has zero shells, cannot use '
                               'with EEG data')
        if bem['coord_frame'] != FIFF.FIFFV_COORD_HEAD:
            raise RuntimeError('Spherical model is not in head coordinates')
    else:
        if neeg > 0 and len(bem['surfs']) == 1:
            raise RuntimeError('Cannot use a homogeneous model in EEG '
                               'calculations')
        logger.info('Employing the head->MRI coordinate transform with the '
                    'BEM model.')
        # fwd_bem_set_head_mri_t: Set the coordinate transformation
        bem['head_mri_t'] = _ensure_trans(mri_head_t, 'head', 'mri')
        logger.info('BEM model %s is now set up' % op.split(bem_extra)[1])
        logger.info('')
    return bem


@verbose
def _prep_meg_channels(info, accurate=True, exclude=(), ignore_ref=False,
                       elekta_defs=False, head_frame=True, do_es=False,
                       verbose=None):
    """Prepare MEG coil definitions for forward calculation.

    Parameters
    ----------
    info : instance of Info
        The measurement information dictionary
    accurate : bool
        If true (default) then use `accurate` coil definitions (more
        integration points)
    exclude : list of str | str
        List of channels to exclude. If 'bads', exclude channels in
        info['bads']
    ignore_ref : bool
        If true, ignore compensation coils
    elekta_defs : bool
        If True, use Elekta's coil definitions, which use different integration
        point geometry. False by default.
    head_frame : bool
        If True (default), use head frame coords. Otherwise, use device frame.
    do_es : bool
        If True, compute and store ex, ey, ez, and r0_exey.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    megcoils : list of dict
        Information for each prepped MEG coil
    compcoils : list of dict
        Information for each prepped MEG coil
    megnames : list of str
        Name of each prepped MEG coil
    meginfo : instance of Info
        Information subselected for just the set of MEG coils
    """
    accuracy = 'accurate' if accurate else 'normal'
    info_extra = 'info'
    meg_info = None
    megnames, megcoils, compcoils = [], [], []

    # Find MEG channels
    picks = pick_types(info, meg=True, eeg=False, ref_meg=False,
                       exclude=exclude)

    # Make sure MEG coils exist
    nmeg = len(picks)
    if nmeg <= 0:
        raise RuntimeError('Could not find any MEG channels')

    # Get channel info and names for MEG channels
    megchs = pick_info(info, picks)['chs']
    megnames = [info['ch_names'][p] for p in picks]
    logger.info('Read %3d MEG channels from %s'
                % (len(picks), info_extra))

    # Get MEG compensation channels
    if not ignore_ref:
        picks = pick_types(info, meg=False, ref_meg=True, exclude=exclude)
        ncomp = len(picks)
        if (ncomp > 0):
            compchs = pick_info(info, picks)['chs']
            logger.info('Read %3d MEG compensation channels from %s'
                        % (ncomp, info_extra))
            # We need to check to make sure these are NOT KIT refs
            if _has_kit_refs(info, picks):
                raise NotImplementedError(
                    'Cannot create forward solution with KIT reference '
                    'channels. Consider using "ignore_ref=True" in '
                    'calculation')
    else:
        ncomp = 0

    # Make info structure to allow making compensator later
    ncomp_data = len(info['comps'])
    ref_meg = True if not ignore_ref else False
    picks = pick_types(info, meg=True, ref_meg=ref_meg, exclude=exclude)
    meg_info = pick_info(info, picks) if nmeg > 0 else None

    # Create coil descriptions with transformation to head or device frame
    templates = _read_coil_defs(elekta_defs=elekta_defs)

    if head_frame:
        _print_coord_trans(info['dev_head_t'])
        transform = info['dev_head_t']
    else:
        transform = None

    megcoils = _create_meg_coils(megchs, accuracy, transform, templates,
                                 do_es=do_es)

    if ncomp > 0:
        logger.info('%d compensation data sets in %s' % (ncomp_data,
                                                         info_extra))
        compcoils = _create_meg_coils(compchs, 'normal', transform, templates,
                                      do_es=do_es)

    # Check that coordinate frame is correct and log it
    if head_frame:
        assert megcoils[0]['coord_frame'] == FIFF.FIFFV_COORD_HEAD
        logger.info('MEG coil definitions created in head coordinates.')
    else:
        assert megcoils[0]['coord_frame'] == FIFF.FIFFV_COORD_DEVICE
        logger.info('MEG coil definitions created in device coordinate.')

    return megcoils, compcoils, megnames, meg_info


@verbose
def _prep_eeg_channels(info, exclude=(), verbose=None):
    """Prepare EEG electrode definitions for forward calculation.

    Parameters
    ----------
    info : instance of Info
        The measurement information dictionary
    exclude : list of str | str
        List of channels to exclude. If 'bads', exclude channels in
        info['bads']
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    eegels : list of dict
        Information for each prepped EEG electrode
    eegnames : list of str
        Name of each prepped EEG electrode
    """
    eegnames, eegels = [], []
    info_extra = 'info'

    # Find EEG electrodes
    picks = pick_types(info, meg=False, eeg=True, ref_meg=False,
                       exclude=exclude)

    # Make sure EEG electrodes exist
    neeg = len(picks)
    if neeg <= 0:
        raise RuntimeError('Could not find any EEG channels')

    # Get channel info and names for EEG channels
    eegchs = pick_info(info, picks)['chs']
    eegnames = [info['ch_names'][p] for p in picks]
    logger.info('Read %3d EEG channels from %s' % (len(picks), info_extra))

    # Create EEG electrode descriptions
    eegels = _create_eeg_els(eegchs)
    logger.info('Head coordinate coil definitions created.')

    return eegels, eegnames


@verbose
def _prepare_for_forward(src, mri_head_t, info, bem, mindist, n_jobs,
                         bem_extra='', trans='', info_extra='',
                         meg=True, eeg=True, ignore_ref=False, fname=None,
                         overwrite=False, verbose=None):
    """Prepare for forward computation."""
    # Read the source locations
    logger.info('')
    # let's make a copy in case we modify something
    src = _ensure_src(src).copy()
    nsource = sum(s['nuse'] for s in src)
    if nsource == 0:
        raise RuntimeError('No sources are active in these source spaces. '
                           '"do_all" option should be used.')
    logger.info('Read %d source spaces a total of %d active source locations'
                % (len(src), nsource))
    # Delete some keys to clean up the source space:
    for key in ['working_dir', 'command_line']:
        if key in src.info:
            del src.info[key]

    # Read the MRI -> head coordinate transformation
    logger.info('')
    _print_coord_trans(mri_head_t)

    # make a new dict with the relevant information
    arg_list = [info_extra, trans, src, bem_extra, fname, meg, eeg,
                mindist, overwrite, n_jobs, verbose]
    cmd = 'make_forward_solution(%s)' % (', '.join([str(a) for a in arg_list]))
    mri_id = dict(machid=np.zeros(2, np.int32), version=0, secs=0, usecs=0)
    info = Info(chs=info['chs'], comps=info['comps'],
                dev_head_t=info['dev_head_t'], mri_file=trans, mri_id=mri_id,
                meas_file=info_extra, meas_id=None, working_dir=os.getcwd(),
                command_line=cmd, bads=info['bads'], mri_head_t=mri_head_t)
    info._update_redundant()
    info._check_consistency()
    logger.info('')

    megcoils, compcoils, megnames, meg_info = [], [], [], []
    eegels, eegnames = [], []

    if meg and len(pick_types(info, ref_meg=False, exclude=[])) > 0:
        megcoils, compcoils, megnames, meg_info = \
            _prep_meg_channels(info, ignore_ref=ignore_ref)
    if eeg and len(pick_types(info, meg=False, eeg=True, ref_meg=False,
                              exclude=[])) > 0:
        eegels, eegnames = _prep_eeg_channels(info)

    # Check that some channels were found
    if len(megcoils + eegels) == 0:
        raise RuntimeError('No MEG or EEG channels found.')

    # pick out final info
    info = pick_info(info, pick_types(info, meg=meg, eeg=eeg, ref_meg=False,
                                      exclude=[]))

    # Transform the source spaces into the appropriate coordinates
    # (will either be HEAD or MRI)
    for s in src:
        transform_surface_to(s, 'head', mri_head_t)
    logger.info('Source spaces are now in %s coordinates.'
                % _coord_frame_name(s['coord_frame']))

    # Prepare the BEM model
    bem = _setup_bem(bem, bem_extra, len(eegnames), mri_head_t)

    # Circumvent numerical problems by excluding points too close to the skull
    if not bem['is_sphere']:
        inner_skull = _bem_find_surface(bem, 'inner_skull')
        _filter_source_spaces(inner_skull, mindist, mri_head_t, src, n_jobs)
        logger.info('')

    rr = np.concatenate([s['rr'][s['vertno']] for s in src])
    if len(rr) < 1:
        raise RuntimeError('No points left in source space after excluding '
                           'points close to inner skull.')

    # deal with free orientations:
    source_nn = np.tile(np.eye(3), (len(rr), 1))
    update_kwargs = dict(nchan=len(info['ch_names']), nsource=len(rr),
                         info=info, src=src, source_nn=source_nn,
                         source_rr=rr, surf_ori=False, mri_head_t=mri_head_t)
    return megcoils, meg_info, compcoils, megnames, eegels, eegnames, rr, \
        info, update_kwargs, bem


@verbose
def make_forward_solution(info, trans, src, bem, fname=None, meg=True,
                          eeg=True, mindist=0.0, ignore_ref=False,
                          overwrite=False, n_jobs=1, verbose=None):
    """Calculate a forward solution for a subject.

    Parameters
    ----------
    info : instance of mne.Info | str
        If str, then it should be a filename to a Raw, Epochs, or Evoked
        file with measurement information. If dict, should be an info
        dict (such as one from Raw, Epochs, or Evoked).
    trans : dict | str | None
        Either a transformation filename (usually made using mne_analyze)
        or an info dict (usually opened using read_trans()).
        If string, an ending of `.fif` or `.fif.gz` will be assumed to
        be in FIF format, any other ending will be assumed to be a text
        file with a 4x4 transformation matrix (like the `--trans` MNE-C
        option). Can be None to use the identity transform.
    src : str | instance of SourceSpaces
        If string, should be a source space filename. Can also be an
        instance of loaded or generated SourceSpaces.
    bem : dict | str
        Filename of the BEM (e.g., "sample-5120-5120-5120-bem-sol.fif") to
        use, or a loaded sphere model (dict).
    fname : str | None
        Destination forward solution filename. If None, the solution
        will not be saved. Deprecated and removed in 0.15, Use
        :func:`mne.write_forward_solution` instead.
    meg : bool
        If True (Default), include MEG computations.
    eeg : bool
        If True (Default), include EEG computations.
    mindist : float
        Minimum distance of sources from inner skull surface (in mm).
    ignore_ref : bool
        If True, do not include reference channels in compensation. This
        option should be True for KIT files, since forward computation
        with reference channels is not currently supported.
    overwrite : bool
        If True, the destination file (if it exists) will be overwritten.
        If False (default), an error will be raised if the file exists.
        Deprecated and removed in 0.15. Use :func:`mne.write_forward_solution`
        instead.
    n_jobs : int
        Number of jobs to run in parallel.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    fwd : instance of Forward
        The forward solution.

    See Also
    --------
    convert_forward_solution

    Notes
    -----
    The ``--grad`` option from MNE-C (to compute gradients) is not implemented
    here.

    To create a fixed-orientation forward solution, use this function
    followed by :func:`mne.convert_forward_solution`.
    """
    # Currently not (sup)ported:
    # 1. --grad option (gradients of the field, not used much)
    # 2. --fixed option (can be computed post-hoc)
    # 3. --mricoord option (probably not necessary)

    # read the transformation from MRI to HEAD coordinates
    # (could also be HEAD to MRI)
    mri_head_t, trans = _get_trans(trans)
    bem_extra = 'dict' if isinstance(bem, dict) else bem
    if fname is not None and op.isfile(fname) and not overwrite:
        raise IOError('file "%s" exists, consider using overwrite=True'
                      % fname)
    if not isinstance(info, (Info, string_types)):
        raise TypeError('info should be an instance of Info or string')
    if isinstance(info, string_types):
        info_extra = op.split(info)[1]
        info = read_info(info, verbose=False)
    else:
        info_extra = 'instance of Info'

    # Report the setup
    logger.info('Source space                 : %s' % src)
    logger.info('MRI -> head transform source : %s' % trans)
    logger.info('Measurement data             : %s' % info_extra)
    if isinstance(bem, dict) and bem['is_sphere']:
        logger.info('Sphere model                 : origin at %s mm'
                    % (bem['r0'],))
        logger.info('Standard field computations')
    else:
        logger.info('BEM model                    : %s' % bem_extra)
        logger.info('Accurate field computations')
    logger.info('Do computations in %s coordinates',
                _coord_frame_name(FIFF.FIFFV_COORD_HEAD))
    logger.info('Free source orientations')
    logger.info('Destination for the solution : %s' % fname)

    megcoils, meg_info, compcoils, megnames, eegels, eegnames, rr, info, \
        update_kwargs, bem = _prepare_for_forward(
            src, mri_head_t, info, bem, mindist, n_jobs, bem_extra, trans,
            info_extra, meg, eeg, ignore_ref, fname, overwrite)
    del (src, mri_head_t, trans, info_extra, bem_extra, mindist,
         meg, eeg, ignore_ref)

    # Time to do the heavy lifting: MEG first, then EEG
    coil_types = ['meg', 'eeg']
    coils = [megcoils, eegels]
    ccoils = [compcoils, None]
    infos = [meg_info, None]
    megfwd, eegfwd = _compute_forwards(rr, bem, coils, ccoils,
                                       infos, coil_types, n_jobs)

    # merge forwards
    fwd = _merge_meg_eeg_fwds(_to_forward_dict(megfwd, megnames),
                              _to_forward_dict(eegfwd, eegnames),
                              verbose=False)
    logger.info('')

    # Don't transform the source spaces back into MRI coordinates (which is
    # done in the C code) because mne-python assumes forward solution source
    # spaces are in head coords.
    fwd.update(**update_kwargs)
    if fname is not None:
        logger.info('writing %s...', fname)
        warn("Parameters 'fname' and 'overwrite' are deprecated and removed in"
             "version 0.15. Use mne.write_forward_solution instead.")
        write_forward_solution(fname, fwd, overwrite, verbose=False)

    logger.info('Finished.')
    return fwd


def make_forward_dipole(dipole, bem, info, trans=None, n_jobs=1, verbose=None):
    """Convert dipole object to source estimate and calculate forward operator.

    The instance of Dipole is converted to a discrete source space,
    which is then combined with a BEM or a sphere model and
    the sensor information in info to form a forward operator.

    The source estimate object (with the forward operator) can be projected to
    sensor-space using :func:`mne.simulation.simulate_evoked`.

    .. note:: If the (unique) time points of the dipole object are unevenly
              spaced, the first output will be a list of single-timepoint
              source estimates.

    Parameters
    ----------
    dipole : instance of Dipole
        Dipole object containing position, orientation and amplitude of
        one or more dipoles. Multiple simultaneous dipoles may be defined by
        assigning them identical times.
    bem : str | dict
        The BEM filename (str) or a loaded sphere model (dict).
    info : instance of Info
        The measurement information dictionary. It is sensor-information etc.,
        e.g., from a real data file.
    trans : str | None
        The head<->MRI transform filename. Must be provided unless BEM
        is a sphere model.
    n_jobs : int
        Number of jobs to run in parallel (used in making forward solution).
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    fwd : instance of Forward
        The forward solution corresponding to the source estimate(s).
    stc : instance of VolSourceEstimate | list of VolSourceEstimate
        The dipoles converted to a discrete set of points and associated
        time courses. If the time points of the dipole are unevenly spaced,
        a list of single-timepoint source estimates are returned.

    See Also
    --------
    mne.simulation.simulate_evoked

    Notes
    -----
    .. versionadded:: 0.12.0
    """
    # Make copies to avoid mangling original dipole
    times = dipole.times.copy()
    pos = dipole.pos.copy()
    amplitude = dipole.amplitude.copy()
    ori = dipole.ori.copy()

    # Convert positions to discrete source space (allows duplicate rr & nn)
    # NB information about dipole orientation enters here, then no more
    sources = dict(rr=pos, nn=ori)
    # Dipole objects must be in the head frame
    sp = _make_discrete_source_space(sources, coord_frame='head')
    src = SourceSpaces([sp])  # dict with working_dir, command_line not nec

    # Forward operator created for channels in info (use pick_info to restrict)
    # Use defaults for most params, including min_dist
    fwd = make_forward_solution(info, trans, src, bem, fname=None,
                                n_jobs=n_jobs, verbose=verbose)
    # Convert from free orientations to fixed (in-place)
    convert_forward_solution(fwd, surf_ori=False, force_fixed=True,
                             copy=False, verbose=None)

    # Check for omissions due to proximity to inner skull in
    # make_forward_solution, which will result in an exception
    if fwd['src'][0]['nuse'] != len(pos):
        inuse = fwd['src'][0]['inuse'].astype(np.bool)
        head = ('The following dipoles are outside the inner skull boundary')
        msg = len(head) * '#' + '\n' + head + '\n'
        for (t, pos) in zip(times[np.logical_not(inuse)],
                            pos[np.logical_not(inuse)]):
            msg += '    t={:.0f} ms, pos=({:.0f}, {:.0f}, {:.0f}) mm\n'.\
                format(t * 1000., pos[0] * 1000.,
                       pos[1] * 1000., pos[2] * 1000.)
        msg += len(head) * '#'
        logger.error(msg)
        raise ValueError('One or more dipoles outside the inner skull.')

    # multiple dipoles (rr and nn) per time instant allowed
    # uneven sampling in time returns list
    timepoints = np.unique(times)
    if len(timepoints) > 1:
        tdiff = np.diff(timepoints)
        if not np.allclose(tdiff, tdiff[0]):
            warn('Unique time points of dipoles unevenly spaced: returned '
                 'stc will be a list, one for each time point.')
            tstep = -1.0
        else:
            tstep = tdiff[0]
    elif len(timepoints) == 1:
        tstep = 0.001

    # Build the data matrix, essentially a block-diagonal with
    # n_rows: number of dipoles in total (dipole.amplitudes)
    # n_cols: number of unique time points in dipole.times
    # amplitude with identical value of times go together in one col (others=0)
    data = np.zeros((len(amplitude), len(timepoints)))  # (n_d, n_t)
    row = 0
    for tpind, tp in enumerate(timepoints):
        amp = amplitude[np.in1d(times, tp)]
        data[row:row + len(amp), tpind] = amp
        row += len(amp)

    if tstep > 0:
        stc = VolSourceEstimate(data, vertices=fwd['src'][0]['vertno'],
                                tmin=timepoints[0],
                                tstep=tstep, subject=None)
    else:  # Must return a list of stc, one for each time point
        stc = []
        for col, tp in enumerate(timepoints):
            stc += [VolSourceEstimate(data[:, col][:, np.newaxis],
                                      vertices=fwd['src'][0]['vertno'],
                                      tmin=tp, tstep=0.001, subject=None)]
    return fwd, stc


def _to_forward_dict(fwd, names, fwd_grad=None,
                     coord_frame=FIFF.FIFFV_COORD_HEAD,
                     source_ori=FIFF.FIFFV_MNE_FREE_ORI):
    """Convert forward solution matrices to dicts."""
    assert names is not None
    if len(fwd) == 0:
        return None
    sol = dict(data=fwd.T, nrow=fwd.shape[1], ncol=fwd.shape[0],
               row_names=names, col_names=[])
    fwd = Forward(sol=sol, source_ori=source_ori, nsource=sol['ncol'],
                  coord_frame=coord_frame, sol_grad=None,
                  nchan=sol['nrow'], _orig_source_ori=source_ori,
                  _orig_sol=sol['data'].copy(), _orig_sol_grad=None)
    if fwd_grad is not None:
        sol_grad = dict(data=fwd_grad.T, nrow=fwd_grad.shape[1],
                        ncol=fwd_grad.shape[0], row_names=names,
                        col_names=[])
        fwd.update(dict(sol_grad=sol_grad),
                   _orig_sol_grad=sol_grad['data'].copy())
    return fwd
