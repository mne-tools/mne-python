# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larsoner@uw.edu>
#
# License: BSD (3-clause)

from ..externals.six import string_types
import os
from os import path as op
import numpy as np

from .. import pick_types, pick_info
from ..io.pick import _has_kit_refs
from ..io import read_info
from ..io.constants import FIFF
from .forward import Forward, write_forward_solution, _merge_meg_eeg_fwds
from ._compute_forward import _compute_forwards
from ..transforms import (invert_transform, transform_surface_to,
                          read_trans, _get_mri_head_t_from_trans_file,
                          apply_trans, _print_coord_trans, _coord_frame_name)
from ..utils import logger, verbose
from ..source_space import (read_source_spaces, _filter_source_spaces,
                            SourceSpaces)
from ..surface import read_bem_solution, _normalize_vectors


def _read_coil_defs(fname=None):
    """Read a coil definition file"""
    if fname is None:
        fname = op.join(op.split(__file__)[0], '..', 'data', 'coil_def.dat')
    big_val = 0.5
    with open(fname, 'r') as fid:
        lines = fid.readlines()
        res = dict(coils=list())
        lines = lines[::-1]
        while len(lines) > 0:
            line = lines.pop()
            if line[0] != '#':
                vals = np.fromstring(line, sep=' ')
                assert len(vals) == 7
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
                res['coils'].append(coil)
    logger.info('%d coil definitions read', len(res['coils']))
    return res


def _create_meg_coil(coilset, ch, acc, t):
    """Create a coil definition using templates, transform if necessary"""
    # Also change the coordinate frame if so desired

    if ch['kind'] not in [FIFF.FIFFV_MEG_CH, FIFF.FIFFV_REF_MEG_CH]:
        raise RuntimeError('%s is not a MEG channel' % ch['ch_name'])

    # Simple linear search from the coil definitions
    d = None
    for coil in coilset['coils']:
        if coil['coil_type'] == (ch['coil_type'] & 0xFFFF) and \
                coil['accuracy'] == acc:
            d = coil

    if d is None:
        raise RuntimeError('Desired coil definition not found '
                           '(type = %d acc = %d)' % (ch['coil_type'], acc))

    # Create the result
    res = dict(chname=ch['ch_name'], desc=None, coil_class=d['coil_class'],
               accuracy=d['accuracy'], base=d['base'], size=d['size'],
               type=ch['coil_type'], w=d['w'])

    if d['desc']:
        res['desc'] = d['desc']

    # Apply a coordinate transformation if so desired
    coil_trans = ch['coil_trans'].copy()  # make sure we don't botch it
    if t is not None:
        coil_trans = np.dot(t['trans'], coil_trans)
        res['coord_frame'] = t['to']
    else:
        res['coord_frame'] = FIFF.FIFFV_COORD_DEVICE

    res['rmag'] = apply_trans(coil_trans, d['rmag'])
    res['cosmag'] = apply_trans(coil_trans, d['cosmag'], False)
    res.update(ex=coil_trans[:3, 0], ey=coil_trans[:3, 1],
               ez=coil_trans[:3, 2], r0=coil_trans[:3, 3])
    return res


def _create_eeg_el(ch, t):
    """Create an electrode definition, transform coords if necessary"""
    if ch['kind'] != FIFF.FIFFV_EEG_CH:
        raise RuntimeError('%s is not an EEG channel. Cannot create an '
                           'electrode definition.' % ch['ch_name'])
    if t is not None and t['from'] != FIFF.FIFFV_COORD_HEAD:
        raise RuntimeError('Inappropriate coordinate transformation')

    r0ex = ch['eeg_loc'][:, :2]
    if r0ex.shape[1] == 1:  # no reference
        w = np.array([1.])
    else:  # has reference
        w = np.array([1., -1.])

    # Optional coordinate transformation
    r0ex = r0ex.T.copy()
    if t is not None:
        r0ex = apply_trans(t['trans'], r0ex)
        coord_frame = t['to']
    else:
        coord_frame = FIFF.FIFFV_COORD_HEAD

    # The electrode location
    cosmag = r0ex.copy()
    _normalize_vectors(cosmag)
    res = dict(chname=ch['ch_name'], coil_class=FIFF.FWD_COILC_EEG, w=w,
               accuracy=FIFF.FWD_COIL_ACCURACY_NORMAL, type=ch['coil_type'],
               coord_frame=coord_frame, rmag=r0ex, cosmag=cosmag)
    return res


def _create_coils(chs, acc=None, t=None, coil_type='meg', coilset=None):
    """Create a set of MEG or EEG coils"""
    if coilset is None:  # auto-read defs if not supplied
        coilset = _read_coil_defs()
    coils = list()
    if coil_type == 'meg':
        for ch in chs:
            coils.append(_create_meg_coil(coilset, ch, acc, t))
    elif coil_type == 'eeg':
        for ch in chs:
            coils.append(_create_eeg_el(ch, t))
    else:
        raise RuntimeError('unknown coil type')
    return coils, coils[0]['coord_frame']  # all get the same coord_frame


@verbose
def make_forward_solution(info, mri, src, bem, fname=None, meg=True, eeg=True,
                          mindist=0.0, ignore_ref=False, overwrite=False,
                          n_jobs=1, verbose=None):
    """Calculate a forward solution for a subject

    Parameters
    ----------
    info : instance of mne.io.meas_info.Info | str
        If str, then it should be a filename to a Raw, Epochs, or Evoked
        file with measurement information. If dict, should be an info
        dict (such as one from Raw, Epochs, or Evoked).
    mri : dict | str
        Either a transformation filename (usually made using mne_analyze)
        or an info dict (usually opened using read_trans()).
        If string, an ending of `.fif` or `.fif.gz` will be assumed to
        be in FIF format, any other ending will be assumed to be a text
        file with a 4x4 transformation matrix (like the `--trans` MNE-C
        option).
    src : str | instance of SourceSpaces
        If string, should be a source space filename. Can also be an
        instance of loaded or generated SourceSpaces.
    bem : str
        Filename of the BEM (e.g., "sample-5120-5120-5120-bem-sol.fif") to
        use.
    fname : str | None
        Destination forward solution filename. If None, the solution
        will not be saved.
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
    n_jobs : int
        Number of jobs to run in parallel.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    fwd : instance of Forward
        The forward solution.

    Notes
    -----
    Some of the forward solution calculation options from the C code
    (e.g., `--grad`, `--fixed`) are not implemented here. For those,
    consider using the C command line tools or the Python wrapper
    `do_forward_solution`.
    """
    # Currently not (sup)ported:
    # 1. EEG Sphere model (not used much)
    # 2. --grad option (gradients of the field, not used much)
    # 3. --fixed option (can be computed post-hoc)
    # 4. --mricoord option (probably not necessary)

    # read the transformation from MRI to HEAD coordinates
    # (could also be HEAD to MRI)
    if isinstance(mri, string_types):
        if not op.isfile(mri):
            raise IOError('mri file "%s" not found' % mri)
        if op.splitext(mri)[1] in ['.fif', '.gz']:
            mri_head_t = read_trans(mri)
        else:
            mri_head_t = _get_mri_head_t_from_trans_file(mri)
    else:  # dict
        mri_head_t = mri
        mri = 'dict'

    if not isinstance(src, string_types):
        if not isinstance(src, SourceSpaces):
            raise TypeError('src must be a string or SourceSpaces')
        src_extra = 'list'
    else:
        src_extra = src
        if not op.isfile(src):
            raise IOError('Source space file "%s" not found' % src)
    if not op.isfile(bem):
        raise IOError('BEM file "%s" not found' % bem)
    if fname is not None and op.isfile(fname) and not overwrite:
        raise IOError('file "%s" exists, consider using overwrite=True'
                      % fname)
    if not isinstance(info, (dict, string_types)):
        raise TypeError('info should be a dict or string')
    if isinstance(info, string_types):
        info_extra = op.split(info)[1]
        info_extra_long = info
        info = read_info(info, verbose=False)
    else:
        info_extra = 'info dict'
        info_extra_long = info_extra
    arg_list = [info_extra, mri, src_extra, bem, fname,  meg, eeg,
                mindist, overwrite, n_jobs, verbose]
    cmd = 'make_forward_solution(%s)' % (', '.join([str(a) for a in arg_list]))

    # set default forward solution coordinate frame to HEAD
    # this could, in principle, be an option
    coord_frame = FIFF.FIFFV_COORD_HEAD

    # Report the setup
    mri_extra = mri if isinstance(mri, string_types) else 'dict'
    logger.info('Source space                 : %s' % src)
    logger.info('MRI -> head transform source : %s' % mri_extra)
    logger.info('Measurement data             : %s' % info_extra_long)
    logger.info('BEM model                    : %s' % bem)
    logger.info('Accurate field computations')
    logger.info('Do computations in %s coordinates',
                _coord_frame_name(coord_frame))
    logger.info('Free source orientations')
    logger.info('Destination for the solution : %s' % fname)

    # Read the source locations
    logger.info('')
    if isinstance(src, string_types):
        logger.info('Reading %s...' % src)
        src = read_source_spaces(src, verbose=False)
    else:
        # let's make a copy in case we modify something
        src = src.copy()
    nsource = sum(s['nuse'] for s in src)
    if nsource == 0:
        raise RuntimeError('No sources are active in these source spaces. '
                           '"do_all" option should be used.')
    logger.info('Read %d source spaces a total of %d active source locations'
                % (len(src), nsource))

    # Read the MRI -> head coordinate transformation
    logger.info('')

    # it's actually usually a head->MRI transform, so we probably need to
    # invert it
    if mri_head_t['from'] == FIFF.FIFFV_COORD_HEAD:
        mri_head_t = invert_transform(mri_head_t)
    if not (mri_head_t['from'] == FIFF.FIFFV_COORD_MRI and
            mri_head_t['to'] == FIFF.FIFFV_COORD_HEAD):
        raise RuntimeError('Incorrect MRI transform provided')
    _print_coord_trans(mri_head_t)

    # make a new dict with the relevant information
    mri_id = dict(machid=np.zeros(2, np.int32), version=0, secs=0, usecs=0)
    info = dict(nchan=info['nchan'], chs=info['chs'], comps=info['comps'],
                ch_names=info['ch_names'], dev_head_t=info['dev_head_t'],
                mri_file=mri_extra, mri_id=mri_id, meas_file=info_extra_long,
                meas_id=None, working_dir=os.getcwd(),
                command_line=cmd, bads=info['bads'])
    meg_head_t = info['dev_head_t']
    logger.info('')

    # MEG channels
    megnames = None
    if meg:
        picks = pick_types(info, meg=True, eeg=False, ref_meg=False,
                           exclude=[])
        nmeg = len(picks)
        if nmeg > 0:
            megchs = pick_info(info, picks)['chs']
            megnames = [info['ch_names'][p] for p in picks]
            logger.info('Read %3d MEG channels from %s'
                        % (len(picks), info_extra))

        # comp channels
        if not ignore_ref:
            picks = pick_types(info, meg=False, ref_meg=True, exclude=[])
            ncomp = len(picks)
            if (ncomp > 0):
                compchs = pick_info(info, picks)['chs']
                logger.info('Read %3d MEG compensation channels from %s'
                            % (ncomp, info_extra))
                # We need to check to make sure these are NOT KIT refs
                if _has_kit_refs(info, picks):
                    err = ('Cannot create forward solution with KIT '
                           'reference channels. Consider using '
                           '"ignore_ref=True" in calculation')
                    raise NotImplementedError(err)
            _print_coord_trans(meg_head_t)
            # make info structure to allow making compensator later
        else:
            ncomp = 0
        ncomp_data = len(info['comps'])
        ref_meg = True if not ignore_ref else False
        picks = pick_types(info, meg=True, ref_meg=ref_meg, exclude=[])
        meg_info = pick_info(info, picks)
    else:
        logger.info('MEG not requested. MEG channels omitted.')
        nmeg = 0
        meg_info = None

    # EEG channels
    eegnames = None
    if eeg:
        picks = pick_types(info, meg=False, eeg=True, ref_meg=False,
                           exclude=[])
        neeg = len(picks)
        if neeg > 0:
            eegchs = pick_info(info, picks)['chs']
            eegnames = [info['ch_names'][p] for p in picks]
            logger.info('Read %3d EEG channels from %s'
                        % (len(picks), info_extra))
    else:
        neeg = 0
        logger.info('EEG not requested. EEG channels omitted.')

    if neeg <= 0 and nmeg <= 0:
        raise RuntimeError('Could not find any MEG or EEG channels')

    # Create coil descriptions with transformation to head or MRI frame
    templates = _read_coil_defs()
    if nmeg > 0 and ncomp > 0:  # Compensation channel information
        logger.info('%d compensation data sets in %s'
                    % (ncomp_data, info_extra))

    meg_xform = meg_head_t
    extra_str = 'Head'

    megcoils, megcf, compcoils, compcf = None, None, None, None
    if nmeg > 0:
        megcoils, megcf = _create_coils(megchs,
                                        FIFF.FWD_COIL_ACCURACY_ACCURATE,
                                        meg_xform, coil_type='meg',
                                        coilset=templates)
        if ncomp > 0:
            compcoils, compcf = _create_coils(compchs,
                                              FIFF.FWD_COIL_ACCURACY_NORMAL,
                                              meg_xform, coil_type='meg',
                                              coilset=templates)
    eegels = None
    if neeg > 0:
        eegels, _ = _create_coils(eegchs, coil_type='eeg')
    logger.info('%s coordinate coil definitions created.' % extra_str)

    # Transform the source spaces into the appropriate coordinates
    # (will either be HEAD or MRI)
    for s in src:
        transform_surface_to(s, coord_frame, mri_head_t)
    logger.info('Source spaces are now in %s coordinates.'
                % _coord_frame_name(coord_frame))

    # Prepare the BEM model
    logger.info('')
    logger.info('Setting up the BEM model using %s...\n' % bem)
    bem_name = bem
    bem = read_bem_solution(bem)
    if neeg > 0 and len(bem['surfs']) == 1:
        raise RuntimeError('Cannot use a homogeneous model in EEG '
                           'calculations')
    logger.info('Employing the head->MRI coordinate transform with the '
                'BEM model.')
    # fwd_bem_set_head_mri_t: Set the coordinate transformation
    to, fro = mri_head_t['to'], mri_head_t['from']
    if fro == FIFF.FIFFV_COORD_HEAD and to == FIFF.FIFFV_COORD_MRI:
        bem['head_mri_t'] = mri_head_t
    elif fro == FIFF.FIFFV_COORD_MRI and to == FIFF.FIFFV_COORD_HEAD:
        bem['head_mri_t'] = invert_transform(mri_head_t)
    else:
        raise RuntimeError('Improper coordinate transform')
    logger.info('BEM model %s is now set up' % op.split(bem_name)[1])
    logger.info('')

    # Circumvent numerical problems by excluding points too close to the skull
    idx = np.where(np.array([s['id'] for s in bem['surfs']])
                   == FIFF.FIFFV_BEM_SURF_ID_BRAIN)[0]
    if len(idx) != 1:
        raise RuntimeError('BEM model does not have the inner skull '
                           'triangulation')
    _filter_source_spaces(bem['surfs'][idx[0]], mindist, mri_head_t, src,
                          n_jobs)
    logger.info('')

    # Time to do the heavy lifting: MEG first, then EEG
    coil_types = ['meg', 'eeg']
    coils = [megcoils, eegels]
    cfs = [megcf, None]
    ccoils = [compcoils, None]
    ccfs = [compcf, None]
    infos = [meg_info, None]
    megfwd, eegfwd = _compute_forwards(src, bem, coils, cfs, ccoils, ccfs,
                                       infos, coil_types, n_jobs)

    # merge forwards into one (creates two Forward objects)
    megfwd = _to_forward_dict(megfwd, None, megnames, coord_frame,
                              FIFF.FIFFV_MNE_FREE_ORI)
    eegfwd = _to_forward_dict(eegfwd, None, eegnames, coord_frame,
                              FIFF.FIFFV_MNE_FREE_ORI)
    fwd = _merge_meg_eeg_fwds(megfwd, eegfwd, verbose=False)
    logger.info('')

    # pick out final dict info
    picks = pick_types(info, meg=meg, eeg=eeg, ref_meg=False, exclude=[])
    info = pick_info(info, picks)
    source_rr = np.concatenate([s['rr'][s['vertno']] for s in src])
    # deal with free orientations:
    nsource = fwd['sol']['data'].shape[1] // 3
    source_nn = np.tile(np.eye(3), (nsource, 1))

    # Don't transform the source spaces back into MRI coordinates (which is
    # done in the C code) because mne-python assumes forward solution source
    # spaces are in head coords. We will delete some keys to clean up the
    # source space, though:
    for key in ['working_dir', 'command_line']:
        if key in src.info:
            del src.info[key]
    fwd.update(dict(nchan=fwd['sol']['data'].shape[0], nsource=nsource,
                    info=info, src=src, source_nn=source_nn,
                    source_rr=source_rr, surf_ori=False,
                    mri_head_t=mri_head_t))
    fwd['info']['mri_head_t'] = mri_head_t
    if fname is not None:
        logger.info('writing %s...', fname)
        write_forward_solution(fname, fwd, overwrite, verbose=False)

    logger.info('Finished.')
    return fwd


def _to_forward_dict(fwd, fwd_grad, names, coord_frame, source_ori):
    """Convert forward solution matrices to dicts"""
    if fwd is not None:
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
