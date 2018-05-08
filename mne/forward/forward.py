# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

from time import time
from copy import deepcopy
import re

import numpy as np
from scipy import linalg, sparse

import shutil
import os
from os import path as op
import tempfile

from ..externals.six import string_types
from ..io import RawArray, Info
from ..io.constants import FIFF
from ..io.open import fiff_open
from ..io.tree import dir_tree_find
from ..io.tag import find_tag, read_tag
from ..io.matrix import (_read_named_matrix, _transpose_named_matrix,
                         write_named_matrix)
from ..io.meas_info import read_bad_channels, write_info
from ..io.pick import (pick_channels_forward, pick_info, pick_channels,
                       pick_types)
from ..io.write import (write_int, start_block, end_block,
                        write_coord_trans, write_ch_info, write_name_list,
                        write_string, start_file, end_file, write_id)
from ..io.base import BaseRaw
from ..evoked import Evoked, EvokedArray
from ..epochs import BaseEpochs
from ..source_space import (_read_source_spaces_from_tree,
                            find_source_space_hemi,
                            _write_source_spaces_to_fid)
from ..source_estimate import VolSourceEstimate
from ..transforms import (transform_surface_to, invert_transform,
                          write_trans)
from ..utils import (_check_fname, get_subjects_dir, has_mne_c, warn,
                     run_subprocess, check_fname, logger, verbose)
from ..label import Label


class Forward(dict):
    """Forward class to represent info from forward solution."""

    def copy(self):
        """Copy the Forward instance."""
        return Forward(deepcopy(self))

    def __repr__(self):
        """Summarize forward info instead of printing all."""
        entr = '<Forward'

        nchan = len(pick_types(self['info'], meg=True, eeg=False, exclude=[]))
        entr += ' | ' + 'MEG channels: %d' % nchan
        nchan = len(pick_types(self['info'], meg=False, eeg=True, exclude=[]))
        entr += ' | ' + 'EEG channels: %d' % nchan

        src_types = np.array([src['type'] for src in self['src']])
        if (src_types == 'surf').all():
            entr += (' | Source space: Surface with %d vertices'
                     % self['nsource'])
        elif (src_types == 'vol').all():
            entr += (' | Source space: Volume with %d grid points'
                     % self['nsource'])
        elif (src_types == 'discrete').all():
            entr += (' | Source space: Discrete with %d dipoles'
                     % self['nsource'])
        else:
            count_string = ''
            if (src_types == 'surf').any():
                count_string += '%d surface, ' % (src_types == 'surf').sum()
            if (src_types == 'vol').any():
                count_string += '%d volume, ' % (src_types == 'vol').sum()
            if (src_types == 'discrete').any():
                count_string += '%d discrete, ' \
                                % (src_types == 'discrete').sum()
            count_string = count_string.rstrip(', ')
            entr += (' | Source space: Mixed (%s) with %d vertices'
                     % (count_string, self['nsource']))

        if self['source_ori'] == FIFF.FIFFV_MNE_UNKNOWN_ORI:
            entr += (' | Source orientation: Unknown')
        elif self['source_ori'] == FIFF.FIFFV_MNE_FIXED_ORI:
            entr += (' | Source orientation: Fixed')
        elif self['source_ori'] == FIFF.FIFFV_MNE_FREE_ORI:
            entr += (' | Source orientation: Free')

        entr += '>'

        return entr


def _block_diag(A, n):
    """Construct a block diagonal from a packed structure.

    You have to try it on a matrix to see what it's doing.

    If A is not sparse, then returns a sparse block diagonal "bd",
    diagonalized from the
    elements in "A".
    "A" is ma x na, comprising bdn=(na/"n") blocks of submatrices.
    Each submatrix is ma x "n", and these submatrices are
    placed down the diagonal of the matrix.

    If A is already sparse, then the operation is reversed, yielding
    a block
    row matrix, where each set of n columns corresponds to a block element
    from the block diagonal.

    Parameters
    ----------
    A : array
        The matrix
    n : int
        The block size
    Returns
    -------
    bd : sparse matrix
        The block diagonal matrix
    """
    if sparse.issparse(A):  # then make block sparse
        raise NotImplemented('sparse reversal not implemented yet')
    ma, na = A.shape
    bdn = na // int(n)  # number of submatrices

    if na % n > 0:
        raise ValueError('Width of matrix must be a multiple of n')

    tmp = np.arange(ma * bdn, dtype=np.int).reshape(bdn, ma)
    tmp = np.tile(tmp, (1, n))
    ii = tmp.ravel()

    jj = np.arange(na, dtype=np.int)[None, :]
    jj = jj * np.ones(ma, dtype=np.int)[:, None]
    jj = jj.T.ravel()  # column indices foreach sparse bd

    bd = sparse.coo_matrix((A.T.ravel(), np.c_[ii, jj].T)).tocsc()

    return bd


def _inv_block_diag(A, n):
    """Construct an inverse block diagonal from a packed structure.

    You have to try it on a matrix to see what it's doing.

    "A" is ma x na, comprising bdn=(na/"n") blocks of submatrices.
    Each submatrix is ma x "n", and the inverses of these submatrices
    are placed down the diagonal of the matrix.

    Parameters
    ----------
    A : array
        The matrix.
    n : int
        The block size.

    Returns
    -------
    bd : sparse matrix
        The block diagonal matrix.
    """
    ma, na = A.shape
    bdn = na // int(n)  # number of submatrices

    if na % n > 0:
        raise ValueError('Width of matrix must be a multiple of n')

    # modify A in-place to invert each sub-block
    A = A.copy()
    for start in range(0, na, 3):
        # this is a view
        A[:, start:start + 3] = linalg.inv(A[:, start:start + 3])

    tmp = np.arange(ma * bdn, dtype=np.int).reshape(bdn, ma)
    tmp = np.tile(tmp, (1, n))
    ii = tmp.ravel()

    jj = np.arange(na, dtype=np.int)[None, :]
    jj = jj * np.ones(ma, dtype=np.int)[:, None]
    jj = jj.T.ravel()  # column indices foreach sparse bd

    bd = sparse.coo_matrix((A.T.ravel(), np.c_[ii, jj].T)).tocsc()

    return bd


def _get_tag_int(fid, node, name, id_):
    """Check we have an appropriate tag."""
    tag = find_tag(fid, node, id_)
    if tag is None:
        fid.close()
        raise ValueError(name + ' tag not found')
    return int(tag.data)


def _read_one(fid, node):
    """Read all interesting stuff for one forward solution."""
    # This function assumes the fid is open as a context manager
    if node is None:
        return None

    one = Forward()
    one['source_ori'] = _get_tag_int(fid, node, 'Source orientation',
                                     FIFF.FIFF_MNE_SOURCE_ORIENTATION)
    one['coord_frame'] = _get_tag_int(fid, node, 'Coordinate frame',
                                      FIFF.FIFF_MNE_COORD_FRAME)
    one['nsource'] = _get_tag_int(fid, node, 'Number of sources',
                                  FIFF.FIFF_MNE_SOURCE_SPACE_NPOINTS)
    one['nchan'] = _get_tag_int(fid, node, 'Number of channels',
                                FIFF.FIFF_NCHAN)
    try:
        one['sol'] = _read_named_matrix(fid, node,
                                        FIFF.FIFF_MNE_FORWARD_SOLUTION,
                                        transpose=True)
        one['_orig_sol'] = one['sol']['data'].copy()
    except Exception:
        logger.error('Forward solution data not found')
        raise

    try:
        fwd_type = FIFF.FIFF_MNE_FORWARD_SOLUTION_GRAD
        one['sol_grad'] = _read_named_matrix(fid, node, fwd_type,
                                             transpose=True)
        one['_orig_sol_grad'] = one['sol_grad']['data'].copy()
    except Exception:
        one['sol_grad'] = None

    if one['sol']['data'].shape[0] != one['nchan'] or \
            (one['sol']['data'].shape[1] != one['nsource'] and
             one['sol']['data'].shape[1] != 3 * one['nsource']):
        raise ValueError('Forward solution matrix has wrong dimensions')

    if one['sol_grad'] is not None:
        if one['sol_grad']['data'].shape[0] != one['nchan'] or \
                (one['sol_grad']['data'].shape[1] != 3 * one['nsource'] and
                 one['sol_grad']['data'].shape[1] != 3 * 3 * one['nsource']):
            raise ValueError('Forward solution gradient matrix has '
                             'wrong dimensions')

    return one


def _read_forward_meas_info(tree, fid):
    """Read light measurement info from forward operator.

    Parameters
    ----------
    tree : tree
        FIF tree structure.
    fid : file id
        The file id.

    Returns
    -------
    info : instance of Info
        The measurement info.
    """
    # This function assumes fid is being used as a context manager
    info = Info()

    # Information from the MRI file
    parent_mri = dir_tree_find(tree, FIFF.FIFFB_MNE_PARENT_MRI_FILE)
    if len(parent_mri) == 0:
        raise ValueError('No parent MEG information found in operator')
    parent_mri = parent_mri[0]

    tag = find_tag(fid, parent_mri, FIFF.FIFF_MNE_FILE_NAME)
    info['mri_file'] = tag.data if tag is not None else None
    tag = find_tag(fid, parent_mri, FIFF.FIFF_PARENT_FILE_ID)
    info['mri_id'] = tag.data if tag is not None else None

    # Information from the MEG file
    parent_meg = dir_tree_find(tree, FIFF.FIFFB_MNE_PARENT_MEAS_FILE)
    if len(parent_meg) == 0:
        raise ValueError('No parent MEG information found in operator')
    parent_meg = parent_meg[0]

    tag = find_tag(fid, parent_meg, FIFF.FIFF_MNE_FILE_NAME)
    info['meas_file'] = tag.data if tag is not None else None
    tag = find_tag(fid, parent_meg, FIFF.FIFF_PARENT_FILE_ID)
    info['meas_id'] = tag.data if tag is not None else None

    # Add channel information
    chs = list()
    for k in range(parent_meg['nent']):
        kind = parent_meg['directory'][k].kind
        pos = parent_meg['directory'][k].pos
        if kind == FIFF.FIFF_CH_INFO:
            tag = read_tag(fid, pos)
            chs.append(tag.data)
    info['chs'] = chs
    info._update_redundant()

    #   Get the MRI <-> head coordinate transformation
    tag = find_tag(fid, parent_mri, FIFF.FIFF_COORD_TRANS)
    coord_head = FIFF.FIFFV_COORD_HEAD
    coord_mri = FIFF.FIFFV_COORD_MRI
    coord_device = FIFF.FIFFV_COORD_DEVICE
    coord_ctf_head = FIFF.FIFFV_MNE_COORD_CTF_HEAD
    if tag is None:
        raise ValueError('MRI/head coordinate transformation not found')
    cand = tag.data
    if cand['from'] == coord_mri and cand['to'] == coord_head:
        info['mri_head_t'] = cand
    else:
        raise ValueError('MRI/head coordinate transformation not found')

    #   Get the MEG device <-> head coordinate transformation
    tag = find_tag(fid, parent_meg, FIFF.FIFF_COORD_TRANS)
    if tag is None:
        raise ValueError('MEG/head coordinate transformation not found')
    cand = tag.data
    if cand['from'] == coord_device and cand['to'] == coord_head:
        info['dev_head_t'] = cand
    elif cand['from'] == coord_ctf_head and cand['to'] == coord_head:
        info['ctf_head_t'] = cand
    else:
        raise ValueError('MEG/head coordinate transformation not found')

    info['bads'] = read_bad_channels(fid, parent_meg)
    # clean up our bad list, old versions could have non-existent bads
    info['bads'] = [bad for bad in info['bads'] if bad in info['ch_names']]

    # Check if a custom reference has been applied
    tag = find_tag(fid, parent_mri, FIFF.FIFF_MNE_CUSTOM_REF)
    if tag is None:
        tag = find_tag(fid, parent_mri, 236)  # Constant 236 used before v0.11

    info['custom_ref_applied'] = bool(tag.data) if tag is not None else False
    info._check_consistency()
    return info


def _subject_from_forward(forward):
    """Get subject id from inverse operator."""
    return forward['src'][0].get('subject_his_id', None)


@verbose
def _merge_meg_eeg_fwds(megfwd, eegfwd, verbose=None):
    """Merge loaded MEG and EEG forward dicts into one dict."""
    if megfwd is not None and eegfwd is not None:
        if (megfwd['sol']['data'].shape[1] != eegfwd['sol']['data'].shape[1] or
                megfwd['source_ori'] != eegfwd['source_ori'] or
                megfwd['nsource'] != eegfwd['nsource'] or
                megfwd['coord_frame'] != eegfwd['coord_frame']):
            raise ValueError('The MEG and EEG forward solutions do not match')

        fwd = megfwd
        fwd['sol']['data'] = np.r_[fwd['sol']['data'], eegfwd['sol']['data']]
        fwd['_orig_sol'] = np.r_[fwd['_orig_sol'], eegfwd['_orig_sol']]
        fwd['sol']['nrow'] = fwd['sol']['nrow'] + eegfwd['sol']['nrow']

        fwd['sol']['row_names'] = (fwd['sol']['row_names'] +
                                   eegfwd['sol']['row_names'])
        if fwd['sol_grad'] is not None:
            fwd['sol_grad']['data'] = np.r_[fwd['sol_grad']['data'],
                                            eegfwd['sol_grad']['data']]
            fwd['_orig_sol_grad'] = np.r_[fwd['_orig_sol_grad'],
                                          eegfwd['_orig_sol_grad']]
            fwd['sol_grad']['nrow'] = (fwd['sol_grad']['nrow'] +
                                       eegfwd['sol_grad']['nrow'])
            fwd['sol_grad']['row_names'] = (fwd['sol_grad']['row_names'] +
                                            eegfwd['sol_grad']['row_names'])

        fwd['nchan'] = fwd['nchan'] + eegfwd['nchan']
        logger.info('    MEG and EEG forward solutions combined')
    elif megfwd is not None:
        fwd = megfwd
    else:
        fwd = eegfwd
    return fwd


@verbose
def read_forward_solution(fname, include=(), exclude=(), verbose=None):
    """Read a forward solution a.k.a. lead field.

    Parameters
    ----------
    fname : string
        The file name, which should end with -fwd.fif or -fwd.fif.gz.
    include : list, optional
        List of names of channels to include. If empty all channels
        are included.
    exclude : list, optional
        List of names of channels to exclude. If empty include all
        channels.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    fwd : instance of Forward
        The forward solution.

    See Also
    --------
    write_forward_solution, make_forward_solution

    Notes
    -----
    Forward solutions, which are derived from an original forward solution with
    free orientation, are always stored on disk as forward solution with free
    orientation in X/Y/Z RAS coordinates. To apply any transformation to the
    forward operator (surface orientation, fixed orienation) please apply
    :func:`convert_forward_solution` after reading the forward solution with
    :func:`read_forward_solution`.

    Forward solutions, which are derived from an original forward solution with
    fixed orientation, are stored on disk as forward solution with fixed
    surface-based orientations. Please note that the transformation to
    surface-based, fixed orienation cannot be reverted after loading the
    forward solution with :func:`read_forward_solution`.
    """
    check_fname(fname, 'forward', ('-fwd.fif', '-fwd.fif.gz',
                                   '_fwd.fif', '_fwd.fif.gz'))

    #   Open the file, create directory
    logger.info('Reading forward solution from %s...' % fname)
    f, tree, _ = fiff_open(fname)
    with f as fid:
        #   Find all forward solutions
        fwds = dir_tree_find(tree, FIFF.FIFFB_MNE_FORWARD_SOLUTION)
        if len(fwds) == 0:
            raise ValueError('No forward solutions in %s' % fname)

        #   Parent MRI data
        parent_mri = dir_tree_find(tree, FIFF.FIFFB_MNE_PARENT_MRI_FILE)
        if len(parent_mri) == 0:
            raise ValueError('No parent MRI information in %s' % fname)
        parent_mri = parent_mri[0]

        src = _read_source_spaces_from_tree(fid, tree, patch_stats=False)
        for s in src:
            s['id'] = find_source_space_hemi(s)

        fwd = None

        #   Locate and read the forward solutions
        megnode = None
        eegnode = None
        for k in range(len(fwds)):
            tag = find_tag(fid, fwds[k], FIFF.FIFF_MNE_INCLUDED_METHODS)
            if tag is None:
                raise ValueError('Methods not listed for one of the forward '
                                 'solutions')

            if tag.data == FIFF.FIFFV_MNE_MEG:
                megnode = fwds[k]
            elif tag.data == FIFF.FIFFV_MNE_EEG:
                eegnode = fwds[k]

        megfwd = _read_one(fid, megnode)
        if megfwd is not None:
            if is_fixed_orient(megfwd):
                ori = 'fixed'
            else:
                ori = 'free'
            logger.info('    Read MEG forward solution (%d sources, '
                        '%d channels, %s orientations)'
                        % (megfwd['nsource'], megfwd['nchan'], ori))

        eegfwd = _read_one(fid, eegnode)
        if eegfwd is not None:
            if is_fixed_orient(eegfwd):
                ori = 'fixed'
            else:
                ori = 'free'
            logger.info('    Read EEG forward solution (%d sources, '
                        '%d channels, %s orientations)'
                        % (eegfwd['nsource'], eegfwd['nchan'], ori))

        fwd = _merge_meg_eeg_fwds(megfwd, eegfwd)

        #   Get the MRI <-> head coordinate transformation
        tag = find_tag(fid, parent_mri, FIFF.FIFF_COORD_TRANS)
        if tag is None:
            raise ValueError('MRI/head coordinate transformation not found')
        mri_head_t = tag.data
        if (mri_head_t['from'] != FIFF.FIFFV_COORD_MRI or
                mri_head_t['to'] != FIFF.FIFFV_COORD_HEAD):
            mri_head_t = invert_transform(mri_head_t)
            if (mri_head_t['from'] != FIFF.FIFFV_COORD_MRI or
                    mri_head_t['to'] != FIFF.FIFFV_COORD_HEAD):
                fid.close()
                raise ValueError('MRI/head coordinate transformation not '
                                 'found')
        fwd['mri_head_t'] = mri_head_t

        #
        # get parent MEG info
        #
        fwd['info'] = _read_forward_meas_info(tree, fid)

        # remove compensation matrcies.
        if len(fwd['info'].get('comps', [])) > 0:
            fwd['info']['comps'] = []
            warn('Removing compensation matrices found in measurement info of '
                 'forward operator. This should not affect application of the '
                 'forward model but it will change the model if it\'s written '
                 'back to disk', UserWarning)

        # MNE environment
        parent_env = dir_tree_find(tree, FIFF.FIFFB_MNE_ENV)
        if len(parent_env) > 0:
            parent_env = parent_env[0]
            tag = find_tag(fid, parent_env, FIFF.FIFF_MNE_ENV_WORKING_DIR)
            if tag is not None:
                fwd['info']['working_dir'] = tag.data
            tag = find_tag(fid, parent_env, FIFF.FIFF_MNE_ENV_COMMAND_LINE)
            if tag is not None:
                fwd['info']['command_line'] = tag.data

    #   Transform the source spaces to the correct coordinate frame
    #   if necessary

    # Make sure forward solution is in either the MRI or HEAD coordinate frame
    if fwd['coord_frame'] not in (FIFF.FIFFV_COORD_MRI, FIFF.FIFFV_COORD_HEAD):
        raise ValueError('Only forward solutions computed in MRI or head '
                         'coordinates are acceptable')

    # Transform each source space to the HEAD or MRI coordinate frame,
    # depending on the coordinate frame of the forward solution
    # NOTE: the function transform_surface_to will also work on discrete and
    # volume sources
    nuse = 0
    for s in src:
        try:
            s = transform_surface_to(s, fwd['coord_frame'], mri_head_t)
        except Exception as inst:
            raise ValueError('Could not transform source space (%s)' % inst)

        nuse += s['nuse']

    # Make sure the number of sources match after transformation
    if nuse != fwd['nsource']:
        raise ValueError('Source spaces do not match the forward solution.')

    logger.info('    Source spaces transformed to the forward solution '
                'coordinate frame')
    fwd['src'] = src

    #   Handle the source locations and orientations
    fwd['source_rr'] = np.concatenate([ss['rr'][ss['vertno'], :]
                                       for ss in src], axis=0)

    #   Store original source orientations
    fwd['_orig_source_ori'] = fwd['source_ori']

    #   Deal with include and exclude
    fwd = pick_channels_forward(fwd, include=include, exclude=exclude)

    if is_fixed_orient(fwd, orig=True):
        fwd['source_nn'] = np.concatenate([_src['nn'][_src['vertno'], :]
                                          for _src in fwd['src']], axis=0)
        fwd['source_ori'] = FIFF.FIFFV_MNE_FIXED_ORI
        fwd['surf_ori'] = True
    else:
        fwd['source_nn'] = np.kron(np.ones((fwd['nsource'], 1)), np.eye(3))
        fwd['source_ori'] = FIFF.FIFFV_MNE_FREE_ORI
        fwd['surf_ori'] = False
    return Forward(fwd)


@verbose
def convert_forward_solution(fwd, surf_ori=False, force_fixed=False,
                             copy=True, use_cps=None, verbose=None):
    """Convert forward solution between different source orientations.

    Parameters
    ----------
    fwd : Forward
        The forward solution to modify.
    surf_ori : bool, optional (default False)
        Use surface-based source coordinate system? Note that force_fixed=True
        implies surf_ori=True.
    force_fixed : bool, optional (default False)
        Force fixed source orientation mode?
    copy : bool
        Whether to return a new instance or modify in place.
    use_cps : None | bool (default None)
        Whether to use cortical patch statistics to define normal
        orientations. Only used when surf_ori and/or force_fixed are True.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    fwd : Forward
        The modified forward solution.
    """
    if use_cps is None:
        if force_fixed:
            use_cps = False
            warn('The default settings controlling the application of '
                 'cortical patch statistics (cps) in the creation of forward '
                 'operators with fixed orientation will be modified in 0.16. '
                 'The cps (if available) will then be applied by default. '
                 'To avoid this warning, set use_cps explicitly to False (the '
                 'current default) or True (the new default).', FutureWarning)
        else:
            use_cps = True

    fwd = fwd.copy() if copy else fwd

    if force_fixed is True:
        surf_ori = True

    if any([src['type'] == 'vol' for src in fwd['src']]) and force_fixed:
        warn('Forward operator was generated with sources from a '
             'volume source space. Conversion to fixed orientation is not '
             'possible. Setting force_fixed to False. surf_ori is ignored for '
             'volume source spaces.')
        force_fixed = False

    if surf_ori:
        if use_cps is True:
            if fwd['src'][0].get('patch_inds') is not None:
                use_ave_nn = True
                logger.info('    Average patch normals will be employed in '
                            'the rotation to the local surface coordinates..'
                            '..')
            else:
                use_ave_nn = False
                logger.info('    No patch info available. The standard source '
                            'space normals will be employed in the rotation '
                            'to the local surface coordinates....')
        else:
            use_ave_nn = False

    # We need to change these entries (only):
    # 1. source_nn
    # 2. sol['data']
    # 3. sol['ncol']
    # 4. sol_grad['data']
    # 5. sol_grad['ncol']
    # 6. source_ori

    if is_fixed_orient(fwd, orig=True) or (force_fixed and not use_ave_nn):
        # Fixed
        fwd['source_nn'] = np.concatenate([s['nn'][s['vertno'], :]
                                           for s in fwd['src']], axis=0)
        if not is_fixed_orient(fwd, orig=True):
            logger.info('    Changing to fixed-orientation forward '
                        'solution with surface-based source orientations...')
            fix_rot = _block_diag(fwd['source_nn'].T, 1)
            # newer versions of numpy require explicit casting here, so *= no
            # longer works
            fwd['sol']['data'] = (fwd['_orig_sol'] *
                                  fix_rot).astype('float32')
            fwd['sol']['ncol'] = fwd['nsource']
            if fwd['sol_grad'] is not None:
                x = sparse.block_diag([fix_rot] * 3)
                fwd['sol_grad']['data'] = fwd['_orig_sol_grad'] * x  # dot prod
                fwd['sol_grad']['ncol'] = 3 * fwd['nsource']
        fwd['source_ori'] = FIFF.FIFFV_MNE_FIXED_ORI
        fwd['surf_ori'] = True

    elif surf_ori:  # Free, surf-oriented
        #   Rotate the local source coordinate systems
        fwd['source_nn'] = np.kron(np.ones((fwd['nsource'], 1)), np.eye(3))
        logger.info('    Converting to surface-based source orientations...')
        #   Actually determine the source orientations
        pp = 0
        for s in fwd['src']:
            if s['type'] in ['surf', 'discrete']:
                for p in range(s['nuse']):
                    #  Project out the surface normal and compute SVD
                    if use_ave_nn is True:
                        nn = s['nn'][s['pinfo'][s['patch_inds'][p]], :]
                        nn = np.sum(nn, axis=0)[:, np.newaxis]
                        nn /= linalg.norm(nn)
                    else:
                        nn = s['nn'][s['vertno'][p], :][:, np.newaxis]
                    U, S, _ = linalg.svd(np.eye(3, 3) - nn * nn.T)
                    #  Make sure that ez is in the direction of nn
                    if np.sum(nn.ravel() * U[:, 2].ravel()) < 0:
                        U *= -1.0
                    fwd['source_nn'][pp:pp + 3, :] = U.T
                    pp += 3
            else:
                pp += 3 * s['nuse']

        #   Rotate the solution components as well
        if force_fixed:
            fwd['source_nn'] = fwd['source_nn'][2::3, :]
            fix_rot = _block_diag(fwd['source_nn'].T, 1)
            # newer versions of numpy require explicit casting here, so *= no
            # longer works
            fwd['sol']['data'] = (fwd['_orig_sol'] *
                                  fix_rot).astype('float32')
            fwd['sol']['ncol'] = fwd['nsource']
            if fwd['sol_grad'] is not None:
                x = sparse.block_diag([fix_rot] * 3)
                fwd['sol_grad']['data'] = fwd['_orig_sol_grad'] * x  # dot prod
                fwd['sol_grad']['ncol'] = 3 * fwd['nsource']
            fwd['source_ori'] = FIFF.FIFFV_MNE_FIXED_ORI
            fwd['surf_ori'] = True
        else:
            surf_rot = _block_diag(fwd['source_nn'].T, 3)
            fwd['sol']['data'] = fwd['_orig_sol'] * surf_rot
            fwd['sol']['ncol'] = 3 * fwd['nsource']
            if fwd['sol_grad'] is not None:
                x = sparse.block_diag([surf_rot] * 3)
                fwd['sol_grad']['data'] = fwd['_orig_sol_grad'] * x  # dot prod
                fwd['sol_grad']['ncol'] = 9 * fwd['nsource']
            fwd['source_ori'] = FIFF.FIFFV_MNE_FREE_ORI
            fwd['surf_ori'] = True

    else:  # Free, cartesian
        logger.info('    Cartesian source orientations...')
        fwd['source_nn'] = np.kron(np.ones((fwd['nsource'], 1)), np.eye(3))
        fwd['sol']['data'] = fwd['_orig_sol'].copy()
        fwd['sol']['ncol'] = 3 * fwd['nsource']
        if fwd['sol_grad'] is not None:
            fwd['sol_grad']['data'] = fwd['_orig_sol_grad'].copy()
            fwd['sol_grad']['ncol'] = 9 * fwd['nsource']
        fwd['source_ori'] = FIFF.FIFFV_MNE_FREE_ORI
        fwd['surf_ori'] = False

    logger.info('    [done]')

    return fwd


@verbose
def write_forward_solution(fname, fwd, overwrite=False, verbose=None):
    """Write forward solution to a file.

    Parameters
    ----------
    fname : str
        File name to save the forward solution to. It should end with -fwd.fif
        or -fwd.fif.gz.
    fwd : Forward
        Forward solution.
    overwrite : bool
        If True, overwrite destination file (if it exists).
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    See Also
    --------
    read_forward_solution

    Notes
    -----
    Forward solutions, which are derived from an original forward solution with
    free orientation, are always stored on disk as forward solution with free
    orientation in X/Y/Z RAS coordinates. Transformations (surface orientation,
    fixed orienation) will be reverted. To reapply any transformation to the
    forward operator please apply :func:`convert_forward_solution` after
    reading the forward solution with :func:`read_forward_solution`.

    Forward solutions, which are derived from an original forward solution with
    fixed orientation, are stored on disk as forward solution with fixed
    surface-based orientations. Please note that the transformation to
    surface-based, fixed orienation cannot be reverted after loading the
    forward solution with :func:`read_forward_solution`.
    """
    check_fname(fname, 'forward', ('-fwd.fif', '-fwd.fif.gz',
                                   '_fwd.fif', '_fwd.fif.gz'))

    # check for file existence
    _check_fname(fname, overwrite)
    fid = start_file(fname)
    start_block(fid, FIFF.FIFFB_MNE)

    #
    # MNE env
    #
    start_block(fid, FIFF.FIFFB_MNE_ENV)
    write_id(fid, FIFF.FIFF_BLOCK_ID)
    data = fwd['info'].get('working_dir', None)
    if data is not None:
        write_string(fid, FIFF.FIFF_MNE_ENV_WORKING_DIR, data)
    data = fwd['info'].get('command_line', None)
    if data is not None:
        write_string(fid, FIFF.FIFF_MNE_ENV_COMMAND_LINE, data)
    end_block(fid, FIFF.FIFFB_MNE_ENV)

    #
    # Information from the MRI file
    #
    start_block(fid, FIFF.FIFFB_MNE_PARENT_MRI_FILE)
    write_string(fid, FIFF.FIFF_MNE_FILE_NAME, fwd['info']['mri_file'])
    if fwd['info']['mri_id'] is not None:
        write_id(fid, FIFF.FIFF_PARENT_FILE_ID, fwd['info']['mri_id'])
    # store the MRI to HEAD transform in MRI file
    write_coord_trans(fid, fwd['info']['mri_head_t'])
    end_block(fid, FIFF.FIFFB_MNE_PARENT_MRI_FILE)

    # write measurement info
    write_forward_meas_info(fid, fwd['info'])

    # invert our original source space transform
    src = list()
    for s in fwd['src']:
        s = deepcopy(s)
        try:
            # returns source space to original coordinate frame
            # usually MRI
            s = transform_surface_to(s, fwd['mri_head_t']['from'],
                                     fwd['mri_head_t'])
        except Exception as inst:
            raise ValueError('Could not transform source space (%s)' % inst)
        src.append(s)

    #
    # Write the source spaces (again)
    #
    _write_source_spaces_to_fid(fid, src)
    n_vert = sum([ss['nuse'] for ss in src])
    if fwd['_orig_source_ori'] == FIFF.FIFFV_MNE_FIXED_ORI:
        n_col = n_vert
    else:
        n_col = 3 * n_vert

    # Undo transformations
    sol = fwd['_orig_sol'].copy()
    if fwd['sol_grad'] is not None:
        sol_grad = fwd['_orig_sol_grad'].copy()
    else:
        sol_grad = None

    if fwd['surf_ori'] is True:
        if fwd['_orig_source_ori'] == FIFF.FIFFV_MNE_FIXED_ORI:
            warn('The forward solution, which is stored on disk now, is based '
                 'on a forward solution with fixed orientation. Please note '
                 'that the transformation to surface-based, fixed orientation '
                 'cannot be reverted after loading the forward solution with '
                 'read_forward_solution.', RuntimeWarning)
        else:
            warn('This forward solution is based on a forward solution with '
                 'free orientation. The original forward solution is stored '
                 'on disk in X/Y/Z RAS coordinates. Any transformation '
                 '(surface orientation or fixed orientation) will be '
                 'reverted. To reapply any transformation to the forward '
                 'operator please apply convert_forward_solution after '
                 'reading the forward solution with read_forward_solution.',
                 RuntimeWarning)

    #
    # MEG forward solution
    #
    picks_meg = pick_types(fwd['info'], meg=True, eeg=False, ref_meg=False,
                           exclude=[])
    picks_eeg = pick_types(fwd['info'], meg=False, eeg=True, ref_meg=False,
                           exclude=[])
    n_meg = len(picks_meg)
    n_eeg = len(picks_eeg)
    row_names_meg = [fwd['sol']['row_names'][p] for p in picks_meg]
    row_names_eeg = [fwd['sol']['row_names'][p] for p in picks_eeg]

    if n_meg > 0:
        meg_solution = dict(data=sol[picks_meg], nrow=n_meg, ncol=n_col,
                            row_names=row_names_meg, col_names=[])
        _transpose_named_matrix(meg_solution)
        start_block(fid, FIFF.FIFFB_MNE_FORWARD_SOLUTION)
        write_int(fid, FIFF.FIFF_MNE_INCLUDED_METHODS, FIFF.FIFFV_MNE_MEG)
        write_int(fid, FIFF.FIFF_MNE_COORD_FRAME, fwd['coord_frame'])
        write_int(fid, FIFF.FIFF_MNE_SOURCE_ORIENTATION,
                  fwd['_orig_source_ori'])
        write_int(fid, FIFF.FIFF_MNE_SOURCE_SPACE_NPOINTS, n_vert)
        write_int(fid, FIFF.FIFF_NCHAN, n_meg)
        write_named_matrix(fid, FIFF.FIFF_MNE_FORWARD_SOLUTION, meg_solution)
        if sol_grad is not None:
            meg_solution_grad = dict(data=sol_grad[picks_meg],
                                     nrow=n_meg, ncol=n_col * 3,
                                     row_names=row_names_meg, col_names=[])
            _transpose_named_matrix(meg_solution_grad)
            write_named_matrix(fid, FIFF.FIFF_MNE_FORWARD_SOLUTION_GRAD,
                               meg_solution_grad)
        end_block(fid, FIFF.FIFFB_MNE_FORWARD_SOLUTION)

    #
    #  EEG forward solution
    #
    if n_eeg > 0:
        eeg_solution = dict(data=sol[picks_eeg], nrow=n_eeg, ncol=n_col,
                            row_names=row_names_eeg, col_names=[])
        _transpose_named_matrix(eeg_solution)
        start_block(fid, FIFF.FIFFB_MNE_FORWARD_SOLUTION)
        write_int(fid, FIFF.FIFF_MNE_INCLUDED_METHODS, FIFF.FIFFV_MNE_EEG)
        write_int(fid, FIFF.FIFF_MNE_COORD_FRAME, fwd['coord_frame'])
        write_int(fid, FIFF.FIFF_MNE_SOURCE_ORIENTATION,
                  fwd['_orig_source_ori'])
        write_int(fid, FIFF.FIFF_NCHAN, n_eeg)
        write_int(fid, FIFF.FIFF_MNE_SOURCE_SPACE_NPOINTS, n_vert)
        write_named_matrix(fid, FIFF.FIFF_MNE_FORWARD_SOLUTION, eeg_solution)
        if sol_grad is not None:
            eeg_solution_grad = dict(data=sol_grad[picks_eeg],
                                     nrow=n_eeg, ncol=n_col * 3,
                                     row_names=row_names_eeg, col_names=[])
            _transpose_named_matrix(eeg_solution_grad)
            write_named_matrix(fid, FIFF.FIFF_MNE_FORWARD_SOLUTION_GRAD,
                               eeg_solution_grad)
        end_block(fid, FIFF.FIFFB_MNE_FORWARD_SOLUTION)

    end_block(fid, FIFF.FIFFB_MNE)
    end_file(fid)


def is_fixed_orient(forward, orig=False):
    """Check if the forward operator is fixed orientation."""
    if orig:  # if we want to know about the original version
        fixed_ori = (forward['_orig_source_ori'] == FIFF.FIFFV_MNE_FIXED_ORI)
    else:  # most of the time we want to know about the current version
        fixed_ori = (forward['source_ori'] == FIFF.FIFFV_MNE_FIXED_ORI)
    return fixed_ori


def write_forward_meas_info(fid, info):
    """Write measurement info stored in forward solution.

    Parameters
    ----------
    fid : file id
        The file id
    info : instance of Info
        The measurement info.
    """
    info._check_consistency()
    #
    # Information from the MEG file
    #
    start_block(fid, FIFF.FIFFB_MNE_PARENT_MEAS_FILE)
    write_string(fid, FIFF.FIFF_MNE_FILE_NAME, info['meas_file'])
    if info['meas_id'] is not None:
        write_id(fid, FIFF.FIFF_PARENT_BLOCK_ID, info['meas_id'])
    # get transformation from CTF and DEVICE to HEAD coordinate frame
    meg_head_t = info.get('dev_head_t', info.get('ctf_head_t'))
    if meg_head_t is None:
        fid.close()
        raise ValueError('Head<-->sensor transform not found')
    write_coord_trans(fid, meg_head_t)

    if 'chs' in info:
        #  Channel information
        write_int(fid, FIFF.FIFF_NCHAN, len(info['chs']))
        for k, c in enumerate(info['chs']):
            #   Scan numbers may have been messed up
            c = deepcopy(c)
            c['scanno'] = k + 1
            write_ch_info(fid, c)
    if 'bads' in info and len(info['bads']) > 0:
        #   Bad channels
        start_block(fid, FIFF.FIFFB_MNE_BAD_CHANNELS)
        write_name_list(fid, FIFF.FIFF_MNE_CH_NAME_LIST, info['bads'])
        end_block(fid, FIFF.FIFFB_MNE_BAD_CHANNELS)

    end_block(fid, FIFF.FIFFB_MNE_PARENT_MEAS_FILE)


@verbose
def compute_orient_prior(forward, loose=0.2, verbose=None):
    """Compute orientation prior.

    Parameters
    ----------
    forward : dict
        Forward operator.
    loose : float in [0, 1]
        The loose orientation parameter.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    orient_prior : array
        Orientation priors.
    """
    is_fixed_ori = is_fixed_orient(forward)
    n_sources = forward['sol']['data'].shape[1]
    loose = float(loose)
    if not (0 <= loose <= 1):
        raise ValueError('loose value should be smaller than 1 and bigger '
                         'than 0, got %s.' % (loose,))
    if loose < 1 and not forward['surf_ori']:
        raise ValueError('Forward operator is not oriented in surface '
                         'coordinates. loose parameter should be 1 '
                         'not %s.' % loose)
    if is_fixed_ori and loose != 0:
        raise ValueError('loose must be 0. with forward operator '
                         'with fixed orientation.')

    orient_prior = np.ones(n_sources, dtype=np.float)
    if not is_fixed_ori and loose < 1:
        logger.info('Applying loose dipole orientations. Loose value '
                    'of %s.' % loose)
        orient_prior[np.mod(np.arange(n_sources), 3) != 2] *= loose

    return orient_prior


def _restrict_gain_matrix(G, info):
    """Restrict gain matrix entries for optimal depth weighting."""
    # Figure out which ones have been used
    if not (len(info['chs']) == G.shape[0]):
        raise ValueError("G.shape[0] and length of info['chs'] do not match: "
                         "%d != %d" % (G.shape[0], len(info['chs'])))
    sel = pick_types(info, meg='grad', ref_meg=False, exclude=[])
    if len(sel) > 0:
        G = G[sel]
        logger.info('    %d planar channels' % len(sel))
    else:
        sel = pick_types(info, meg='mag', ref_meg=False, exclude=[])
        if len(sel) > 0:
            G = G[sel]
            logger.info('    %d magnetometer or axial gradiometer '
                        'channels' % len(sel))
        else:
            sel = pick_types(info, meg=False, eeg=True, exclude=[])
            if len(sel) > 0:
                G = G[sel]
                logger.info('    %d EEG channels' % len(sel))
            else:
                warn('Could not find MEG or EEG channels')
    return G


def compute_depth_prior(G, gain_info, is_fixed_ori, exp=0.8, limit=10.0,
                        patch_areas=None, limit_depth_chs=False):
    """Compute weighting for depth prior."""
    logger.info('Creating the depth weighting matrix...')

    # If possible, pick best depth-weighting channels
    if limit_depth_chs is True:
        G = _restrict_gain_matrix(G, gain_info)

    # Compute the gain matrix
    if is_fixed_ori:
        d = np.sum(G ** 2, axis=0)
    else:
        n_pos = G.shape[1] // 3
        d = np.zeros(n_pos)
        for k in range(n_pos):
            Gk = G[:, 3 * k:3 * (k + 1)]
            d[k] = linalg.svdvals(np.dot(Gk.T, Gk))[0]

    # XXX Currently the fwd solns never have "patch_areas" defined
    if patch_areas is not None:
        d /= patch_areas ** 2
        logger.info('    Patch areas taken into account in the depth '
                    'weighting')

    w = 1.0 / d
    ws = np.sort(w)
    weight_limit = limit ** 2
    if limit_depth_chs is False:
        # match old mne-python behavor
        ind = np.argmin(ws)
        n_limit = ind
        limit = ws[ind] * weight_limit
        wpp = (np.minimum(w / limit, 1)) ** exp
    else:
        # match C code behavior
        limit = ws[-1]
        n_limit = len(d)
        if ws[-1] > weight_limit * ws[0]:
            ind = np.where(ws > weight_limit * ws[0])[0][0]
            limit = ws[ind]
            n_limit = ind

    logger.info('    limit = %d/%d = %f'
                % (n_limit + 1, len(d),
                   np.sqrt(limit / ws[0])))
    scale = 1.0 / limit
    logger.info('    scale = %g exp = %g' % (scale, exp))
    wpp = np.minimum(w / limit, 1) ** exp

    depth_prior = wpp if is_fixed_ori else np.repeat(wpp, 3)

    return depth_prior


def _stc_src_sel(src, stc):
    """Select the vertex indices of a source space using a source estimate."""
    if isinstance(stc, VolSourceEstimate):
        vertices = [stc.vertices]
    else:
        vertices = stc.vertices
    if not len(src) == len(vertices):
        raise RuntimeError('Mismatch between number of source spaces (%s) and '
                           'STC vertices (%s)' % (len(src), len(vertices)))
    src_sels = []
    offset = 0
    for s, v in zip(src, vertices):
        src_sel = np.intersect1d(s['vertno'], v)
        src_sel = np.searchsorted(s['vertno'], src_sel)
        src_sels.append(src_sel + offset)
        offset += len(s['vertno'])
    src_sel = np.concatenate(src_sels)
    return src_sel


def _fill_measurement_info(info, fwd, sfreq):
    """Fill the measurement info of a Raw or Evoked object."""
    sel = pick_channels(info['ch_names'], fwd['sol']['row_names'])
    info = pick_info(info, sel)
    info['bads'] = []

    # this is probably correct based on what's done in meas_info.py...
    info['meas_id'] = fwd['info']['meas_id']
    info['file_id'] = info['meas_id']

    now = time()
    sec = np.floor(now)
    usec = 1e6 * (now - sec)

    info['meas_date'] = np.array([sec, usec], dtype=np.int32)
    info['highpass'] = 0.0
    info['lowpass'] = sfreq / 2.0
    info['sfreq'] = sfreq
    info['projs'] = []

    return info


@verbose
def _apply_forward(fwd, stc, start=None, stop=None, verbose=None):
    """Apply forward model and return data, times, ch_names."""
    if not is_fixed_orient(fwd):
        raise ValueError('Only fixed-orientation forward operators are '
                         'supported.')

    if np.all(stc.data > 0):
        warn('Source estimate only contains currents with positive values. '
             'Use pick_ori="normal" when computing the inverse to compute '
             'currents not current magnitudes.')

    max_cur = np.max(np.abs(stc.data))
    if max_cur > 1e-7:  # 100 nAm threshold for warning
        warn('The maximum current magnitude is %0.1f nAm, which is very large.'
             ' Are you trying to apply the forward model to noise-normalized '
             '(dSPM, sLORETA, or eLORETA) values? The result will only be '
             'correct if currents (in units of Am) are used.'
             % (1e9 * max_cur))

    src_sel = _stc_src_sel(fwd['src'], stc)
    if isinstance(stc, VolSourceEstimate):
        n_src = len(stc.vertices)
    else:
        n_src = sum([len(v) for v in stc.vertices])
    if len(src_sel) != n_src:
        raise RuntimeError('Only %i of %i SourceEstimate vertices found in '
                           'fwd' % (len(src_sel), n_src))

    gain = fwd['sol']['data'][:, src_sel]

    logger.info('Projecting source estimate to sensor space...')
    data = np.dot(gain, stc.data[:, start:stop])
    logger.info('[done]')

    times = deepcopy(stc.times[start:stop])

    return data, times


@verbose
def apply_forward(fwd, stc, info, start=None, stop=None, use_cps=None,
                  verbose=None):
    """Project source space currents to sensor space using a forward operator.

    The sensor space data is computed for all channels present in fwd. Use
    pick_channels_forward or pick_types_forward to restrict the solution to a
    subset of channels.

    The function returns an Evoked object, which is constructed from
    evoked_template. The evoked_template should be from the same MEG system on
    which the original data was acquired. An exception will be raised if the
    forward operator contains channels that are not present in the template.


    Parameters
    ----------
    fwd : Forward
        Forward operator to use.
    stc : SourceEstimate
        The source estimate from which the sensor space data is computed.
    info : instance of Info
        Measurement info to generate the evoked.
    start : int, optional
        Index of first time sample (index not time is seconds).
    stop : int, optional
        Index of first time sample not to include (index not time is seconds).
    use_cps : None | bool (default None)
        Whether to use cortical patch statistics to define normal
        orientations when converting to fixed orientation (if necessary).

        .. versionadded:: 0.15
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    evoked : Evoked
        Evoked object with computed sensor space data.

    See Also
    --------
    apply_forward_raw: Compute sensor space data and return a Raw object.
    """
    # make sure evoked_template contains all channels in fwd
    for ch_name in fwd['sol']['row_names']:
        if ch_name not in info['ch_names']:
            raise ValueError('Channel %s of forward operator not present in '
                             'evoked_template.' % ch_name)

    # project the source estimate to the sensor space
    if not is_fixed_orient(fwd):
        fwd = convert_forward_solution(fwd, force_fixed=True, use_cps=use_cps)
    data, times = _apply_forward(fwd, stc, start, stop)

    # fill the measurement info
    sfreq = float(1.0 / stc.tstep)
    info_out = _fill_measurement_info(info, fwd, sfreq)

    evoked = EvokedArray(data, info_out, times[0], nave=1)

    evoked.times = times
    evoked.first = int(np.round(evoked.times[0] * sfreq))
    evoked.last = evoked.first + evoked.data.shape[1] - 1

    return evoked


@verbose
def apply_forward_raw(fwd, stc, info, start=None, stop=None,
                      verbose=None):
    """Project source space currents to sensor space using a forward operator.

    The sensor space data is computed for all channels present in fwd. Use
    pick_channels_forward or pick_types_forward to restrict the solution to a
    subset of channels.

    The function returns a Raw object, which is constructed using provided
    info. The info object should be from the same MEG system on which the
    original data was acquired. An exception will be raised if the forward
    operator contains channels that are not present in the info.

    Parameters
    ----------
    fwd : Forward
        Forward operator to use. Has to be fixed-orientation.
    stc : SourceEstimate
        The source estimate from which the sensor space data is computed.
    info : instance of Info
        The measurement info.
    start : int, optional
        Index of first time sample (index not time is seconds).
    stop : int, optional
        Index of first time sample not to include (index not time is seconds).
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    raw : Raw object
        Raw object with computed sensor space data.

    See Also
    --------
    apply_forward: Compute sensor space data and return an Evoked object.
    """
    # make sure info contains all channels in fwd
    for ch_name in fwd['sol']['row_names']:
        if ch_name not in info['ch_names']:
            raise ValueError('Channel %s of forward operator not present in '
                             'info.' % ch_name)

    # project the source estimate to the sensor space
    data, times = _apply_forward(fwd, stc, start, stop)

    sfreq = 1.0 / stc.tstep
    info = _fill_measurement_info(info, fwd, sfreq)
    info['projs'] = []
    # store sensor data in Raw object using the info
    raw = RawArray(data, info)
    raw.preload = True

    raw._first_samps = np.array([int(np.round(times[0] * sfreq))])
    raw._last_samps = np.array([raw.first_samp + raw._data.shape[1] - 1])
    raw._projector = None
    raw._update_times()
    return raw


def restrict_forward_to_stc(fwd, stc):
    """Restrict forward operator to active sources in a source estimate.

    Parameters
    ----------
    fwd : Forward
        Forward operator.
    stc : SourceEstimate
        Source estimate.

    Returns
    -------
    fwd_out : dict
        Restricted forward operator.

    See Also
    --------
    restrict_forward_to_label
    """
    fwd_out = deepcopy(fwd)
    src_sel = _stc_src_sel(fwd['src'], stc)

    fwd_out['source_rr'] = fwd['source_rr'][src_sel]
    fwd_out['nsource'] = len(src_sel)

    if is_fixed_orient(fwd):
        idx = src_sel
        if fwd['sol_grad'] is not None:
            idx_grad = (3 * src_sel[:, None] + np.arange(3)).ravel()
    else:
        idx = (3 * src_sel[:, None] + np.arange(3)).ravel()
        if fwd['sol_grad'] is not None:
            idx_grad = (9 * src_sel[:, None] + np.arange(9)).ravel()

    fwd_out['source_nn'] = fwd['source_nn'][idx]
    fwd_out['sol']['data'] = fwd['sol']['data'][:, idx]
    if fwd['sol_grad'] is not None:
        fwd_out['sol_grad']['data'] = fwd['sol_grad']['data'][:, idx_grad]
    fwd_out['sol']['ncol'] = len(idx)

    if is_fixed_orient(fwd, orig=True):
        idx = src_sel
        if fwd['sol_grad'] is not None:
            idx_grad = (3 * src_sel[:, None] + np.arange(3)).ravel()
    else:
        idx = (3 * src_sel[:, None] + np.arange(3)).ravel()
        if fwd['sol_grad'] is not None:
            idx_grad = (9 * src_sel[:, None] + np.arange(9)).ravel()

    fwd_out['_orig_sol'] = fwd['_orig_sol'][:, idx]
    if fwd['sol_grad'] is not None:
        fwd_out['_orig_sol_grad'] = fwd['_orig_sol_grad'][:, idx_grad]

    for i in range(2):
        fwd_out['src'][i]['vertno'] = stc.vertices[i]
        fwd_out['src'][i]['nuse'] = len(stc.vertices[i])
        fwd_out['src'][i]['inuse'] = fwd['src'][i]['inuse'].copy()
        fwd_out['src'][i]['inuse'].fill(0)
        fwd_out['src'][i]['inuse'][stc.vertices[i]] = 1
        fwd_out['src'][i]['use_tris'] = np.array([[]], int)
        fwd_out['src'][i]['nuse_tri'] = np.array([0])

    return fwd_out


def restrict_forward_to_label(fwd, labels):
    """Restrict forward operator to labels.

    Parameters
    ----------
    fwd : Forward
        Forward operator.
    labels : label object | list
        Label object or list of label objects.

    Returns
    -------
    fwd_out : dict
        Restricted forward operator.

    See Also
    --------
    restrict_forward_to_stc
    """
    message = 'labels must be instance of Label or a list of Label.'
    vertices = [np.array([], int), np.array([], int)]

    if not isinstance(labels, list):
        labels = [labels]

    # Get vertices separately of each hemisphere from all label
    for label in labels:
        if not isinstance(label, Label):
            raise TypeError(message + ' Instead received %s' % type(label))
        i = 0 if label.hemi == 'lh' else 1
        vertices[i] = np.append(vertices[i], label.vertices)
    # Remove duplicates and sort
    vertices = [np.unique(vert_hemi) for vert_hemi in vertices]

    fwd_out = deepcopy(fwd)
    fwd_out['source_rr'] = np.zeros((0, 3))
    fwd_out['nsource'] = 0
    fwd_out['source_nn'] = np.zeros((0, 3))
    fwd_out['sol']['data'] = np.zeros((fwd['sol']['data'].shape[0], 0))
    fwd_out['_orig_sol'] = np.zeros((fwd['_orig_sol'].shape[0], 0))
    if fwd['sol_grad'] is not None:
        fwd_out['sol_grad']['data'] = np.zeros(
            (fwd['sol_grad']['data'].shape[0], 0))
        fwd_out['_orig_sol_grad'] = np.zeros(
            (fwd['_orig_sol_grad'].shape[0], 0))
    fwd_out['sol']['ncol'] = 0
    nuse_lh = fwd['src'][0]['nuse']

    for i in range(2):
        fwd_out['src'][i]['vertno'] = np.array([], int)
        fwd_out['src'][i]['nuse'] = 0
        fwd_out['src'][i]['inuse'] = fwd['src'][i]['inuse'].copy()
        fwd_out['src'][i]['inuse'].fill(0)
        fwd_out['src'][i]['use_tris'] = np.array([[]], int)
        fwd_out['src'][i]['nuse_tri'] = np.array([0])

        # src_sel is idx to cols in fwd that are in any label per hemi
        src_sel = np.intersect1d(fwd['src'][i]['vertno'], vertices[i])
        src_sel = np.searchsorted(fwd['src'][i]['vertno'], src_sel)

        # Reconstruct each src
        vertno = fwd['src'][i]['vertno'][src_sel]
        fwd_out['src'][i]['inuse'][vertno] = 1
        fwd_out['src'][i]['nuse'] += len(vertno)
        fwd_out['src'][i]['vertno'] = np.where(fwd_out['src'][i]['inuse'])[0]

        # Reconstruct part of fwd that is not sol data
        src_sel += i * nuse_lh  # Add column shift to right hemi
        fwd_out['source_rr'] = np.vstack([fwd_out['source_rr'],
                                          fwd['source_rr'][src_sel]])
        fwd_out['nsource'] += len(src_sel)

        if is_fixed_orient(fwd):
            idx = src_sel
            if fwd['sol_grad'] is not None:
                idx_grad = (3 * src_sel[:, None] + np.arange(3)).ravel()
        else:
            idx = (3 * src_sel[:, None] + np.arange(3)).ravel()
            if fwd['sol_grad'] is not None:
                idx_grad = (9 * src_sel[:, None] + np.arange(9)).ravel()

        fwd_out['source_nn'] = np.vstack(
            [fwd_out['source_nn'], fwd['source_nn'][idx]])
        fwd_out['sol']['data'] = np.hstack(
            [fwd_out['sol']['data'], fwd['sol']['data'][:, idx]])
        if fwd['sol_grad'] is not None:
            fwd_out['sol_grad']['data'] = np.hstack(
                [fwd_out['sol_grad']['data'],
                 fwd['sol_rad']['data'][:, idx_grad]])
        fwd_out['sol']['ncol'] += len(idx)

        if is_fixed_orient(fwd, orig=True):
            idx = src_sel
            if fwd['sol_grad'] is not None:
                idx_grad = (3 * src_sel[:, None] + np.arange(3)).ravel()
        else:
            idx = (3 * src_sel[:, None] + np.arange(3)).ravel()
            if fwd['sol_grad'] is not None:
                idx_grad = (9 * src_sel[:, None] + np.arange(9)).ravel()

        fwd_out['_orig_sol'] = np.hstack(
            [fwd_out['_orig_sol'], fwd['_orig_sol'][:, idx]])
        if fwd['sol_grad'] is not None:
            fwd_out['_orig_sol_grad'] = np.hstack(
                [fwd_out['_orig_sol_grad'],
                 fwd['_orig_sol_grad'][:, idx_grad]])

    return fwd_out


def _do_forward_solution(subject, meas, fname=None, src=None, spacing=None,
                         mindist=None, bem=None, mri=None, trans=None,
                         eeg=True, meg=True, fixed=False, grad=False,
                         mricoord=False, overwrite=False, subjects_dir=None,
                         verbose=None):
    """Calculate a forward solution for a subject using MNE-C routines.

    This is kept around for testing purposes.

    This function wraps to mne_do_forward_solution, so the mne
    command-line tools must be installed and accessible from Python.

    Parameters
    ----------
    subject : str
        Name of the subject.
    meas : Raw | Epochs | Evoked | str
        If Raw or Epochs, a temporary evoked file will be created and
        saved to a temporary directory. If str, then it should be a
        filename to a file with measurement information the mne
        command-line tools can understand (i.e., raw or evoked).
    fname : str | None
        Destination forward solution filename. If None, the solution
        will be created in a temporary directory, loaded, and deleted.
    src : str | None
        Source space name. If None, the MNE default is used.
    spacing : str
        The spacing to use. Can be ``'#'`` for spacing in mm, ``'ico#'`` for a
        recursively subdivided icosahedron, or ``'oct#'`` for a recursively
        subdivided octahedron (e.g., ``spacing='ico4'``). Default is 7 mm.
    mindist : float | str | None
        Minimum distance of sources from inner skull surface (in mm).
        If None, the MNE default value is used. If string, 'all'
        indicates to include all points.
    bem : str | None
        Name of the BEM to use (e.g., "sample-5120-5120-5120"). If None
        (Default), the MNE default will be used.
    mri : str | None
        The name of the trans file in FIF format.
        If None, trans must not be None.
    trans : dict | str | None
        File name of the trans file in text format.
        If None, mri must not be None.
    eeg : bool
        If True (Default), include EEG computations.
    meg : bool
        If True (Default), include MEG computations.
    fixed : bool
        If True, make a fixed-orientation forward solution (Default:
        False). Note that fixed-orientation inverses can still be
        created from free-orientation forward solutions.
    grad : bool
        If True, compute the gradient of the field with respect to the
        dipole coordinates as well (Default: False).
    mricoord : bool
        If True, calculate in MRI coordinates (Default: False).
    overwrite : bool
        If True, the destination file (if it exists) will be overwritten.
        If False (default), an error will be raised if the file exists.
    subjects_dir : None | str
        Override the SUBJECTS_DIR environment variable.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    See Also
    --------
    make_forward_solution

    Returns
    -------
    fwd : Forward
        The generated forward solution.
    """
    if not has_mne_c():
        raise RuntimeError('mne command line tools could not be found')

    # check for file existence
    temp_dir = tempfile.mkdtemp()
    if fname is None:
        fname = op.join(temp_dir, 'temp-fwd.fif')
    _check_fname(fname, overwrite)

    if not isinstance(subject, string_types):
        raise ValueError('subject must be a string')

    # check for meas to exist as string, or try to make evoked
    if isinstance(meas, string_types):
        if not op.isfile(meas):
            raise IOError('measurement file "%s" could not be found' % meas)
    elif isinstance(meas, (BaseRaw, BaseEpochs, Evoked)):
        meas_file = op.join(temp_dir, 'info.fif')
        write_info(meas_file, meas.info)
        meas = meas_file
    else:
        raise ValueError('meas must be string, Raw, Epochs, or Evoked')

    # deal with trans/mri
    if mri is not None and trans is not None:
        raise ValueError('trans and mri cannot both be specified')
    if mri is None and trans is None:
        # MNE allows this to default to a trans/mri in the subject's dir,
        # but let's be safe here and force the user to pass us a trans/mri
        raise ValueError('Either trans or mri must be specified')

    if trans is not None:
        if not isinstance(trans, string_types):
            raise ValueError('trans must be a string')
        if not op.isfile(trans):
            raise IOError('trans file "%s" not found' % trans)
    if mri is not None:
        # deal with trans
        if not isinstance(mri, string_types):
            if isinstance(mri, dict):
                mri_data = deepcopy(mri)
                mri = op.join(temp_dir, 'mri-trans.fif')
                try:
                    write_trans(mri, mri_data)
                except Exception:
                    raise IOError('mri was a dict, but could not be '
                                  'written to disk as a transform file')
            else:
                raise ValueError('trans must be a string or dict (trans)')
        if not op.isfile(mri):
            raise IOError('trans file "%s" could not be found' % trans)

    # deal with meg/eeg
    if not meg and not eeg:
        raise ValueError('meg or eeg (or both) must be True')

    path, fname = op.split(fname)
    if not op.splitext(fname)[1] == '.fif':
        raise ValueError('Forward name does not end with .fif')
    path = op.abspath(path)

    # deal with mindist
    if mindist is not None:
        if isinstance(mindist, string_types):
            if not mindist.lower() == 'all':
                raise ValueError('mindist, if string, must be "all"')
            mindist = ['--all']
        else:
            mindist = ['--mindist', '%g' % mindist]

    # src, spacing, bem
    if src is not None:
        if not isinstance(src, string_types):
            raise ValueError('src must be a string or None')
    if spacing is not None:
        if not isinstance(spacing, string_types):
            raise ValueError('spacing must be a string or None')
    if bem is not None:
        if not isinstance(bem, string_types):
            raise ValueError('bem must be a string or None')

    # put together the actual call
    cmd = ['mne_do_forward_solution',
           '--subject', subject,
           '--meas', meas,
           '--fwd', fname,
           '--destdir', path]
    if src is not None:
        cmd += ['--src', src]
    if spacing is not None:
        if spacing.isdigit():
            pass  # spacing in mm
        else:
            # allow both "ico4" and "ico-4" style values
            match = re.match(r"(oct|ico)-?(\d+)$", spacing)
            if match is None:
                raise ValueError("Invalid spacing parameter: %r" % spacing)
            spacing = '-'.join(match.groups())
        cmd += ['--spacing', spacing]
    if mindist is not None:
        cmd += mindist
    if bem is not None:
        cmd += ['--bem', bem]
    if mri is not None:
        cmd += ['--mri', '%s' % mri]
    if trans is not None:
        cmd += ['--trans', '%s' % trans]
    if not meg:
        cmd.append('--eegonly')
    if not eeg:
        cmd.append('--megonly')
    if fixed:
        cmd.append('--fixed')
    if grad:
        cmd.append('--grad')
    if mricoord:
        cmd.append('--mricoord')
    if overwrite:
        cmd.append('--overwrite')

    env = os.environ.copy()
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)
    env['SUBJECTS_DIR'] = subjects_dir

    try:
        logger.info('Running forward solution generation command with '
                    'subjects_dir %s' % subjects_dir)
        run_subprocess(cmd, env=env)
    except Exception:
        raise
    else:
        fwd = read_forward_solution(op.join(path, fname), verbose=False)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return fwd


@verbose
def average_forward_solutions(fwds, weights=None):
    """Average forward solutions.

    Parameters
    ----------
    fwds : list of Forward
        Forward solutions to average. Each entry (dict) should be a
        forward solution.
    weights : array | None
        Weights to apply to each forward solution in averaging. If None,
        forward solutions will be equally weighted. Weights must be
        non-negative, and will be adjusted to sum to one.

    Returns
    -------
    fwd : Forward
        The averaged forward solution.
    """
    # check for fwds being a list
    if not isinstance(fwds, list):
        raise TypeError('fwds must be a list')
    if not len(fwds) > 0:
        raise ValueError('fwds must not be empty')

    # check weights
    if weights is None:
        weights = np.ones(len(fwds))
    weights = np.asanyarray(weights)  # in case it's a list, convert it
    if not np.all(weights >= 0):
        raise ValueError('weights must be non-negative')
    if not len(weights) == len(fwds):
        raise ValueError('weights must be None or the same length as fwds')
    w_sum = np.sum(weights)
    if not w_sum > 0:
        raise ValueError('weights cannot all be zero')
    weights /= w_sum

    # check our forward solutions
    for fwd in fwds:
        # check to make sure it's a forward solution
        if not isinstance(fwd, dict):
            raise TypeError('Each entry in fwds must be a dict')
        # check to make sure the dict is actually a fwd
        check_keys = ['info', 'sol_grad', 'nchan', 'src', 'source_nn', 'sol',
                      'source_rr', 'source_ori', 'surf_ori', 'coord_frame',
                      'mri_head_t', 'nsource']
        if not all(key in fwd for key in check_keys):
            raise KeyError('forward solution dict does not have all standard '
                           'entries, cannot compute average.')

    # check forward solution compatibility
    if any(fwd['sol'][k] != fwds[0]['sol'][k]
           for fwd in fwds[1:] for k in ['nrow', 'ncol']):
        raise ValueError('Forward solutions have incompatible dimensions')
    if any(fwd[k] != fwds[0][k] for fwd in fwds[1:]
           for k in ['source_ori', 'surf_ori', 'coord_frame']):
        raise ValueError('Forward solutions have incompatible orientations')

    # actually average them (solutions and gradients)
    fwd_ave = deepcopy(fwds[0])
    fwd_ave['sol']['data'] *= weights[0]
    fwd_ave['_orig_sol'] *= weights[0]
    for fwd, w in zip(fwds[1:], weights[1:]):
        fwd_ave['sol']['data'] += w * fwd['sol']['data']
        fwd_ave['_orig_sol'] += w * fwd['_orig_sol']
    if fwd_ave['sol_grad'] is not None:
        fwd_ave['sol_grad']['data'] *= weights[0]
        fwd_ave['_orig_sol_grad'] *= weights[0]
        for fwd, w in zip(fwds[1:], weights[1:]):
            fwd_ave['sol_grad']['data'] += w * fwd['sol_grad']['data']
            fwd_ave['_orig_sol_grad'] += w * fwd['_orig_sol_grad']
    return fwd_ave
