# -*- coding: utf-8 -*-
# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Matti Hämäläinen <msh@nmr.mgh.harvard.edu>
#          Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD (3-clause)

from copy import deepcopy
from math import sqrt
import numpy as np
from scipy import linalg

from ._eloreta import _compute_eloreta
from ..fixes import _safe_svd
from ..io.base import BaseRaw
from ..io.constants import FIFF
from ..io.open import fiff_open
from ..io.tag import find_tag
from ..io.matrix import (_read_named_matrix, _transpose_named_matrix,
                         write_named_matrix)
from ..io.proj import (_read_proj, make_projector, _write_proj,
                       _needs_eeg_average_ref_proj)
from ..io.tree import dir_tree_find
from ..io.write import (write_int, write_float_matrix, start_file,
                        start_block, end_block, end_file, write_float,
                        write_coord_trans, write_string)

from ..io.pick import channel_type, pick_info, pick_types, pick_channels
from ..cov import (compute_whitener, _read_cov, _write_cov, Covariance,
                   prepare_noise_cov)
from ..epochs import BaseEpochs
from ..evoked import EvokedArray, Evoked
from ..forward import (compute_depth_prior, _read_forward_meas_info,
                       is_fixed_orient, compute_orient_prior,
                       convert_forward_solution, _select_orient_forward)
from ..forward.forward import write_forward_meas_info, _triage_loose
from ..source_space import (_read_source_spaces_from_tree, _get_src_nn,
                            find_source_space_hemi, _get_vertno,
                            _write_source_spaces_to_fid, label_src_vertno_sel)
from ..surface import _normal_orth
from ..transforms import _ensure_trans, transform_surface_to
from ..source_estimate import _make_stc, _get_src_type
from ..utils import (check_fname, logger, verbose, warn, _validate_type,
                     _check_compensation_grade, _check_option,
                     _check_depth, _check_src_normal)


INVERSE_METHODS = ('MNE', 'dSPM', 'sLORETA', 'eLORETA')


class InverseOperator(dict):
    """InverseOperator class to represent info from inverse operator."""

    def copy(self):
        """Return a copy of the InverseOperator."""
        return InverseOperator(deepcopy(self))

    def __repr__(self):  # noqa: D105
        """Summarize inverse info instead of printing all."""
        entr = '<InverseOperator'

        nchan = len(pick_types(self['info'], meg=True, eeg=False))
        entr += ' | ' + 'MEG channels: %d' % nchan
        nchan = len(pick_types(self['info'], meg=False, eeg=True))
        entr += ' | ' + 'EEG channels: %d' % nchan

        entr += (' | Source space: %s with %d sources'
                 % (self['src'].kind, self['nsource']))
        source_ori = {FIFF.FIFFV_MNE_UNKNOWN_ORI: 'Unknown',
                      FIFF.FIFFV_MNE_FIXED_ORI: 'Fixed',
                      FIFF.FIFFV_MNE_FREE_ORI: 'Free'}
        entr += ' | Source orientation: %s' % source_ori[self['source_ori']]
        entr += '>'

        return entr


def _pick_channels_inverse_operator(ch_names, inv):
    """Return data channel indices to be used knowing an inverse operator.

    Unlike ``pick_channels``, this respects the order of ch_names.
    """
    sel = list()
    for name in inv['noise_cov'].ch_names:
        try:
            sel.append(ch_names.index(name))
        except ValueError:
            raise ValueError('The inverse operator was computed with '
                             'channel %s which is not present in '
                             'the data. You should compute a new inverse '
                             'operator restricted to the good data '
                             'channels.' % name)
    return sel


@verbose
def read_inverse_operator(fname, verbose=None):
    """Read the inverse operator decomposition from a FIF file.

    Parameters
    ----------
    fname : str
        The name of the FIF file, which ends with -inv.fif or -inv.fif.gz.
    %(verbose)s

    Returns
    -------
    inv : instance of InverseOperator
        The inverse operator.

    See Also
    --------
    write_inverse_operator, make_inverse_operator
    """
    check_fname(fname, 'inverse operator', ('-inv.fif', '-inv.fif.gz',
                                            '_inv.fif', '_inv.fif.gz'))

    #
    #   Open the file, create directory
    #
    logger.info('Reading inverse operator decomposition from %s...'
                % fname)
    f, tree, _ = fiff_open(fname)
    with f as fid:
        #
        #   Find all inverse operators
        #
        invs = dir_tree_find(tree, FIFF.FIFFB_MNE_INVERSE_SOLUTION)
        if invs is None or len(invs) < 1:
            raise Exception('No inverse solutions in %s' % fname)

        invs = invs[0]
        #
        #   Parent MRI data
        #
        parent_mri = dir_tree_find(tree, FIFF.FIFFB_MNE_PARENT_MRI_FILE)
        if len(parent_mri) == 0:
            raise Exception('No parent MRI information in %s' % fname)
        parent_mri = parent_mri[0]  # take only first one

        logger.info('    Reading inverse operator info...')
        #
        #   Methods and source orientations
        #
        tag = find_tag(fid, invs, FIFF.FIFF_MNE_INCLUDED_METHODS)
        if tag is None:
            raise Exception('Modalities not found')

        inv = dict()
        inv['methods'] = int(tag.data)

        tag = find_tag(fid, invs, FIFF.FIFF_MNE_SOURCE_ORIENTATION)
        if tag is None:
            raise Exception('Source orientation constraints not found')

        inv['source_ori'] = int(tag.data)

        tag = find_tag(fid, invs, FIFF.FIFF_MNE_SOURCE_SPACE_NPOINTS)
        if tag is None:
            raise Exception('Number of sources not found')

        inv['nsource'] = int(tag.data)
        inv['nchan'] = 0
        #
        #   Coordinate frame
        #
        tag = find_tag(fid, invs, FIFF.FIFF_MNE_COORD_FRAME)
        if tag is None:
            raise Exception('Coordinate frame tag not found')

        inv['coord_frame'] = tag.data

        #
        #   Units
        #
        tag = find_tag(fid, invs, FIFF.FIFF_MNE_INVERSE_SOURCE_UNIT)
        unit_dict = {FIFF.FIFF_UNIT_AM: 'Am',
                     FIFF.FIFF_UNIT_AM_M2: 'Am/m^2',
                     FIFF.FIFF_UNIT_AM_M3: 'Am/m^3'}
        inv['units'] = unit_dict.get(int(getattr(tag, 'data', -1)), None)

        #
        #   The actual source orientation vectors
        #
        tag = find_tag(fid, invs, FIFF.FIFF_MNE_INVERSE_SOURCE_ORIENTATIONS)
        if tag is None:
            raise Exception('Source orientation information not found')

        inv['source_nn'] = tag.data
        logger.info('    [done]')
        #
        #   The SVD decomposition...
        #
        logger.info('    Reading inverse operator decomposition...')
        tag = find_tag(fid, invs, FIFF.FIFF_MNE_INVERSE_SING)
        if tag is None:
            raise Exception('Singular values not found')

        inv['sing'] = tag.data
        inv['nchan'] = len(inv['sing'])
        #
        #   The eigenleads and eigenfields
        #
        inv['eigen_leads_weighted'] = False
        inv['eigen_leads'] = _read_named_matrix(
            fid, invs, FIFF.FIFF_MNE_INVERSE_LEADS, transpose=True)
        if inv['eigen_leads'] is None:
            inv['eigen_leads_weighted'] = True
            inv['eigen_leads'] = _read_named_matrix(
                fid, invs, FIFF.FIFF_MNE_INVERSE_LEADS_WEIGHTED,
                transpose=True)
        if inv['eigen_leads'] is None:
            raise ValueError('Eigen leads not found in inverse operator.')
        #
        #   Having the eigenleads as cols is better for the inverse calcs
        #
        inv['eigen_fields'] = _read_named_matrix(fid, invs,
                                                 FIFF.FIFF_MNE_INVERSE_FIELDS)
        logger.info('    [done]')
        #
        #   Read the covariance matrices
        #
        inv['noise_cov'] = Covariance(
            **_read_cov(fid, invs, FIFF.FIFFV_MNE_NOISE_COV, limited=True))
        logger.info('    Noise covariance matrix read.')

        inv['source_cov'] = _read_cov(fid, invs, FIFF.FIFFV_MNE_SOURCE_COV)
        logger.info('    Source covariance matrix read.')
        #
        #   Read the various priors
        #
        inv['orient_prior'] = _read_cov(fid, invs,
                                        FIFF.FIFFV_MNE_ORIENT_PRIOR_COV)
        if inv['orient_prior'] is not None:
            logger.info('    Orientation priors read.')

        inv['depth_prior'] = _read_cov(fid, invs,
                                       FIFF.FIFFV_MNE_DEPTH_PRIOR_COV)
        if inv['depth_prior'] is not None:
            logger.info('    Depth priors read.')

        inv['fmri_prior'] = _read_cov(fid, invs, FIFF.FIFFV_MNE_FMRI_PRIOR_COV)
        if inv['fmri_prior'] is not None:
            logger.info('    fMRI priors read.')

        #
        #   Read the source spaces
        #
        inv['src'] = _read_source_spaces_from_tree(fid, tree,
                                                   patch_stats=False)

        for s in inv['src']:
            s['id'] = find_source_space_hemi(s)

        #
        #   Get the MRI <-> head coordinate transformation
        #
        tag = find_tag(fid, parent_mri, FIFF.FIFF_COORD_TRANS)
        if tag is None:
            raise Exception('MRI/head coordinate transformation not found')
        mri_head_t = _ensure_trans(tag.data, 'mri', 'head')

        inv['mri_head_t'] = mri_head_t

        #
        # get parent MEG info
        #
        inv['info'] = _read_forward_meas_info(tree, fid)

        #
        #   Transform the source spaces to the correct coordinate frame
        #   if necessary
        #
        if inv['coord_frame'] not in (FIFF.FIFFV_COORD_MRI,
                                      FIFF.FIFFV_COORD_HEAD):
            raise Exception('Only inverse solutions computed in MRI or '
                            'head coordinates are acceptable')

        #
        #  Number of averages is initially one
        #
        inv['nave'] = 1
        #
        #  We also need the SSP operator
        #
        inv['projs'] = _read_proj(fid, tree)

        #
        #  Some empty fields to be filled in later
        #
        inv['proj'] = []       # This is the projector to apply to the data
        inv['whitener'] = []   # This whitens the data
        # This the diagonal matrix implementing regularization and the inverse
        inv['reginv'] = []
        inv['noisenorm'] = []  # These are the noise-normalization factors
        #
        nuse = 0
        for k in range(len(inv['src'])):
            try:
                inv['src'][k] = transform_surface_to(inv['src'][k],
                                                     inv['coord_frame'],
                                                     mri_head_t)
            except Exception as inst:
                raise Exception('Could not transform source space (%s)' % inst)

            nuse += inv['src'][k]['nuse']

        logger.info('    Source spaces transformed to the inverse solution '
                    'coordinate frame')
        #
        #   Done!
        #

    return InverseOperator(inv)


@verbose
def write_inverse_operator(fname, inv, verbose=None):
    """Write an inverse operator to a FIF file.

    Parameters
    ----------
    fname : str
        The name of the FIF file, which ends with -inv.fif or -inv.fif.gz.
    inv : dict
        The inverse operator.
    %(verbose)s

    See Also
    --------
    read_inverse_operator
    """
    check_fname(fname, 'inverse operator', ('-inv.fif', '-inv.fif.gz',
                                            '_inv.fif', '_inv.fif.gz'))
    _validate_type(inv, InverseOperator, 'inv')

    #
    #   Open the file, create directory
    #
    logger.info('Write inverse operator decomposition in %s...' % fname)

    # Create the file and save the essentials
    fid = start_file(fname)
    start_block(fid, FIFF.FIFFB_MNE)

    #
    #   Parent MEG measurement info
    #
    write_forward_meas_info(fid, inv['info'])

    #
    #   Parent MRI data
    #
    start_block(fid, FIFF.FIFFB_MNE_PARENT_MRI_FILE)
    write_string(fid, FIFF.FIFF_MNE_FILE_NAME, inv['info']['mri_file'])
    write_coord_trans(fid, inv['mri_head_t'])
    end_block(fid, FIFF.FIFFB_MNE_PARENT_MRI_FILE)

    #
    #   Write SSP operator
    #
    _write_proj(fid, inv['projs'])

    #
    #   Write the source spaces
    #
    if 'src' in inv:
        _write_source_spaces_to_fid(fid, inv['src'])

    start_block(fid, FIFF.FIFFB_MNE_INVERSE_SOLUTION)

    logger.info('    Writing inverse operator info...')

    write_int(fid, FIFF.FIFF_MNE_INCLUDED_METHODS, inv['methods'])
    write_int(fid, FIFF.FIFF_MNE_COORD_FRAME, inv['coord_frame'])

    udict = {'Am': FIFF.FIFF_UNIT_AM,
             'Am/m^2': FIFF.FIFF_UNIT_AM_M2,
             'Am/m^3': FIFF.FIFF_UNIT_AM_M3}
    if 'units' in inv and inv['units'] is not None:
        write_int(fid, FIFF.FIFF_MNE_INVERSE_SOURCE_UNIT, udict[inv['units']])

    write_int(fid, FIFF.FIFF_MNE_SOURCE_ORIENTATION, inv['source_ori'])
    write_int(fid, FIFF.FIFF_MNE_SOURCE_SPACE_NPOINTS, inv['nsource'])
    if 'nchan' in inv:
        write_int(fid, FIFF.FIFF_NCHAN, inv['nchan'])
    elif 'nchan' in inv['info']:
        write_int(fid, FIFF.FIFF_NCHAN, inv['info']['nchan'])
    write_float_matrix(fid, FIFF.FIFF_MNE_INVERSE_SOURCE_ORIENTATIONS,
                       inv['source_nn'])
    write_float(fid, FIFF.FIFF_MNE_INVERSE_SING, inv['sing'])

    #
    #   write the covariance matrices
    #
    logger.info('    Writing noise covariance matrix.')
    _write_cov(fid, inv['noise_cov'])

    logger.info('    Writing source covariance matrix.')
    _write_cov(fid, inv['source_cov'])

    #
    #   write the various priors
    #
    logger.info('    Writing orientation priors.')
    if inv['depth_prior'] is not None:
        _write_cov(fid, inv['depth_prior'])
    if inv['orient_prior'] is not None:
        _write_cov(fid, inv['orient_prior'])
    if inv['fmri_prior'] is not None:
        _write_cov(fid, inv['fmri_prior'])

    write_named_matrix(fid, FIFF.FIFF_MNE_INVERSE_FIELDS, inv['eigen_fields'])

    #
    #   The eigenleads and eigenfields
    #
    if inv['eigen_leads_weighted']:
        kind = FIFF.FIFF_MNE_INVERSE_LEADS_WEIGHTED
    else:
        kind = FIFF.FIFF_MNE_INVERSE_LEADS
    _transpose_named_matrix(inv['eigen_leads'])
    write_named_matrix(fid, kind, inv['eigen_leads'])
    _transpose_named_matrix(inv['eigen_leads'])

    #
    #   Done!
    #
    logger.info('    [done]')

    end_block(fid, FIFF.FIFFB_MNE_INVERSE_SOLUTION)
    end_block(fid, FIFF.FIFFB_MNE)
    end_file(fid)

    fid.close()

###############################################################################
# Compute inverse solution


def combine_xyz(vec, square=False):
    """Compute the three Cartesian components of a vector or matrix together.

    Parameters
    ----------
    vec : 2d array of shape [3 n x p]
        Input [ x1 y1 z1 ... x_n y_n z_n ] where x1 ... z_n
        can be vectors

    Returns
    -------
    comb : array
        Output vector [sqrt(x1^2+y1^2+z1^2), ..., sqrt(x_n^2+y_n^2+z_n^2)]
    """
    if vec.ndim != 2:
        raise ValueError('Input must be 2D')
    if (vec.shape[0] % 3) != 0:
        raise ValueError('Input must have 3N rows')
    if np.iscomplexobj(vec):
        vec = np.abs(vec)
    comb = vec[0::3] ** 2
    comb += vec[1::3] ** 2
    comb += vec[2::3] ** 2
    if not square:
        comb = np.sqrt(comb)
    return comb


def _check_ch_names(inv, info):
    """Check that channels in inverse operator are measurements."""
    inv_ch_names = inv['eigen_fields']['col_names']

    if inv['noise_cov'].ch_names != inv_ch_names:
        raise ValueError('Channels in inverse operator eigen fields do not '
                         'match noise covariance channels.')
    data_ch_names = info['ch_names']

    missing_ch_names = sorted(set(inv_ch_names) - set(data_ch_names))
    n_missing = len(missing_ch_names)
    if n_missing > 0:
        raise ValueError('%d channels in inverse operator ' % n_missing +
                         'are not present in the data (%s)' % missing_ch_names)
    _check_compensation_grade(inv['info'], info, 'inverse')


def _check_or_prepare(inv, nave, lambda2, method, method_params, prepared,
                      copy=True):
    """Check if inverse was prepared, or prepare it."""
    if not prepared:
        inv = prepare_inverse_operator(
            inv, nave, lambda2, method, method_params, copy=copy)
    elif 'colorer' not in inv:
        raise ValueError('inverse operator has not been prepared, but got '
                         'argument prepared=True. Either pass prepared=False '
                         'or use prepare_inverse_operator.')
    return inv


@verbose
def prepare_inverse_operator(orig, nave, lambda2, method='dSPM',
                             method_params=None, copy=True, verbose=None):
    """Prepare an inverse operator for actually computing the inverse.

    Parameters
    ----------
    orig : dict
        The inverse operator structure read from a file.
    nave : int
        Number of averages (scales the noise covariance).
    lambda2 : float
        The regularization factor. Recommended to be 1 / SNR**2.
    method : "MNE" | "dSPM" | "sLORETA" | "eLORETA"
        Use minimum norm, dSPM (default), sLORETA, or eLORETA.
    method_params : dict | None
        Additional options for eLORETA. See Notes of :func:`apply_inverse`.

        .. versionadded:: 0.16
    copy : bool | str
        If True (default), copy the inverse. False will not copy.
        Can be "non-src" to avoid copying the source space, which typically
        is not modified and can be large in memory.

        .. versionadded:: 0.21
    %(verbose)s

    Returns
    -------
    inv : instance of InverseOperator
        Prepared inverse operator.
    """
    if nave <= 0:
        raise ValueError('The number of averages should be positive')

    _validate_type(copy, (bool, str), 'copy')
    if isinstance(copy, str):
        _check_option('copy', copy, ('non-src',), extra='when a string')
    logger.info('Preparing the inverse operator for use...')
    inv = orig
    if copy:
        src = orig['src']
        if copy == 'non-src':
            orig['src'] = None
        try:
            inv = orig.copy()
        finally:
            orig['src'] = src
        if copy == 'non-src':
            inv['src'] = src
    del orig

    #
    #   Scale some of the stuff
    #
    scale = float(inv['nave']) / nave
    inv['noise_cov']['data'] = scale * inv['noise_cov']['data']
    # deal with diagonal case
    if inv['noise_cov']['data'].ndim == 1:
        logger.info('    Diagonal noise covariance found')
        inv['noise_cov']['eig'] = inv['noise_cov']['data']
        inv['noise_cov']['eigvec'] = np.eye(len(inv['noise_cov']['data']))

    inv['noise_cov']['eig'] = scale * inv['noise_cov']['eig']
    inv['source_cov']['data'] = scale * inv['source_cov']['data']
    #
    if inv['eigen_leads_weighted']:
        inv['eigen_leads']['data'] = sqrt(scale) * inv['eigen_leads']['data']

    logger.info('    Scaled noise and source covariance from nave = %d to'
                ' nave = %d' % (inv['nave'], nave))
    inv['nave'] = nave
    #
    #   Create the diagonal matrix for computing the regularized inverse
    #
    inv['reginv'] = _compute_reginv(inv, lambda2)
    logger.info('    Created the regularized inverter')
    #
    #   Create the projection operator
    #
    inv['proj'], ncomp, _ = make_projector(inv['projs'],
                                           inv['noise_cov']['names'])
    if ncomp > 0:
        logger.info('    Created an SSP operator (subspace dimension = %d)'
                    % ncomp)
    else:
        logger.info('    The projection vectors do not apply to these '
                    'channels.')

    #
    #   Create the whitener
    #

    inv['whitener'], _, inv['colorer'] = compute_whitener(
        inv['noise_cov'], pca='white', return_colorer=True)

    #
    #   Finally, compute the noise-normalization factors
    #
    inv['noisenorm'] = []
    if method == 'eLORETA':
        _compute_eloreta(inv, lambda2, method_params)
    elif method != 'MNE':
        logger.info('    Computing noise-normalization factors (%s)...'
                    % method)
        # Here we have::
        #
        #     inv['reginv'] = sing / (sing ** 2 + lambda2)
        #
        # where ``sing`` are the singular values of the whitened gain matrix.
        if method == "dSPM":
            # dSPM normalization
            noise_weight = inv['reginv']
        elif method == 'sLORETA':
            # sLORETA normalization is given by the square root of the
            # diagonal entries of the resolution matrix R, which is
            # the product of the inverse and forward operators as:
            #
            #     w = diag(diag(R)) ** 0.5
            #
            noise_weight = (inv['reginv'] *
                            np.sqrt((1. + inv['sing'] ** 2 / lambda2)))

        noise_norm = np.zeros(inv['eigen_leads']['nrow'])
        nrm2, = linalg.get_blas_funcs(('nrm2',), (noise_norm,))
        if inv['eigen_leads_weighted']:
            for k in range(inv['eigen_leads']['nrow']):
                one = inv['eigen_leads']['data'][k, :] * noise_weight
                noise_norm[k] = nrm2(one)
        else:
            for k in range(inv['eigen_leads']['nrow']):
                one = (sqrt(inv['source_cov']['data'][k]) *
                       inv['eigen_leads']['data'][k, :] * noise_weight)
                noise_norm[k] = nrm2(one)

        #
        #   Compute the final result
        #
        if inv['source_ori'] == FIFF.FIFFV_MNE_FREE_ORI:
            #
            #   The three-component case is a little bit more involved
            #   The variances at three consecutive entries must be squared and
            #   added together
            #
            #   Even in this case return only one noise-normalization factor
            #   per source location
            #
            noise_norm = combine_xyz(noise_norm[:, None]).ravel()
        inv['noisenorm'] = 1.0 / np.abs(noise_norm)
        logger.info('[done]')
    else:
        inv['noisenorm'] = []

    return InverseOperator(inv)


@verbose
def _assemble_kernel(inv, label, method, pick_ori, use_cps=True, verbose=None):
    """Assemble the kernel.

    Simple matrix multiplication followed by combination of the current
    components. This does all the data transformations to compute the weights
    for the eigenleads.

    Parameters
    ----------
    inv : instance of InverseOperator
        The inverse operator to use. This object contains the matrices that
        will be multiplied to assemble the kernel.
    label : Label | None
        Restricts the source estimates to a given label. If None,
        source estimates will be computed for the entire source space.
    method : "MNE" | "dSPM" | "sLORETA" | "eLORETA"
        Use minimum norm, dSPM, sLORETA, or eLORETA.
    pick_ori : None | "normal" | "vector"
        Which orientation to pick (only matters in the case of 'normal').
    %(use_cps_restricted)s

    Returns
    -------
    K : array, shape (n_vertices, n_channels) | (3 * n_vertices, n_channels)
        The kernel matrix. Multiply this with the data to obtain the source
        estimate.
    noise_norm : array, shape (n_vertices, n_samples) | (3 * n_vertices, n_samples)
        Normalization to apply to the source estimate in order to obtain dSPM
        or sLORETA solutions.
    vertices : list of length 2
        Vertex numbers for lh and rh hemispheres that correspond to the
        vertices in the source estimate. When the label parameter has been
        set, these correspond to the vertices in the label. Otherwise, all
        vertex numbers are returned.
    source_nn : array, shape (3 * n_vertices, 3)
        The direction in carthesian coordicates of the direction of the source
        dipoles.
    """  # noqa: E501
    eigen_leads = inv['eigen_leads']['data']
    source_cov = inv['source_cov']['data']
    if method in ('dSPM', 'sLORETA'):
        noise_norm = inv['noisenorm'][:, np.newaxis]
    else:
        noise_norm = None

    src = inv['src']
    vertno = _get_vertno(src)
    source_nn = inv['source_nn']

    if label is not None:
        vertno, src_sel = label_src_vertno_sel(label, src)

        if method not in ["MNE", "eLORETA"]:
            noise_norm = noise_norm[src_sel]

        if inv['source_ori'] == FIFF.FIFFV_MNE_FREE_ORI:
            src_sel = 3 * src_sel
            src_sel = np.c_[src_sel, src_sel + 1, src_sel + 2]
            src_sel = src_sel.ravel()

        eigen_leads = eigen_leads[src_sel]
        source_cov = source_cov[src_sel]
        source_nn = source_nn[src_sel]

    # vector or normal, might need to rotate
    if pick_ori == 'normal' and all(s['type'] == 'surf' for s in src) and \
            np.allclose(inv['source_nn'].reshape(inv['nsource'], 3, 3),
                        np.eye(3), atol=1e-6):
        offset = 0
        eigen_leads = np.reshape(
            eigen_leads, (-1, 3, eigen_leads.shape[1])).copy()
        source_nn = np.reshape(source_nn, (-1, 3, 3)).copy()
        for s, v in zip(src, vertno):
            sl = slice(offset, offset + len(v))
            source_nn[sl] = _normal_orth(_get_src_nn(s, use_cps, v))
            eigen_leads[sl] = np.matmul(source_nn[sl], eigen_leads[sl])
            # No need to rotate source_cov because it should be uniform
            # (loose=1., and depth weighting is uniform across columns)
            offset = sl.stop
        eigen_leads.shape = (-1, eigen_leads.shape[2])
        source_nn.shape = (-1, 3)

    if pick_ori == "normal":
        if not inv['source_ori'] == FIFF.FIFFV_MNE_FREE_ORI:
            raise ValueError('Picking normal orientation can only be done '
                             'with a free orientation inverse operator.')

        is_loose = 0 < inv['orient_prior']['data'][0] <= 1
        if not is_loose:
            raise ValueError('Picking normal orientation can only be done '
                             'when working with loose orientations.')

    trans = np.dot(inv['eigen_fields']['data'],
                   np.dot(inv['whitener'], inv['proj']))
    trans *= inv['reginv'][:, None]

    #
    #   Transformation into current distributions by weighting the eigenleads
    #   with the weights computed above
    #
    K = np.dot(eigen_leads, trans)
    if inv['eigen_leads_weighted']:
        #
        #     R^0.5 has been already factored in
        #
        logger.info('    Eigenleads already weighted ... ')
    else:
        #
        #     R^0.5 has to be factored in
        #
        logger.info('    Eigenleads need to be weighted ...')
        K *= np.sqrt(source_cov)[:, np.newaxis]

    if pick_ori == 'normal':
        K = K[2::3]

    return K, noise_norm, vertno, source_nn


def _check_ori(pick_ori, source_ori, src, allow_vector=True):
    """Check pick_ori."""
    _check_option('pick_ori', pick_ori, [None, 'normal', 'vector'])
    _check_src_normal(pick_ori, src)


def _check_reference(inst, ch_names=None):
    """Check for EEG ref."""
    info = inst.info
    if ch_names is not None:
        picks = [ci for ci, ch_name in enumerate(info['ch_names'])
                 if ch_name in ch_names]
        info = pick_info(info, sel=picks)
    if _needs_eeg_average_ref_proj(info):
        raise ValueError(
            'EEG average reference (using a projector) is mandatory for '
            'modeling, use the method set_eeg_reference(projection=True)')
    if info['custom_ref_applied']:
        raise ValueError('Custom EEG reference is not allowed for inverse '
                         'modeling.')


def _subject_from_inverse(inverse_operator):
    """Get subject id from inverse operator."""
    return inverse_operator['src']._subject


@verbose
def apply_inverse(evoked, inverse_operator, lambda2=1. / 9., method="dSPM",
                  pick_ori=None, prepared=False, label=None,
                  method_params=None, return_residual=False, use_cps=True,
                  verbose=None):
    """Apply inverse operator to evoked data.

    Parameters
    ----------
    evoked : Evoked object
        Evoked data.
    inverse_operator : instance of InverseOperator
        Inverse operator.
    lambda2 : float
        The regularization parameter.
    method : "MNE" | "dSPM" | "sLORETA" | "eLORETA"
        Use minimum norm :footcite:`HamalainenIlmoniemi1994`,
        dSPM (default) :footcite:`DaleEtAl2000`,
        sLORETA :footcite:`Pascual-Marqui2002`, or
        eLORETA :footcite:`Pascual-Marqui2011`.
    %(pick_ori)s
    prepared : bool
        If True, do not call :func:`prepare_inverse_operator`.
    label : Label | None
        Restricts the source estimates to a given label. If None,
        source estimates will be computed for the entire source space.
    method_params : dict | None
        Additional options for eLORETA. See Notes for details.

        .. versionadded:: 0.16
    return_residual : bool
        If True (default False), return the residual evoked data.
        Cannot be used with ``method=='eLORETA'``.

        .. versionadded:: 0.17
    %(use_cps_restricted)s

        .. versionadded:: 0.20
    %(verbose)s

    Returns
    -------
    stc : SourceEstimate | VectorSourceEstimate | VolSourceEstimate
        The source estimates.
    residual : instance of Evoked
        The residual evoked data, only returned if return_residual is True.

    See Also
    --------
    apply_inverse_raw : Apply inverse operator to raw object.
    apply_inverse_epochs : Apply inverse operator to epochs object.

    Notes
    -----
    Currently only the ``method='eLORETA'`` has additional options.
    It performs an iterative fit with a convergence criterion, so you can
    pass a ``method_params`` :class:`dict` with string keys mapping to values
    for:

        'eps' : float
            The convergence epsilon (default 1e-6).
        'max_iter' : int
            The maximum number of iterations (default 20).
            If less regularization is applied, more iterations may be
            necessary.
        'force_equal' : bool
            Force all eLORETA weights for each direction for a given
            location equal. The default is None, which means ``True`` for
            loose-orientation inverses and ``False`` for free- and
            fixed-orientation inverses. See below.

    The eLORETA paper :footcite:`Pascual-Marqui2011` defines how to compute
    inverses for fixed- and
    free-orientation inverses. In the free orientation case, the X/Y/Z
    orientation triplet for each location is effectively multiplied by a
    3x3 weight matrix. This is the behavior obtained with
    ``force_equal=False`` parameter.

    However, other noise normalization methods (dSPM, sLORETA) multiply all
    orientations for a given location by a single value.
    Using ``force_equal=True`` mimics this behavior by modifying the iterative
    algorithm to choose uniform weights (equivalent to a 3x3 diagonal matrix
    with equal entries).

    It is necessary to use ``force_equal=True``
    with loose orientation inverses (e.g., ``loose=0.2``), otherwise the
    solution resembles a free-orientation inverse (``loose=1.0``).
    It is thus recommended to use ``force_equal=True`` for loose orientation
    and ``force_equal=False`` for free orientation inverses. This is the
    behavior used when the parameter ``force_equal=None`` (default behavior).

    References
    ----------
    .. footbibliography::
    """
    out = _apply_inverse(
        evoked, inverse_operator, lambda2, method, pick_ori, prepared, label,
        method_params, return_residual, use_cps)
    logger.info('[done]')
    return out


def _log_exp_var(data, est, prefix='    '):
    res = data - est
    var_exp = 1 - ((res * res.conj()).sum().real /
                   (data * data.conj()).sum().real)
    var_exp *= 100
    logger.info(f'{prefix}Explained {var_exp:5.1f}% variance')
    return var_exp


def _apply_inverse(evoked, inverse_operator, lambda2, method, pick_ori,
                   prepared, label, method_params, return_residual, use_cps):
    _validate_type(evoked, Evoked, 'evoked')
    _check_reference(evoked, inverse_operator['info']['ch_names'])
    _check_option('method', method, INVERSE_METHODS)
    _check_ori(pick_ori, inverse_operator['source_ori'],
               inverse_operator['src'])
    #
    #   Set up the inverse according to the parameters
    #
    nave = evoked.nave

    _check_ch_names(inverse_operator, evoked.info)

    inv = _check_or_prepare(inverse_operator, nave, lambda2, method,
                            method_params, prepared, copy='non-src')
    del inverse_operator

    #
    #   Pick the correct channels from the data
    #
    sel = _pick_channels_inverse_operator(evoked.ch_names, inv)
    logger.info('Applying inverse operator to "%s"...' % (evoked.comment,))
    logger.info('    Picked %d channels from the data' % len(sel))
    logger.info('    Computing inverse...')
    K, noise_norm, vertno, source_nn = _assemble_kernel(
        inv, label, method, pick_ori, use_cps=use_cps)
    sol = np.dot(K, evoked.data[sel])  # apply imaging kernel
    logger.info('    Computing residual...')
    # x̂(t) = G ĵ(t) = C ** 1/2 U Π w(t)
    # where the diagonal matrix Π has elements πk = λk γk
    Pi = inv['sing'] * inv['reginv']
    data_w = np.dot(inv['whitener'],  # C ** -0.5
                    np.dot(inv['proj'], evoked.data[sel]))
    w_t = np.dot(inv['eigen_fields']['data'], data_w)  # U.T @ data
    data_est = np.dot(inv['colorer'],  # C ** 0.5
                      np.dot(inv['eigen_fields']['data'].T,  # U
                             Pi[:, np.newaxis] * w_t))
    data_est_w = np.dot(inv['whitener'], np.dot(inv['proj'], data_est))
    _log_exp_var(data_w, data_est_w)
    if return_residual:
        residual = evoked.copy()
        residual.data[sel] -= data_est
    is_free_ori = (inv['source_ori'] == FIFF.FIFFV_MNE_FREE_ORI and
                   pick_ori != 'normal')

    if is_free_ori and pick_ori != 'vector':
        logger.info('    Combining the current components...')
        sol = combine_xyz(sol)

    if noise_norm is not None:
        logger.info('    %s...' % (method,))
        if is_free_ori and pick_ori == 'vector':
            noise_norm = noise_norm.repeat(3, axis=0)
        sol *= noise_norm

    tstep = 1.0 / evoked.info['sfreq']
    tmin = float(evoked.times[0])
    subject = _subject_from_inverse(inv)
    src_type = _get_src_type(inv['src'], vertno)
    stc = _make_stc(sol, vertno, tmin=tmin, tstep=tstep, subject=subject,
                    vector=(pick_ori == 'vector'), source_nn=source_nn,
                    src_type=src_type)

    return (stc, residual) if return_residual else stc


@verbose
def apply_inverse_raw(raw, inverse_operator, lambda2, method="dSPM",
                      label=None, start=None, stop=None, nave=1,
                      time_func=None, pick_ori=None, buffer_size=None,
                      prepared=False, method_params=None, use_cps=True,
                      verbose=None):
    """Apply inverse operator to Raw data.

    Parameters
    ----------
    raw : Raw object
        Raw data.
    inverse_operator : dict
        Inverse operator.
    lambda2 : float
        The regularization parameter.
    method : "MNE" | "dSPM" | "sLORETA" | "eLORETA"
        Use minimum norm, dSPM (default), sLORETA, or eLORETA.
    label : Label | None
        Restricts the source estimates to a given label. If None,
        source estimates will be computed for the entire source space.
    start : int
        Index of first time sample (index not time is seconds).
    stop : int
        Index of first time sample not to include (index not time is seconds).
    nave : int
        Number of averages used to regularize the solution.
        Set to 1 on raw data.
    time_func : callable
        Linear function applied to sensor space time series.
    %(pick_ori)s
    buffer_size : int (or None)
        If not None, the computation of the inverse and the combination of the
        current components is performed in segments of length buffer_size
        samples. While slightly slower, this is useful for long datasets as it
        reduces the memory requirements by approx. a factor of 3 (assuming
        buffer_size << data length).
        Note that this setting has no effect for fixed-orientation inverse
        operators.
    prepared : bool
        If True, do not call :func:`prepare_inverse_operator`.
    method_params : dict | None
        Additional options for eLORETA. See Notes of :func:`apply_inverse`.

        .. versionadded:: 0.16
    %(use_cps_restricted)s

        .. versionadded:: 0.20
    %(verbose)s

    Returns
    -------
    stc : SourceEstimate | VectorSourceEstimate | VolSourceEstimate
        The source estimates.

    See Also
    --------
    apply_inverse_epochs : Apply inverse operator to epochs object.
    apply_inverse : Apply inverse operator to evoked object.
    """
    _validate_type(raw, BaseRaw, 'raw')
    _check_reference(raw, inverse_operator['info']['ch_names'])
    _check_option('method', method, INVERSE_METHODS)
    _check_ori(pick_ori, inverse_operator['source_ori'],
               inverse_operator['src'])
    _check_ch_names(inverse_operator, raw.info)

    #
    #   Set up the inverse according to the parameters
    #
    inv = _check_or_prepare(inverse_operator, nave, lambda2, method,
                            method_params, prepared)

    #
    #   Pick the correct channels from the data
    #
    sel = _pick_channels_inverse_operator(raw.ch_names, inv)
    logger.info('Applying inverse to raw...')
    logger.info('    Picked %d channels from the data' % len(sel))
    logger.info('    Computing inverse...')

    data, times = raw[sel, start:stop]

    if time_func is not None:
        data = time_func(data)

    K, noise_norm, vertno, source_nn = _assemble_kernel(
        inv, label, method, pick_ori, use_cps)

    is_free_ori = (inverse_operator['source_ori'] ==
                   FIFF.FIFFV_MNE_FREE_ORI and pick_ori != 'normal')

    if buffer_size is not None and is_free_ori:
        # Process the data in segments to conserve memory
        n_seg = int(np.ceil(data.shape[1] / float(buffer_size)))
        logger.info('    computing inverse and combining the current '
                    'components (using %d segments)...' % (n_seg))

        # Allocate space for inverse solution
        n_times = data.shape[1]

        n_dipoles = K.shape[0] if pick_ori == 'vector' else K.shape[0] // 3
        sol = np.empty((n_dipoles, n_times), dtype=np.result_type(K, data))

        for pos in range(0, n_times, buffer_size):
            sol_chunk = np.dot(K, data[:, pos:pos + buffer_size])
            if pick_ori != 'vector':
                sol_chunk = combine_xyz(sol_chunk)
            sol[:, pos:pos + buffer_size] = sol_chunk

            logger.info('        segment %d / %d done..'
                        % (pos / buffer_size + 1, n_seg))
    else:
        sol = np.dot(K, data)
        if is_free_ori and pick_ori != 'vector':
            logger.info('    combining the current components...')
            sol = combine_xyz(sol)
    if noise_norm is not None:
        if pick_ori == 'vector' and is_free_ori:
            noise_norm = noise_norm.repeat(3, axis=0)
        sol *= noise_norm

    tmin = float(times[0])
    tstep = 1.0 / raw.info['sfreq']
    subject = _subject_from_inverse(inverse_operator)
    src_type = _get_src_type(inverse_operator['src'], vertno)
    stc = _make_stc(sol, vertno, tmin=tmin, tstep=tstep, subject=subject,
                    vector=(pick_ori == 'vector'), source_nn=source_nn,
                    src_type=src_type)
    logger.info('[done]')

    return stc


def _apply_inverse_epochs_gen(epochs, inverse_operator, lambda2, method='dSPM',
                              label=None, nave=1, pick_ori=None,
                              prepared=False, method_params=None,
                              use_cps=True, verbose=None):
    """Generate inverse solutions for epochs. Used in apply_inverse_epochs."""
    _validate_type(epochs, BaseEpochs, 'epochs')
    _check_option('method', method, INVERSE_METHODS)
    _check_ori(pick_ori, inverse_operator['source_ori'],
               inverse_operator['src'])
    _check_ch_names(inverse_operator, epochs.info)

    #
    #   Set up the inverse according to the parameters
    #
    inv = _check_or_prepare(inverse_operator, nave, lambda2, method,
                            method_params, prepared)

    #
    #   Pick the correct channels from the data
    #
    sel = _pick_channels_inverse_operator(epochs.ch_names, inv)
    logger.info('Picked %d channels from the data' % len(sel))
    logger.info('Computing inverse...')
    K, noise_norm, vertno, source_nn = _assemble_kernel(
        inv, label, method, pick_ori, use_cps)

    tstep = 1.0 / epochs.info['sfreq']
    tmin = epochs.times[0]

    is_free_ori = not (is_fixed_orient(inverse_operator) or
                       pick_ori == 'normal')

    if pick_ori == 'vector' and noise_norm is not None:
        noise_norm = noise_norm.repeat(3, axis=0)

    if not is_free_ori and noise_norm is not None:
        # premultiply kernel with noise normalization
        K *= noise_norm

    subject = _subject_from_inverse(inverse_operator)
    try:
        total = ' / %d' % (len(epochs),)  # len not always defined
    except RuntimeError:
        total = ' / %d (at most)' % (len(epochs.events),)
    for k, e in enumerate(epochs):
        logger.info('Processing epoch : %d%s' % (k + 1, total))
        if is_free_ori:
            # Compute solution and combine current components (non-linear)
            sol = np.dot(K, e[sel])  # apply imaging kernel

            if pick_ori != 'vector':
                logger.info('combining the current components...')
                sol = combine_xyz(sol)

            if noise_norm is not None:
                sol *= noise_norm
        else:
            # Linear inverse: do computation here or delayed
            if len(sel) < K.shape[1]:
                sol = (K, e[sel])
            else:
                sol = np.dot(K, e[sel])

        src_type = _get_src_type(inverse_operator['src'], vertno)
        stc = _make_stc(sol, vertno, tmin=tmin, tstep=tstep, subject=subject,
                        vector=(pick_ori == 'vector'), source_nn=source_nn,
                        src_type=src_type)

        yield stc

    logger.info('[done]')


@verbose
def apply_inverse_epochs(epochs, inverse_operator, lambda2, method="dSPM",
                         label=None, nave=1, pick_ori=None,
                         return_generator=False, prepared=False,
                         method_params=None, use_cps=True, verbose=None):
    """Apply inverse operator to Epochs.

    Parameters
    ----------
    epochs : Epochs object
        Single trial epochs.
    inverse_operator : dict
        Inverse operator.
    lambda2 : float
        The regularization parameter.
    method : "MNE" | "dSPM" | "sLORETA" | "eLORETA"
        Use minimum norm, dSPM (default), sLORETA, or eLORETA.
    label : Label | None
        Restricts the source estimates to a given label. If None,
        source estimates will be computed for the entire source space.
    nave : int
        Number of averages used to regularize the solution.
        Set to 1 on single Epoch by default.
    %(pick_ori)s
    return_generator : bool
        Return a generator object instead of a list. This allows iterating
        over the stcs without having to keep them all in memory.
    prepared : bool
        If True, do not call :func:`prepare_inverse_operator`.
    method_params : dict | None
        Additional options for eLORETA. See Notes of :func:`apply_inverse`.

        .. versionadded:: 0.16
    %(use_cps_restricted)s

        .. versionadded:: 0.20
    %(verbose)s

    Returns
    -------
    stc : list of (SourceEstimate | VectorSourceEstimate | VolSourceEstimate)
        The source estimates for all epochs.

    See Also
    --------
    apply_inverse_raw : Apply inverse operator to raw object.
    apply_inverse : Apply inverse operator to evoked object.
    """
    stcs = _apply_inverse_epochs_gen(
        epochs, inverse_operator, lambda2, method=method, label=label,
        nave=nave, pick_ori=pick_ori, verbose=verbose, prepared=prepared,
        method_params=method_params, use_cps=use_cps)

    if not return_generator:
        # return a list
        stcs = [stc for stc in stcs]

    return stcs


@verbose
def apply_inverse_cov(cov, info, inverse_operator, nave=1, lambda2=1 / 9,
                      method="dSPM", pick_ori=None, prepared=False,
                      label=None, method_params=None, use_cps=True,
                      verbose=None):
    """Apply inverse operator to covariance data.

    Parameters
    ----------
    cov : instance of Covariance
        Covariance data, computed on the time segment for which to compute
        source power.
    info : dict
        The measurement info to specify the channels to include.
    inverse_operator : instance of InverseOperator
        Inverse operator.
    nave : int
        Number of averages used to regularize the solution.
    lambda2 : float
        The regularization parameter.
    method : "MNE" | "dSPM" | "sLORETA" | "eLORETA"
        Use minimum norm, dSPM (default), sLORETA, or eLORETA.
    %(pick_ori-novec)s
    prepared : bool
        If True, do not call :func:`prepare_inverse_operator`.
    label : Label | None
        Restricts the source estimates to a given label. If None,
        source estimates will be computed for the entire source space.
    method_params : dict | None
        Additional options for eLORETA. See Notes for details.
    %(use_cps)s
    %(verbose)s

    Returns
    -------
    stc : SourceEstimate | VectorSourceEstimate | VolSourceEstimate
        The source estimates.

    See Also
    --------
    apply_inverse : Apply inverse operator to evoked object.
    apply_inverse_raw : Apply inverse operator to raw object.
    apply_inverse_epochs : Apply inverse operator to epochs object.

    Notes
    -----
    .. versionadded:: 0.20

    This code is based on the original research code from
    :footcite:`Sabbagh2020` and has been useful to correct for individual field
    spread using source localization in the context of predictive modeling.

    References
    ----------
    .. footbibliography::
    """
    _validate_type(cov, Covariance, cov)
    _validate_type(inverse_operator, InverseOperator, 'inverse_operator')
    sel = _pick_channels_inverse_operator(cov['names'], inverse_operator)
    use_names = [cov['names'][idx] for idx in sel]
    info = pick_info(
        info, pick_channels(info['ch_names'], use_names, ordered=True))
    evoked = EvokedArray(
        np.eye(len(info['ch_names'])), info, nave=nave, comment='cov')
    is_free_ori = (inverse_operator['source_ori'] == FIFF.FIFFV_MNE_FREE_ORI)
    _check_option('pick_ori', pick_ori, (None, 'normal'))
    if is_free_ori and pick_ori is None:
        use_ori = 'vector'
        combine = True
    else:
        use_ori = pick_ori
        combine = False
    stc = _apply_inverse(
        evoked, inverse_operator, lambda2, method, use_ori, prepared, label,
        method_params, return_residual=False, use_cps=use_cps)
    # apply (potentially rotated in the vector case) operator twice
    K = np.reshape(stc.data, (-1, stc.data.shape[-1]))
    # diagonal entries of A @ B are given by (A * B.T).sum(axis=1), so this is
    # equivalent to np.diag(K @ cov.data[sel][:, sel] @ K.T)[:, np.newaxis]:
    sol = cov.data[sel][:, sel] @ K.T
    sol = np.sum(K * sol.T, axis=1, keepdims=True)
    # Reshape back to (n_src, ..., 1)
    sol.shape = stc.data.shape[:-1] + (1,)
    stc = stc.__class__(
        sol, stc.vertices, stc.tmin, stc.tstep, stc.subject, stc.verbose)
    if combine:  # combine the three directions
        logger.info('    Combining the current components...')
        np.sqrt(stc.data, out=stc.data)
        stc = stc.magnitude()
        stc.data *= stc.data
    logger.info('[done]')
    return stc


###############################################################################
# Assemble the inverse operator

def _prepare_forward(forward, info, noise_cov, fixed, loose, rank, pca,
                     use_cps, exp, limit_depth_chs, combine_xyz,
                     allow_fixed_depth, limit):
    """Prepare a gain matrix and noise covariance for localization."""
    # Steps (according to MNE-C, we change the order of various steps
    # because our I/O is already done, and we create the objects
    # on the fly more easily):
    #
    # 1. Read the bad channels
    # 2. Read the necessary data from the forward solution matrix file
    # 3. Load the projection data
    # 4. Load the sensor noise covariance matrix and attach it to the forward
    # 5. Compose the depth-weighting matrix
    # 6. Compose the source covariance matrix
    # 7. Apply fMRI weighting (not done)
    # 8. Apply the linear projection to the forward solution
    # 9. Apply whitening to the forward computation matrix
    # 10. Exclude the source space points within the labels (not done)
    # 11. Do appropriate source weighting to the forward computation matrix
    #

    # make a copy immediately so we do it exactly once
    forward = forward.copy()

    # Deal with "fixed" and "loose"
    loose = _triage_loose(forward['src'], loose, fixed)
    del fixed

    # Deal with "depth"
    if exp is not None:
        exp = float(exp)
        if not (0 <= exp <= 1):
            raise ValueError('depth exponent should be a scalar between '
                             '0 and 1, got %s' % (exp,))
        exp = exp or None  # alias 0. -> None

    # put the forward solution in correct orientation
    # (delaying for the case of fixed ori with depth weighting if
    # allow_fixed_depth is True)
    if loose.get('surface', 1.) == 0. and len(loose) == 1:
        if not is_fixed_orient(forward):
            if allow_fixed_depth:
                # can convert now
                logger.info('Converting forward solution to fixed orietnation')
                convert_forward_solution(
                    forward, force_fixed=True, use_cps=True, copy=False)
        elif exp is not None and not allow_fixed_depth:
            raise ValueError(
                'For a fixed orientation inverse solution with depth '
                'weighting, the forward solution must be free-orientation and '
                'in surface orientation')
    else:  # loose or free ori
        if is_fixed_orient(forward):
            raise ValueError(
                'Forward operator has fixed orientation and can only '
                'be used to make a fixed-orientation inverse '
                'operator.')
        if loose.get('surface', 1.) < 1. and not forward['surf_ori']:
            logger.info('Converting forward solution to surface orientation')
            convert_forward_solution(
                forward, surf_ori=True, use_cps=use_cps, copy=False)

    forward, info_picked = _select_orient_forward(forward, info, noise_cov,
                                                  copy=False)
    logger.info("Selected %d channels" % (len(info_picked['ch_names'],)))

    if exp is None:
        depth_prior = None
    else:
        depth_prior = compute_depth_prior(
            forward, info_picked, exp=exp, limit_depth_chs=limit_depth_chs,
            combine_xyz=combine_xyz, limit=limit, noise_cov=noise_cov,
            rank=rank)

    # Deal with fixed orientation forward / inverse
    if loose.get('surface', 1.) == 0. and len(loose) == 1:
        orient_prior = None
        if not is_fixed_orient(forward):
            if depth_prior is not None:
                # Convert the depth prior into a fixed-orientation one
                logger.info('    Picked elements from a free-orientation '
                            'depth-weighting prior into the fixed-orientation '
                            'one')
                depth_prior = depth_prior[2::3]
            convert_forward_solution(
                forward, surf_ori=True, force_fixed=True,
                use_cps=use_cps, copy=False)
    else:
        if loose.get('surface', 1.) < 1:
            assert forward['surf_ori']
        # In theory we could have orient_prior=None for loose=1., but
        # the MNE-C code does not do this
        orient_prior = compute_orient_prior(forward, loose=loose)

    logger.info('Whitening the forward solution.')
    noise_cov = prepare_noise_cov(
        noise_cov, info, info_picked['ch_names'], rank)
    whitener, _ = compute_whitener(
        noise_cov, info, info_picked['ch_names'], pca=pca, verbose=False,
        rank=rank)
    gain = np.dot(whitener, forward['sol']['data'])

    logger.info('Creating the source covariance matrix')
    source_std = np.ones(gain.shape[1], dtype=gain.dtype)
    if depth_prior is not None:
        source_std *= depth_prior
    if orient_prior is not None:
        source_std *= orient_prior
    np.sqrt(source_std, out=source_std)
    gain *= source_std
    # Adjusting Source Covariance matrix to make trace of G*R*G' equal
    # to number of sensors.
    logger.info('Adjusting source covariance matrix.')
    trace_GRGT = linalg.norm(gain, ord='fro') ** 2
    n_nzero = (noise_cov['eig'] > 0).sum()
    scale = np.sqrt(n_nzero / trace_GRGT)
    source_std *= scale
    gain *= scale

    return (forward, info_picked, gain, depth_prior, orient_prior, source_std,
            trace_GRGT, noise_cov, whitener)


@verbose
def make_inverse_operator(info, forward, noise_cov, loose='auto', depth=0.8,
                          fixed='auto', rank=None, use_cps=True, verbose=None):
    """Assemble inverse operator.

    Parameters
    ----------
    info : dict
        The measurement info to specify the channels to include.
        Bad channels in info['bads'] are not used.
    forward : dict
        Forward operator.
    noise_cov : instance of Covariance
        The noise covariance matrix.
    %(loose)s
    %(depth)s
    fixed : bool | 'auto'
        Use fixed source orientations normal to the cortical mantle. If True,
        the loose parameter must be "auto" or 0. If 'auto', the loose value
        is used.
    %(rank_None)s
    %(use_cps)s
    %(verbose)s

    Returns
    -------
    inv : instance of InverseOperator
        Inverse operator.

    Notes
    -----
    For different sets of options (**loose**, **depth**, **fixed**) to work,
    the forward operator must have been loaded using a certain configuration
    (i.e., with **force_fixed** and **surf_ori** set appropriately). For
    example, given the desired inverse type (with representative choices
    of **loose** = 0.2 and **depth** = 0.8 shown in the table in various
    places, as these are the defaults for those parameters):

        +---------------------+-----------+-----------+-----------+-----------------+--------------+
        | Inverse desired                             | Forward parameters allowed                 |
        +=====================+===========+===========+===========+=================+==============+
        |                     | **loose** | **depth** | **fixed** | **force_fixed** | **surf_ori** |
        +---------------------+-----------+-----------+-----------+-----------------+--------------+
        | | Loose constraint, | 0.2       | 0.8       | False     | False           | True         |
        | | Depth weighted    |           |           |           |                 |              |
        +---------------------+-----------+-----------+-----------+-----------------+--------------+
        | | Loose constraint  | 0.2       | None      | False     | False           | True         |
        +---------------------+-----------+-----------+-----------+-----------------+--------------+
        | | Free orientation, | 1.0       | 0.8       | False     | False           | True         |
        | | Depth weighted    |           |           |           |                 |              |
        +---------------------+-----------+-----------+-----------+-----------------+--------------+
        | | Free orientation  | 1.0       | None      | False     | False           | True | False |
        +---------------------+-----------+-----------+-----------+-----------------+--------------+
        | | Fixed constraint, | 0.0       | 0.8       | True      | False           | True         |
        | | Depth weighted    |           |           |           |                 |              |
        +---------------------+-----------+-----------+-----------+-----------------+--------------+
        | | Fixed constraint  | 0.0       | None      | True      | True            | True         |
        +---------------------+-----------+-----------+-----------+-----------------+--------------+

    Also note that, if the source space (as stored in the forward solution)
    has patch statistics computed, these are used to improve the depth
    weighting. Thus slightly different results are to be expected with
    and without this information.
    """  # noqa: E501
    # For now we always have pca='white'. It does not seem to affect
    # calculations and is also backward-compatible with MNE-C
    depth = _check_depth(depth, 'depth_mne')
    forward, gain_info, gain, depth_prior, orient_prior, source_std, \
        trace_GRGT, noise_cov, _ = _prepare_forward(
            forward, info, noise_cov, fixed, loose, rank, pca='white',
            use_cps=use_cps, **depth)
    # no need to copy any attributes of forward here because there is
    # a deepcopy in _prepare_forward
    inv = dict(
        projs=deepcopy(gain_info['projs']), eigen_leads_weighted=False,
        source_ori=forward['source_ori'], mri_head_t=forward['mri_head_t'],
        nsource=forward['nsource'], units='Am',
        coord_frame=forward['coord_frame'], source_nn=forward['source_nn'],
        src=forward['src'], fmri_prior=None, info=deepcopy(forward['info']))
    inv['info']['bads'] = [bad for bad in info['bads']
                           if bad in forward['info']['ch_names']]
    inv['info']._check_consistency()
    del fixed, loose, depth, use_cps, forward

    # Decompose the combined matrix
    logger.info('Computing SVD of whitened and weighted lead field matrix.')
    eigen_fields, sing, eigen_leads = _safe_svd(gain, full_matrices=False)
    del gain
    logger.info('    largest singular value = %g' % np.max(sing))
    logger.info('    scaling factor to adjust the trace = %g' % trace_GRGT)

    # MNE-ify everything for output
    eigen_fields = dict(data=eigen_fields.T, col_names=gain_info['ch_names'],
                        row_names=[], nrow=eigen_fields.shape[1],
                        ncol=eigen_fields.shape[0])
    eigen_leads = dict(data=eigen_leads.T, nrow=eigen_leads.shape[1],
                       ncol=eigen_leads.shape[0], row_names=[],
                       col_names=[])
    has_meg = False
    has_eeg = False
    for idx in range(gain_info['nchan']):
        ch_type = channel_type(gain_info, idx)
        if ch_type == 'eeg':
            has_eeg = True
        if (ch_type == 'mag') or (ch_type == 'grad'):
            has_meg = True
    if has_eeg and has_meg:
        methods = FIFF.FIFFV_MNE_MEG_EEG
    elif has_meg:
        methods = FIFF.FIFFV_MNE_MEG
    else:
        methods = FIFF.FIFFV_MNE_EEG

    if orient_prior is not None:
        orient_prior = dict(data=orient_prior,
                            kind=FIFF.FIFFV_MNE_ORIENT_PRIOR_COV,
                            bads=[], diag=True, names=[], eig=None,
                            eigvec=None, dim=orient_prior.size, nfree=1,
                            projs=[])
    if depth_prior is not None:
        depth_prior = dict(data=depth_prior,
                           kind=FIFF.FIFFV_MNE_DEPTH_PRIOR_COV,
                           bads=[], diag=True, names=[], eig=None,
                           eigvec=None, dim=depth_prior.size, nfree=1,
                           projs=[])
    source_cov = dict(data=source_std * source_std, dim=source_std.size,
                      kind=FIFF.FIFFV_MNE_SOURCE_COV, diag=True,
                      names=[], projs=[], eig=None, eigvec=None,
                      nfree=1, bads=[])
    inv.update(
        eigen_fields=eigen_fields, eigen_leads=eigen_leads, sing=sing, nave=1.,
        depth_prior=depth_prior, source_cov=source_cov, noise_cov=noise_cov,
        orient_prior=orient_prior, methods=methods)
    return InverseOperator(inv)


def _compute_reginv(inv, lambda2):
    """Safely compute reginv from sing."""
    sing = np.array(inv['sing'], dtype=np.float64)
    reginv = np.zeros_like(sing)
    n_nzero = compute_rank_inverse(inv)
    sing = sing[:n_nzero]
    with np.errstate(invalid='ignore'):  # if lambda2==0
        reginv[:n_nzero] = np.where(
            sing > 0, sing / (sing ** 2 + lambda2), 0)
    return reginv


def compute_rank_inverse(inv):
    """Compute the rank of a linear inverse operator (MNE, dSPM, etc.).

    Parameters
    ----------
    inv : instance of InverseOperator
        The inverse operator.

    Returns
    -------
    rank : int
        The rank of the inverse operator.
    """
    # this code shortened from prepare_inverse_operator
    eig = inv['noise_cov']['eig']
    if not inv['noise_cov']['diag']:
        rank = np.sum(eig > 0)
    else:
        ncomp = make_projector(inv['projs'], inv['noise_cov']['names'])[1]
        rank = inv['noise_cov']['dim'] - ncomp
    return rank


# #############################################################################
# SNR Estimation

@verbose
def estimate_snr(evoked, inv, verbose=None):
    r"""Estimate the SNR as a function of time for evoked data.

    Parameters
    ----------
    evoked : instance of Evoked
        Evoked instance.
    inv : instance of InverseOperator
        The inverse operator.
    %(verbose)s

    Returns
    -------
    snr : ndarray, shape (n_times,)
        The SNR estimated from the whitened data (i.e., GFP of whitened data).
    snr_est : ndarray, shape (n_times,)
        The SNR estimated using the mismatch between the unregularized
        solution and the regularized solution.

    Notes
    -----
    ``snr_est`` is estimated by using different amounts of inverse
    regularization and checking the mismatch between predicted and
    measured whitened data.

    In more detail, given our whitened inverse obtained from SVD:

    .. math::

        \tilde{M} = R^\frac{1}{2}V\Gamma U^T

    The values in the diagonal matrix :math:`\Gamma` are expressed in terms
    of the chosen regularization :math:`\lambda\approx\frac{1}{\rm{SNR}^2}`
    and singular values :math:`\lambda_k` as:

    .. math::

        \gamma_k = \frac{1}{\lambda_k}\frac{\lambda_k^2}{\lambda_k^2 + \lambda^2}

    We also know that our predicted data is given by:

    .. math::

        \hat{x}(t) = G\hat{j}(t)=C^\frac{1}{2}U\Pi w(t)

    And thus our predicted whitened data is just:

    .. math::

        \hat{w}(t) = U\Pi w(t)

    Where :math:`\Pi` is diagonal with entries entries:

    .. math::

        \lambda_k\gamma_k = \frac{\lambda_k^2}{\lambda_k^2 + \lambda^2}

    If we use no regularization, note that :math:`\Pi` is just the
    identity matrix. Here we test the squared magnitude of the difference
    between unregularized solution and regularized solutions, choosing the
    biggest regularization that achieves a :math:`\chi^2`-test significance
    of 0.001.

    .. versionadded:: 0.9.0
    """  # noqa: E501
    from scipy.stats import chi2
    _check_reference(evoked, inv['info']['ch_names'])
    _check_ch_names(inv, evoked.info)
    inv = prepare_inverse_operator(inv, evoked.nave, 1. / 9., 'MNE',
                                   copy='non-src')
    sel = _pick_channels_inverse_operator(evoked.ch_names, inv)
    logger.info('Picked %d channels from the data' % len(sel))
    data_white = np.dot(inv['whitener'], np.dot(inv['proj'], evoked.data[sel]))
    data_white_ef = np.dot(inv['eigen_fields']['data'], data_white)
    n_ch, n_times = data_white.shape

    # Adapted from mne_analyze/regularization.c, compute_regularization
    n_ch_eff = compute_rank_inverse(inv)
    n_zero = n_ch - n_ch_eff
    logger.info('Effective nchan = %d - %d = %d'
                % (n_ch, n_zero, n_ch_eff))
    del n_ch
    signal = np.sum(data_white ** 2, axis=0)  # sum of squares across channels
    snr = signal / n_ch_eff

    # Adapted from noise_regularization
    lambda2_est = np.empty(n_times)
    lambda2_est.fill(10.)
    remaining = np.ones(n_times, bool)

    # deal with low SNRs
    bad = (snr <= 1)
    lambda2_est[bad] = np.inf
    remaining[bad] = False

    # parameters
    lambda_mult = 0.99
    sing2 = (inv['sing'] * inv['sing'])[:, np.newaxis]
    val = chi2.isf(1e-3, n_ch_eff)
    for n_iter in range(1000):
        # get_mne_weights (ew=error_weights)
        # (split newaxis creation here for old numpy)
        f = sing2 / (sing2 + lambda2_est[np.newaxis][:, remaining])
        f[inv['sing'] == 0] = 0
        ew = data_white_ef[:, remaining] * (1.0 - f)
        # check condition
        err = np.sum(ew * ew, axis=0)
        remaining[np.where(remaining)[0][err < val]] = False
        if not remaining.any():
            break
        lambda2_est[remaining] *= lambda_mult
    else:
        warn('SNR estimation did not converge')
    snr_est = 1.0 / np.sqrt(lambda2_est)
    snr = np.sqrt(snr)
    return snr, snr_est
