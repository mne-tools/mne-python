# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Olaf Hauk <olaf.hauk@mrc-cbu.cam.ac.uk>
#
# License: BSD (3-clause)

import warnings
from copy import deepcopy
from math import sqrt
import numpy as np
from scipy import linalg

from ..io.constants import FIFF
from ..io.open import fiff_open
from ..io.tag import find_tag
from ..io.matrix import (_read_named_matrix, _transpose_named_matrix,
                         write_named_matrix)
from ..io.proj import _read_proj, make_projector, _write_proj
from ..io.tree import dir_tree_find
from ..io.write import (write_int, write_float_matrix, start_file,
                        start_block, end_block, end_file, write_float,
                        write_coord_trans, write_string)

from ..epochs import EpochsArray
from ..io.pick import channel_type, pick_info
from ..cov import prepare_noise_cov, _read_cov, _write_cov
from ..forward import (compute_depth_prior, read_forward_meas_info,
                       write_forward_meas_info, is_fixed_orient,
                       compute_orient_prior, _to_fixed_ori)
from ..source_space import (read_source_spaces_from_tree,
                            find_source_space_hemi, _get_vertno,
                            _write_source_spaces_to_fid, label_src_vertno_sel)
from ..transforms import invert_transform, transform_surface_to
from ..source_estimate import _make_stc
from ..utils import check_fname, logger, verbose
from functools import reduce


def _pick_channels_inverse_operator(ch_names, inv):
    """Gives the indices of the data channel to be used knowing
    an inverse operator
    """
    sel = []
    for name in inv['noise_cov']['names']:
        if name in ch_names:
            sel.append(ch_names.index(name))
        else:
            raise ValueError('The inverse operator was computed with '
                             'channel %s which is not present in '
                             'the data. You should compute a new inverse '
                             'operator restricted to the good data '
                             'channels.' % name)
    return sel


@verbose
def read_inverse_operator(fname, verbose=None):
    """Read the inverse operator decomposition from a FIF file

    Parameters
    ----------
    fname : string
        The name of the FIF file, which ends with -inv.fif or -inv.fif.gz.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    inv : dict
        The inverse operator.
    """
    check_fname(fname, 'inverse operator', ('-inv.fif', '-inv.fif.gz'))

    #
    #   Open the file, create directory
    #
    logger.info('Reading inverse operator decomposition from %s...'
                % fname)
    fid, tree, _ = fiff_open(fname, preload=True)
    #
    #   Find all inverse operators
    #
    invs = dir_tree_find(tree, FIFF.FIFFB_MNE_INVERSE_SOLUTION)
    if invs is None or len(invs) < 1:
        fid.close()
        raise Exception('No inverse solutions in %s' % fname)

    invs = invs[0]
    #
    #   Parent MRI data
    #
    parent_mri = dir_tree_find(tree, FIFF.FIFFB_MNE_PARENT_MRI_FILE)
    if len(parent_mri) == 0:
        fid.close()
        raise Exception('No parent MRI information in %s' % fname)
    parent_mri = parent_mri[0]  # take only first one

    logger.info('    Reading inverse operator info...')
    #
    #   Methods and source orientations
    #
    tag = find_tag(fid, invs, FIFF.FIFF_MNE_INCLUDED_METHODS)
    if tag is None:
        fid.close()
        raise Exception('Modalities not found')

    inv = dict()
    inv['methods'] = int(tag.data)

    tag = find_tag(fid, invs, FIFF.FIFF_MNE_SOURCE_ORIENTATION)
    if tag is None:
        fid.close()
        raise Exception('Source orientation constraints not found')

    inv['source_ori'] = int(tag.data)

    tag = find_tag(fid, invs, FIFF.FIFF_MNE_SOURCE_SPACE_NPOINTS)
    if tag is None:
        fid.close()
        raise Exception('Number of sources not found')

    inv['nsource'] = int(tag.data)
    inv['nchan'] = 0
    #
    #   Coordinate frame
    #
    tag = find_tag(fid, invs, FIFF.FIFF_MNE_COORD_FRAME)
    if tag is None:
        fid.close()
        raise Exception('Coordinate frame tag not found')

    inv['coord_frame'] = tag.data

    #
    #   Units
    #
    tag = find_tag(fid, invs, FIFF.FIFF_MNE_INVERSE_SOURCE_UNIT)
    if tag is not None:
        if tag.data == FIFF.FIFF_UNIT_AM:
            inv['units'] = 'Am'
        elif tag.data == FIFF.FIFF_UNIT_AM_M2:
            inv['units'] = 'Am/m^2'
        elif tag.data == FIFF.FIFF_UNIT_AM_M3:
            inv['units'] = 'Am/m^3'
        else:
            inv['units'] = None
    else:
        inv['units'] = None
    #
    #   The actual source orientation vectors
    #
    tag = find_tag(fid, invs, FIFF.FIFF_MNE_INVERSE_SOURCE_ORIENTATIONS)
    if tag is None:
        fid.close()
        raise Exception('Source orientation information not found')

    inv['source_nn'] = tag.data
    logger.info('    [done]')
    #
    #   The SVD decomposition...
    #
    logger.info('    Reading inverse operator decomposition...')
    tag = find_tag(fid, invs, FIFF.FIFF_MNE_INVERSE_SING)
    if tag is None:
        fid.close()
        raise Exception('Singular values not found')

    inv['sing'] = tag.data
    inv['nchan'] = len(inv['sing'])
    #
    #   The eigenleads and eigenfields
    #
    inv['eigen_leads_weighted'] = False
    eigen_leads = _read_named_matrix(fid, invs, FIFF.FIFF_MNE_INVERSE_LEADS)
    if eigen_leads is None:
        inv['eigen_leads_weighted'] = True
        eigen_leads = _read_named_matrix(fid, invs,
                                         FIFF.FIFF_MNE_INVERSE_LEADS_WEIGHTED)
    if eigen_leads is None:
        raise ValueError('Eigen leads not found in inverse operator.')
    #
    #   Having the eigenleads as columns is better for the inverse calculations
    #
    inv['eigen_leads'] = _transpose_named_matrix(eigen_leads, copy=False)
    inv['eigen_fields'] = _read_named_matrix(fid, invs,
                                             FIFF.FIFF_MNE_INVERSE_FIELDS)
    logger.info('    [done]')
    #
    #   Read the covariance matrices
    #
    inv['noise_cov'] = _read_cov(fid, invs, FIFF.FIFFV_MNE_NOISE_COV)
    logger.info('    Noise covariance matrix read.')

    inv['source_cov'] = _read_cov(fid, invs, FIFF.FIFFV_MNE_SOURCE_COV)
    logger.info('    Source covariance matrix read.')
    #
    #   Read the various priors
    #
    inv['orient_prior'] = _read_cov(fid, invs, FIFF.FIFFV_MNE_ORIENT_PRIOR_COV)
    if inv['orient_prior'] is not None:
        logger.info('    Orientation priors read.')

    inv['depth_prior'] = _read_cov(fid, invs, FIFF.FIFFV_MNE_DEPTH_PRIOR_COV)
    if inv['depth_prior'] is not None:
        logger.info('    Depth priors read.')

    inv['fmri_prior'] = _read_cov(fid, invs, FIFF.FIFFV_MNE_FMRI_PRIOR_COV)
    if inv['fmri_prior'] is not None:
        logger.info('    fMRI priors read.')

    #
    #   Read the source spaces
    #
    inv['src'] = read_source_spaces_from_tree(fid, tree, add_geom=False)

    for s in inv['src']:
        s['id'] = find_source_space_hemi(s)

    #
    #   Get the MRI <-> head coordinate transformation
    #
    tag = find_tag(fid, parent_mri, FIFF.FIFF_COORD_TRANS)
    if tag is None:
        fid.close()
        raise Exception('MRI/head coordinate transformation not found')
    else:
        mri_head_t = tag.data
        if mri_head_t['from'] != FIFF.FIFFV_COORD_MRI or \
                        mri_head_t['to'] != FIFF.FIFFV_COORD_HEAD:
            mri_head_t = invert_transform(mri_head_t)
            if mri_head_t['from'] != FIFF.FIFFV_COORD_MRI or \
                        mri_head_t['to'] != FIFF.FIFFV_COORD_HEAD:
                fid.close()
                raise Exception('MRI/head coordinate transformation '
                                'not found')

    inv['mri_head_t'] = mri_head_t

    #
    # get parent MEG info
    #
    inv['info'] = read_forward_meas_info(tree, fid)

    #
    #   Transform the source spaces to the correct coordinate frame
    #   if necessary
    #
    if inv['coord_frame'] != FIFF.FIFFV_COORD_MRI and \
            inv['coord_frame'] != FIFF.FIFFV_COORD_HEAD:
        fid.close()
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
    inv['reginv'] = []     # This the diagonal matrix implementing
                           # regularization and the inverse
    inv['noisenorm'] = []  # These are the noise-normalization factors
    #
    nuse = 0
    for k in range(len(inv['src'])):
        try:
            inv['src'][k] = transform_surface_to(inv['src'][k],
                                                 inv['coord_frame'],
                                                 mri_head_t)
        except Exception as inst:
            fid.close()
            raise Exception('Could not transform source space (%s)' % inst)

        nuse += inv['src'][k]['nuse']

    logger.info('    Source spaces transformed to the inverse solution '
                'coordinate frame')
    #
    #   Done!
    #
    fid.close()

    return inv


@verbose
def write_inverse_operator(fname, inv, verbose=None):
    """Write an inverse operator to a FIF file

    Parameters
    ----------
    fname : string
        The name of the FIF file, which ends with -inv.fif or -inv.fif.gz.
    inv : dict
        The inverse operator.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
    """
    check_fname(fname, 'inverse operator', ('-inv.fif', '-inv.fif.gz'))

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

    if 'units' in inv:
        if inv['units'] == 'Am':
            write_int(fid, FIFF.FIFF_MNE_INVERSE_SOURCE_UNIT,
                      FIFF.FIFF_UNIT_AM)
        elif inv['units'] == 'Am/m^2':
            write_int(fid, FIFF.FIFF_MNE_INVERSE_SOURCE_UNIT,
                      FIFF.FIFF_UNIT_AM_M2)
        elif inv['units'] == 'Am/m^3':
            write_int(fid, FIFF.FIFF_MNE_INVERSE_SOURCE_UNIT,
                      FIFF.FIFF_UNIT_AM_M3)

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
        write_named_matrix(fid, FIFF.FIFF_MNE_INVERSE_LEADS_WEIGHTED,
                           _transpose_named_matrix(inv['eigen_leads']))
    else:
        write_named_matrix(fid, FIFF.FIFF_MNE_INVERSE_LEADS,
                           _transpose_named_matrix(inv['eigen_leads']))

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
    """Compute the three Cartesian components of a vector or matrix together

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

    n, p = vec.shape
    if np.iscomplexobj(vec):
        vec = np.abs(vec)
    comb = vec[0::3] ** 2
    comb += vec[1::3] ** 2
    comb += vec[2::3] ** 2
    if not square:
        comb = np.sqrt(comb)
    return comb


def _check_ch_names(inv, info):
    """Check that channels in inverse operator are measurements"""

    inv_ch_names = inv['eigen_fields']['col_names']

    if inv['noise_cov']['names'] != inv_ch_names:
        raise ValueError('Channels in inverse operator eigen fields do not '
                         'match noise covariance channels.')
    data_ch_names = info['ch_names']

    missing_ch_names = list()
    for ch_name in inv_ch_names:
        if ch_name not in data_ch_names:
            missing_ch_names.append(ch_name)
    n_missing = len(missing_ch_names)
    if n_missing > 0:
        raise ValueError('%d channels in inverse operator ' % n_missing +
                         'are not present in the data (%s)' % missing_ch_names)


@verbose
def prepare_inverse_operator(orig, nave, lambda2, method, verbose=None):
    """Prepare an inverse operator for actually computing the inverse

    Parameters
    ----------
    orig : dict
        The inverse operator structure read from a file.
    nave : int
        Number of averages (scales the noise covariance).
    lambda2 : float
        The regularization factor. Recommended to be 1 / SNR**2.
    method : "MNE" | "dSPM" | "sLORETA"
        Use mininum norm, dSPM or sLORETA.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    inv : dict
        Prepared inverse operator.
    """
    if nave <= 0:
        raise ValueError('The number of averages should be positive')

    logger.info('Preparing the inverse operator for use...')
    inv = deepcopy(orig)
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
    sing = np.array(inv['sing'], dtype=np.float64)
    inv['reginv'] = sing / (sing ** 2 + lambda2)
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
    if not inv['noise_cov']['diag']:
        inv['whitener'] = np.zeros((inv['noise_cov']['dim'],
                                    inv['noise_cov']['dim']))
        #
        #   Omit the zeroes due to projection
        #
        eig = inv['noise_cov']['eig']
        nzero = (eig > 0)
        inv['whitener'][nzero, nzero] = 1.0 / np.sqrt(eig[nzero])
        #
        #   Rows of eigvec are the eigenvectors
        #
        inv['whitener'] = np.dot(inv['whitener'], inv['noise_cov']['eigvec'])
        logger.info('    Created the whitener using a full noise '
                    'covariance matrix (%d small eigenvalues omitted)'
                    % (inv['noise_cov']['dim'] - np.sum(nzero)))
    else:
        #
        #   No need to omit the zeroes due to projection
        #
        inv['whitener'] = np.diag(1.0 /
                                  np.sqrt(inv['noise_cov']['data'].ravel()))
        logger.info('    Created the whitener using a diagonal noise '
                    'covariance matrix (%d small eigenvalues discarded)'
                    % ncomp)

    #
    #   Finally, compute the noise-normalization factors
    #
    if method in ["dSPM", 'sLORETA']:
        if method == "dSPM":
            logger.info('    Computing noise-normalization factors '
                        '(dSPM)...')
            noise_weight = inv['reginv']
        else:
            logger.info('    Computing noise-normalization factors '
                        '(sLORETA)...')
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
            #   The variances at three consequtive entries must be squared and
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

    return inv


@verbose
def _assemble_kernel(inv, label, method, pick_ori, verbose=None):
    #
    #   Simple matrix multiplication followed by combination of the
    #   current components
    #
    #   This does all the data transformations to compute the weights for the
    #   eigenleads
    #
    eigen_leads = inv['eigen_leads']['data']
    source_cov = inv['source_cov']['data'][:, None]
    if method != "MNE":
        noise_norm = inv['noisenorm'][:, None]

    src = inv['src']
    vertno = _get_vertno(src)

    if label is not None:
        vertno, src_sel = label_src_vertno_sel(label, inv['src'])

        if method != "MNE":
            noise_norm = noise_norm[src_sel]

        if inv['source_ori'] == FIFF.FIFFV_MNE_FREE_ORI:
            src_sel = 3 * src_sel
            src_sel = np.c_[src_sel, src_sel + 1, src_sel + 2]
            src_sel = src_sel.ravel()

        eigen_leads = eigen_leads[src_sel]
        source_cov = source_cov[src_sel]

    if pick_ori == "normal":
        if not inv['source_ori'] == FIFF.FIFFV_MNE_FREE_ORI:
            raise ValueError('Picking normal orientation can only be done '
                             'with a free orientation inverse operator.')

        is_loose = 0 < inv['orient_prior']['data'][0] < 1
        if not is_loose:
            raise ValueError('Picking normal orientation can only be done '
                             'when working with loose orientations.')

        # keep only the normal components
        eigen_leads = eigen_leads[2::3]
        source_cov = source_cov[2::3]

    trans = inv['reginv'][:, None] * reduce(np.dot,
                                            [inv['eigen_fields']['data'],
                                             inv['whitener'],
                                             inv['proj']])
    #
    #   Transformation into current distributions by weighting the eigenleads
    #   with the weights computed above
    #
    if inv['eigen_leads_weighted']:
        #
        #     R^0.5 has been already factored in
        #
        logger.info('(eigenleads already weighted)...')
        K = np.dot(eigen_leads, trans)
    else:
        #
        #     R^0.5 has to be factored in
        #
        logger.info('(eigenleads need to be weighted)...')
        K = np.sqrt(source_cov) * np.dot(eigen_leads, trans)

    if method == "MNE":
        noise_norm = None

    return K, noise_norm, vertno


def _check_method(method):
    if method not in ["MNE", "dSPM", "sLORETA"]:
        raise ValueError('method parameter should be "MNE" or "dSPM" '
                         'or "sLORETA".')
    return method


def _check_ori(pick_ori, pick_normal):
    if pick_normal is not None:
        warnings.warn('DEPRECATION: The pick_normal parameter has been '
                      'changed to pick_ori. Please update your code.')
        pick_ori = pick_normal
    if pick_ori is True:
        warnings.warn('DEPRECATION: The pick_ori parameter should now be None '
                      'or "normal".')
        pick_ori = "normal"
    elif pick_ori is False:
        warnings.warn('DEPRECATION: The pick_ori parameter should now be None '
                      'or "normal".')
        pick_ori = None

    if pick_ori not in [None, "normal"]:
        raise ValueError('The pick_ori parameter should now be None or '
                         '"normal".')
    return pick_ori


def _subject_from_inverse(inverse_operator):
    """Get subject id from inverse operator"""
    return inverse_operator['src'][0].get('subject_his_id', None)


@verbose
def apply_inverse(evoked, inverse_operator, lambda2, method="dSPM",
                  pick_ori=None, verbose=None, pick_normal=None):
    """Apply inverse operator to evoked data

    Computes a L2-norm inverse solution
    Actual code using these principles might be different because
    the inverse operator is often reused across data sets.

    Parameters
    ----------
    evoked : Evoked object
        Evoked data.
    inverse_operator: dict
        Inverse operator read with mne.read_inverse_operator.
    lambda2 : float
        The regularization parameter.
    method : "MNE" | "dSPM" | "sLORETA"
        Use mininum norm, dSPM or sLORETA.
    pick_ori : None | "normal"
        If "normal", rather than pooling the orientations by taking the norm,
        only the radial component is kept. This is only implemented
        when working with loose orientations.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    stc : SourceEstimate | VolSourceEstimate
        The source estimates
    """
    method = _check_method(method)
    pick_ori = _check_ori(pick_ori, pick_normal)
    #
    #   Set up the inverse according to the parameters
    #
    nave = evoked.nave

    _check_ch_names(inverse_operator, evoked.info)

    inv = prepare_inverse_operator(inverse_operator, nave, lambda2, method)
    #
    #   Pick the correct channels from the data
    #
    sel = _pick_channels_inverse_operator(evoked.ch_names, inv)
    logger.info('Picked %d channels from the data' % len(sel))
    logger.info('Computing inverse...')
    K, noise_norm, _ = _assemble_kernel(inv, None, method, pick_ori)
    sol = np.dot(K, evoked.data[sel])  # apply imaging kernel

    is_free_ori = (inverse_operator['source_ori'] == FIFF.FIFFV_MNE_FREE_ORI
                   and pick_ori is None)

    if is_free_ori:
        logger.info('combining the current components...')
        sol = combine_xyz(sol)

    if noise_norm is not None:
        logger.info('(dSPM)...')
        sol *= noise_norm

    tstep = 1.0 / evoked.info['sfreq']
    tmin = float(evoked.times[0])
    vertno = _get_vertno(inv['src'])
    subject = _subject_from_inverse(inverse_operator)

    stc = _make_stc(sol, vertices=vertno, tmin=tmin, tstep=tstep,
                    subject=subject)
    logger.info('[done]')

    return stc


@verbose
def apply_inverse_raw(raw, inverse_operator, lambda2, method="dSPM",
                      label=None, start=None, stop=None, nave=1,
                      time_func=None, pick_ori=None,
                      buffer_size=None, verbose=None,
                      pick_normal=None):
    """Apply inverse operator to Raw data

    Computes a L2-norm inverse solution
    Actual code using these principles might be different because
    the inverse operator is often reused across data sets.

    Parameters
    ----------
    raw : Raw object
        Raw data.
    inverse_operator : dict
        Inverse operator read with mne.read_inverse_operator.
    lambda2 : float
        The regularization parameter.
    method : "MNE" | "dSPM" | "sLORETA"
        Use mininum norm, dSPM or sLORETA.
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
    pick_ori : None | "normal"
        If "normal", rather than pooling the orientations by taking the norm,
        only the radial component is kept. This is only implemented
        when working with loose orientations.
    buffer_size : int (or None)
        If not None, the computation of the inverse and the combination of the
        current components is performed in segments of length buffer_size
        samples. While slightly slower, this is useful for long datasets as it
        reduces the memory requirements by approx. a factor of 3 (assuming
        buffer_size << data length).
        Note that this setting has no effect for fixed-orientation inverse
        operators.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    stc : SourceEstimate | VolSourceEstimate
        The source estimates.
    """
    method = _check_method(method)
    pick_ori = _check_ori(pick_ori, pick_normal)

    _check_ch_names(inverse_operator, raw.info)

    #
    #   Set up the inverse according to the parameters
    #
    inv = prepare_inverse_operator(inverse_operator, nave, lambda2, method)
    #
    #   Pick the correct channels from the data
    #
    sel = _pick_channels_inverse_operator(raw.ch_names, inv)
    logger.info('Picked %d channels from the data' % len(sel))
    logger.info('Computing inverse...')

    data, times = raw[sel, start:stop]

    if time_func is not None:
        data = time_func(data)

    K, noise_norm, vertno = _assemble_kernel(inv, label, method, pick_ori)

    is_free_ori = (inverse_operator['source_ori'] == FIFF.FIFFV_MNE_FREE_ORI
                   and pick_ori is None)

    if buffer_size is not None and is_free_ori:
        # Process the data in segments to conserve memory
        n_seg = int(np.ceil(data.shape[1] / float(buffer_size)))
        logger.info('computing inverse and combining the current '
                    'components (using %d segments)...' % (n_seg))

        # Allocate space for inverse solution
        n_times = data.shape[1]
        sol = np.empty((K.shape[0] // 3, n_times),
                       dtype=(K[0, 0] * data[0, 0]).dtype)

        for pos in range(0, n_times, buffer_size):
            sol[:, pos:pos + buffer_size] = \
                combine_xyz(np.dot(K, data[:, pos:pos + buffer_size]))

            logger.info('segment %d / %d done..'
                        % (pos / buffer_size + 1, n_seg))
    else:
        sol = np.dot(K, data)
        if is_free_ori:
            logger.info('combining the current components...')
            sol = combine_xyz(sol)

    if noise_norm is not None:
        sol *= noise_norm

    tmin = float(times[0])
    tstep = 1.0 / raw.info['sfreq']
    subject = _subject_from_inverse(inverse_operator)
    stc = _make_stc(sol, vertices=vertno, tmin=tmin, tstep=tstep,
                    subject=subject)
    logger.info('[done]')

    return stc


def point_spread_function(inverse_operator, forward, labels, method='dSPM',
                          lambda2=1 / 9., pick_ori=None, mode='mean',
                          svd_comp=1):
    """Compute point-spread functions (PSFs) for linear estimators

    Compute point-spread functions (PSF) in labels for a combination of inverse
    operator and forward solution. PSFs are computed for test sources that are
    perpendicular to cortical surface

    Parameters
    ----------
    inverse_operator: dict
        Inverse operator read with mne.read_inverse_operator.
    forward: dict
        Forward solution, created with "surf_ori=True" and "force_fixed=False"
        Note: (Bad) channels not included in forward solution will not be used
        in PSF computation.
    labels: list of Label
        Labels for which PSFs shall be computed.
    method: 'MNE' | 'dSPM' | 'sLORETA'
        Inverse method for which PSFs shall be computed (for apply_inverse).
    lambda2 : float
        The regularization parameter (for apply_inverse).
    pick_ori : None | "normal"
        If "normal", rather than pooling the orientations by taking the norm,
        only the radial component is kept. This is only implemented
        when working with loose orientations (for apply_inverse).
    mode: 'mean' | 'sum' | 'svd' |
        PSFs can be computed for different summary measures with labels:
        'sum' or 'mean': sum or means of sub-leadfields for labels
        This corresponds to situations where labels can be assumed to be
        homogeneously activated.
        'svd': SVD components of sub-leadfields for labels
        This is better suited for situations where activation patterns are
        assumed to be more variable.
        "sub-leadfields" are the parts of the forward solutions that belong to
        vertices within invidual labels
    svd_comp: integer
        Number of SVD components for which PSFs will be computed and output
        (irrelevant for 'sum' and 'mean'). Explained variances within
        sub-leadfields are shown in screen output

    Returns
    -------
    stc_psf : SourceEstimate
        The PSFs for the specified labels
        If mode='svd': svd_comp components per label are created
        (i.e. svd_comp successive time points in mne_analyze)
        The last sample is the summed PSF across all labels
        Scaling of PSFs is arbitrary, and may differ greatly among methods
        (especially for MNE compared to noise-normalized estimates)
    evoked_fwd: Evoked
        Forward solutions corresponding to PSFs in stc_psf
        If mode='svd': svd_comp components per label are created
        (i.e. svd_comp successive time points in mne_analyze)
        The last sample is the summed forward solution across all labels
        (sum is taken across summary measures)
    label_singvals: list of numpy arrays
        Singular values of svd for sub-leadfields
        Provides information about how well labels are represented by chosen
        components. Explained variances within sub-leadfields are shown in
        screen output
    """
    if mode.lower() not in ['mean', 'svd']:
        raise ValueError('mode must be ''svd'' or ''mean''. Got %s.' %
                         mode.lower())

    logger.info("\nAbout to process %d labels" % len(labels))

    # get whole leadfield matrix with normal dipole components
    leadfield = forward['sol']['data'][:, 2::3]

    # in order to convert sub-leadfield matrix to evoked data type (pretending
    # it's an epoch, see in loop below), uses 'info' from forward solution,
    # need to add 'sfreq' and 'proj'
    info = forward['info']
    info['sfreq'] = 1000.  # add sfreq or it won't work
    info['projs'] = []  # add projs

    # will contain means of subleadfields for all labels
    label_psf_summary = np.array(0)
    # if mode='svd', this will collect all SVD singular values for labels
    label_singvals = []

    # loop over labels
    for ll in labels:
        logger.info(ll)
        if ll.hemi == 'rh':
            # for RH labels, add number of LH vertices
            offset = forward['src'][0]['vertno'].shape[0]
            # remember whether we are in the LH or RH
            this_hemi = 1
        elif ll.hemi == 'lh':
            offset = 0
            this_hemi = 0

        # get vertices on cortical surface inside label
        idx = np.intersect1d(ll.vertices, forward['src'][this_hemi]['vertno'])

        # get vertices in source space inside label
        fwd_idx = np.searchsorted(forward['src'][this_hemi]['vertno'], idx)

        # get sub-leadfield matrix for label vertices
        sub_leadfield = leadfield[:, fwd_idx + offset]

        # compute summary data for labels
        if mode.lower() == 'sum':  # sum across forward solutions in label
            logger.info("Computing sums within labels")
            this_label_psf_summary = sub_leadfield.sum(axis=1)

        elif mode.lower() == 'mean':
            logger.info("Computing means within labels")
            this_label_psf_summary = sub_leadfield.mean(axis=1)

        elif mode.lower() == 'svd':  # takes svd of forward solutions in label
            logger.info("Computing SVD within labels, using %d component(s)"
                        % svd_comp)

            # compute SVD of sub-leadfield
            u_svd, s_svd, _ = np.linalg.svd(sub_leadfield,
                                            full_matrices=False,
                                            compute_uv=True)

            # keep singular values (might be useful to some people)
            label_singvals.append(s_svd)

            # get first svd_comp components, weighted with their corresponding
            # singular values
            logger.info("first 5 singular values:")
            logger.info(s_svd[0:5])
            logger.info("(This tells you something about variability of "
                        "forward solutions in sub-leadfield for label)")
            # explained variance by chosen components within sub-leadfield
            my_comps = s_svd[0:svd_comp]
            comp_var = (100 * np.sum(np.power(my_comps, 2)) /
                        np.sum(np.power(s_svd, 2)))
            logger.info("Your %d component(s) explain(s) %.1f%% "
                        "variance.\n" % (svd_comp, comp_var))
            this_label_psf_summary = np.dot(u_svd[:, 0:svd_comp],
                                            np.diag(s_svd[0:svd_comp]))
            # transpose required for conversion to "evoked"
            this_label_psf_summary = this_label_psf_summary.T

        # initialise or append to existing collection
        if label_psf_summary.shape == ():
            label_psf_summary = this_label_psf_summary
        else:
            label_psf_summary = np.vstack((label_psf_summary,
                                           this_label_psf_summary))

    # compute sum across forward solutions for labels, append to end
    label_psf_summary = np.vstack((label_psf_summary,
                                   label_psf_summary.sum(axis=0)))
    # transpose required for conversion to "evoked"
    label_psf_summary = label_psf_summary.T

    # convert sub-leadfield matrix to evoked data type (a bit of a hack)
    evoked_fwd = EpochsArray(label_psf_summary[None, :, :], info,
                             np.zeros((1, 3), dtype=int)).average()

    # compute PSFs by applying inverse operator to sub-leadfields
    logger.info("About to apply inverse operator for method='%s' and "
                "lambda2=%f\n" % (method, lambda2))

    stc_psf = apply_inverse(evoked_fwd, inverse_operator, lambda2,
                            method=method, pick_ori=pick_ori)

    return stc_psf, evoked_fwd, label_singvals


def _get_matrix_from_inverse_operator(inverse_operator, forward, labels=None,
                                      method='dSPM', lambda2=3, mode='mean',
                                      svd_comp=1):
    """

    Get inverse matrix from an inverse operator for specific parameter settings
    Currently works only for fixed/loose orientation constraints
    For loose orientation constraint, the CTFs are computed for the radial
    component (pick_ori='normal')

    Parameters
    ----------
    inverse_operator : dict
        Inverse operator read with mne.read_inverse_operator.
    forward : dict
         The forward operator.
    method : 'MNE' | 'dSPM' | 'sLORETA'
        Inverse methods (for apply_inverse).
    labels : list of Label | None
        Labels for which CTFs shall be computed. If None, inverse matrix for
        all vertices will be returned.
    lambda2 : float
        The regularization parameter (for apply_inverse).
    pick_ori : None | "normal"
        pick_ori : None | "normal"
        If "normal", rather than pooling the orientations by taking the norm,
        only the radial component is kept. This is only implemented
        when working with loose orientations (for apply_inverse).
        Determines whether whole inverse matrix G will have one or three rows
        per vertex. This will also affect summary measures for labels.
    mode : 'mean' | 'sum' | 'svd'
        CTFs can be computed for different summary measures with labels:
        'sum' or 'mean': sum or means of sub-inverse for labels
        This corresponds to situations where labels can be assumed to be
        homogeneously activated.
        'svd': SVD components of sub-inverse for labels
        This is better suited for situations where activation patterns are
        assumed to be more variable.
        "sub-inverse" is the part of the inverse matrix that belongs to
        vertices within invidual labels.
    svd_comp : integer
        Number of SVD components for which CTFs will be computed and output
        (irrelevant for 'sum' and 'mean'). Explained variances within
        sub-inverses are shown in screen output.

    Returns
    -------
    invmat : list numpy arrays
        Inverse matrix associated with inverse operator and specified
        parameters.
    label_singvals : list of numpy arrays
        Singular values of svd for sub-inverses
        Provides information about how well labels are represented by chosen
        components. Explained variances within sub-inverses are shown in
        screen output.
    """
    if labels:
        logger.info("\nAbout to process %d labels" % len(labels))
    else:
        logger.info("\nComputing whole inverse operator.")

    # in order to convert sub-leadfield matrix to evoked data type (pretending
    # it's an epoch, see in loop below), uses 'info' from forward solution,
    # need to add 'sfreq' and 'proj'
    info = forward['info']
    info['sfreq'] = 1000.  # add sfreq or it won't work
    info['projs'] = []  # add projs

    # create identity matrix as input for inverse operator
    id_mat = np.eye(forward['nchan'])

    # convert identity matrix to evoked data type (pretending it's an epoch)
    ev_id = EpochsArray(id_mat[None, :, :], info,
                        np.zeros((1, 3), dtype=int)).average()

    snr = 3.0
    lambda2 = 1.0 / snr ** 2

    # apply inverse operator to identity matrix in order to get inverse matrix
    # free orientation constraint not possible because apply_inverse would
    # combined components
    invmat_mat_op = apply_inverse(ev_id, inverse_operator, lambda2=lambda2,
                                  method=method, pick_ori='normal')

    logger.info("\nDimension of inverse matrix:")
    logger.info(invmat_mat_op.shape)

    # turn source estimate into numpty array
    invmat_mat = invmat_mat_op.data

    invmat_summary = np.array(0)
    # if mode='svd', label_singvals will collect all SVD singular values for
    # labels
    label_singvals = []

    if labels:
        for ll in labels:
            print ll
            if ll.hemi == 'rh':
                # for RH labels, add number of LH vertices
                offset = forward['src'][0]['vertno'].shape[0]
                # remember whether we are in the LH or RH
                this_hemi = 1
            elif ll.hemi == 'lh':
                offset = 0
                this_hemi = 0
            else:
                print "Cannot determine hemisphere of label.\n"

            # get vertices on cortical surface inside label
            idx = np.intersect1d(ll.vertices,
                                 forward['src'][this_hemi]['vertno'])

            # get vertices in source space inside label
            fwd_idx = np.searchsorted(forward['src'][this_hemi]['vertno'], idx)

            # get sub-inverse for label vertices, one row per vertex
            invmat_lbl = invmat_mat[fwd_idx + offset, :]

            # compute summary data for labels
            if mode.lower() == 'sum':  # takes sum across estimators in label
                logger.info("Computing sums within labels")
                this_invmat_summary = invmat_lbl.sum(axis=0)

            elif mode.lower() == 'mean':
                logger.info("Computing means within labels")
                this_invmat_summary = invmat_lbl.mean(axis=0)

            elif mode.lower() == 'svd':  # takes svd of sub-inverse in label
                logger.info("Computing SVD within labels, using %d "
                            "component(s)" % svd_comp)

                # compute SVD of sub-inverse
                u_svd, s_svd, _ = np.linalg.svd(invmat_lbl.T,
                                                full_matrices=False,
                                                compute_uv=True)

                # keep singular values (might be useful to some people)
                label_singvals.append(s_svd)

                # get first svd_comp components, weighted with their
                # corresponding singular values
                logger.info("first 5 singular values:")
                logger.info(s_svd[0:5])
                logger.info("(This tells you something about variability of "
                            "estimators in sub-inverse for label)")
                # explained variance by chosen components within sub-inverse
                my_comps = s_svd[0:svd_comp]
                comp_var = (100 * np.sum(np.power(my_comps, 2)) /
                            np.sum(np.power(s_svd, 2)))
                logger.info("Your %d component(s) explain(s) %.1f%% "
                            "variance.\n" % (svd_comp, comp_var))
                this_invmat_summary = np.dot(u_svd[:, 0:svd_comp],
                                             np.diag(s_svd[0:svd_comp]))
                this_invmat_summary = this_invmat_summary.T

            if invmat_summary.shape == ():
                invmat_summary = this_invmat_summary
            else:
                invmat_summary = np.vstack((invmat_summary,
                                            this_invmat_summary))

        invmat = invmat_summary

    else:   # no labels provided: return whole matrix
        invmat = invmat_mat

    return invmat, label_singvals


def cross_talk_function(inverse_operator, forward, labels,
                        method='dSPM', lambda2=1 / 9.,
                        mode='mean', svd_comp=1):
    """Compute cross-talk functions (CTFs) for linear estimators

    Compute cross-talk functions (CTF) in labels for a combination of inverse
    operator and forward solution. CTFs are computed for test sources that are
    perpendicular to cortical surface.

    Parameters
    ----------
    inverse_operator : dict
        Inverse operator read with mne.read_inverse_operator.
    forward : dict
         Forward solution, created with "force_fixed=True"
         Note: (Bad) channels not included in forward solution will not be used
         in CTF computation.
    method : 'MNE' | 'dSPM' | 'sLORETA'
        Inverse method for which CTFs shall be computed.
    labels : list of Label
        Labels for which CTFs shall be computed.
    lambda2 : float
        The regularization parameter.
    mode : 'mean' | 'sum' | 'svd'
        CTFs can be computed for different summary measures with labels:
        'sum' or 'mean': sum or means of sub-inverses for labels
        This corresponds to situations where labels can be assumed to be
        homogeneously activated.
        'svd': SVD components of sub-inverses for labels
        This is better suited for situations where activation patterns are
        assumed to be more variable. "sub-inverse" is the part of the inverse
        matrix that belongs to vertices within invidual labels.
    svd_comp : int
        Number of SVD components for which CTFs will be computed and output
        (irrelevant for 'sum' and 'mean'). Explained variances within
        sub-inverses are shown in screen output.

    Returns
    -------
    stc_ctf : SourceEstimate
        The CTFs for the specified labels
        If mode='svd': svd_comp components per label are created
        (i.e. svd_comp successive time points in mne_analyze)
        The last sample is the summed CTF across all labels
    label_singvals : list of numpy arrays
        Singular values of svd for sub-inverses
        Provides information about how well labels are represented by chosen
        components. Explained variances within sub-inverses are shown in screen
        output.
    """

    # get the inverse matrix corresponding to inverse operator
    invmat, label_singvals = _get_matrix_from_inverse_operator(inverse_operator,
                                                               forward,
                                                               labels=labels,
                                                               method=method,
                                                               lambda2=lambda2,
                                                               mode=mode,
                                                               svd_comp=svd_comp)

    # get the leadfield matrix from forward solution
    leadfield = forward['sol']['data']

    # compute cross-talk functions (CTFs)
    ctfs = np.dot(invmat, leadfield)

    # compute sum across forward solutions for labels, append to end
    ctfs = np.vstack((ctfs, ctfs.sum(axis=0)))

    # create a dummy source estimate and put in the CTFs, in order to write
    # them to STC file

    # in order to convert sub-leadfield matrix to evoked data type (pretending
    # it's an epoch, see in loop below), uses 'info' from forward solution,
    # need to add 'sfreq' and 'proj'
    info = forward['info']
    info['sfreq'] = 1000.  # add sfreq or it won't work
    info['projs'] = []  # add projs

    # create identity matrix as input for inverse operator
    id_mat = np.eye(forward['nchan'])

    # convert identity matrix to evoked data type (pretending it's an epoch)
    ev_id = EpochsArray(id_mat[None, :, 0:len(labels) + 1], info,
                        np.zeros((1, 3), dtype=int)).average()

    # apply inverse operator to dummy data to create dummy source estimate, in
    # this case fixed orientation constraint
    stc_ctf = apply_inverse(ev_id, inverse_operator, lambda2=3, method='MNE')

    # insert CTF into source estimate object
    stc_ctf._data = ctfs.T

    return stc_ctf, label_singvals


def _apply_inverse_epochs_gen(epochs, inverse_operator, lambda2, method='dSPM',
                              label=None, nave=1, pick_ori=None,
                              verbose=None, pick_normal=None):
    """ see apply_inverse_epochs """
    method = _check_method(method)
    pick_ori = _check_ori(pick_ori, pick_normal)

    _check_ch_names(inverse_operator, epochs.info)

    #
    #   Set up the inverse according to the parameters
    #
    inv = prepare_inverse_operator(inverse_operator, nave, lambda2, method)
    #
    #   Pick the correct channels from the data
    #
    sel = _pick_channels_inverse_operator(epochs.ch_names, inv)
    logger.info('Picked %d channels from the data' % len(sel))
    logger.info('Computing inverse...')
    K, noise_norm, vertno = _assemble_kernel(inv, label, method, pick_ori)

    tstep = 1.0 / epochs.info['sfreq']
    tmin = epochs.times[0]

    is_free_ori = (inverse_operator['source_ori'] == FIFF.FIFFV_MNE_FREE_ORI
                   and pick_ori is None)

    if not is_free_ori and noise_norm is not None:
        # premultiply kernel with noise normalization
        K *= noise_norm

    subject = _subject_from_inverse(inverse_operator)
    for k, e in enumerate(epochs):
        logger.info('Processing epoch : %d' % (k + 1))
        if is_free_ori:
            # Compute solution and combine current components (non-linear)
            sol = np.dot(K, e[sel])  # apply imaging kernel
            if is_free_ori:
                logger.info('combining the current components...')
                sol = combine_xyz(sol)

                if noise_norm is not None:
                    sol *= noise_norm
        else:
            # Linear inverse: do computation here or delayed
            if len(sel) < K.shape[0]:
                sol = (K, e[sel])
            else:
                sol = np.dot(K, e[sel])

        stc = _make_stc(sol, vertices=vertno, tmin=tmin, tstep=tstep,
                        subject=subject)

        yield stc

    logger.info('[done]')


@verbose
def apply_inverse_epochs(epochs, inverse_operator, lambda2, method="dSPM",
                         label=None, nave=1, pick_ori=None,
                         return_generator=False, verbose=None,
                         pick_normal=None):
    """Apply inverse operator to Epochs

    Computes a L2-norm inverse solution on each epochs and returns
    single trial source estimates.

    Parameters
    ----------
    epochs : Epochs object
        Single trial epochs.
    inverse_operator : dict
        Inverse operator read with mne.read_inverse_operator.
    lambda2 : float
        The regularization parameter.
    method : "MNE" | "dSPM" | "sLORETA"
        Use mininum norm, dSPM or sLORETA.
    label : Label | None
        Restricts the source estimates to a given label. If None,
        source estimates will be computed for the entire source space.
    nave : int
        Number of averages used to regularize the solution.
        Set to 1 on single Epoch by default.
    pick_ori : None | "normal"
        If "normal", rather than pooling the orientations by taking the norm,
        only the radial component is kept. This is only implemented
        when working with loose orientations.
    return_generator : bool
        Return a generator object instead of a list. This allows iterating
        over the stcs without having to keep them all in memory.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    stc : list of SourceEstimate or VolSourceEstimate
        The source estimates for all epochs.
    """
    stcs = _apply_inverse_epochs_gen(epochs, inverse_operator, lambda2,
                                     method=method, label=label, nave=nave,
                                     pick_ori=pick_ori, verbose=verbose,
                                     pick_normal=pick_normal)

    if not return_generator:
        # return a list
        stcs = [stc for stc in stcs]

    return stcs


def _xyz2lf(Lf_xyz, normals):
    """Reorient leadfield to one component matching the normal to the cortex

    This program takes a leadfield matix computed for dipole components
    pointing in the x, y, and z directions, and outputs a new lead field
    matrix for dipole components pointing in the normal direction of the
    cortical surfaces and in the two tangential directions to the cortex
    (that is on the tangent cortical space). These two tangential dipole
    components are uniquely determined by the SVD (reduction of variance).

    Parameters
    ----------
    Lf_xyz: array of shape [n_sensors, n_positions x 3]
        Leadfield
    normals : array of shape [n_positions, 3]
        Normals to the cortex

    Returns
    -------
    Lf_cortex : array of shape [n_sensors, n_positions x 3]
        Lf_cortex is a leadfield matrix for dipoles in rotated orientations, so
        that the first column is the gain vector for the cortical normal dipole
        and the following two column vectors are the gain vectors for the
        tangential orientations (tangent space of cortical surface).
    """
    n_sensors, n_dipoles = Lf_xyz.shape
    n_positions = n_dipoles // 3
    Lf_xyz = Lf_xyz.reshape(n_sensors, n_positions, 3)
    n_sensors, n_positions, _ = Lf_xyz.shape
    Lf_cortex = np.zeros_like(Lf_xyz)

    for k in range(n_positions):
        lf_normal = np.dot(Lf_xyz[:, k, :], normals[k])
        lf_normal_n = lf_normal[:, None] / linalg.norm(lf_normal)
        P = np.eye(n_sensors, n_sensors) - np.dot(lf_normal_n, lf_normal_n.T)
        lf_p = np.dot(P, Lf_xyz[:, k, :])
        U, s, Vh = linalg.svd(lf_p)
        Lf_cortex[:, k, 0] = lf_normal
        Lf_cortex[:, k, 1:] = np.c_[U[:, 0] * s[0], U[:, 1] * s[1]]

    Lf_cortex = Lf_cortex.reshape(n_sensors, n_dipoles)
    return Lf_cortex


###############################################################################
# Assemble the inverse operator

@verbose
def _prepare_forward(forward, info, noise_cov, pca=False, verbose=None):
    """Util function to prepare forward solution for inverse solvers
    """
    fwd_ch_names = [c['ch_name'] for c in forward['info']['chs']]
    ch_names = [c['ch_name'] for c in info['chs']
                if (c['ch_name'] not in info['bads']
                    and c['ch_name'] not in noise_cov['bads'])
                and (c['ch_name'] in fwd_ch_names
                     and c['ch_name'] in noise_cov.ch_names)]

    if not len(info['bads']) == len(noise_cov['bads']) or \
            not all([b in noise_cov['bads'] for b in info['bads']]):
        logger.info('info["bads"] and noise_cov["bads"] do not match, '
                    'excluding bad channels from both')

    n_chan = len(ch_names)
    logger.info("Computing inverse operator with %d channels." % n_chan)

    #
    #   Handle noise cov
    #
    noise_cov = prepare_noise_cov(noise_cov, info, ch_names)

    #   Omit the zeroes due to projection
    eig = noise_cov['eig']
    nzero = (eig > 0)
    n_nzero = sum(nzero)

    if pca:
        #   Rows of eigvec are the eigenvectors
        whitener = noise_cov['eigvec'][nzero] / np.sqrt(eig[nzero])[:, None]
        logger.info('Reducing data rank to %d' % n_nzero)
    else:
        whitener = np.zeros((n_chan, n_chan), dtype=np.float)
        whitener[nzero, nzero] = 1.0 / np.sqrt(eig[nzero])
        #   Rows of eigvec are the eigenvectors
        whitener = np.dot(whitener, noise_cov['eigvec'])

    gain = forward['sol']['data']

    fwd_idx = [fwd_ch_names.index(name) for name in ch_names]
    gain = gain[fwd_idx]
    info_idx = [info['ch_names'].index(name) for name in ch_names]
    fwd_info = pick_info(info, info_idx)

    logger.info('Total rank is %d' % n_nzero)

    return fwd_info, gain, noise_cov, whitener, n_nzero


@verbose
def make_inverse_operator(info, forward, noise_cov, loose=0.2, depth=0.8,
                          fixed=False, limit_depth_chs=True, verbose=None):
    """Assemble inverse operator

    Parameters
    ----------
    info : dict
        The measurement info to specify the channels to include.
        Bad channels in info['bads'] are not used.
    forward : dict
        Forward operator.
    noise_cov : Covariance
        The noise covariance matrix.
    loose : None | float in [0, 1]
        Value that weights the source variances of the dipole components
        defining the tangent space of the cortical surfaces. Requires surface-
        based, free orientation forward solutions.
    depth : None | float in [0, 1]
        Depth weighting coefficients. If None, no depth weighting is performed.
    fixed : bool
        Use fixed source orientations normal to the cortical mantle. If True,
        the loose parameter is ignored.
    limit_depth_chs : bool
        If True, use only grad channels in depth weighting (equivalent to MNE
        C code). If grad chanels aren't present, only mag channels will be
        used (if no mag, then eeg). If False, use all channels.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    inv : dict
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
        | | Free orientation, | None      | 0.8       | False     | False           | True         |
        | | Depth weighted    |           |           |           |                 |              |
        +---------------------+-----------+-----------+-----------+-----------------+--------------+
        | | Free orientation  | None      | None      | False     | False           | True | False |
        +---------------------+-----------+-----------+-----------+-----------------+--------------+
        | | Fixed constraint, | None      | 0.8       | True      | False           | True         |
        | | Depth weighted    |           |           |           |                 |              |
        +---------------------+-----------+-----------+-----------+-----------------+--------------+
        | | Fixed constraint  | None      | None      | True      | True            | True         |
        +---------------------+-----------+-----------+-----------+-----------------+--------------+

    Also note that, if the source space (as stored in the forward solution)
    has patch statistics computed, these are used to improve the depth
    weighting. Thus slightly different results are to be expected with
    and without this information.
    """
    is_fixed_ori = is_fixed_orient(forward)

    if fixed and loose is not None:
        warnings.warn("When invoking make_inverse_operator with fixed=True, "
                      "the loose parameter is ignored.")
        loose = None

    if is_fixed_ori and not fixed:
        raise ValueError('Forward operator has fixed orientation and can only '
                         'be used to make a fixed-orientation inverse '
                         'operator.')
    if fixed:
        if depth is not None:
            if is_fixed_ori or not forward['surf_ori']:
                raise ValueError('For a fixed orientation inverse solution '
                                 'with depth weighting, the forward solution '
                                 'must be free-orientation and in surface '
                                 'orientation')
        elif forward['surf_ori'] is False:
            raise ValueError('For a fixed orientation inverse solution '
                             'without depth weighting, the forward solution '
                             'must be in surface orientation')

    # depth=None can use fixed fwd, depth=0<x<1 must use free ori
    if depth is not None:
        if not (0 < depth <= 1):
            raise ValueError('depth should be a scalar between 0 and 1')
        if is_fixed_ori or not forward['surf_ori']:
            raise ValueError('You need a free-orientation, surface-oriented '
                             'forward solution to do depth weighting even '
                             'when calculating a fixed-orientation inverse.')

    if loose is not None:
        if not (0 <= loose <= 1):
            raise ValueError('loose value should be smaller than 1 and bigger '
                             'than 0, or None for not loose orientations.')
        if loose < 1 and not forward['surf_ori']:
            raise ValueError('Forward operator is not oriented in surface '
                             'coordinates. A loose inverse operator requires '
                             'a surface-based, free orientation forward '
                             'operator.')

    #
    # 1. Read the bad channels
    # 2. Read the necessary data from the forward solution matrix file
    # 3. Load the projection data
    # 4. Load the sensor noise covariance matrix and attach it to the forward
    #

    gain_info, gain, noise_cov, whitener, n_nzero = \
        _prepare_forward(forward, info, noise_cov)

    #
    # 5. Compose the depth-weighting matrix
    #

    if depth is not None:
        patch_areas = forward.get('patch_areas', None)
        depth_prior = compute_depth_prior(gain, gain_info, is_fixed_ori,
                                          exp=depth, patch_areas=patch_areas,
                                          limit_depth_chs=limit_depth_chs)
    else:
        depth_prior = np.ones(gain.shape[1], dtype=gain.dtype)

    # Deal with fixed orientation forward / inverse
    if fixed:
        if depth is not None:
            # Convert the depth prior into a fixed-orientation one
            logger.info('    Picked elements from a free-orientation '
                        'depth-weighting prior into the fixed-orientation one')
        if not is_fixed_ori:
            # Convert to the fixed orientation forward solution now
            depth_prior = depth_prior[2::3]
            forward = deepcopy(forward)
            _to_fixed_ori(forward)
            is_fixed_ori = is_fixed_orient(forward)
            gain_info, gain, noise_cov, whitener, n_nzero = \
                _prepare_forward(forward, info, noise_cov, verbose=False)

    logger.info("Computing inverse operator with %d channels."
                % len(gain_info['ch_names']))

    #
    # 6. Compose the source covariance matrix
    #

    logger.info('Creating the source covariance matrix')
    source_cov = depth_prior.copy()
    depth_prior = dict(data=depth_prior, kind=FIFF.FIFFV_MNE_DEPTH_PRIOR_COV,
                       bads=[], diag=True, names=[], eig=None,
                       eigvec=None, dim=depth_prior.size, nfree=1,
                       projs=[])

    # apply loose orientations
    if not is_fixed_ori:
        orient_prior = compute_orient_prior(forward, loose=loose)
        source_cov *= orient_prior
        orient_prior = dict(data=orient_prior,
                            kind=FIFF.FIFFV_MNE_ORIENT_PRIOR_COV,
                            bads=[], diag=True, names=[], eig=None,
                            eigvec=None, dim=orient_prior.size, nfree=1,
                            projs=[])
    else:
        orient_prior = None

    # 7. Apply fMRI weighting (not done)

    #
    # 8. Apply the linear projection to the forward solution
    # 9. Apply whitening to the forward computation matrix
    #
    logger.info('Whitening the forward solution.')
    gain = np.dot(whitener, gain)

    # 10. Exclude the source space points within the labels (not done)

    #
    # 11. Do appropriate source weighting to the forward computation matrix
    #

    # Adjusting Source Covariance matrix to make trace of G*R*G' equal
    # to number of sensors.
    logger.info('Adjusting source covariance matrix.')
    source_std = np.sqrt(source_cov)
    gain *= source_std[None, :]
    trace_GRGT = linalg.norm(gain, ord='fro') ** 2
    scaling_source_cov = n_nzero / trace_GRGT
    source_cov *= scaling_source_cov
    gain *= sqrt(scaling_source_cov)

    source_cov = dict(data=source_cov, dim=source_cov.size,
                      kind=FIFF.FIFFV_MNE_SOURCE_COV, diag=True,
                      names=[], projs=[], eig=None, eigvec=None,
                      nfree=1, bads=[])

    # now np.trace(np.dot(gain, gain.T)) == n_nzero
    # logger.info(np.trace(np.dot(gain, gain.T)), n_nzero)

    #
    # 12. Decompose the combined matrix
    #

    logger.info('Computing SVD of whitened and weighted lead field '
                'matrix.')
    eigen_fields, sing, eigen_leads = linalg.svd(gain, full_matrices=False)
    logger.info('    largest singular value = %g' % np.max(sing))
    logger.info('    scaling factor to adjust the trace = %g' % trace_GRGT)

    eigen_fields = dict(data=eigen_fields.T, col_names=gain_info['ch_names'],
                        row_names=[], nrow=eigen_fields.shape[1],
                        ncol=eigen_fields.shape[0])
    eigen_leads = dict(data=eigen_leads.T, nrow=eigen_leads.shape[1],
                       ncol=eigen_leads.shape[0], row_names=[],
                       col_names=[])
    nave = 1.0

    # Handle methods
    has_meg = False
    has_eeg = False
    ch_idx = [k for k, c in enumerate(info['chs'])
              if c['ch_name'] in gain_info['ch_names']]
    for idx in ch_idx:
        ch_type = channel_type(info, idx)
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

    # We set this for consistency with mne C code written inverses
    if depth is None:
        depth_prior = None
    inv_op = dict(eigen_fields=eigen_fields, eigen_leads=eigen_leads,
                  sing=sing, nave=nave, depth_prior=depth_prior,
                  source_cov=source_cov, noise_cov=noise_cov,
                  orient_prior=orient_prior, projs=deepcopy(info['projs']),
                  eigen_leads_weighted=False, source_ori=forward['source_ori'],
                  mri_head_t=deepcopy(forward['mri_head_t']),
                  methods=methods, nsource=forward['nsource'],
                  coord_frame=forward['coord_frame'],
                  source_nn=forward['source_nn'].copy(),
                  src=deepcopy(forward['src']), fmri_prior=None)
    inv_info = deepcopy(forward['info'])
    inv_info['bads'] = deepcopy(info['bads'])
    inv_op['units'] = 'Am'
    inv_op['info'] = inv_info

    return inv_op


def compute_rank_inverse(inv):
    """Compute the rank of a linear inverse operator (MNE, dSPM, etc.)

    Parameters
    ----------
    inv : dict
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
