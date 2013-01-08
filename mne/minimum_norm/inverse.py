# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import warnings
from copy import deepcopy
from math import sqrt
import numpy as np
from scipy import linalg

import logging
logger = logging.getLogger('mne')

from ..fiff.constants import FIFF
from ..fiff.open import fiff_open
from ..fiff.tag import find_tag
from ..fiff.matrix import _read_named_matrix, _transpose_named_matrix, \
                          write_named_matrix
from ..fiff.proj import read_proj, make_projector, write_proj
from ..fiff.tree import dir_tree_find
from ..fiff.write import write_int, write_float_matrix, start_file, \
                         start_block, end_block, end_file, write_float, \
                         write_coord_trans

from ..fiff.cov import read_cov, write_cov
from ..fiff.pick import channel_type
from ..cov import prepare_noise_cov
from ..forward import compute_depth_prior, read_forward_meas_info, \
                      write_forward_meas_info, is_fixed_orient, \
                      compute_orient_prior
from ..source_space import read_source_spaces_from_tree, \
                           find_source_space_hemi, _get_vertno, \
                           write_source_spaces_to_fid, label_src_vertno_sel
from ..transforms import invert_transform, transform_source_space_to
from ..source_estimate import SourceEstimate
from .. import verbose


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
        The name of the FIF file.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    inv : dict
        The inverse operator.
    """
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
    if invs is None:
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
    #   The actual source orientation vectors
    #
    tag = find_tag(fid, invs, FIFF.FIFF_MNE_INVERSE_SOURCE_ORIENTATIONS)
    if tag is None:
        fid.close()
        raise Exception('Source orientation information not found')

    inv['source_nn'] = tag.data
    logger.info('[done]')
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
    inv['eigen_leads'] = _transpose_named_matrix(eigen_leads)
    inv['eigen_fields'] = _read_named_matrix(fid, invs,
                                             FIFF.FIFF_MNE_INVERSE_FIELDS)
    logger.info('[done]')
    #
    #   Read the covariance matrices
    #
    inv['noise_cov'] = read_cov(fid, invs, FIFF.FIFFV_MNE_NOISE_COV)
    logger.info('    Noise covariance matrix read.')

    inv['source_cov'] = read_cov(fid, invs, FIFF.FIFFV_MNE_SOURCE_COV)
    logger.info('    Source covariance matrix read.')
    #
    #   Read the various priors
    #
    inv['orient_prior'] = read_cov(fid, invs,
                                   FIFF.FIFFV_MNE_ORIENT_PRIOR_COV)
    if inv['orient_prior'] is not None:
        logger.info('    Orientation priors read.')

    inv['depth_prior'] = read_cov(fid, invs,
                                      FIFF.FIFFV_MNE_DEPTH_PRIOR_COV)
    if inv['depth_prior'] is not None:
        logger.info('    Depth priors read.')

    inv['fmri_prior'] = read_cov(fid, invs, FIFF.FIFFV_MNE_FMRI_PRIOR_COV)
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
    inv['projs'] = read_proj(fid, tree)
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
            inv['src'][k] = transform_source_space_to(inv['src'][k],
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
        The name of the FIF file.
    inv : dict
        The inverse operator.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
    """
    #
    #   Open the file, create directory
    #
    logger.info('Write inverse operator decomposition in %s...' % fname)

    # Create the file and save the essentials
    fid = start_file(fname)

    start_block(fid, FIFF.FIFFB_MNE_INVERSE_SOLUTION)

    logger.info('    Writing inverse operator info...')

    write_int(fid, FIFF.FIFF_MNE_INCLUDED_METHODS, inv['methods'])
    write_int(fid, FIFF.FIFF_MNE_SOURCE_ORIENTATION, inv['source_ori'])
    write_int(fid, FIFF.FIFF_MNE_SOURCE_SPACE_NPOINTS, inv['nsource'])
    write_int(fid, FIFF.FIFF_MNE_COORD_FRAME, inv['coord_frame'])
    write_float_matrix(fid, FIFF.FIFF_MNE_INVERSE_SOURCE_ORIENTATIONS,
                       inv['source_nn'])
    write_float(fid, FIFF.FIFF_MNE_INVERSE_SING, inv['sing'])

    #
    #   The eigenleads and eigenfields
    #
    if inv['eigen_leads_weighted']:
        write_named_matrix(fid, FIFF.FIFF_MNE_INVERSE_LEADS_WEIGHTED,
                           _transpose_named_matrix(inv['eigen_leads']))
    else:
        write_named_matrix(fid, FIFF.FIFF_MNE_INVERSE_LEADS,
                           _transpose_named_matrix(inv['eigen_leads']))

    write_named_matrix(fid, FIFF.FIFF_MNE_INVERSE_FIELDS, inv['eigen_fields'])
    logger.info('[done]')
    #
    #   write the covariance matrices
    #
    logger.info('    Writing noise covariance matrix.')
    write_cov(fid, inv['noise_cov'])

    logger.info('    Writing source covariance matrix.')
    write_cov(fid, inv['source_cov'])
    #
    #   write the various priors
    #
    logger.info('    Writing orientation priors.')
    if inv['orient_prior'] is not None:
        write_cov(fid, inv['orient_prior'])
    write_cov(fid, inv['depth_prior'])

    if inv['fmri_prior'] is not None:
        write_cov(fid, inv['fmri_prior'])

    #
    #   Parent MRI data
    #
    start_block(fid, FIFF.FIFFB_MNE_PARENT_MRI_FILE)
    #   write the MRI <-> head coordinate transformation
    write_coord_trans(fid, inv['mri_head_t'])
    end_block(fid, FIFF.FIFFB_MNE_PARENT_MRI_FILE)

    #
    #   Parent MEG measurement info
    #
    write_forward_meas_info(fid, inv['info'])

    #
    #   Write the source spaces
    #
    if 'src' in inv:
        write_source_spaces_to_fid(fid, inv['src'])

    #
    #  We also need the SSP operator
    #
    write_proj(fid, inv['projs'])
    #
    #   Done!
    #

    end_block(fid, FIFF.FIFFB_MNE_INVERSE_SOLUTION)
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
        raise Exception('Input must be 2D')
    if (vec.shape[0] % 3) != 0:
        raise Exception('Input must have 3N rows')

    n, p = vec.shape
    if np.iscomplexobj(vec):
        vec = np.abs(vec)
    comb = vec[0::3] ** 2
    comb += vec[1::3] ** 2
    comb += vec[2::3] ** 2
    if not square:
        comb = np.sqrt(comb)
    return comb


def _chech_ch_names(inv, info):
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
    if inv['noise_cov']['eig'] is None:
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
def _assemble_kernel(inv, label, method, pick_normal, verbose=None):
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

    if pick_normal:
        if not inv['source_ori'] == FIFF.FIFFV_MNE_FREE_ORI:
            raise ValueError('Pick normal can only be used with a free '
                             'orientation inverse operator.')

        is_loose = 0 < inv['orient_prior']['data'][0] < 1
        if not is_loose:
            raise ValueError('The pick_normal parameter is only valid '
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


def _check_method(method, dSPM):
    if dSPM is not None:
        warnings.warn('DEPRECATION: The dSPM parameter has been changed to '
                      'method. Please update your code')
        method = dSPM
    if method is True:
        warnings.warn('DEPRECATION:Inverse method should now be "MNE" or '
                      '"dSPM" or "sLORETA".')
        method = "dSPM"
    if method is False:
        warnings.warn('DEPRECATION:Inverse method should now be "MNE" or '
                      '"dSPM" or "sLORETA".')
        method = "MNE"

    if method not in ["MNE", "dSPM", "sLORETA"]:
        raise ValueError('method parameter should be "MNE" or "dSPM" '
                         'or "sLORETA".')
    return method


@verbose
def apply_inverse(evoked, inverse_operator, lambda2, method="dSPM",
                  pick_normal=False, dSPM=None, verbose=None):
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
    pick_normal : bool
        If True, rather than pooling the orientations by taking the norm,
        only the radial component is kept. This is only implemented
        when working with loose orientations.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    stc : SourceEstimate
        The source estimates
    """
    method = _check_method(method, dSPM)
    #
    #   Set up the inverse according to the parameters
    #
    nave = evoked.nave

    _chech_ch_names(inverse_operator, evoked.info)

    inv = prepare_inverse_operator(inverse_operator, nave, lambda2, method)
    #
    #   Pick the correct channels from the data
    #
    sel = _pick_channels_inverse_operator(evoked.ch_names, inv)
    logger.info('Picked %d channels from the data' % len(sel))
    logger.info('Computing inverse...')
    K, noise_norm, _ = _assemble_kernel(inv, None, method, pick_normal)
    sol = np.dot(K, evoked.data[sel])  # apply imaging kernel

    is_free_ori = (inverse_operator['source_ori'] == FIFF.FIFFV_MNE_FREE_ORI
                   and not pick_normal)

    if is_free_ori:
        logger.info('combining the current components...')
        sol = combine_xyz(sol)

    if noise_norm is not None:
        logger.info('(dSPM)...')
        sol *= noise_norm

    tstep = 1.0 / evoked.info['sfreq']
    tmin = float(evoked.first) / evoked.info['sfreq']
    vertno = _get_vertno(inv['src'])
    stc = SourceEstimate(sol, vertices=vertno, tmin=tmin, tstep=tstep)
    logger.info('[done]')

    return stc


@verbose
def apply_inverse_raw(raw, inverse_operator, lambda2, method="dSPM",
                      label=None, start=None, stop=None, nave=1,
                      time_func=None, pick_normal=False,
                      buffer_size=None, dSPM=None, verbose=None):
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
    label : Label
        Restricts the source estimates to a given label.
    start : int
        Index of first time sample (index not time is seconds).
    stop : int
        Index of first time sample not to include (index not time is seconds).
    nave : int
        Number of averages used to regularize the solution.
        Set to 1 on raw data.
    time_func : callable
        Linear function applied to sensor space time series.
    pick_normal : bool
        If True, rather than pooling the orientations by taking the norm,
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
    stc : SourceEstimate
        The source estimates.
    """
    method = _check_method(method, dSPM)

    _chech_ch_names(inverse_operator, raw.info)

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

    K, noise_norm, vertno = _assemble_kernel(inv, label, method, pick_normal)

    is_free_ori = (inverse_operator['source_ori'] == FIFF.FIFFV_MNE_FREE_ORI
                   and not pick_normal)

    if buffer_size is not None and is_free_ori:
        # Process the data in segments to conserve memory
        n_seg = int(np.ceil(data.shape[1] / float(buffer_size)))
        logger.info('computing inverse and combining the current '
                    'components (using %d segments)...' % (n_seg))

        # Allocate space for inverse solution
        n_times = data.shape[1]
        sol = np.empty((K.shape[0] / 3, n_times),
                       dtype=(K[0, 0] * data[0, 0]).dtype)

        for pos in xrange(0, n_times, buffer_size):
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
    stc = SourceEstimate(sol, vertices=vertno, tmin=tmin, tstep=tstep)
    logger.info('[done]')

    return stc


def _apply_inverse_epochs_gen(epochs, inverse_operator, lambda2, method="dSPM",
                              label=None, nave=1, pick_normal=False, dSPM=None,
                              verbose=None):
    """ see apply_inverse_epochs """
    method = _check_method(method, dSPM)

    _chech_ch_names(inverse_operator, epochs.info)

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
    K, noise_norm, vertno = _assemble_kernel(inv, label, method, pick_normal)

    tstep = 1.0 / epochs.info['sfreq']
    tmin = epochs.times[0]

    is_free_ori = (inverse_operator['source_ori'] == FIFF.FIFFV_MNE_FREE_ORI
                   and not pick_normal)

    for k, e in enumerate(epochs):
        logger.info("Processing epoch : %d" % (k + 1))
        sol = np.dot(K, e[sel])  # apply imaging kernel

        if is_free_ori:
            logger.info('combining the current components...')
            sol = combine_xyz(sol)

        if noise_norm is not None:
            sol *= noise_norm

        yield SourceEstimate(sol, vertices=vertno, tmin=tmin, tstep=tstep)

    logger.info('[done]')


@verbose
def apply_inverse_epochs(epochs, inverse_operator, lambda2, method="dSPM",
                         label=None, nave=1, pick_normal=False, dSPM=None,
                         return_generator=False, verbose=None):
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
    label : Label
        Restricts the source estimates to a given label.
    nave : int
        Number of averages used to regularize the solution.
        Set to 1 on single Epoch by default.
    pick_normal : bool
        If True, rather than pooling the orientations by taking the norm,
        only the radial component is kept. This is only implemented
        when working with loose orientations.
    return_generator : bool
        Return a generator object instead of a list. This allows iterating
        over the stcs without having to keep them all in memory.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    stc : list of SourceEstimate
        The source estimates for all epochs.
    """

    stcs = _apply_inverse_epochs_gen(epochs, inverse_operator, lambda2,
                                     method=method, label=label, nave=nave,
                                     pick_normal=pick_normal, dSPM=dSPM,
                                     verbose=verbose)

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
    normals: array of shape [n_positions, 3]
        Normals to the cortex

    Returns
    -------
    Lf_cortex: array of shape [n_sensors, n_positions x 3]
        Lf_cortex is a leadfield matrix for dipoles in rotated orientations, so
        that the first column is the gain vector for the cortical normal dipole
        and the following two column vectors are the gain vectors for the
        tangential orientations (tangent space of cortical surface).
    """
    n_sensors, n_dipoles = Lf_xyz.shape
    n_positions = n_dipoles / 3
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
                if (c['ch_name'] not in info['bads'])
                and (c['ch_name'] in fwd_ch_names)]
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
        whitener = np.zeros((n_nzero, n_chan), dtype=np.float)
        whitener = 1.0 / np.sqrt(eig[nzero])
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

    logger.info('Total rank is %d' % n_nzero)

    return ch_names, gain, noise_cov, whitener, n_nzero


@verbose
def make_inverse_operator(info, forward, noise_cov, loose=0.2, depth=0.8,
                          verbose=None):
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
        defining the tangent space of the cortical surfaces. Should be None
        for fixed-orientation forward solutions and for forward solutions
        whose source coordinate system is not surface based.
    depth : None | float in [0, 1]
        Depth weighting coefficients. If None, no depth weighting is performed.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    inv : dict
        Inverse operator.
    """
    is_fixed_ori = is_fixed_orient(forward)
    if is_fixed_ori and loose is not None:
        warnings.warn('Ignoring loose parameter with forward operator with '
                      'fixed orientation.')
        loose = None

    if not forward['surf_ori'] and loose is not None:
        raise ValueError('Forward operator is not oriented in surface '
                         'coordinates. loose parameter should be None '
                         'not %s.' % loose)

    if loose is not None and not (0 <= loose <= 1):
        raise ValueError('loose value should be smaller than 1 and bigger than'
                         ' 0, or None for not loose orientations.')
    if depth is not None and not (0 < depth <= 1):
        raise ValueError('depth should be a scalar between 0 and 1')

    ch_names, gain, noise_cov, whitener, n_nzero = \
        _prepare_forward(forward, info, noise_cov)

    n_dipoles = gain.shape[1]

    # Handle depth prior scaling
    if depth is not None:
        depth_prior = compute_depth_prior(gain, exp=depth, forward=forward,
                                          ch_names=ch_names)
    else:
        depth_prior = np.ones(n_dipoles, dtype=gain.dtype)

    logger.info("Computing inverse operator with %d channels."
                % len(ch_names))

    # Whiten lead field.
    logger.info('Whitening lead field matrix.')
    gain = np.dot(whitener, gain)

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

    logger.info('Computing SVD of whitened and weighted lead field '
                'matrix.')
    eigen_fields, sing, eigen_leads = linalg.svd(gain, full_matrices=False)

    eigen_fields = dict(data=eigen_fields.T, col_names=ch_names, row_names=[],
                        nrow=eigen_fields.shape[1], ncol=eigen_fields.shape[0])
    eigen_leads = dict(data=eigen_leads.T, nrow=eigen_leads.shape[1],
                       ncol=eigen_leads.shape[0], row_names=[],
                       col_names=[])
    nave = 1.0

    # Handle methods
    has_meg = False
    has_eeg = False
    ch_idx = [k for k, c in enumerate(info['chs']) if c['ch_name'] in ch_names]
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
