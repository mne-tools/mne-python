# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD (3-clause)

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
from ..io.proj import _needs_eeg_average_ref_proj
from ..io.tree import dir_tree_find
from ..io.write import (write_int, write_float_matrix, start_file,
                        start_block, end_block, end_file, write_float,
                        write_coord_trans, write_string)

from ..io.pick import channel_type, pick_info, pick_types
from ..cov import prepare_noise_cov, _read_cov, _write_cov, Covariance
from ..forward import (compute_depth_prior, _read_forward_meas_info,
                       write_forward_meas_info, is_fixed_orient,
                       compute_orient_prior, convert_forward_solution)
from ..source_space import (_read_source_spaces_from_tree,
                            find_source_space_hemi, _get_vertno,
                            _write_source_spaces_to_fid, label_src_vertno_sel)
from ..transforms import _ensure_trans, transform_surface_to
from ..source_estimate import _make_stc
from ..utils import check_fname, logger, verbose, warn
from functools import reduce


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

        # XXX TODO: This and the __repr__ in SourceSpaces should call a
        # function _get_name_str() in source_space.py
        if self['src'][0]['type'] == 'surf':
            entr += (' | Source space: Surface with %d vertices'
                     % self['nsource'])
        elif self['src'][0]['type'] == 'vol':
            entr += (' | Source space: Volume with %d grid points'
                     % self['nsource'])
        elif self['src'][0]['type'] == 'discrete':
            entr += (' | Source space: Discrete with %d dipoles'
                     % self['nsource'])

        source_ori = {FIFF.FIFFV_MNE_UNKNOWN_ORI: 'Unknown',
                      FIFF.FIFFV_MNE_FIXED_ORI: 'Fixed',
                      FIFF.FIFFV_MNE_FREE_ORI: 'Free'}
        entr += ' | Source orientation: %s' % source_ori[self['source_ori']]
        entr += '>'

        return entr


def _pick_channels_inverse_operator(ch_names, inv):
    """Data channel indices to be used knowing an inverse operator.

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
    fname : string
        The name of the FIF file, which ends with -inv.fif or -inv.fif.gz.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    inv : instance of InverseOperator
        The inverse operator.

    See Also
    --------
    write_inverse_operator, make_inverse_operator
    """
    check_fname(fname, 'inverse operator', ('-inv.fif', '-inv.fif.gz'))

    #
    #   Open the file, create directory
    #
    logger.info('Reading inverse operator decomposition from %s...'
                % fname)
    f, tree, _ = fiff_open(fname, preload=True)
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
    fname : string
        The name of the FIF file, which ends with -inv.fif or -inv.fif.gz.
    inv : dict
        The inverse operator.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    See Also
    --------
    read_inverse_operator
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
    """Check that channels in inverse operator are measurements."""
    inv_ch_names = inv['eigen_fields']['col_names']

    if inv['noise_cov'].ch_names != inv_ch_names:
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
    """Prepare an inverse operator for actually computing the inverse.

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
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    inv : instance of InverseOperator
        Prepared inverse operator.
    """
    if nave <= 0:
        raise ValueError('The number of averages should be positive')

    logger.info('Preparing the inverse operator for use...')
    inv = orig.copy()
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

    return InverseOperator(inv)


@verbose
def _assemble_kernel(inv, label, method, pick_ori, verbose=None):
    """Assemble the kernel."""
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
    """Check the method."""
    if method not in ["MNE", "dSPM", "sLORETA"]:
        raise ValueError('method parameter should be "MNE" or "dSPM" '
                         'or "sLORETA".')
    return method


def _check_ori(pick_ori):
    """Check pick_ori."""
    if pick_ori is not None and pick_ori != 'normal':
        raise RuntimeError('pick_ori must be None or "normal", not %s'
                           % pick_ori)
    return pick_ori


def _check_reference(inst):
    """Check for EEG ref."""
    if _needs_eeg_average_ref_proj(inst.info):
        raise ValueError('EEG average reference is mandatory for inverse '
                         'modeling, use set_eeg_reference method.')
    if inst.info['custom_ref_applied']:
        raise ValueError('Custom EEG reference is not allowed for inverse '
                         'modeling.')


def _subject_from_inverse(inverse_operator):
    """Get subject id from inverse operator."""
    return inverse_operator['src'][0].get('subject_his_id', None)


@verbose
def apply_inverse(evoked, inverse_operator, lambda2=1. / 9.,
                  method="dSPM", pick_ori=None,
                  prepared=False, label=None, verbose=None):
    """Apply inverse operator to evoked data.

    Parameters
    ----------
    evoked : Evoked object
        Evoked data.
    inverse_operator: instance of InverseOperator
        Inverse operator returned from `mne.read_inverse_operator`,
        `prepare_inverse_operator` or `make_inverse_operator`.
    lambda2 : float
        The regularization parameter.
    method : "MNE" | "dSPM" | "sLORETA"
        Use mininum norm, dSPM or sLORETA.
    pick_ori : None | "normal"
        If "normal", rather than pooling the orientations by taking the norm,
        only the radial component is kept. This is only implemented
        when working with loose orientations.
    prepared : bool
        If True, do not call `prepare_inverse_operator`.
    label : Label | None
        Restricts the source estimates to a given label. If None,
        source estimates will be computed for the entire source space.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    stc : SourceEstimate | VolSourceEstimate
        The source estimates

    See Also
    --------
    apply_inverse_raw : Apply inverse operator to raw object
    apply_inverse_epochs : Apply inverse operator to epochs object
    """
    _check_reference(evoked)
    method = _check_method(method)
    pick_ori = _check_ori(pick_ori)
    #
    #   Set up the inverse according to the parameters
    #
    nave = evoked.nave

    _check_ch_names(inverse_operator, evoked.info)

    if not prepared:
        inv = prepare_inverse_operator(inverse_operator, nave, lambda2, method)
    else:
        inv = inverse_operator
    #
    #   Pick the correct channels from the data
    #
    sel = _pick_channels_inverse_operator(evoked.ch_names, inv)
    logger.info('Picked %d channels from the data' % len(sel))
    logger.info('Computing inverse...')
    K, noise_norm, vertno = _assemble_kernel(inv, label, method, pick_ori)
    sol = np.dot(K, evoked.data[sel])  # apply imaging kernel

    is_free_ori = (inverse_operator['source_ori'] ==
                   FIFF.FIFFV_MNE_FREE_ORI and pick_ori is None)

    if is_free_ori:
        logger.info('combining the current components...')
        sol = combine_xyz(sol)

    if noise_norm is not None:
        logger.info('(dSPM)...')
        sol *= noise_norm

    tstep = 1.0 / evoked.info['sfreq']
    tmin = float(evoked.times[0])
    subject = _subject_from_inverse(inverse_operator)

    stc = _make_stc(sol, vertices=vertno, tmin=tmin, tstep=tstep,
                    subject=subject)
    logger.info('[done]')

    return stc


@verbose
def apply_inverse_raw(raw, inverse_operator, lambda2, method="dSPM",
                      label=None, start=None, stop=None, nave=1,
                      time_func=None, pick_ori=None, buffer_size=None,
                      prepared=False, verbose=None):
    """Apply inverse operator to Raw data.

    Parameters
    ----------
    raw : Raw object
        Raw data.
    inverse_operator : dict
        Inverse operator returned from `mne.read_inverse_operator`,
        `prepare_inverse_operator` or `make_inverse_operator`.
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
    prepared : bool
        If True, do not call `prepare_inverse_operator`.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    stc : SourceEstimate | VolSourceEstimate
        The source estimates.

    See Also
    --------
    apply_inverse_epochs : Apply inverse operator to epochs object
    apply_inverse : Apply inverse operator to evoked object
    """
    _check_reference(raw)
    method = _check_method(method)
    pick_ori = _check_ori(pick_ori)

    _check_ch_names(inverse_operator, raw.info)

    #
    #   Set up the inverse according to the parameters
    #
    if not prepared:
        inv = prepare_inverse_operator(inverse_operator, nave, lambda2, method)
    else:
        inv = inverse_operator
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

    is_free_ori = (inverse_operator['source_ori'] ==
                   FIFF.FIFFV_MNE_FREE_ORI and pick_ori is None)

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


def _apply_inverse_epochs_gen(epochs, inverse_operator, lambda2, method='dSPM',
                              label=None, nave=1, pick_ori=None,
                              prepared=False, verbose=None):
    """Generator for apply_inverse_epochs."""
    method = _check_method(method)
    pick_ori = _check_ori(pick_ori)

    _check_ch_names(inverse_operator, epochs.info)

    #
    #   Set up the inverse according to the parameters
    #
    if not prepared:
        inv = prepare_inverse_operator(inverse_operator, nave, lambda2, method)
    else:
        inv = inverse_operator
    #
    #   Pick the correct channels from the data
    #
    sel = _pick_channels_inverse_operator(epochs.ch_names, inv)
    logger.info('Picked %d channels from the data' % len(sel))
    logger.info('Computing inverse...')
    K, noise_norm, vertno = _assemble_kernel(inv, label, method, pick_ori)

    tstep = 1.0 / epochs.info['sfreq']
    tmin = epochs.times[0]

    is_free_ori = (inverse_operator['source_ori'] ==
                   FIFF.FIFFV_MNE_FREE_ORI and pick_ori is None)

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
                         return_generator=False,
                         prepared=False, verbose=None):
    """Apply inverse operator to Epochs.

    Parameters
    ----------
    epochs : Epochs object
        Single trial epochs.
    inverse_operator : dict
        Inverse operator returned from `mne.read_inverse_operator`,
        `prepare_inverse_operator` or `make_inverse_operator`.
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
    prepared : bool
        If True, do not call `prepare_inverse_operator`.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    stc : list of SourceEstimate or VolSourceEstimate
        The source estimates for all epochs.

    See Also
    --------
    apply_inverse_raw : Apply inverse operator to raw object
    apply_inverse : Apply inverse operator to evoked object
    """
    _check_reference(epochs)
    stcs = _apply_inverse_epochs_gen(epochs, inverse_operator, lambda2,
                                     method=method, label=label, nave=nave,
                                     pick_ori=pick_ori, verbose=verbose,
                                     prepared=prepared)

    if not return_generator:
        # return a list
        stcs = [stc for stc in stcs]

    return stcs


'''
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
'''


###############################################################################
# Assemble the inverse operator

@verbose
def _prepare_forward(forward, info, noise_cov, pca=False, rank=None,
                     verbose=None):
    """Prepare forward solution for inverse solvers."""
    # fwd['sol']['row_names'] may be different order from fwd['info']['chs']
    fwd_sol_ch_names = forward['sol']['row_names']
    ch_names = [c['ch_name'] for c in info['chs']
                if ((c['ch_name'] not in info['bads'] and
                     c['ch_name'] not in noise_cov['bads']) and
                    (c['ch_name'] in fwd_sol_ch_names and
                     c['ch_name'] in noise_cov.ch_names))]

    if not len(info['bads']) == len(noise_cov['bads']) or \
            not all(b in noise_cov['bads'] for b in info['bads']):
        logger.info('info["bads"] and noise_cov["bads"] do not match, '
                    'excluding bad channels from both')

    n_chan = len(ch_names)
    logger.info("Computing inverse operator with %d channels." % n_chan)

    #
    #   Handle noise cov
    #
    noise_cov = prepare_noise_cov(noise_cov, info, ch_names, rank)

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

    # This actually reorders the gain matrix to conform to the info ch order
    fwd_idx = [fwd_sol_ch_names.index(name) for name in ch_names]
    gain = gain[fwd_idx]
    # Any function calling this helper will be using the returned fwd_info
    # dict, so fwd['sol']['row_names'] becomes obsolete and is NOT re-ordered

    info_idx = [info['ch_names'].index(name) for name in ch_names]
    fwd_info = pick_info(info, info_idx)

    logger.info('Total rank is %d' % n_nzero)

    return fwd_info, gain, noise_cov, whitener, n_nzero


@verbose
def make_inverse_operator(info, forward, noise_cov, loose=0.2, depth=0.8,
                          fixed=False, limit_depth_chs=True, rank=None,
                          verbose=None):
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
    rank : None | int | dict
        Specified rank of the noise covariance matrix. If None, the rank is
        detected automatically. If int, the rank is specified for the MEG
        channels. A dictionary with entries 'eeg' and/or 'meg' can be used
        to specify the rank for each modality.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`).

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
    """  # noqa: E501
    is_fixed_ori = is_fixed_orient(forward)

    if fixed and loose is not None:
        warn('When invoking make_inverse_operator with fixed=True, the loose '
             'parameter is ignored.')
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
        if is_fixed_ori:
            raise ValueError('You need a free-orientation, surface-oriented '
                             'forward solution to do depth weighting even '
                             'when calculating a fixed-orientation inverse.')
        if not forward['surf_ori']:
            forward = convert_forward_solution(forward, surf_ori=True)
        assert forward['surf_ori']
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
        _prepare_forward(forward, info, noise_cov, rank=rank)
    forward['info']._check_consistency()

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
            forward = convert_forward_solution(
                forward, surf_ori=forward['surf_ori'], force_fixed=True)
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
    inv_info['bads'] = [bad for bad in info['bads']
                        if bad in inv_info['ch_names']]
    inv_info._check_consistency()
    inv_op['units'] = 'Am'
    inv_op['info'] = inv_info

    return InverseOperator(inv_op)


def compute_rank_inverse(inv):
    """Compute the rank of a linear inverse operator (MNE, dSPM, etc.).

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
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`).

    Returns
    -------
    snr : ndarray, shape (n_times,)
        The SNR estimated from the whitened data.
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

        \\tilde{M} = R^\\frac{1}{2}V\\Gamma U^T

    The values in the diagonal matrix :math:`\\Gamma` are expressed in terms
    of the chosen regularization :math:`\\lambda\\approx\\frac{1}{\\rm{SNR}^2}`
    and singular values :math:`\\lambda_k` as:

    .. math::

        \\gamma_k = \\frac{1}{\\lambda_k}\\frac{\\lambda_k^2}{\\lambda_k^2 + \\lambda^2}

    We also know that our predicted data is given by:

    .. math::

        \\hat{x}(t) = G\\hat{j}(t)=C^\\frac{1}{2}U\\Pi w(t)

    And thus our predicted whitened data is just:

    .. math::

        \\hat{w}(t) = U\\Pi w(t)

    Where :math:`\\Pi` is diagonal with entries entries:

    .. math::

        \\lambda_k\\gamma_k = \\frac{\\lambda_k^2}{\\lambda_k^2 + \\lambda^2}

    If we use no regularization, note that :math:`\\Pi` is just the
    identity matrix. Here we test the squared magnitude of the difference
    between unregularized solution and regularized solutions, choosing the
    biggest regularization that achieves a :math:`\\chi^2`-test significance
    of 0.001.

    .. versionadded:: 0.9.0
    """  # noqa: E501
    from scipy.stats import chi2
    _check_reference(evoked)
    _check_ch_names(inv, evoked.info)
    inv = prepare_inverse_operator(inv, evoked.nave, 1. / 9., 'MNE')
    sel = _pick_channels_inverse_operator(evoked.ch_names, inv)
    logger.info('Picked %d channels from the data' % len(sel))
    data_white = np.dot(inv['whitener'], np.dot(inv['proj'], evoked.data[sel]))
    data_white_ef = np.dot(inv['eigen_fields']['data'], data_white)
    n_ch, n_times = data_white.shape

    # Adapted from mne_analyze/regularization.c, compute_regularization
    n_zero = (inv['noise_cov']['eig'] <= 0).sum()
    logger.info('Effective nchan = %d - %d = %d'
                % (n_ch, n_zero, n_ch - n_zero))
    signal = np.sum(data_white ** 2, axis=0)  # sum of squares across channels
    noise = n_ch - n_zero
    snr = signal / noise

    # Adapted from noise_regularization
    lambda2_est = np.empty(n_times)
    lambda2_est.fill(10.)
    remaining = np.ones(n_times, bool)

    # deal with low SNRs
    bad = (snr <= 1)
    lambda2_est[bad] = 100.
    remaining[bad] = False

    # parameters
    lambda_mult = 0.9
    sing2 = (inv['sing'] * inv['sing'])[:, np.newaxis]
    val = chi2.isf(1e-3, n_ch - 1)
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
