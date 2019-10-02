"""Coompute resolution matrix for linear estimators."""
from copy import deepcopy

import numpy as np

from mne import pick_channels, EvokedArray
from mne.utils import logger
from mne.forward.forward import is_fixed_orient, convert_forward_solution

from mne.minimum_norm import apply_inverse
from mne.beamformer import make_lcmv


def make_resolution_matrix(forward, inverse_operator, method, lambda2):
    """Compute resolution matrix for linear inverse operator.

    Parameters
    ----------
    forward: dict
        Forward Operator.
    inverse_operator: Instance of InverseOperator
        Inverse operator.
    method: str
        Inverse method to use (MNE, dSPM, sLORETA).
    lambda2: float
        The regularisation parameter.

    Returns
    -------
        resmat: numpy array, shape (n_dipoles, n_dipoles)
        Resolution matrix (inverse operator times forward operator).
    """
    # make sure forward and inverse operator match
    inv = inverse_operator
    fwd = _convert_forward_match_inv(forward, inv)

    # don't include bad channels
    # only use good channels from inverse operator
    bads_inv = inv['info']['bads']

    # good channels
    ch_names = [c for c in inv['info']['ch_names'] if (c not in bads_inv)]

    # get leadfield matrix from forward solution
    leadfield = _pick_leadfield(fwd['sol']['data'], fwd, ch_names)

    invmat = _get_matrix_from_inverse_operator(inv, fwd,
                                               method=method, lambda2=lambda2)

    resmat = invmat.dot(leadfield)

    dims = resmat.shape

    print('Dimensions of resolution matrix: %d by %d.' % (dims[0], dims[1]))

    return resmat


def _convert_forward_match_inv(fwd, inv):
    """Helper to ensure forward and inverse operators match."""
    # did inverse operator use fixed orientation?
    is_fixed_inv = inv['eigen_leads']['data'].shape[0] == inv['nsource']

    # ...or loose orientation?
    if not is_fixed_inv:
        is_loose = not (inv['orient_prior']['data'] == 1.).all()

    # if inv op is fixed or loose, do the same with fwd
    if is_fixed_inv:
        if not is_fixed_orient(fwd):
            fwd = convert_forward_solution(fwd, force_fixed=True)
    elif is_loose:
        if not fwd['surf_ori']:
            fwd = convert_forward_solution(fwd, surf_ori=True)
    else:  # free orientation, change surface orientation for fwd
        if fwd['surf_ori']:
            fwd = convert_forward_solution(fwd, surf_ori=False)

    # The following finds a difference, probably because too sensitive
    # # check if source spaces for inv and fwd are the same
    # differences = object_diff(fwd['src'], inv['src'])

    # if differences:
    #     raise RuntimeError('fwd["src"] and inv["src"] did not match: %s'
    #                        % (differences,))
    return fwd


def _pick_leadfield(leadfield, forward, ch_names):
    """Helper to pick out correct lead field components."""
    picks_fwd = pick_channels(forward['info']['ch_names'], ch_names)
    return leadfield[picks_fwd]


def _prepare_info(inverse_operator):
    """Helper to get a usable dict."""
    # in order to convert sub-leadfield matrix to evoked data type (pretending
    # it's an epoch, see in loop below), uses 'info' from inverse solution
    # because this has all the correct projector information
    info = deepcopy(inverse_operator['info'])
    info['sfreq'] = 1000.  # necessary
    info['projs'] = inverse_operator['projs']
    return info


def _get_matrix_from_inverse_operator(inverse_operator, forward, method='dSPM',
                                      lambda2=1. / 9.):
    """Get inverse matrix from an inverse operator.

    Currently works only for fixed/loose orientation constraints
    For loose orientation constraint, the CTFs are computed for the normal
    component (pick_ori='normal').
    Parameters
    ----------
    inverse_operator : instance of InverseOperator
        The inverse operator.
    forward : dict
        The forward operator.
    method : 'MNE' | 'dSPM' | 'sLORETA'
        Inverse methods (for apply_inverse).
    lambda2 : float
        The regularization parameter (for apply_inverse).
    Returns
    -------
    invmat : array, shape (n_dipoles, n_channels)
        Inverse matrix associated with inverse operator and specified
        parameters.
    """
    # make sure forward and inverse operators match
    _convert_forward_match_inv(forward, inverse_operator)

    info_inv = _prepare_info(inverse_operator)

    # only use channels that are good for inverse operator and forward sol
    ch_names_inv = info_inv['ch_names']
    n_chs_inv = len(ch_names_inv)
    bads_inv = inverse_operator['info']['bads']

    # indices of bad channels
    ch_idx_bads = [ch_names_inv.index(ch) for ch in bads_inv]

    # create identity matrix as input for inverse operator
    # set elements to zero for non-selected channels
    id_mat = np.eye(n_chs_inv)

    # convert identity matrix to evoked data type (pretending it's an epoch)
    ev_id = EvokedArray(id_mat, info=info_inv, tmin=0.)

    # apply inverse operator to identity matrix in order to get inverse matrix
    # free orientation constraint not possible because apply_inverse would
    # combine components

    # check if inverse operator uses fixed source orientations
    is_fixed_inv = inverse_operator['eigen_leads']['data'].shape[0] == \
        inverse_operator['nsource']

    # choose pick_ori according to inverse operator
    if is_fixed_inv:
        pick_ori = None
    else:
        pick_ori = 'vector'

    # columns for bad channels will be zero
    invmat_op = apply_inverse(ev_id, inverse_operator, lambda2=lambda2,
                              method=method, pick_ori=pick_ori)

    # turn source estimate into numpty array
    invmat = invmat_op.data

    dims = invmat.shape

    # remove columns for bad channels
    # take into account it may be 3D array
    invmat = np.delete(invmat, ch_idx_bads, axis=len(dims) - 1)

    # if 3D array, i.e. multiple values per location (fixed and loose),
    # reshape into 2D array
    if len(dims) == 3:
        v0o1 = invmat[0, 1].copy()
        v3o2 = invmat[3, 2].copy()
        invmat = invmat.reshape(dims[0] * dims[1], dims[2])
        # make sure that reshaping worked
        assert np.array_equal(v0o1, invmat[1])
        assert np.array_equal(v3o2, invmat[11])

    logger.info("Dimension of Inverse Matrix: %s" % str(invmat.shape))

    return invmat


def _get_matrix_from_lcmv_beamformer(info, forward, data_cov, reg=0.05,
                                     noise_cov=None,
                                     pick_ori=None, rank='info',
                                     weight_norm='unit-noise-gain',
                                     reduce_rank=False, depth=None,
                                     verbose=None):
    """Compute matrix for LCMV spatial filter.

    Parameters
    ----------
    info : dict
        The measurement info to specify the channels to include.
        Bad channels in info['bads'] are not used.
    forward : dict
        Forward operator.
    data_cov : instance of Covariance
        The data covariance.
    reg : float
        The regularization for the whitened data covariance.
    noise_cov : instance of Covariance
        The noise covariance. If provided, whitening will be done. Providing a
        noise covariance is mandatory if you mix sensor types, e.g.
        gradiometers with magnetometers or EEG with MEG.
    pick_ori : None | 'normal' | 'max-power' | 'vector'
        For forward solutions with fixed orientation, None (default) must be
        used and a scalar beamformer is computed. For free-orientation forward
        solutions, a vector beamformer is computed and:

            None
                Pools the orientations by taking the norm.
            'normal'
                Keeps only the radial component.
            'max-power'
                Selects orientations that maximize output source power at
                each location.
            'vector'
                Keeps the currents for each direction separate

    %(rank_info)s
    weight_norm : 'unit-noise-gain' | 'nai' | None
        If 'unit-noise-gain', the unit-noise gain minimum variance beamformer
        will be computed (Borgiotti-Kaplan beamformer) [2]_,
        if 'nai', the Neural Activity Index [1]_ will be computed,
        if None, the unit-gain LCMV beamformer [2]_ will be computed.
    reduce_rank : bool
        If True, the rank of the leadfield will be reduced by 1 for each
        spatial location. Setting reduce_rank to True is typically necessary
        if you use a single sphere model for MEG.
    %(depth)s

        .. versionadded:: 0.18
    %(verbose)s

    Returns
    -------
    filtmat: array, (n_dipoles, n_channels)
    The beamformer filters as matrix.
    """

    # compute beamformer filters
    filters = make_lcmv(info, forward, data_cov, reg=0.05, noise_cov=noise_cov,
                        pick_ori=pick_ori, rank=rank,
                        weight_norm=weight_norm,
                        reduce_rank=reduce_rank, depth=depth,
                        verbose=verbose)

    filtmat = filters['weights']

    return filtmat
