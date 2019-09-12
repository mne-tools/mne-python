"""Functions shared between different beamformer types."""

# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Roman Goj <roman.goj@gmail.com>
#          Britta Westner <britta.wstnr@gmail.com>
#
# License: BSD (3-clause)

from copy import deepcopy

import numpy as np
from scipy import linalg

from ..cov import Covariance, make_ad_hoc_cov
from ..forward.forward import is_fixed_orient, _restrict_forward_to_src_sel
from ..io.proj import make_projector, Projection
from ..minimum_norm.inverse import _get_vertno, _prepare_forward
from ..source_space import label_src_vertno_sel
from ..utils import (verbose, check_fname, _reg_pinv, _check_option, logger,
                     _pl, _svd_lwork, _repeated_svd, _repeated_pinv2,
                     _inv_lwork, _repeated_inv, _eig_lwork, _repeated_eig,
                     LinAlgError)
from ..time_frequency.csd import CrossSpectralDensity

from ..externals.h5io import read_hdf5, write_hdf5


def _check_proj_match(info, filters):
    """Check whether SSP projections in data and spatial filter match."""
    proj_data, _, _ = make_projector(info['projs'],
                                     filters['ch_names'])
    if not np.allclose(proj_data, filters['proj'],
                       atol=np.finfo(float).eps, rtol=1e-13):
        raise ValueError('The SSP projections present in the data '
                         'do not match the projections used when '
                         'calculating the spatial filter.')


def _check_src_type(filters):
    """Check whether src_type is in filters and set custom warning."""
    if 'src_type' not in filters:
        filters['src_type'] = None
    warn_text = ('The spatial filter does not contain src_type and a robust '
                 'guess of src_type is not possible without src. Consider '
                 'recomputing the filter.')
    return filters, warn_text


def _prepare_beamformer_input(info, forward, label=None, pick_ori=None,
                              noise_cov=None, rank=None, pca=False, loose=None,
                              combine_xyz='fro', exp=None, limit=None,
                              allow_fixed_depth=True, limit_depth_chs=False):
    """Input preparation common for LCMV, DICS, and RAP-MUSIC."""
    _check_option('pick_ori', pick_ori,
                  ('normal', 'max-power', 'vector', None))

    # Restrict forward solution to selected vertices
    if label is not None:
        _, src_sel = label_src_vertno_sel(label, forward['src'])
        forward = _restrict_forward_to_src_sel(forward, src_sel)

    if loose is None:
        loose = 0. if is_fixed_orient(forward) else 1.
    if noise_cov is None:
        noise_cov = make_ad_hoc_cov(info, std=1.)
    forward, info_picked, gain, _, orient_prior, _, trace_GRGT, noise_cov, \
        whitener = _prepare_forward(
            forward, info, noise_cov, 'auto', loose, rank=rank, pca=pca,
            use_cps=True, exp=exp, limit_depth_chs=limit_depth_chs,
            combine_xyz=combine_xyz, limit=limit,
            allow_fixed_depth=allow_fixed_depth)
    is_free_ori = not is_fixed_orient(forward)  # could have been changed
    nn = forward['source_nn']
    if is_free_ori:  # take Z coordinate
        nn = nn[2::3]
    nn = nn.copy()
    vertno = _get_vertno(forward['src'])
    if forward['surf_ori']:
        nn[...] = [0, 0, 1]  # align to local +Z coordinate
    if pick_ori is not None and not is_free_ori:
        raise ValueError(
            'Normal or max-power orientation (got %r) can only be picked when '
            'a forward operator with free orientation is used.' % (pick_ori,))
    if pick_ori == 'normal' and not forward['surf_ori']:
        raise ValueError('Normal orientation can only be picked when a '
                         'forward operator oriented in surface coordinates is '
                         'used.')
    if pick_ori == 'normal' and not forward['src'].kind == 'surface':
        raise ValueError('Normal orientation can only be picked when a '
                         'forward operator with a surface-based source space '
                         'is used.')
    del forward, info

    # Undo the scaling that MNE prefers
    scale = np.sqrt((noise_cov['eig'] > 0).sum() / trace_GRGT)
    gain /= scale
    if orient_prior is not None:
        orient_std = np.sqrt(orient_prior)
    else:
        orient_std = np.ones(gain.shape[1])

    # Get the projector
    proj, ncomp, _ = make_projector(
        info_picked['projs'], info_picked['ch_names'])
    return (is_free_ori, info_picked, proj, vertno, gain, whitener, nn,
            orient_std)


def _normalized_weights(W_stacked, G, Cm_inv_sq, reduce_rank, nn, n_sources, n_orient):
    """Compute the normalized weights in max-power orientation.

    Uses Eq. 4.47 from [1]_.

    Parameters
    ----------
    W : ndarray, shape (3, n_channels)
        The set of un-normalized filters at a single source point.
    G : ndarray, shape (n_channels, 3)
        The leadfield at a single source point.
    Cm_inv_sq : nsarray, snape (n_channels, n_channels)
        The squared inverse covariance matrix.
    reduce_rank : bool
        Whether to reduce the rank of the filter by one.
    nn : ndarray, shape (3,)
        The source normal.
    sk : ndarray, shape (3,)
        The source prior.
    svd_lwork : int
        The svd lwork value.
    inv_lwork : int
        The inv lwork value.
    eig_lwork : int
        The eig lwork value.

    Returns
    -------
    Wk : ndarray, shape (n_dipoles, n_channels)
        The normalized beamformer filters at the source point in the direction
        of max power.

    References
    ----------
    .. [1] Sekihara & Nagarajan. Adaptive spatial filters for electromagnetic
           brain imaging (2008) Springer Science & Business Media
    """
    G_stacked_T = G.T.reshape(n_sources, n_orient, -1)
    tmp = np.matmul(Cm_inv_sq, G)
    tmp = tmp.reshape(-1, n_orient, n_sources, order='F').transpose(2, 0, 1)
    norm_inv = np.matmul(G_stacked_T, tmp)

    evals_denom, evecs_denom = np.linalg.eigh(norm_inv)
    evals_denom = evals_denom[:, np.newaxis, :]
    evals_sign = np.sign(evals_denom)
    evals_denom *= evals_sign
    evecs_denom *= evals_sign
    if reduce_rank:
        # Use pseudo inverse computation setting smallest
        # component to zero if the leadfield is not full rank
        # here we can use eigh to compute pinv; dmalt
        # norm = _reg_pinv(norm_inv, rank=norm_inv.shape[0] - 1,
        #                  svd_lwork=svd_lwork)[0]
        U = np.zeros_like(evecs_denom)
        U[:, :, 1:3] = (
            evecs_denom[:, :, 1:3] / np.sqrt(evals_denom[:, :, 1:3])
        )
    else:
        # Use straight inverse with full rank leadfield
        try:
            # norm = _repeated_inv(norm_inv, inv_lwork)
            with np.errstate(divide='raise'):
                U = np.divide(evecs_denom, np.sqrt(evals_denom))
        except (FloatingPointError, Exception):
            raise ValueError(
                'Singular matrix detected when estimating spatial filters. '
                'Consider reducing the rank of the forward operator by using '
                'reduce_rank=True.'
            )
    # Reapply source covariance after inversion
    # It seems like this does nothing since _prepare_beamformer_input
    # is always called with the default loose=None; dmalt
    # norm *= sk
    # norm *= sk[:, np.newaxis]

    # Determine orientation of max power
    # assert power.dtype in (np.float64, np.complex128)  # LCMV, DICS
    # eig_vals, eig_vecs = _repeated_eig(power, eig_lwork)

    G_stacked = G_stacked_T.transpose(0, 2, 1)
    A = np.matmul(W_stacked, G_stacked)
    A_hat = np.matmul(U.transpose(0, 2, 1), np.matmul(A, U))
    evals_A_hat, evecs_a_hat = np.linalg.eigh(A_hat)

    evecs_generalized = np.matmul(U, evecs_a_hat)
    max_power_ori = evecs_generalized[:, :, -1]

    if not np.iscomplexobj(A) and np.iscomplexobj(evecs_generalized):
        raise ValueError('The eigenspectrum of the leadfield at this voxel is '
                         'complex. Consider reducing the rank of the '
                         'leadfield by using reduce_rank=True.')

    # set the (otherwise arbitrary) sign to match the normal
    # sign = np.sign(np.dot(max_power_ori, nn)) or 1
    cos_with_normal = np.matmul(
        max_power_ori[:, np.newaxis, :], nn[:, :, np.newaxis]
    )
    sign = np.sign(
        cos_with_normal,
        where=cos_with_normal != 0,
        out=np.ones_like(cos_with_normal)
    )
    max_power_ori[:, :, np.newaxis] *= sign

    # Compute the filter in the orientation of max power
    # -------- filters in the orientation of max power -------- #
    W_max_ori = np.matmul(max_power_ori[:, np.newaxis, :], W_stacked).squeeze()
    # --------------------------------------------------------- #
    # -------- unit-noise-gain filters normalization -------- #
    tmp = np.matmul(norm_inv, max_power_ori[:, :, np.newaxis])
    pwr = np.matmul(max_power_ori[:, np.newaxis, :], tmp).ravel()
    denom = np.sqrt(pwr)
    W_max_ori /= np.expand_dims(denom, axis=1)
    # -------------------------------------------------------- #
    # Wk[:] = np.dot(max_power_ori, Wk)
    # Gk = np.dot(Gk, max_power_ori)
    # denom = np.dot(Gk.T, np.dot(Cm_inv_sq, Gk))
    # denom = np.sqrt(denom)
    # Wk /= denom
    # W = W_stacked.reshape(-1, n_sources * n_orient)
    W_stacked[:, 0, :] = W_max_ori
    W = W_stacked.reshape(n_sources * n_orient, -1)

    return W


# @profile
def _compute_beamformer(G, Cm, reg, n_orient, weight_norm, pick_ori,
                        reduce_rank, rank, inversion, nn, orient_std):
    """Compute a spatial beamformer filter (LCMV or DICS).

    For more detailed information on the parameters, see the docstrings of
    `make_lcmv` and `make_dics`.

    Parameters
    ----------
    G : ndarray, shape (n_dipoles, n_channels)
        The leadfield.
    Cm : ndarray, shape (n_channels, n_channels)
        The data covariance matrix.
    reg : float
        Regularization parameter.
    n_orient : int
        Number of dipole orientations defined at each source point
    weight_norm : None | 'unit-noise-gain' | 'nai'
        The weight normalization scheme to use.
    pick_ori : None | 'normal' | 'max-power'
        The source orientation to compute the beamformer in.
    reduce_rank : bool
        Whether to reduce the rank by one during computation of the filter.
    rank : dict | None | 'full' | 'info'
        See compute_rank.
    inversion : 'matrix' | 'single'
        The inversion scheme to compute the weights.
    nn : ndarray, shape (n_dipoles, 3)
        The source normals.
    orient_std : ndarray, shape (n_dipoles,)
        The std of the orientation prior used in weighting the lead fields.

    Returns
    -------
    W : ndarray, shape (n_dipoles, n_channels)
        The beamformer filter weights.
    """
    # Tikhonov regularization using reg parameter to control for
    # trade-off between spatial resolution and noise sensitivity
    # eq. 25 in Gross and Ioannides, 1999 Phys. Med. Biol. 44 2081
    Cm_inv, loading_factor, rank = _reg_pinv(Cm, reg, rank)
    Cm_inv_sq = Cm_inv.dot(Cm_inv)

    # Compute spatial filters
    W = np.dot(G.T, Cm_inv)
    assert orient_std.shape == (G.shape[1],)
    n_sources = G.shape[1] // n_orient
    assert nn.shape == (n_sources, 3)

    logger.info('Computing beamformer filters for %d source%s'
                % (n_sources, _pl(n_sources)))
    # svd_lwork = _svd_lwork((3, 3), Cm.dtype)  # for real or complex
    # real_svd_lwork = _svd_lwork((3, 3))  # for one that will always be real
    # eig_lwork = _eig_lwork((3, 3), Cm.dtype)
    # inv_lwork = _inv_lwork((3, 3), Cm.dtype)
    # plan
    # compute matrices A and B if ung or just A
    #

    # for k in range(n_sources):
    # this_sl = slice(n_orient * k, n_orient * k + n_orient)
    # Wk, Gk, sk = W[this_sl], G[:, this_sl], orient_std[this_sl]

    G_stacked_T = G.T.reshape(n_sources, n_orient, -1)
    G_stacked = G_stacked_T.transpose(0, 2, 1)
    W_stacked = W.reshape(n_sources, n_orient, -1)

    if (inversion == 'matrix' and pick_ori == 'max-power'
            and weight_norm in ['unit-noise-gain', 'nai']):
        # In this case, take a shortcut to compute the filter
        # Wk[:] = _normalized_weights(
        #     W, G, Cm_inv_sq, reduce_rank, nn, orient_std,
        #     svd_lwork, inv_lwork, eig_lwork, n_sources, n_orient)
        W = _normalized_weights(
            W_stacked, G, Cm_inv_sq, reduce_rank, nn, n_sources, n_orient
        )
    else:
        # Compute power at the source
        C_stacked = np.matmul(W_stacked, G_stacked)
        # Ck = np.dot(Wk, Gk)

        # Normalize the spatial filters
        if n_orient > 1:  # XXX: not sure why 'and len(Wk) > 1' is checked
            # Free source orientation
            if inversion == 'single':
                # Invert for each dipole separately using plain division
                diag = np.diagonal(C_stacked, axis1=1, axis2=2)
                with np.errstate(divide='ignore'):
                    # XXX: Still can create infs. Need to raise error
                    # and suggest dimensionality reduction; as in
                    # _normalized_weights; dmalt
                    # norm = np.diag(1. / np.diag(Ck))
                    W_stacked = W_stacked / diag[:, :, np.newaxis]
            elif inversion == 'matrix':
                # Invert for all dipoles simultaneously using matrix
                # inversion.
                assert C_stacked.shape == (n_sources, 3, 3)
                # XXX: reduce_rank doesn't have any effect here;
                # compared to similar place in _normalized_weights; dmalt
                # -------- pseudo-invert C_stacked for each block -------- #
                # Faster then doing pinv since it utilizes symmetry of matrices
                # in C_stacked
                # norm = _repeated_pinv2(Ck, svd_lwork)
                evals, evecs = np.linalg.eigh(C_stacked)
                tmp = np.divide(
                    evecs,
                    evals[:, np.newaxis, :],
                    where=evals[:, np.newaxis, :] != 0,
                    out=np.zeros_like(evecs)
                )
                norm = np.matmul(tmp, evecs.transpose(0, 2, 1))
                # -------------------------------------------------------- #
                W_stacked = np.matmul(norm, W_stacked)
            # Reapply source covariance after inversion

            # XXX: Do we need this?
            # It seems that sk always consists of ones; dmalt
            # norm *= sk
            # norm *= sk[:, np.newaxis]
        else:
            assert C_stacked.shape == (n_sources, 1, 1)
            # Fixed source orientation

            # Vectorized version of
            # norm = np.eye(1) if Ck[0, 0] == 0. else 1. / Ck
            # Wk[:] = norm * Wk
            np.divide(
                W_stacked, C_stacked, where=C_stacked != 0, out=W_stacked
            )

        if pick_ori == 'max-power':
            # XXX: If I use 'max-power' with 'single' and 'unit-noise-gain'
            # it looks like it's gonna crash since power isn't defined; dmalt

            # Compute the power
            if inversion == 'single' and weight_norm is not None:
                # First make the filters unit gain, then apply them to the
                # cov matrix to compute power.
                W_stacked_norm = (
                    W_stacked
                    / np.sqrt(np.sum(W_stacked ** 2, axis=-1, keepdims=True))
                )
                power = np.matmul(
                    np.matmul(W_stacked_norm, Cm),
                    W_stacked_norm.transpose(0, 2, 1)
                )
            elif weight_norm is None:
                # Compute power by applying the spatial filters to
                # the cov matrix.
                power = np.matmul(
                    np.matmul(W_stacked, Cm), W_stacked.transpose(0, 2, 1)
                )
                # power = Wk.dot(Cm).dot(Wk.T)

            # Compute the direction of max power
            # can use eigh here; dmalt
            # u, s, _ = _repeated_svd(power.real, real_svd_lwork)
            evals, evecs = np.linalg.eigh(power.real)
            # max_power_ori = u[:, 0]
            max_power_ori = evecs[:, :, -1]
            max_power_ori = max_power_ori[:, np.newaxis, :]
            # sign = np.sign((max_power_ori * nn).sum(axis=1))
            cos_with_normal = np.matmul(max_power_ori, nn[:, :, np.newaxis])
            sign = np.sign(
                cos_with_normal,
                where=cos_with_normal != 0,
                out=np.ones_like(cos_with_normal)
            )
            max_power_ori *= sign
            # sign = np.sign(np.dot(max_power_ori, nn)) or 1

            # set the (otherwise arbitrary) sign to match the normal
            # sign = np.sign(np.dot(nn[k], max_power_ori)) or 1  # avoid 0
            # max_power_ori *= sign

            # Re-compute the filter in the direction of max power
            W_max_ori = np.matmul(max_power_ori, W_stacked).squeeze()
            # Wk[:] = max_power_ori.dot(Wk)
            W_stacked[:, 0, :] = W_max_ori
        W = W_stacked.reshape(n_sources * n_orient, -1)

    if pick_ori == 'normal':
        W = W[2::3]
    elif pick_ori == 'max-power':
        W = W[0::3]

    # Re-scale the filter weights according to the selected weight
    # normalization scheme
    if weight_norm in ['unit-noise-gain', 'nai']:
        if pick_ori in [None, 'vector'] and n_orient > 1:
            # Rescale each set of 3 filters
            W = W.reshape(-1, 3, W.shape[1])
            noise_norm = np.sqrt(np.sum(W ** 2, axis=(1, 2), keepdims=True))
        else:
            # Rescale each filter separately
            noise_norm = np.sqrt(np.sum(W ** 2, axis=1, keepdims=True))

        if weight_norm == 'nai':
            # Estimate noise level based on covariance matrix, taking the
            # first eigenvalue that falls outside the signal subspace or the
            # loading factor used during regularization, whichever is largest.
            if rank > len(Cm):  # XXX: shouldn't it be '>=' ??; dmalt
                # Covariance matrix is full rank, no noise subspace!
                # Use the loading factor as noise ceiling.
                if loading_factor == 0:
                    raise RuntimeError(
                        'Cannot compute noise subspace with a full-rank '
                        'covariance matrix and no regularization. Try '
                        'manually specifying the rank of the covariance '
                        'matrix or using regularization.')
                noise = loading_factor
            else:
                noise, _ = linalg.eigh(Cm)
                noise = noise[-rank]
                noise = max(noise, loading_factor)
            noise_norm *= np.sqrt(noise)

        # Apply the normalization
        if np.all(noise_norm == 0.):
            noise_norm_inv = 0.  # avoid division by 0
        else:
            noise_norm_inv = 1 / noise_norm
        W *= noise_norm_inv
        W = W.reshape(-1, W.shape[-1])

    logger.info('Filter computation complete')
    return W


def _compute_power(Cm, W, n_orient):
    """Use beamformer filters to compute source power.

    Parameters
    ----------
    Cm : ndarray, shape (n_channels, n_channels)
        Data covariance matrix or CSD matrix.
    W : ndarray, shape (nvertices*norient, nchannels)
        Beamformer weights.

    Returns
    -------
    power : ndarray, shape (nvertices,)
        Source power.
    """
    n_sources = W.shape[0] // n_orient

    source_power = np.zeros(n_sources)
    for k in range(n_sources):
        Wk = W[n_orient * k: n_orient * k + n_orient]
        power = Wk.dot(Cm).dot(Wk.T)

        if n_orient > 1:  # Pool the orientations
            source_power[k] = np.abs(power.trace())
        else:
            source_power[k] = np.abs(power)

    return source_power


class Beamformer(dict):
    """A computed beamformer.

    Notes
    -----
    .. versionadded:: 0.17
    """

    def copy(self):
        """Copy the beamformer.

        Returns
        -------
        beamformer : instance of Beamformer
            A deep copy of the beamformer.
        """
        return deepcopy(self)

    def __repr__(self):  # noqa: D105
        n_verts = sum(len(v) for v in self['vertices'])
        n_channels = len(self['ch_names'])
        if self['subject'] is None:
            subject = 'unknown'
        else:
            subject = '"%s"' % (self['subject'],)
        out = ('<Beamformer  |  %s, subject %s, %s vert, %s ch'
               % (self['kind'], subject, n_verts, n_channels))
        if self['pick_ori'] is not None:
            out += ', %s ori' % (self['pick_ori'],)
        if self['weight_norm'] is not None:
            out += ', %s norm' % (self['weight_norm'],)
        if self.get('inversion') is not None:
            out += ', %s inversion' % (self['inversion'],)
        if 'rank' in self:
            out += ', rank %s' % (self['rank'],)
        out += '>'
        return out

    @verbose
    def save(self, fname, overwrite=False, verbose=None):
        """Save the beamformer filter.

        Parameters
        ----------
        fname : str
            The filename to use to write the HDF5 data.
            Should end in ``'-lcmv.h5'`` or ``'-dics.h5'``.
        overwrite : bool
            If True, overwrite the file (if it exists).
        %(verbose)s
        """
        ending = '-%s.h5' % (self['kind'].lower(),)
        check_fname(fname, self['kind'], (ending,))
        csd_orig = None
        try:
            if 'csd' in self:
                csd_orig = self['csd']
                self['csd'] = self['csd'].__getstate__()
            write_hdf5(fname, self, overwrite=overwrite, title='mnepython')
        finally:
            if csd_orig is not None:
                self['csd'] = csd_orig


def read_beamformer(fname):
    """Read a beamformer filter.

    Parameters
    ----------
    fname : str
        The filename of the HDF5 file.

    Returns
    -------
    filter : instance of Beamformer
        The beamformer filter.
    """
    beamformer = read_hdf5(fname, title='mnepython')
    if 'csd' in beamformer:
        beamformer['csd'] = CrossSpectralDensity(**beamformer['csd'])
    # h5io seems to cast `bool` to `int` on round-trip, probably a bug
    # we should fix at some point (if possible -- could be HDF5 limitation)
    for key in ('normalize_fwd', 'is_free_ori', 'is_ssp'):
        if key in beamformer:
            beamformer[key] = bool(beamformer[key])
    for key in ('data_cov', 'noise_cov'):
        if beamformer.get(key) is not None:
            for pi, p in enumerate(beamformer[key]['projs']):
                p = Projection(**p)
                p['active'] = bool(p['active'])
                beamformer[key]['projs'][pi] = p
            beamformer[key] = Covariance(
                *[beamformer[key].get(arg)
                  for arg in ('data', 'names', 'bads', 'projs', 'nfree', 'eig',
                              'eigvec', 'method', 'loglik')])
    return Beamformer(beamformer)
