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
                     _pl, _check_src_normal, check_version, _sym_inv, warn)
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
    _check_src_normal(pick_ori, forward['src'])
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


def _reduce_leadfield_rank(G):
    """Reduce the rank of the leadfield."""
    # decompose lead field
    u, s, v = np.linalg.svd(G, full_matrices=False)

    # backproject, omitting one direction (equivalent to setting the smallest
    # singular value to zero)
    G = np.matmul(u[:, :, :-1], s[:, :-1, np.newaxis] * v[:, :-1, :])

    return G


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
    assert Cm.shape == (G.shape[0],) * 2
    # Because of whitening, Cm is only guaranteed to be positive
    # semidefinite when loading is added, otherwise it can have
    # negative eigenvalues (?)
    # So make sure _reg_pinv handles them properly by using abs
    s, _ = np.linalg.eigh(Cm)
    if not (s >= -s.max() * 1e-7).all():
        # This shouldn't ever happen, but just in case
        warn('data covariance does not appear to be positive semidefinite, '
             'results will likely be incorrect')
    Cm_inv, loading_factor, rank = _reg_pinv(Cm, reg, rank)

    assert orient_std.shape == (G.shape[1],)
    n_sources = G.shape[1] // n_orient
    assert nn.shape == (n_sources, 3)

    logger.info('Computing beamformer filters for %d source%s'
                % (n_sources, _pl(n_sources)))
    n_channels = G.shape[0]
    assert n_orient in (3, 1)
    Gk = np.reshape(G.T, (n_sources, n_orient, n_channels)).transpose(0, 2, 1)
    assert Gk.shape == (n_sources, n_channels, n_orient)
    sk = np.reshape(orient_std, (n_sources, n_orient))
    del G, orient_std
    pinv_kwargs = dict()
    if check_version('numpy', '1.17'):
        pinv_kwargs['hermitian'] = True

    _check_option('reduce_rank', reduce_rank, (True, False))

    # inversion of the denominator
    _check_option('inversion', inversion, ('matrix', 'single'))
    if reduce_rank and inversion == 'single':
        raise ValueError('reduce_rank cannot be used with inversion="single"; '
                         'consider using inversion="matrix" if you have a '
                         'rank-deficient forward model (i.e., from a sphere '
                         'model with MEG channels), otherwise consider using '
                         'reduce_rank=False')
    if n_orient > 1:
        _, Gk_s, _ = np.linalg.svd(Gk, full_matrices=False)
        assert Gk_s.shape == (n_sources, n_orient)
        if not reduce_rank and (Gk_s[:, 0] > 1e6 * Gk_s[:, 2]).any():
            raise ValueError(
                'Singular matrix detected when estimating spatial filters. '
                'Consider reducing the rank of the forward operator by using '
                'reduce_rank=True.')
        del Gk_s

    # rank reduction of the lead field
    if reduce_rank:
        Gk = _reduce_leadfield_rank(Gk)

    #
    # 1. Compute max power orientation if applicable
    #
    if pick_ori == 'max-power':
        # compute the numerator of the weight formula
        numer = np.matmul(np.matmul(Gk.transpose(0, 2, 1),
                                    Cm_inv[np.newaxis]), Gk)
        if weight_norm is None:
            # compute covariance as shortcut, cf Van Veen 1997
            # TODO is _sym_inv okay or should be np.linalg.pinv() ?
            ori_pick = _sym_inv(numer, reduce_rank)
        else:
            assert weight_norm in ['unit-noise-gain', 'nai']
            # compute power, cf Sekihara & Nagarajan 2008, eq. 4.47
            Cm_inv_sq = Cm_inv.dot(Cm_inv)
            denom = np.matmul(np.matmul(Gk.transpose(0, 2, 1),
                                        Cm_inv_sq[np.newaxis]), Gk)
            # TODO is _sym_inv okay or should be np.linalg.pinv() ?
            ori_pick = np.matmul(_sym_inv(denom, reduce_rank), numer)

        # TODO: single inversion for ori selection is currently missing
        # # Compute the power
        # if inversion == 'matrix':
        #     # Compute power by applying the spatial filters to the cov matrix
        #     mult = numer
        # else:
        #     assert inversion == 'single'
        #     # First make the filters unit gain, then apply them to the
        #     # cov matrix to compute power.
        #     mult = numer / np.linalg.norm(numer, axis=2, keepdims=True)
        # power = np.matmul(np.matmul(mult, Cm),
        #                   mult.conj().transpose(0, 2, 1))
        # assert ori_pick.shape == (n_sources, 3, 3)

        # pick eigenvector that corresponds to maximum eigenvalue:
        eig_vals, eig_vecs = np.linalg.eig(ori_pick.real)  # not Hermitian!
        # sort eigenvectors by eigenvalues for picking:
        order = np.argsort(np.abs(eig_vals), axis=-1)[..., ::-1]
        # eig_vals = np.take_along_axis(eig_vals, order, axis=-1)
        eig_vecs = np.take_along_axis(eig_vecs, order[..., np.newaxis, :],
                                      axis=-1)
        max_power_ori = eig_vecs[:, :, 0]  # sorted eigenvecs in columns
        assert max_power_ori.shape == (n_sources, 3)

        # TODO bring back - this completely destroys the relationship to
        # external tests now. Probably should not be done on lead field?
        # set the (otherwise arbitrary) sign to match the normal
        # signs = np.sign(np.sum(max_power_ori[:, np.newaxis] * nn, axis=1))
        # signs[signs == 0] = 1.
        # max_power_ori *= signs

        # Compute the lead field for the optimal orientation
        Gk = np.matmul(Gk, max_power_ori[:, :, np.newaxis])
        n_orient = 1

    #
    # 2. Compute numerator and denominator of beamformer formula (unit-gain)
    #

    # numer = G.T @ Cm_inv
    # denom = G.T @ Cm_inv @ G
    numer = np.matmul(Gk.transpose(0, 2, 1), Cm_inv[np.newaxis])
    denom = np.matmul(numer, Gk)

    #
    # 3. Invert the denominator
    #

    if n_orient > 1:
        # Free source orientation
        if inversion == 'single':
            # Invert for each dipole separately using plain division
            diags = np.diagonal(denom, axis1=1, axis2=2)
            assert not reduce_rank   # guaranteed above
            with np.errstate(divide='ignore'):
                diags = 1. / diags
            # set the diagonal of each 3x3
            norm = np.zeros((n_sources, n_orient, n_orient), denom.dtype)
            for k in range(n_sources):
                norm[k].flat[::4] = diags[k]
        else:
            assert inversion == 'matrix'
            assert denom.shape[1:] == (3, 3)
            # TODO is any of this still needed? - I guess not?
            # Invert for all dipoles simultaneously using matrix inversion.
            # if pick_ori == 'max-power' and weight_norm is not None:
            #     # G.T @ Cm_inv @ Cm_inv @ G == Wk @ Wk.H
            #     norm_inv = np.matmul(numer, np.swapaxes(numer, -2, -1).conj())
            # else:
            #     # Wk @ Gk
            #     norm_inv = denom
            norm = _sym_inv(denom, reduce_rank)
        # Reapply source covariance after inversion
        norm *= sk[:, :, np.newaxis]
        norm *= sk[:, np.newaxis, :]
    else:  # n_orient == 1
        assert denom.shape[1:] == (1, 1)
        # Fixed source orientation
        with np.errstate(divide='ignore'):
            norm = 1. / denom
        norm[~np.isfinite(norm)] = 1.
    assert norm.shape == (n_sources, n_orient, n_orient)
    assert numer.shape == (n_sources, n_orient, n_channels)

    # do the normalization of the filters
    W = np.matmul(norm, numer)  # np.dot for each source

    #
    # 4. Do the picking and reshaping
    #

    if pick_ori == 'max-power':
        W = np.squeeze(W)
    elif pick_ori == 'normal':
        W = W[:, 2]
        max_power_ori = None
    else:
        W = W.reshape(n_sources * n_orient, n_channels)
        max_power_ori = None

    del Gk, numer, sk

    #
    # 5. Re-scale filter weights according to the selected weight_norm
    #
    # Weight normalization is done by computing:
    # W_ung = inv(sqrt(W_ug @ W_ug.T)) @ W_ug
    # with W_ung referring to the unit-noise-gain (weight normalized) filter
    # and W_ug referring to the above-calculated unit-gain filter stored in W

    if weight_norm is not None:
        if pick_ori in [None, 'vector'] and n_orient > 1:
            # Rescale each set of 3 filters
            # TODO is this conform with Sekihara & Nagarajan 2008 for vector
            # beamformers?
            W = W.reshape(-1, 3, W.shape[-1])
            noise_norm = np.linalg.norm(W, axis=(1, 2), keepdims=True)
        else:
            # Compute sqrt(W_ug @ W_ug.T)
            noise_norm = np.linalg.norm(W, axis=1, keepdims=True)

        if weight_norm == 'nai':
            # Estimate noise level based on covariance matrix, taking the
            # first eigenvalue that falls outside the signal subspace or the
            # loading factor used during regularization, whichever is largest.
            if rank > len(Cm):
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

        # Apply the normalization: W_ung = W_ug / noise_norm
        noise_norm_inv = 1 / np.where(noise_norm > 0, noise_norm, 1.)
        W *= noise_norm_inv
        W = W.reshape(-1, W.shape[-1])

    logger.info('Filter computation complete')
    return W, max_power_ori


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
        power = Wk.dot(Cm).dot(Wk.conj().T)

        if n_orient > 1:  # Pool the orientations
            source_power[k] = power.trace().real
        else:
            source_power[k] = power.real

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
