"""Functions shared between different beamformer types."""

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
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
from ..utils import verbose, check_fname, _reg_pinv, _check_option
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


def _normalized_weights(Wk, Gk, Cm_inv_sq, reduce_rank, nn, sk):
    """Compute the normalized weights in max-power orientation.

    Uses Eq. 4.47 from [1]_.

    Parameters
    ----------
    Wk : ndarray, shape (3, n_channels)
        The set of un-normalized filters at a single source point.
    Gk : ndarray, shape (n_channels, 3)
        The leadfield at a single source point.
    Cm_inv_sq : nsarray, snape (n_channels, n_channels)
        The squared inverse covariance matrix.
    reduce_rank : bool
        Whether to reduce the rank of the filter by one.
    nn : ndarray, shape (3,)
        The source normal.
    sk : ndarray, shape (3,)
        The source prior.

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
    norm_inv = np.dot(Gk.T, np.dot(Cm_inv_sq, Gk))
    if reduce_rank:
        # Use pseudo inverse computation setting smallest
        # component to zero if the leadfield is not full rank
        norm = _reg_pinv(norm_inv, rank=norm_inv.shape[0] - 1)[0]
    else:
        # Use straight inverse with full rank leadfield
        try:
            norm = linalg.inv(norm_inv)
        except np.linalg.linalg.LinAlgError:
            raise ValueError(
                'Singular matrix detected when estimating spatial filters. '
                'Consider reducing the rank of the forward operator by using '
                'reduce_rank=True.'
            )
    # Reapply source covariance after inversion
    norm *= sk
    norm *= sk[:, np.newaxis]
    power = np.dot(norm, np.dot(Wk, Gk))

    # Determine orientation of max power
    eig_vals, eig_vecs = linalg.eig(power)
    if not np.iscomplex(power).any() and np.iscomplex(eig_vecs).any():
        raise ValueError('The eigenspectrum of the leadfield at this voxel is '
                         'complex. Consider reducing the rank of the '
                         'leadfield by using reduce_rank=True.')

    idx_max = eig_vals.argmax()
    max_power_ori = eig_vecs[:, idx_max]

    # set the (otherwise arbitrary) sign to match the normal
    sign = np.sign(np.dot(max_power_ori, nn)) or 1
    max_power_ori *= sign

    # Compute the filter in the orientation of max power
    Wk[:] = np.dot(max_power_ori, Wk)
    Gk = np.dot(Gk, max_power_ori)
    denom = np.dot(Gk.T, np.dot(Cm_inv_sq, Gk))
    denom = np.sqrt(denom)
    Wk /= denom

    return Wk


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

    for k in range(n_sources):
        this_sl = slice(n_orient * k, n_orient * k + n_orient)
        Wk, Gk, sk = W[this_sl], G[:, this_sl], orient_std[this_sl]

        if (inversion == 'matrix' and pick_ori == 'max-power' and
                weight_norm in ['unit-noise-gain', 'nai']):
            # In this case, take a shortcut to compute the filter
            Wk[:] = _normalized_weights(
                Wk, Gk, Cm_inv_sq, reduce_rank, nn[k], sk)
        else:
            # Compute power at the source
            Ck = np.dot(Wk, Gk)

            # Normalize the spatial filters
            if Wk.ndim == 2 and len(Wk) > 1:
                # Free source orientation
                if inversion == 'single':
                    # Invert for each dipole separately using plain division
                    with np.errstate(divide='ignore'):
                        norm = np.diag(1. / np.diag(Ck))
                elif inversion == 'matrix':
                    # Invert for all dipoles simultaneously using matrix
                    # inversion.
                    norm = linalg.pinv2(Ck)
                # Reapply source covariance after inversion
                norm *= sk
                norm *= sk[:, np.newaxis]
            else:
                assert Ck.shape == (1, 1)
                # Fixed source orientation
                norm = np.eye(1) if Ck[0, 0] == 0. else 1. / Ck
            Wk[:] = np.dot(norm, Wk)

            if pick_ori == 'max-power':
                # Compute the power
                if inversion == 'single' and weight_norm is not None:
                    # First make the filters unit gain, then apply them to the
                    # cov matrix to compute power.
                    Wk_norm = Wk / np.sqrt(np.sum(Wk ** 2, axis=1,
                                                  keepdims=True))
                    power = Wk_norm.dot(Cm).dot(Wk_norm.T)
                elif weight_norm is None:
                    # Compute power by applying the spatial filters to
                    # the cov matrix.
                    power = Wk.dot(Cm).dot(Wk.T)

                # Compute the direction of max power
                u, s, _ = np.linalg.svd(power.real)
                max_power_ori = u[:, 0]

                # set the (otherwise arbitrary) sign to match the normal
                sign = np.sign(np.dot(nn[k], max_power_ori)) or 1  # avoid 0
                max_power_ori *= sign

                # Re-compute the filter in the direction of max power
                Wk[:] = max_power_ori.dot(Wk)

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

        # Apply the normalization
        if np.all(noise_norm == 0.):
            noise_norm_inv = 0.  # avoid division by 0
        else:
            noise_norm_inv = 1 / noise_norm
        W *= noise_norm_inv
        W = W.reshape(-1, W.shape[-1])

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
