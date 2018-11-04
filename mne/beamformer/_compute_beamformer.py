"""Functions shared between different beamformer types."""

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Roman Goj <roman.goj@gmail.com>
#          Britta Westner <britta.wstnr@gmail.com>
#
# License: BSD (3-clause)

from copy import deepcopy
import operator

import numpy as np
from scipy import linalg

from ..cov import Covariance
from ..io.constants import FIFF
from ..io.proj import make_projector, Projection
from ..io.pick import (pick_channels_forward, pick_info)
from ..minimum_norm.inverse import _get_vertno
from ..source_space import label_src_vertno_sel
from ..utils import logger, warn, verbose, check_fname, _reg_pinv
from ..channels.channels import _contains_ch_type
from ..time_frequency.csd import CrossSpectralDensity

from ..externals.h5io import read_hdf5, write_hdf5
from ..externals.six import string_types


def _check_rank(rank):
    """Check rank parameter and deal with deprecation."""
    if isinstance(rank, string_types):
        # XXX we can use rank='' to deprecate to get to None eventually:
        # if rank == '':
        #     warn('The rank parameter default in 0.18 of "full" will change '
        #          'to None in 0.19, set it explicitly to avoid this warning',
        #          DeprecationWarning)
        #     rank = 'full'
        if rank != 'full':
            raise ValueError('rank, if str, must be "full", got %s' % (rank,))
    elif rank is not None and not isinstance(rank, dict):
        try:
            rank = int(operator.index(rank))
        except TypeError:
            raise TypeError('rank must be None, dict, "full", or int-like, '
                            'got %s (type %s)' % (rank, type(rank)))
    return rank


def _setup_picks(info, forward, data_cov=None, noise_cov=None):
    """Return good channels common to forward model and covariance matrices."""
    # get a list of all channel names:
    fwd_ch_names = forward['info']['ch_names']

    # handle channels from forward model and info:
    ch_names = _compare_ch_names(info['ch_names'], fwd_ch_names, info['bads'])

    # inform about excluding channels:
    if (data_cov is not None and set(info['bads']) != set(data_cov['bads']) and
            (len(set(ch_names).intersection(data_cov['bads'])) > 0)):
        logger.info('info["bads"] and data_cov["bads"] do not match, '
                    'excluding bad channels from both.')
    if (noise_cov is not None and
            set(info['bads']) != set(noise_cov['bads']) and
            (len(set(ch_names).intersection(noise_cov['bads'])) > 0)):
        logger.info('info["bads"] and noise_cov["bads"] do not match, '
                    'excluding bad channels from both.')

    # handle channels from data cov if data cov is not None
    # Note: data cov is supposed to be None in tf_lcmv
    if data_cov is not None:
        ch_names = _compare_ch_names(ch_names, data_cov.ch_names,
                                     data_cov['bads'])

    # handle channels from noise cov if noise cov available:
    if noise_cov is not None:
        ch_names = _compare_ch_names(ch_names, noise_cov.ch_names,
                                     noise_cov['bads'])

    picks = [info['ch_names'].index(k) for k in ch_names if k in
             info['ch_names']]
    return picks


def _compare_ch_names(names1, names2, bads):
    """Return channel names of common and good channels."""
    ch_names = [ch for ch in names1 if ch not in bads and ch in names2]
    return ch_names


def _check_one_ch_type(info, picks, noise_cov, method):
    """Check number of sensor types and presence of noise covariance matrix."""
    info_pick = pick_info(info, sel=picks)
    ch_types =\
        [_contains_ch_type(info_pick, tt) for tt in ('mag', 'grad', 'eeg')]
    if method == 'lcmv' and sum(ch_types) > 1 and noise_cov is None:
        raise ValueError('Source reconstruction with several sensor types '
                         'requires a noise covariance matrix to be '
                         'able to apply whitening.')
    elif method == 'dics' and sum(ch_types) > 1:
        warn('The use of several sensor types with the DICS beamformer is '
             'not heavily tested yet.')


def _pick_channels_spatial_filter(ch_names, filters):
    """Return data channel indices to be used with spatial filter.

    Unlike ``pick_channels``, this respects the order of ch_names.
    """
    sel = []
    # first check for channel discrepancies between filter and data:
    for ch_name in filters['ch_names']:
        if ch_name not in ch_names:
            raise ValueError('The spatial filter was computed with channel %s '
                             'which is not present in the data. You should '
                             'compute a new spatial filter restricted to the '
                             'good data channels.' % ch_name)
    # then compare list of channels and get selection based on data:
    sel = [ii for ii, ch_name in enumerate(ch_names)
           if ch_name in filters['ch_names']]
    return sel


def _check_proj_match(info, filters):
    """Check whether SSP projections in data and spatial filter match."""
    proj_data, _, _ = make_projector(info['projs'],
                                     filters['ch_names'])
    if not np.array_equal(proj_data, filters['proj']):
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


def _prepare_beamformer_input(info, forward, label, picks, pick_ori,
                              fwd_norm=None):
    """Input preparation common for all beamformer functions.

    Check input values, prepare channel list and gain matrix. For documentation
    of parameters, please refer to _apply_lcmv.
    """
    is_free_ori = forward['source_ori'] == FIFF.FIFFV_MNE_FREE_ORI

    if pick_ori in ['normal', 'max-power', 'vector']:
        if not is_free_ori:
            raise ValueError(
                'Normal or max-power orientation can only be picked '
                'when a forward operator with free orientation is used.')
    elif pick_ori is not None:
        raise ValueError('pick_ori must be one of "normal", "max-power", '
                         '"vector", or None, got %s' % (pick_ori,))
    if pick_ori == 'normal' and not forward['surf_ori']:
        # XXX eventually this could just call convert_forward_solution
        raise ValueError('Normal orientation can only be picked when a '
                         'forward operator oriented in surface coordinates is '
                         'used.')
    if pick_ori == 'normal' and not forward['src'][0]['type'] == 'surf':
        raise ValueError('Normal orientation can only be picked when a '
                         'forward operator with a surface-based source space '
                         'is used.')
    # Restrict forward solution to selected channels
    info_ch_names = [ch['ch_name'] for ch in info['chs']]
    ch_names = [info_ch_names[k] for k in picks]
    fwd_ch_names = forward['sol']['row_names']
    # Keep channels in forward present in info:
    fwd_ch_names = [ch for ch in fwd_ch_names if ch in info_ch_names]
    forward = pick_channels_forward(forward, fwd_ch_names)
    picks_forward = [fwd_ch_names.index(ch) for ch in ch_names]

    # Get gain matrix (forward operator)
    nn = forward['source_nn']
    if is_free_ori:  # take Z coordinate
        nn = nn[2::3]
    nn = nn.copy()
    if forward['surf_ori']:
        nn[...] = [0, 0, 1]  # align to local +Z coordinate
    if label is not None:
        vertno, src_sel = label_src_vertno_sel(label, forward['src'])
        nn = nn[src_sel]

        if is_free_ori:
            src_sel = 3 * src_sel
            src_sel = np.c_[src_sel, src_sel + 1, src_sel + 2]
            src_sel = src_sel.ravel()

        G = forward['sol']['data'][:, src_sel]
    else:
        vertno = _get_vertno(forward['src'])
        G = forward['sol']['data']

    # Apply SSPs
    proj, ncomp, _ = make_projector(info['projs'], fwd_ch_names)

    if info['projs']:
        G = np.dot(proj, G)

    # Pick after applying the projections. This makes a copy of G, so further
    # operations can be safely done in-place.
    G = G[picks_forward]
    proj = proj[np.ix_(picks_forward, picks_forward)]

    # Normalize the leadfield if requested
    if fwd_norm == 'dipole':  # each orientation separately
        G /= np.linalg.norm(G, axis=0)
    elif fwd_norm == 'vertex':  # all three orientations per loc jointly
        depth_prior = np.sum(G ** 2, axis=0)
        if is_free_ori:
            depth_prior = depth_prior.reshape(-1, 3).sum(axis=1)
        # Spherical leadfield can be zero at the center
        depth_prior[depth_prior == 0.] = np.min(
            depth_prior[depth_prior != 0.])
        if is_free_ori:
            depth_prior = np.repeat(depth_prior, 3)
        source_weighting = np.sqrt(1. / depth_prior)
        G *= source_weighting[np.newaxis, :]
    elif fwd_norm is not None:
        raise ValueError('Got invalid value for "fwd_norm". Valid '
                         'values are: "dipole", "vertex" or None.')

    return is_free_ori, ch_names, proj, vertno, G, nn


def _normalized_weights(Wk, Gk, Cm_inv_sq, reduce_rank, nn):
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
    assert Wk.shape[0] == Gk.shape[1] == 3
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
    sign = np.sign(np.dot(max_power_ori, nn))
    sign = 1 if sign == 0 else sign
    max_power_ori *= sign

    # Compute the filter in the orientation of max power
    Wk[:] = np.dot(max_power_ori, Wk)
    Gk = np.dot(Gk, max_power_ori)
    denom = np.dot(Gk.T, np.dot(Cm_inv_sq, Gk))
    denom = np.sqrt(denom)
    Wk /= denom

    return Wk


def _compute_beamformer(G, Cm, reg, n_orient, weight_norm, pick_ori,
                        reduce_rank, rank, inversion, nn):
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
    rank : int | None | 'full'
        This controls the effective rank of the covariance matrix when
        computing the inverse. The rank can be set explicitly by specifying an
        integer value. If ``None``, the rank will be automatically estimated.
        Since applying regularization will always make the covariance matrix
        full rank, the rank is estimated before regularization in this case. If
        'full', the rank will be estimated after regularization and hence
        will mean using the full rank, unless ``reg=0`` is used.
    inversion : 'matrix' | 'single'
        The inversion scheme to compute the weights.
    nn : ndarray, shape (n_dipoles, 3)
        The source normals.

    Returns
    -------
    W : ndarray, shape (n_dipoles, n_channels)
        The beamformer filter weights.
    """
    # Tikhonov regularization using reg parameter to control for
    # trade-off between spatial resolution and noise sensitivity
    # eq. 25 in Gross and Ioannides, 1999 Phys. Med. Biol. 44 2081
    Cm_inv, loading_factor, rank = _reg_pinv(Cm, reg, rank)

    if (inversion == 'matrix' and pick_ori == 'max-power' and
            weight_norm in ['unit-noise-gain', 'nai']):
        Cm_inv_sq = Cm_inv.dot(Cm_inv)

    # Compute spatial filters
    W = np.dot(G.T, Cm_inv)
    n_sources = G.shape[1] // n_orient
    assert nn.shape == (n_sources, 3)

    for k in range(n_sources):
        Wk = W[n_orient * k: n_orient * k + n_orient]
        Gk = G[:, n_orient * k: n_orient * k + n_orient]

        # Compute power at the source
        Ck = np.dot(Wk, Gk)

        if (inversion == 'matrix' and pick_ori == 'max-power' and
                weight_norm in ['unit-noise-gain', 'nai']):
            # In this case, take a shortcut to compute the filter
            Wk[:] = _normalized_weights(Wk, Gk, Cm_inv_sq, reduce_rank, nn[k])
        else:
            # Normalize the spatial filters
            if Wk.ndim == 2 and len(Wk) > 1:
                # Free source orientation
                if inversion == 'single':
                    # Invert for each dipole separately using plain division
                    Wk /= np.diag(Ck)[:, np.newaxis]
                elif inversion == 'matrix':
                    # Invert for all dipoles simultaneously using matrix
                    # inversion.
                    Wk[:] = np.dot(linalg.pinv(Ck, 0.1), Wk)
            else:
                # Fixed source orientation
                if not np.all(Ck == 0.):
                    Wk /= Ck

            if pick_ori == 'max-power':
                # Compute the power
                if inversion == 'single' and weight_norm == 'unit-noise-gain':
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
                sign = np.sign(np.dot(nn[k], max_power_ori))
                sign = 1 if sign == 0 else sign  # corner case
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
            if rank >= len(Cm):
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
        verbose : bool, str, int, or None
            If not None, override default verbose level (see
            :func:`mne.verbose` and :ref:`Logging documentation <tut_logging>`
            for more).
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
