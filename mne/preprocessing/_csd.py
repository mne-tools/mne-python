# Copyright 2003-2010 Jürgen Kayser <rjk23@columbia.edu>
# Copyright 2017 Federico Raimondo <federaimondo@gmail.com> and
#                Denis A. Engemann <dengemann@gmail.com>
#
#
# The original CSD Toolbox can be find at
# http://psychophysiology.cpmc.columbia.edu/Software/CSDtoolbox/

# Authors: Denis A. Engeman <denis.engemann@gmail.com>
#          Alex Rockhill <aprockhill@mailbox.org>
#
# License: Relicensed under BSD-3-Clause and adapted with
#          permission from authors of original GPL code

import numpy as np

from .. import pick_types
from ..utils import (_validate_type, _ensure_int, _check_preload, verbose,
                     logger)
from ..io import BaseRaw
from ..io.constants import FIFF
from ..epochs import BaseEpochs, make_fixed_length_epochs
from ..evoked import Evoked
from ..bem import fit_sphere_to_headshape
from ..channels.interpolation import _calc_g, _calc_h


def _prepare_G(G, lambda2):
    G.flat[::len(G) + 1] += lambda2
    # compute the CSD
    Gi = np.linalg.inv(G)

    TC = Gi.sum(0)
    sgi = np.sum(TC)  # compute sum total

    return Gi, TC, sgi


def _compute_csd(G_precomputed, H, radius):
    """Compute the CSD."""
    n_channels = H.shape[0]
    data = np.eye(n_channels)
    mu = data.mean(0)
    Z = data - mu

    Gi, TC, sgi = G_precomputed

    Cp2 = np.dot(Gi, Z)
    c02 = np.sum(Cp2, axis=0) / sgi
    C2 = Cp2 - np.dot(TC[:, np.newaxis], c02[np.newaxis, :])
    X = np.dot(C2.T, H).T / radius ** 2
    return X


@verbose
def compute_current_source_density(inst, sphere='auto', lambda2=1e-5,
                                   stiffness=4, n_legendre_terms=50,
                                   copy=True, *, verbose=None):
    """Get the current source density (CSD) transformation.

    Transformation based on spherical spline surface Laplacian
    :footcite:`PerrinEtAl1987,PerrinEtAl1989,Cohen2014,KayserTenke2015`.

    This function can be used to re-reference the signal using a Laplacian
    (LAP) "reference-free" transformation.

    Parameters
    ----------
    inst : instance of Raw, Epochs or Evoked
        The data to be transformed.
    sphere : array-like, shape (4,) | str
        The sphere, head-model of the form (x, y, z, r) where x, y, z
        is the center of the sphere and r is the radius in meters.
        Can also be "auto" to use a digitization-based fit.
    lambda2 : float
        Regularization parameter, produces smoothness. Defaults to 1e-5.
    stiffness : float
        Stiffness of the spline.
    n_legendre_terms : int
        Number of Legendre terms to evaluate.
    copy : bool
        Whether to overwrite instance data or create a copy.
    %(verbose)s

    Returns
    -------
    inst_csd : instance of Raw, Epochs or Evoked
        The transformed data. Output type will match input type.

    Notes
    -----
    .. versionadded:: 0.20

    References
    ----------
    .. footbibliography::
    """
    _validate_type(inst, (BaseEpochs, BaseRaw, Evoked), 'inst')
    _check_preload(inst, 'Computing CSD')

    if inst.info['custom_ref_applied'] == FIFF.FIFFV_MNE_CUSTOM_REF_CSD:
        raise ValueError('CSD already applied, should not be reapplied')

    _validate_type(copy, (bool), 'copy')
    inst = inst.copy() if copy else inst

    picks = pick_types(inst.info, meg=False, eeg=True, exclude=[])

    if any([ch in np.array(inst.ch_names)[picks] for ch in inst.info['bads']]):
        raise ValueError('CSD cannot be computed with bad EEG channels. Either'
                         ' drop (inst.drop_channels(inst.info[\'bads\']) '
                         'or interpolate (`inst.interpolate_bads()`) '
                         'bad EEG channels.')

    if len(picks) == 0:
        raise ValueError('No EEG channels found.')

    _validate_type(lambda2, 'numeric', 'lambda2')
    if not 0 <= lambda2 < 1:
        raise ValueError('lambda2 must be between 0 and 1, got %s' % lambda2)

    _validate_type(stiffness, 'numeric', 'stiffness')
    if stiffness < 0:
        raise ValueError('stiffness must be non-negative got %s' % stiffness)

    n_legendre_terms = _ensure_int(n_legendre_terms, 'n_legendre_terms')
    if n_legendre_terms < 1:
        raise ValueError('n_legendre_terms must be greater than 0, '
                         'got %s' % n_legendre_terms)

    if isinstance(sphere, str) and sphere == 'auto':
        radius, origin_head, origin_device = fit_sphere_to_headshape(inst.info)
        x, y, z = origin_head - origin_device
        sphere = (x, y, z, radius)
    try:
        sphere = np.array(sphere, float)
        x, y, z, radius = sphere
    except Exception:
        raise ValueError(
            f'sphere must be "auto" or array-like with shape (4,), '
            f'got {sphere}')
    _validate_type(x, 'numeric', 'x')
    _validate_type(y, 'numeric', 'y')
    _validate_type(z, 'numeric', 'z')
    _validate_type(radius, 'numeric', 'radius')
    if radius <= 0:
        raise ValueError('sphere radius must be greater than 0, '
                         'got %s' % radius)

    pos = np.array([inst.info['chs'][pick]['loc'][:3] for pick in picks])
    if not np.isfinite(pos).all() or np.isclose(pos, 0.).all(1).any():
        raise ValueError('Zero or infinite position found in chs')
    pos -= (x, y, z)

    # Project onto a unit sphere to compute the cosine similarity:
    pos /= np.linalg.norm(pos, axis=1, keepdims=True)
    cos_dist = np.clip(np.dot(pos, pos.T), -1, 1)
    # This is equivalent to doing one minus half the squared Euclidean:
    # from scipy.spatial.distance import squareform, pdist
    # cos_dist = 1 - squareform(pdist(pos, 'sqeuclidean')) / 2.
    del pos

    G = _calc_g(cos_dist, stiffness=stiffness,
                n_legendre_terms=n_legendre_terms)
    H = _calc_h(cos_dist, stiffness=stiffness,
                n_legendre_terms=n_legendre_terms)

    G_precomputed = _prepare_G(G, lambda2)

    trans_csd = _compute_csd(G_precomputed=G_precomputed,
                             H=H, radius=radius)

    epochs = [inst._data] if not isinstance(inst, BaseEpochs) else inst._data
    for epo in epochs:
        epo[picks] = np.dot(trans_csd, epo[picks])
    with inst.info._unlock():
        inst.info['custom_ref_applied'] = FIFF.FIFFV_MNE_CUSTOM_REF_CSD
    for pick in picks:
        inst.info['chs'][pick].update(coil_type=FIFF.FIFFV_COIL_EEG_CSD,
                                      unit=FIFF.FIFF_UNIT_V_M2)

    # Remove rejection thresholds for EEG
    if isinstance(inst, BaseEpochs):
        if inst.reject and 'eeg' in inst.reject:
            del inst.reject['eeg']
        if inst.flat and 'eeg' in inst.flat:
            del inst.flat['eeg']

    return inst


@verbose
def compute_bridged_electrodes(inst, lm_cutoff=16, epoch_threshold=0.5,
                               l_freq=0.5, h_freq=30, epoch_duration=2,
                               bw_method=None, verbose=None):
    r"""Compute bridged EEG electrodes using the intrinsic Hjorth algorithm.

    First, an electrical distance matrix is computed by taking the pairwise
    variance between electrodes. Local minimums in this matrix below
    ``lm_cutoff`` are indicative of bridging between a pair of electrodes.
    Pairs of electrodes are marked as bridged as long as their electrical
    distance is below ``lm_cutoff`` on more than the ``epoch_threshold``
    proportion of epochs.

    Based on :footcite:`TenkeKayser2001,GreischarEtAl2004,DelormeMakeig2004`
    and the `EEGLAB implementation
    <https://psychophysiology.cpmc.columbia.edu/software/eBridge/index.html>`_.

    Parameters
    ----------
    inst : instance of Raw, Epochs or Evoked
        The data to compute electrode bridging on.
    lm_cutoff : float
        The distance in :math:`{\mu}V^2` cutoff below which to
        search for a local minimum (lm) indicative of bridging.
        EEGLAB defaults to 5 :math:`{\mu}V^2`. MNE defaults to
        16 :math:`{\mu}V^2` to be conservative based on the distributions in
        :footcite:t:`GreischarEtAl2004`.
    epoch_threshold : float
        The proportion of epochs with electrical distance less than
        ``lm_cutoff`` in order to consider the channel bridged.
        The default is 0.5.
    l_freq : float
        The low cutoff frequency to use. Default is 0.5 Hz.
    h_freq : float
        The high cutoff frequency to use. Default is 30 Hz.
    epoch_duration : float
        The time in seconds to divide the raw into fixed-length epochs
        to check for consistent bridging. Only used if ``inst`` is
        :class:`mne.io.BaseRaw`. The default is 2 seconds.
    bw_method : None
        ``bw_method`` to pass to :class:`scipy.stats.gaussian_kde`.
    %(verbose)s

    Returns
    -------
    bridged_idx : list of tuple
        The indices of channels marked as bridged with each bridged
        pair stored as a tuple.
    ed_matrix : ndarray of float, shape (n_epochs, n_channels, n_channels)
        The electrical distance matrix for each pair of EEG electrodes.

    Notes
    -----
    .. versionadded:: 1.1

    References
    ----------
    .. footbibliography::
    """
    from scipy.stats import gaussian_kde
    from scipy.optimize import minimize_scalar
    _check_preload(inst, 'Computing bridged electrodes')
    inst = inst.copy()  # don't modify original
    picks = pick_types(inst.info, eeg=True)
    if len(picks) == 0:
        raise RuntimeError('No EEG channels found, cannot compute '
                           'electrode bridging')
    # first, filter
    inst.filter(l_freq=l_freq, h_freq=h_freq, picks=picks, verbose=False)

    if isinstance(inst, BaseRaw):
        inst = make_fixed_length_epochs(inst, duration=epoch_duration,
                                        preload=True, verbose=False)

    # standardize shape
    data = inst.get_data(picks=picks)
    if isinstance(inst, Evoked):
        data = data[np.newaxis, ...]  # expand evoked

    # next, compute electrical distance matrix, upper triangular
    n_epochs = data.shape[0]
    ed_matrix = np.zeros((n_epochs, picks.size, picks.size)) * np.nan
    for i in range(picks.size):
        for j in range(i + 1, picks.size):
            ed_matrix[:, i, j] = np.var(data[:, i] - data[:, j], axis=1)

    # scale, fill in other half, diagonal
    ed_matrix *= 1e12  # scale to muV**2

    # initialize bridged indices
    bridged_idx = list()

    # if not enough values below local minimum cutoff, return no bridges
    ed_flat = ed_matrix[~np.isnan(ed_matrix)]
    if ed_flat[ed_flat < lm_cutoff].size / n_epochs < epoch_threshold:
        return bridged_idx, ed_matrix

    # kernel density estimation
    kde = gaussian_kde(ed_flat[ed_flat < lm_cutoff])
    with np.errstate(invalid='ignore'):
        local_minimum = float(minimize_scalar(
            lambda x: kde(x) if x < lm_cutoff and x > 0 else np.inf).x)
    logger.info(f'Local minimum {local_minimum} found')

    # find electrodes that are below the cutoff local minimum on
    # `epochs_threshold` proportion of epochs
    for i in range(picks.size):
        for j in range(i + 1, picks.size):
            bridged_count = np.sum(ed_matrix[:, i, j] < local_minimum)
            if bridged_count / n_epochs > epoch_threshold:
                logger.info('Bridge detected between '
                            f'{inst.ch_names[picks[i]]} and '
                            f'{inst.ch_names[picks[j]]}')
                bridged_idx.append((picks[i], picks[j]))

    return bridged_idx, ed_matrix
