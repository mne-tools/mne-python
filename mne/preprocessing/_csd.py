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
from ..utils import _validate_type, _ensure_int, _check_preload
from ..io import BaseRaw
from ..io.constants import FIFF
from ..epochs import BaseEpochs
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


def compute_current_source_density(inst, sphere='auto', lambda2=1e-5,
                                   stiffness=4, n_legendre_terms=50,
                                   copy=True):
    """Get the current source density (CSD) transformation.

    Transformation based on spherical spline surface Laplacian
    :footcite:`PerrinEtAl1987,PerrinEtAl1989,Cohen2014,KayserTenke2015`.

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
    return inst
