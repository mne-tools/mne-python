# Copyright 2003-2010 Jürgen Kayser <rjk23@columbia.edu>
# Copyright 2017 Federico Raimondo <federaimondo@gmail.com> and
#                Denis A. Engemann <dengemann@gmail.com>
#
#
# The original CSD Toolbox can be find at
# http://psychophysiology.cpmc.columbia.edu/Software/CSDtoolbox/

# Authors: Denis A. Engeman <denis.engemann@gmail.com>
#          Alex Rockhill <aprockhill206@gmail.com>
#
# License: Relicensed under BSD (3-clause) and adapted with
#          permission from authors of original GPL code

import numpy as np

from scipy import linalg

from .. import pick_types
from ..utils import _validate_type, _ensure_int
from ..io import BaseRaw
from ..io.constants import FIFF
from ..epochs import BaseEpochs
from ..evoked import Evoked
from ..bem import fit_sphere_to_headshape
from ..channels.interpolation import _calc_g, _calc_h


def _prepare_G(G, lambda2):
    G.flat[::len(G) + 1] += lambda2
    # compute the CSD
    Gi = linalg.inv(G)

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

    Transformation based on spherical spline surface Laplacian [1]_ [2]_ [3]_.

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
    This function applies an average reference to the data if copy is False.
    Do not transform CSD data to source space.

    .. versionadded:: 0.20

    References
    ----------
    .. [1] Perrin F, Bertrand O, Pernier J. "Scalp current density mapping:
           Value and estimation from potential data." IEEE Trans Biomed Eng.
           1987;34(4):283–288.
    .. [2] Perrin F, Pernier J, Bertrand O, Echallier JF. "Spherical splines
           for scalp potential and current density mapping."
           Electroenceph Clin Neurophysiol. 1989;72(2):184–187.
    .. [3] Kayser J, Tenke CE. "On the benefits of using surface Laplacian
           (Current Source Density) methodology in electrophysiology.
           Int J Psychophysiol. 2015 Sep; 97(3): 171–173.
    """
    _validate_type(inst, (BaseEpochs, BaseRaw, Evoked), 'inst')

    if inst.info['custom_ref_applied'] == FIFF.FIFFV_MNE_CUSTOM_REF_CSD:
        raise ValueError('CSD already applied, should not be reapplied')

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

    if sphere == 'auto':
        radius, origin_head, origin_device = fit_sphere_to_headshape(inst.info)
        x, y, z = origin_head - origin_device
        sphere = (x, y, z, radius)
    _validate_type(sphere, tuple, 'sphere')
    x, y, z, radius = sphere
    _validate_type(x, 'numeric', 'x')
    _validate_type(y, 'numeric', 'y')
    _validate_type(z, 'numeric', 'z')
    _validate_type(radius, 'numeric', 'radius')
    if radius <= 0:
        raise ValueError('sphere radius must be greater than 0, '
                         'got %s' % radius)

    _validate_type(copy, (bool), 'copy')

    pos = np.array([inst.info['chs'][pick]['loc'][:3] for pick in picks])
    if not np.isfinite(pos).all() or np.isclose(pos, 0.).all(1).any():
        raise ValueError('Zero or infinite position found in chs')
    pos -= (x, y, z)

    G = _calc_g(np.dot(pos, pos.T), stiffness=stiffness,
                n_legendre_terms=n_legendre_terms)
    H = _calc_h(np.dot(pos, pos.T), stiffness=stiffness,
                n_legendre_terms=n_legendre_terms)

    G_precomputed = _prepare_G(G, lambda2)

    trans_csd = _compute_csd(G_precomputed=G_precomputed,
                             H=H, radius=radius)

    epochs = [inst._data] if not isinstance(inst, BaseEpochs) else inst._data
    for epo in epochs:
        epo[picks] = np.dot(trans_csd, epo[picks])
    inst.info['custom_ref_applied'] = FIFF.FIFFV_MNE_CUSTOM_REF_CSD
    for pick in picks:
        inst.info['chs'][pick].update(coil_type=FIFF.FIFFV_COIL_EEG_CSD,
                                      unit=FIFF.FIFF_UNIT_V_M2)
    return inst

# References
# ----------
#
# [1] Perrin F, Bertrand O, Pernier J. "Scalp current density mapping:
#     Value and estimation from potential data." IEEE Trans Biomed Eng.
#     1987;34(4):283–288.
#
# [2] Perrin F, Pernier J, Bertrand O, Echallier JF. "Spherical splines
#     for scalp potential and current density mapping."
#     [Corrigenda EEG 02274, EEG Clin. Neurophysiol., 1990, 76, 565]
#     Electroenceph Clin Neurophysiol. 1989;72(2):184–187.
#
# [3] Kayser J, Tenke CE. "On the benefits of using surface Laplacian
#     (Current Source Density) methodology in electrophysiology."
#     Int J Psychophysiol. 2015 Sep; 97(3): 171–173.
