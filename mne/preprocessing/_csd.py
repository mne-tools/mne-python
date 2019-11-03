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
from warnings import warn

from scipy.linalg import inv

from mne import pick_types, pick_info
from mne.utils import _validate_type, _check_ch_locs, _ensure_int
from mne.io import BaseRaw
from mne.io.constants import FIFF
from mne.epochs import BaseEpochs
from mne.evoked import Evoked
from mne.bem import fit_sphere_to_headshape
from mne.channels.interpolation import _calc_g, _calc_h


def _prepare_G(G, lambda2):
    G.flat[::len(G) + 1] += lambda2
    # compute the CSD
    Gi = inv(G)

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
    C2 = Cp2 - np.dot(TC[:, None], c02[None, :])
    X = np.dot(C2.T, H).T / radius ** 2
    return X


def compute_current_source_density(inst, lambda2=1e-5, stiffness=4,
                                   n_legendre_terms=50, sphere='auto',
                                   copy=True):
    """Get the current source density (CSD) transformation.

    Transformation based on spherical spline surface Laplacian.

    .. note:: This function applies an average reference to the data.
              Do not transform CSD data to source space.

    Parameters
    ----------
    inst : instance of Raw, Epochs or Evoked
        The data to be transformed.
    lambda2 : float
        Regularization parameter, produces smoothnes. Defaults to 1e-5.
    stiffness : float
        Stiffness of the spline. Also referred to as `m`.
    n_legendre_terms : int
        Number of Legendre terms to evaluate.
    sphere : array-like, shape (4,)
        The sphere, head-model of the form (x, y, z, r) where x, y, z
        is the center of the sphere and r is the radius in meters.
    copy : bool
        Whether to overwrite instance data or create a copy.

    Returns
    -------
    inst_csd : instance of Epochs or Evoked
        The transformed data. Output type will match input type.
    """
    _validate_type(inst, (BaseEpochs, BaseRaw, Evoked), 'inst')

    if inst.info['custom_ref_applied'] == -1:
        raise ValueError('CSD already applied, should not be reapplied')

    inst = inst.copy() if copy else inst

    if inst.info['bads']:
        warn('Deleting bad channels for consistency')
        inst.drop_channels(inst.info['bads'])

    picks = pick_types(inst.info, meg=False, eeg=True, exclude='bads')

    if len(picks) == 0:
        raise ValueError('No EEG channels found.')

    if lambda2 is None:
        lambda2 = 1e-5

    _validate_type(lambda2, 'numeric', 'lambda2')
    if 0 > lambda2 or lambda2 > 1:
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
    else:
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

    if not _check_ch_locs(inst.info['chs']):
        raise ValueError('Zero or infinite position found in chs')

    pos = np.array([inst.info['chs'][pick]['loc'][:3] for pick in picks])
    pos -= (x, y, z)

    G = _calc_g(np.dot(pos, pos.T), stiffness=stiffness,
                num_lterms=n_legendre_terms)
    H = _calc_h(np.dot(pos, pos.T), stiffness=stiffness,
                num_lterms=n_legendre_terms)

    G_precomputed = _prepare_G(G, lambda2)

    trans_csd = _compute_csd(G_precomputed=G_precomputed,
                             H=H, radius=radius)

    if isinstance(inst, BaseEpochs):
        for epo in inst._data:
            epo[picks] = np.dot(trans_csd, epo[picks])
    else:
        inst._data = np.dot(trans_csd, inst._data[picks])

    pick_info(inst.info, picks, copy=False)
    inst.info['custom_ref_applied'] = -1
    for pick in picks:
        inst.info['chs'][pick].update(kind=FIFF.FIFFV_EEG_CSD_CH,
                                      coil_type=FIFF.FIFFV_COIL_EEG_CSD,
                                      unit=FIFF.FIFF_UNIT_V_M2)
    return inst

# References
# ----------
#
# [1] Perrin F, Bertrand O, Pernier J. "Scalp current density mapping:
#     Value and estimation from potential data." IEEE Trans Biomed Eng.
#     1987;34(4):283–288.
#
# [1] Perrin F, Pernier J, Bertrand O, Echallier JF. "Spherical splines
#     for scalp potential and current density mapping."
#     [Corrigenda EEG 02274, EEG Clin. Neurophysiol., 1990, 76, 565]
#     Electroenceph Clin Neurophysiol. 1989;72(2):184–187.
#
# [2] Kayser J, Tenke CE. "On the benefits of using surface Laplacian
#     (Current Source Density) methodology in electrophysiology."
#     Int J Psychophysiol. 2015 Sep; 97(3): 171–173.
