"""Dynamic Imaging of Coherent Sources (DICS).
"""

# Authors: Roman Goj <roman.goj@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from scipy import linalg

import logging
logger = logging.getLogger('mne')

from ..fiff.constants import FIFF
from ..fiff.proj import make_projector
from ..fiff.pick import pick_types, pick_channels_forward
from ..minimum_norm.inverse import _get_vertno
from ..source_space import label_src_vertno_sel
from .. import verbose


@verbose
def dics_epochs(epochs, info, tmin, forward, noise_csd, data_csd, reg=0.1,
                label=None, picks=None, pick_ori=None, verbose=None):
    """ DICS

    Parameters
    ----------
    data : array or list / iterable
        Sensor space data. If data.ndim == 2 a single observation is assumed
        and a single stc is returned. If data.ndim == 3 or if data is
        a list / iterable, a list of stc's is returned.
    info : dict
        Measurement info.
    tmin : float
        Time of first sample.
    forward : dict
        Forward operator.
    noise_cov : Covariance
        The noise covariance.
    data_cov : Covariance
        The data covariance.
    reg : float
        The regularization for the whitened data covariance.
    label : Label | None
        Restricts the LCMV solution to a given label.
    picks : array of int | None
        Indices (in info) of data channels. If None, MEG and EEG data channels
        (without bad channels) will be used.
    pick_ori : None | 'normal' | 'max-power'
        If 'normal', rather than pooling the orientations by taking the norm,
        only the radial component is kept. If 'max-power', the source
        orientation that maximizes output source power is chosen.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    stc : SourceEstimate (or list of SourceEstimate)
        Source time courses.
    """
    # TODO: DICS, in the original 2001 paper, used a free orientation beamformer,
    # however selection of the max-power orientation was also employed depending on
    # whether a dominant component was present
    pick_ori = None

    is_free_ori = forward['source_ori'] == FIFF.FIFFV_MNE_FREE_ORI

    if pick_ori in ['normal', 'max-power'] and not is_free_ori:
        raise ValueError('Normal or max-power orientation can only be picked '
                         'when a forward operator with free orientation is '
                         'used.')
    if pick_ori == 'normal' and not forward['surf_ori']:
        raise ValueError('Normal orientation can only be picked when a '
                         'forward operator oriented in surface coordinates is '
                         'used.')
    if pick_ori == 'normal' and not forward['src'][0]['type'] == 'surf':
        raise ValueError('Normal orientation can only be picked when a '
                         'forward operator with a surface-based source space '
                         'is used.')

    # Set picks again for epochs
    picks = pick_types(epochs.info, meg=True, eeg=False, eog=False, stim=False,
                       exclude='bads')

    ch_names = [epochs.info['ch_names'][k] for k in picks]

    # Restrict forward solution to selected channels
    forward = pick_channels_forward(forward, include=ch_names)

    # Get gain matrix (forward operator)
    if label is not None:
        vertno, src_sel = label_src_vertno_sel(label, forward['src'])

        if is_free_ori:
            src_sel = 3 * src_sel
            src_sel = np.c_[src_sel, src_sel + 1, src_sel + 2]
            src_sel = src_sel.ravel()

        G = forward['sol']['data'][:, src_sel]
    else:
        vertno = _get_vertno(forward['src'])
        G = forward['sol']['data']

    # Apply SSPs
    proj, ncomp, _ = make_projector(epochs.info['projs'], ch_names)
    G = np.dot(proj, G)

    Cm = data_csd.data

    # Cm += reg * np.trace(Cm) / len(Cm) * np.eye(len(Cm))
    Cm_inv = linalg.pinv(Cm, reg)

    # Compute spatial filters
    W = np.dot(G.T, Cm_inv)
    n_orient = 3 if is_free_ori else 1
    n_sources = G.shape[1] // n_orient
    source_power = np.zeros(n_sources)
    for k in range(n_sources):
        Wk = W[n_orient * k: n_orient * k + n_orient]
        Gk = G[:, n_orient * k: n_orient * k + n_orient]
        Ck = np.dot(Wk, Gk)

        # Find source orientation maximizing output source power
        # TODO: max-power is not used in this example, however DICS does employ
        # orientation picking when one eigen value is much larger than the other
        if pick_ori == 'max-power':
            eig_vals, eig_vecs = linalg.eigh(Ck)

            # Choosing the eigenvector associated with the middle eigenvalue.
            # The middle and not the minimal eigenvalue is used because MEG is
            # insensitive to one (radial) of the three dipole orientations and
            # therefore the smallest eigenvalue reflects mostly noise.
            for i in range(3):
                if i != eig_vals.argmax() and i != eig_vals.argmin():
                    idx_middle = i

            # TODO: The eigenvector associated with the smallest eigenvalue
            # should probably be used when using combined EEG and MEG data
            max_ori = eig_vecs[:, idx_middle]

            Wk[:] = np.dot(max_ori, Wk)
            Ck = np.dot(max_ori, np.dot(Ck, max_ori))
            is_free_ori = False

        if is_free_ori:
            # Free source orientation
            Wk[:] = np.dot(linalg.pinv(Ck, 0.1), Wk)
        else:
            # Fixed source orientation
            Wk /= Ck

        # TODO: Vectorize outside of the loop?
        source_power[k] = np.real_if_close(np.dot(Wk, np.dot(data_csd.data,
                                                             Wk.conj().T)).trace())

    # Preparing noise normalization
    # TODO: Noise normalization in DICS should takes into account noise CSD
    noise_norm = np.sum((W * W.conj()), axis=1)
    noise_norm = np.real_if_close(noise_norm)
    if is_free_ori:
        noise_norm = np.sum(np.reshape(noise_norm, (-1, 3)), axis=1)
    noise_norm = np.sqrt(noise_norm)

    # Applying noise normalization
    source_power /= noise_norm

    return source_power
