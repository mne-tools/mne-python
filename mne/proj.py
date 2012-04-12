# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import numpy as np
from scipy import linalg

from . import fiff
from .fiff.pick import pick_types


def read_proj(fname):
    """Read projections from a FIF file.

    Parameters
    ----------
    fname: string
        The name of file containing the projections vectors.

    Returns
    -------
    projs: list
        The list of projection vectors.
    """
    fid, tree, _ = fiff.fiff_open(fname)
    projs = fiff.proj.read_proj(fid, tree)
    return projs


def write_proj(fname, projs):
    """Write projections to a FIF file.

    Parameters
    ----------
    fname: string
        The name of file containing the projections vectors.

    projs: list
        The list of projection vectors.
    """
    fid = fiff.write.start_file(fname)
    fiff.proj.write_proj(fid, projs)
    fiff.write.end_file(fid)


def _compute_proj(data, info, n_grad, n_mag, n_eeg, desc_prefix):

    mag_ind = pick_types(info, meg='mag')
    grad_ind = pick_types(info, meg='grad')
    eeg_ind = pick_types(info, meg=False, eeg=True)

    if (n_grad > 0) and len(grad_ind) == 0:
        print "No gradiometers found. Forcing n_grad to 0"
        n_grad = 0
    if (n_mag > 0) and len(mag_ind) == 0:
        print "No magnetometers found. Forcing n_mag to 0"
        n_mag = 0
    if (n_eeg > 0) and len(eeg_ind) == 0:
        print "No EEG channels found. Forcing n_eeg to 0"
        n_eeg = 0

    ch_names = info['ch_names']
    grad_names, mag_names, eeg_names = ([ch_names[k] for k in ind]
                                     for ind in [grad_ind, mag_ind, eeg_ind])

    projs = []
    for n, ind, names, desc in zip([n_grad, n_mag, n_eeg],
                      [grad_ind, mag_ind, eeg_ind],
                      [grad_names, mag_names, eeg_names],
                      ['planar', 'axial', 'eeg']):
        if n == 0:
            continue
        data_ind = data[ind][:, ind]
        U = linalg.svd(data_ind, full_matrices=False,
                                         overwrite_a=True)[0][:, :n]
        for k, u in enumerate(U.T):
            proj_data = dict(col_names=names, row_names=None,
                             data=u[np.newaxis, :], nrow=1, ncol=u.size)
            this_desc = "%s-%s-PCA-%02d" % (desc, desc_prefix, k + 1)
            print "Adding projection: %s" % this_desc
            proj = dict(active=False, data=proj_data, desc=this_desc, kind=1)
            projs.append(proj)

    return projs


def compute_proj_epochs(epochs, n_grad=2, n_mag=2, n_eeg=2):
    """Compute SSP (spatial space projection) vectors on Epochs

    Parameters
    ----------
    epochs: instance of Epochs
        The epochs containing the artifact
    n_grad: int
        Number of vectors for gradiometers
    n_mag: int
        Number of vectors for gradiometers
    n_eeg: int
        Number of vectors for gradiometers

    Returns
    -------
    projs: list
        List of projection vectors
    """
    data = sum(np.dot(e, e.T) for e in epochs)  # compute data covariance
    event_id = epochs.event_id
    if event_id is None:
        event_id = 0
    desc_prefix = "%-d-%-.3f-%-.3f" % (event_id, epochs.tmin, epochs.tmax)
    return _compute_proj(data, epochs.info, n_grad, n_mag, n_eeg, desc_prefix)


def compute_proj_evoked(evoked, n_grad=2, n_mag=2, n_eeg=2):
    """Compute SSP (spatial space projection) vectors on Evoked

    Parameters
    ----------
    evoked: instance of Evoked
        The Evoked obtained by averaging the artifact
    n_grad: int
        Number of vectors for gradiometers
    n_mag: int
        Number of vectors for gradiometers
    n_eeg: int
        Number of vectors for gradiometers

    Returns
    -------
    projs: list
        List of projection vectors
    """
    data = np.dot(evoked.data, evoked.data.T)  # compute data covariance
    desc_prefix = "%-.3f-%-.3f" % (evoked.times[0], evoked.times[-1])
    return _compute_proj(data, evoked.info, n_grad, n_mag, n_eeg, desc_prefix)
