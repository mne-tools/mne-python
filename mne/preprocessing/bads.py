# Authors: Denis Engemann <denis.engemann@gmail.com>
#          Ana Radanovic <radanovica@protonmail.com>
#          Erica Peterson <nordme@uw.edu>
# License: BSD-3-Clause

import numpy as np
from scipy.stats import zscore


from ..utils import _validate_type

from ..io import BaseRaw
from ..epochs import Epochs
from ..evoked import Evoked
from ..time_frequency.spectrum import BaseSpectrum


def _find_outliers(X, threshold=3.0, max_iter=2, tail=0):
    """Find outliers based on iterated Z-scoring.

    This procedure compares the absolute z-score against the threshold.
    After excluding local outliers, the comparison is repeated until no
    local outlier is present any more.

    Parameters
    ----------
    X : np.ndarray of float, shape (n_elemenets,)
        The scores for which to find outliers.
    threshold : float
        The value above which a feature is classified as outlier.
    max_iter : int
        The maximum number of iterations.
    tail : {0, 1, -1}
        Whether to search for outliers on both extremes of the z-scores (0),
        or on just the positive (1) or negative (-1) side.

    Returns
    -------
    bad_idx : np.ndarray of int, shape (n_features)
        The outlier indices.
    """
    my_mask = np.zeros(len(X), dtype=bool)
    for _ in range(max_iter):
        X = np.ma.masked_array(X, my_mask)
        if tail == 0:
            this_z = np.abs(zscore(X))
        elif tail == 1:
            this_z = zscore(X)
        elif tail == -1:
            this_z = -zscore(X)
        else:
            raise ValueError("Tail parameter %s not recognised." % tail)
        local_bad = this_z > threshold
        my_mask = np.max([my_mask, local_bad], 0)
        if not np.any(local_bad):
            break

    bad_idx = np.where(my_mask)[0]
    return bad_idx


def unify_bad_channels(insts):
    """Unify bad channels across a list of instances, using the union.

    Parameters
    ----------
    insts : list
        List of instances (:class:`~mne.io.Raw`, :class:`~mne.Epochs`,
        :class:`~mne.Evoked`, :class:`~mne.time_frequency.Spectrum`,
        :class:`~mne.time_frequency.EpochSpectrum`) across which to unify bad channels.

    Returns
    -------
    insts : list
        List of instances with bad channels unified across instances.

    Notes
    -----
    This function operates in-place.

    .. versionadded:: 1.6
    """
    # check input is not an empty list
    if len(insts) == 0:
        raise ValueError("Be sure insts is not empty list")

    # first check that each object is mne object
    inst_type = type(insts[0])
    valid_types = (BaseRaw, Epochs, Evoked, BaseSpectrum)
    for inst in insts:
        _validate_type(inst, valid_types, "instance type")
        if type(inst) != inst_type:
            raise ValueError("All insts must be the same type")

    # check that input is a list
    if not isinstance(insts, list):
        raise ValueError(f"insts must be a *list* of mne objects, got {type(insts)}")

    # check that all channels have the same name and same number
    ch_names = insts[0].ch_names
    diff_chns = []
    for inst in insts[1:]:
        if inst.ch_names != ch_names:
            dif = set(inst.info.ch_names) ^ (set(ch_names))
            diff_chns.extend(list(dif))

    if len(diff_chns) > 0:
        raise ValueError(
            "Channel names are not consistent across instances."
            f" Mismatch channels are {diff_chns}"
        )

    # then iterate through the insts to gather bads
    all_bads = dict()
    for inst in insts:
        # using dictionary method to remove duplicates & preserve order
        all_bads.update(dict.fromkeys(inst.info["bads"]))
    all_bads = list(all_bads)

    # apply bads set to all instances
    new_insts = []

    for inst in insts:
        inst.info["bads"] = all_bads
        new_insts.append(inst)

    return new_insts
