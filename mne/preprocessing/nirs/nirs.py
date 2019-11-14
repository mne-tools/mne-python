# Authors: Robert Luke <mail@robertluke.net>
#          Eric Larson <larson.eric.d@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#
# License: BSD (3-clause)

import numpy as np
from scipy import linalg

from ...io.pick import _picks_to_idx
from ...utils import fill_doc


@fill_doc
def source_detector_distances(info, picks=None):
    r"""Determine the distance between NIRS source and detectors.

    Parameters
    ----------
    info : Info
        The measurement info.
    %(picks_all)s

    Returns
    -------
    dists : array of float
        Array containing distances in meters.
        Of shape equal to number of channels, or shape of picks if supplied.
    """
    dist = [linalg.norm(ch['loc'][3:6] - ch['loc'][6:9])
            for ch in info['chs']]
    picks = _picks_to_idx(info, picks)
    return np.array(dist, float)[picks]


def short_channels(info, threshold=0.01):
    r"""Determine which NIRS channels are short.

    Channels with a source to detector distance of less than
    `threshold` are reported as short. The default threshold is 0.01 m.

    Parameters
    ----------
    info : Info
        The measurement info.
    threshold : float
        The threshold distance for what is considered short in meters.

    Returns
    -------
    short : array of bool
        Array indicating which channels are short.
        Of shape equal to number of channels.
    """
    return source_detector_distances(info) < threshold
