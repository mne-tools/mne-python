# Authors: Alexandre Barachant <alexandre.barachant@gmail.com>
#
# License: BSD (3-clause)


import numpy as np
from scipy import linalg

from .. import Covariance, Evoked
from ..io.pick import (pick_types, pick_channels, pick_info)


def _least_square_evoked(data, events, events_id, nmin, nmax):
    """Least square estimation of evoked response from data
    return evoked data and toeplitz matrices.
    """
    window = nmax - nmin
    ne, ns = data.shape
    to = {}

    for eid in events_id:
        # select events by type
        ix_ev = events[:, -1] == events_id[eid]

        # build toeplitz matrix
        trig = np.zeros((ns, 1))
        trig[events[ix_ev, 0] + nmin] = 1
        toep = linalg.toeplitz(trig[0:window], trig)
        to[eid] = np.matrix(toep)

    # Concatenate toeplitz
    to_tot = np.concatenate(to.values())

    # least square estimation
    evo = linalg.pinv(to_tot * to_tot.T) * to_tot * data.T

    # parse evoked response
    evoked_data = {}
    for i, eid in enumerate(events_id):
        evoked_data[eid] = np.array(evo[(i * window):(i + 1) * window, :]).T

    return evoked_data, to
