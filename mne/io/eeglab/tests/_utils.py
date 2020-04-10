# Author: Joan Massich <mailsik@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
from mne.channels import make_dig_montage
from mne.transforms import _sph_to_cart, _topo_to_sph


# XXX: This is a workaround to get the previous behavior.
def _read_eeglab_montage(fname):
    """Read an EEGLAB digitization file.

    Parameters
    ----------
    fname : str
        The filepath of Polhemus ISOTrak formatted file.
        File extension is expected to be '.loc', '.locs' or '.eloc'.

    Returns
    -------
    montage : instance of DigMontage
        The montage.

    See Also
    --------
    make_dig_montage
    """
    ch_names = np.genfromtxt(fname, dtype=str, usecols=3).tolist()
    topo = np.loadtxt(fname, dtype=float, usecols=[1, 2])
    sph = _topo_to_sph(topo)
    pos = _sph_to_cart(sph)
    pos[:, [0, 1]] = pos[:, [1, 0]] * [-1, 1]

    return make_dig_montage(
        ch_pos=dict(zip(ch_names, pos)),
        coord_frame='head',
    )
