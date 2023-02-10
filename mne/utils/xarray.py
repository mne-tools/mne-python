import numpy as np

from .. import Epochs, Evoked
from ..time_frequency import AverageTFR


def to_xarray(mne_inst):
    """Convert MNE object instance to xarray DataArray.

    Parameters
    ----------
    mne_inst : Epochs | Evoked
        The MNE object to convert.

    Returns
    -------
    xarr : DataArray
        The xarray object.
    """
    from xarray import DataArray
    from mne.utils import _validate_type

    _validate_type(mne_inst, (Epochs, Evoked, AverageTFR))

    if isinstance(mne_inst, Epochs):
        data = mne_inst.get_data()
        dims = ('chan', 'trial', 'time')
    elif isinstance(mne_inst, Evoked):
        data = mne_inst.data
        dims = ('chan', 'time')
    else:
        raise ValueError('MNE instance must be Epochs or Evoked.')

    coords = dict(chan=mne_inst.ch_names)
    if 'time' in dims:
        coords['time'] = mne_inst.times
    if 'trial' in dims:
        coords['trial'] = np.arange(mne_inst.n_epochs)

    return DataArray(data, dims=dims, coords=coords)
