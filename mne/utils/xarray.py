import numpy as np

from .. import Epochs, Evoked
from ..time_frequency import AverageTFR


def to_xarray(mne_inst):
    '''Convert MNE instance to xarray DataArray.'''
    from xarray import DataArray
    from mne.utils import _validate_type

    _validate_type(mne_inst, (Epochs, Evoked, AverageTFR))

    coords = dict(chan=mne_inst.ch_names)
    if isinstance(mne_inst, Epochs):
        data = mne_inst.get_data()
        dims = ('chan', 'trial', 'time')
    elif isinstance(mne_inst, Evoked):
        data = mne_inst.data
        dims = ('chan', 'time')
    elif isinstance(mne_inst, AverageTFR):
        data = mne_inst.data
        dims = ('chan', 'freq', 'time')
    else:
        raise ValueError('MNE instance must be Epochs, Evoked or AverageTFR.')

    if 'time' in dims:
        coords['time'] = mne_inst.times
    if 'freq' in dims:
        coords['freq'] = mne_inst.freqs
    if 'trial' in dims:
        coords['trial'] = np.arange(mne_inst.n_epochs)

    return DataArray(data, dims=dims, coords=coords)
