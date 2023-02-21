import numpy as np


def to_xarray(inst, picks=None):
    """Convert MNE object instance to xarray DataArray.

    Parameters
    ----------
    inst : Epochs | Evoked
        The MNE object to convert.
    picks : list of str | array-like of int | None
        Channels to include. If None only good data channels are kept.
        (I couldn't use @fill_doc here, so I put this temporarily here)

    Returns
    -------
    xarr : DataArray
        The xarray object.
    """
    from xarray import DataArray
    from .. import Epochs, Evoked
    from . import _validate_type
    from ..io.pick import _picks_to_idx

    _validate_type(inst, (Epochs, Evoked))

    if isinstance(inst, Epochs):
        dims = ('chan', 'epoch', 'time')
    elif isinstance(inst, Evoked):
        dims = ('chan', 'time')
    else:
        raise ValueError('MNE instance must be Epochs or Evoked.')

    coords = dict(chan=inst.ch_names)
    if picks is not None:
        picks = _picks_to_idx(inst.info, picks, exclude=[])
        coords['chan'] = [inst.ch_names[pck] for pck in picks]

    if 'time' in dims:
        coords['time'] = inst.times
    if 'epoch' in dims:
        coords['epoch'] = np.arange(inst.n_epochs)

    data = inst.get_data(picks=picks)
    xarr = DataArray(data, dims=dims, coords=coords)

    # add channel types as additional dimension coordinate
    xarr = xarr.assign_coords(
        ch_type=('chan', inst.get_channel_types(picks=picks))
    )
    return xarr
