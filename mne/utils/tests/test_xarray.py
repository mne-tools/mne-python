import numpy as np
import mne
from mne.utils import requires_xarray


@requires_xarray
def test_conversion_to_xarray():
    """Test conversion of mne object to xarray DataArray."""
    import xarray as xr

    info = mne.create_info(list('abcd'), sfreq=250)
    data = np.random.rand(4, 350)
    evoked = mne.EvokedArray(data, info, tmin=-0.5)

    xarr = evoked.to_xarray()
    assert isinstance(xarr, xr.DataArray)
    assert xarr.shape == evoked.data.shape
    assert xarr.dims == ('chan', 'time')
    assert xarr.coords['chan'].data.tolist() == evoked.ch_names
    assert (xarr.coords['time'].data == evoked.times).all()

    xarr = evoked.to_xarray(picks=['b', 'd'])
    assert xarr.shape == (2, 350)
    assert xarr.coords['chan'].data.tolist() == ['b', 'd']
    assert (xarr.data == evoked.data[[1, 3]]).all()
