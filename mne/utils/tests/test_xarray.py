import numpy as np
import mne


def test_conversion_to_xarray():
    import xarray as xr
    from mne.utils.xarray import to_xarray

    info = mne.create_info(list('abcd'), sfreq=250)
    data = np.random.rand(4, 350)
    erp = mne.EvokedArray(data, info, tmin=-0.5)

    erp_x = to_xarray(erp)
    assert isinstance(erp_x, xr.DataArray)
    assert erp_x.shape == (4, 350)
    assert erp_x.dims == ('chan', 'time')
    assert erp_x.coords['chan'].data.tolist() == erp.ch_names
    assert (erp_x.coords['time'].data == erp.times).all()
