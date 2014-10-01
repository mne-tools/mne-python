import numpy as np
from mne.montages import read_montage

skip = False
try:
    from mayavi import mlab
except ImportError:
    try:
        from enthought.mayavi import mlab
    except ImportError:
        skip = True
requires_mayavi = np.testing.dec.skipif(skip, 'Requires mayavi')


@requires_mayavi
def test_plot_montage():
    """Test plotting montages
    """
    read_montage('easycap-M1').plot()
