import numpy as np
from mne.montages import read_montage

lacks_mayavi = False
try:
    from mayavi import mlab
except ImportError:
    try:
        from enthought.mayavi import mlab
    except ImportError:
        lacks_mayavi = True
requires_mayavi = np.testing.dec.skipif(lacks_mayavi, 'Requires mayavi')


@requires_mayavi
def test_plot_montage():
    """Test plotting montages
    """
    read_montage('easycap-M1').plot()
