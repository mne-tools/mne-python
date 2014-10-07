from mne.montages import read_montage
from ...utils import requires_mayavi


@requires_mayavi
def test_plot_montage():
    """Test plotting montages
    """
    read_montage('easycap-M1').plot()
