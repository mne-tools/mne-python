# Authors: Denis Engemann <denis.engemann@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#
# License: Simplified BSD

# Set our plotters to test mode
import matplotlib
matplotlib.use('Agg')  # for testing don't use X server

from mne.channels import read_montage  # noqa


def test_plot_montage():
    """Test plotting montages
    """
    read_montage('easycap-M1').plot()
