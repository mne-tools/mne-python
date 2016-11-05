# Authors: Denis Engemann <denis.engemann@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Teon Brooks <teon.brooks@gmail.com>
#
# License: Simplified BSD

# Set our plotters to test mode
import os.path as op

import matplotlib

from mne.channels import read_montage, read_dig_montage

matplotlib.use('Agg')  # for testing don't use X server


p_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'kit', 'tests', 'data')
elp = op.join(p_dir, 'test_elp.txt')
hsp = op.join(p_dir, 'test_hsp.txt')
hpi = op.join(p_dir, 'test_mrk.sqd')
point_names = ['nasion', 'lpa', 'rpa', '1', '2', '3', '4', '5']


def test_plot_montage():
    """Test plotting montages
    """
    m = read_montage('easycap-M1')
    m.plot()
    m.plot(show_names=True)
    d = read_dig_montage(hsp, hpi, elp, point_names)
    d.plot()
    d.plot(show_names=True)
