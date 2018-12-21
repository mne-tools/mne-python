# Authors: Denis Engemann <denis.engemann@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Teon Brooks <teon.brooks@gmail.com>
#
# License: Simplified BSD

# Set our plotters to test mode
import os.path as op

import pytest
import matplotlib.pyplot as plt

from mne.channels import read_montage, read_dig_montage

p_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'kit', 'tests', 'data')
elp = op.join(p_dir, 'test_elp.txt')
hsp = op.join(p_dir, 'test_hsp.txt')
hpi = op.join(p_dir, 'test_mrk.sqd')
point_names = ['nasion', 'lpa', 'rpa', '1', '2', '3', '4', '5']
io_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
fif_fname = op.join(io_dir, 'test_raw.fif')


def test_plot_montage():
    """Test plotting montages."""
    m = read_montage('easycap-M1')
    m.plot()
    plt.close('all')
    m.plot(kind='3d')
    plt.close('all')
    m.plot(kind='3d', show_names=True)
    plt.close('all')
    m.plot(kind='topomap')
    plt.close('all')
    m.plot(kind='topomap', show_names=True)
    plt.close('all')
    d = read_dig_montage(hsp, hpi, elp, point_names)
    assert '0 channels' in repr(d)
    with pytest.raises(RuntimeError, match='No valid channel positions'):
        d.plot()
    d = read_dig_montage(fif=fif_fname)
    assert '61 channels' in repr(d)
    # XXX this is broken; dm.point_names is used. Sometimes we say this should
    # Just contain the HPI coils, other times that it's all channels (e.g.,
    # EEG channels). But there is redundancy with this and dm.dig_ch_pos.
    # This should be addressed in the pending big refactoring.
    # d.plot()
    # plt.close('all')


def test_plot_defect_montage():
    """Test plotting defect montages (i.e. with duplicate labels)."""
    # montage name and number of unique labels
    montages = [('standard_1005', 342), ('standard_postfixed', 85),
                ('standard_primed', 85), ('standard_1020', 93)]
    for name, n in montages:
        m = read_montage(name)
        fig = m.plot()
        collection = fig.axes[0].collections[0]
        assert collection._edgecolors.shape[0] == n
        assert collection._facecolors.shape[0] == n
        assert collection._offsets.shape[0] == n
