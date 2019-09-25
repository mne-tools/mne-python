# Authors: Denis Engemann <denis.engemann@gmail.com>
#          Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Teon Brooks <teon.brooks@gmail.com>
#
# License: Simplified BSD

# Set our plotters to test mode
import os.path as op
import numpy as np

import pytest
import matplotlib.pyplot as plt

from mne.io.kit import read_mrk
from mne.channels import (read_dig_fif, make_dig_montage,
                          read_polhemus_fastscan,
                          make_standard_montage)

p_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'kit', 'tests', 'data')
elp = op.join(p_dir, 'test_elp.txt')
hsp = op.join(p_dir, 'test_hsp.txt')
hpi = op.join(p_dir, 'test_mrk.sqd')
point_names = ['nasion', 'lpa', 'rpa', '1', '2', '3', '4', '5']
io_dir = op.join(op.dirname(__file__), '..', '..', 'io', 'tests', 'data')
fif_fname = op.join(io_dir, 'test_raw.fif')


def test_plot_montage():
    """Test plotting montages."""
    m = make_standard_montage('easycap-M1')
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

    montage = make_dig_montage(hsp=read_polhemus_fastscan(hsp),
                               hpi=read_mrk(hpi))
    elp_points = read_polhemus_fastscan(elp)
    ch_pos = {"EEG%03d" % (k + 1): pos for k, pos in enumerate(elp_points[8:])}
    montage += make_dig_montage(nasion=elp_points[0],
                                lpa=elp_points[1],
                                rpa=elp_points[2],
                                ch_pos=ch_pos)
    assert '0 channels' in repr(montage)
    with pytest.raises(RuntimeError, match='No valid channel positions'):
        montage.plot()
    d = read_dig_fif(fname=fif_fname)
    assert '61 channels' in repr(d)
    # XXX this is broken; dm.point_names is used. Sometimes we say this should
    # Just contain the HPI coils, other times that it's all channels (e.g.,
    # EEG channels). But there is redundancy with this and dm.dig_ch_pos.
    # This should be addressed in the pending big refactoring.
    # d.plot()
    # plt.close('all')


@pytest.mark.parametrize('name, n', [
    ('standard_1005', 342), ('standard_postfixed', 85),
    ('standard_primed', 85), ('standard_1020', 93)
])
def test_plot_defect_montage(name, n):
    """Test plotting defect montages (i.e. with duplicate labels)."""
    # montage name and number of unique labels
    m = make_standard_montage(name)
    n -= 3  # new montage does not have fiducials
    fig = m.plot()
    collection = fig.axes[0].collections[0]
    assert collection._edgecolors.shape[0] == n
    assert collection._facecolors.shape[0] == n
    assert collection._offsets.shape[0] == n


def test_plot_digmontage():
    """Test plot DigMontage."""
    montage = make_dig_montage(
        ch_pos=dict(zip(list('abc'), np.eye(3))),
        coord_frame='head'
    )
    montage.plot()
    plt.close('all')
