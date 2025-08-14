# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# Set our plotters to test mode

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from mne.channels import make_dig_montage, make_standard_montage, read_dig_fif

p_dir = Path(__file__).parents[2] / "io" / "kit" / "tests" / "data"
elp = p_dir / "test_elp.txt"
hsp = p_dir / "test_hsp.txt"
hpi = p_dir / "test_mrk.sqd"
io_dir = Path(__file__).parents[2] / "io" / "tests" / "data"
fif_fname = io_dir / "test_raw.fif"


def test_plot_montage():
    """Test plotting montages."""
    m = make_standard_montage("easycap-M1")
    m.plot()
    plt.close("all")
    m.plot(kind="3d")
    plt.close("all")
    m.plot(kind="3d", show_names=True)
    plt.close("all")
    m.plot(kind="topomap")
    plt.close("all")
    m.plot(kind="topomap", show_names=True)
    plt.close("all")
    m.plot(sphere="eeglab")
    plt.close("all")

    N_HSP, N_HPI = 2, 1
    montage = make_dig_montage(
        nasion=[1, 1, 1],
        lpa=[2, 2, 2],
        rpa=[3, 3, 3],
        hsp=np.full((N_HSP, 3), 4),
        hpi=np.full((N_HPI, 3), 4),
        coord_frame="head",
    )
    assert "0 channels" in repr(montage)
    with pytest.raises(RuntimeError, match="No valid channel positions"):
        montage.plot()
    d = read_dig_fif(fname=fif_fname, verbose="error")
    assert "61 channels" in repr(d)
    # XXX this is broken; dm.point_names is used. Sometimes we say this should
    # Just contain the HPI coils, other times that it's all channels (e.g.,
    # EEG channels). But there is redundancy with this and dm.dig_ch_pos.
    # This should be addressed in the pending big refactoring.
    # d.plot()
    # plt.close('all')


@pytest.mark.parametrize(
    "name, n",
    [
        ("standard_1005", 342),
        ("standard_postfixed", 85),
        ("standard_primed", 85),
        ("standard_1020", 93),
    ],
)
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
        ch_pos=dict(zip(list("abc"), np.eye(3))), coord_frame="head"
    )
    montage.plot()
    plt.close("all")
