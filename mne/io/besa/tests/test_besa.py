"""
Test reading BESA fileformats.
"""
import inspect
import numpy as np
import pytest
from pathlib import Path

from mne.io import read_evoked_besa


FILE = Path(inspect.getfile(inspect.currentframe()))
data_dir = FILE.parent / 'data'
avr_file = data_dir / 'simulation.avr'
avr_file_oldstyle = data_dir / 'simulation_oldstyle.avr'
mul_file = data_dir / 'simulation.mul'


@pytest.mark.filterwarnings("ignore:Fiducial point nasion not found")
def test_read_evoked_besa_avr(tmp_path):
    """Test reading BESA .avr files."""
    # Check new style .avr file
    ev = read_evoked_besa(avr_file)
    assert len(ev.ch_names) == len(ev.data) == 33
    assert ev.info['sfreq'] == 200
    assert ev.tmin == -0.1
    assert len(ev.times) == 200
    # Check electrode positions
    assert all([np.isnan(ch['loc'][:6]).sum() == 0 for ch in ev.info['chs']])

    # Check head_size parameter
    ev = read_evoked_besa(avr_file, head_size=1)
    assert np.ptp([ch['loc'][:3] for ch in ev.info['chs']]).round(2) == 2

    # Check old style .avr file
    ev = read_evoked_besa(avr_file_oldstyle)
    assert len(ev.ch_names) == len(ev.data) == 33
    assert ev.info['sfreq'] == 200
    assert ev.tmin == -0.1
    assert len(ev.times) == 200
    assert ev.ch_names[:5] == ['CH01', 'CH02', 'CH03', 'CH04', 'CH05']

    # Create BESA file with missing header fields
    with open(f'{tmp_path}/missing.avr', 'w') as f:
        f.write('DI= 5\n0\n')
    ev = read_evoked_besa(f'{tmp_path}/missing.avr')
    assert len(ev.ch_names) == len(ev.data) == 1
    assert ev.info['sfreq'] == 200
    assert ev.tmin == 0
    assert len(ev.times) == 1
    assert ev.ch_names == ['CH01']

    # The DI field (sample frequency) must exist
    with open(f'{tmp_path}/missing.avr', 'w') as f:
        f.write('Npts= 1  TSB= 0  SB= 1.00  SC= 500.0\n0\n')
    with pytest.raises(RuntimeError, match='No "DI" field present'):
        ev = read_evoked_besa(f'{tmp_path}/missing.avr')


def test_read_evoked_besa_mul():
    """Test reading BESA .mul files."""
    pass
