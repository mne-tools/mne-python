"""Test reading BESA fileformats."""
import inspect
import pytest
from pathlib import Path

from mne.io import read_evoked_besa
from mne.channels import read_custom_montage


FILE = Path(inspect.getfile(inspect.currentframe()))
data_dir = FILE.parent / 'data'
avr_file = data_dir / 'simulation.avr'
avr_file_oldstyle = data_dir / 'simulation_oldstyle.avr'
mul_file = data_dir / 'simulation.mul'
montage = read_custom_montage(data_dir / 'simulation.elp')


@pytest.mark.filterwarnings("ignore:Fiducial point nasion not found")
@pytest.mark.parametrize('fname', (avr_file, avr_file_oldstyle, mul_file))
def test_read_evoked_besa(fname):
    """Test reading MESA .avr and .mul files."""
    ev = read_evoked_besa(fname)
    assert len(ev.ch_names) == len(ev.data) == 33
    assert ev.info['sfreq'] == 200
    assert ev.tmin == -0.1
    assert len(ev.times) == 200
    assert ev.ch_names == montage.ch_names
    assert ev.comment == 'simulation'


def test_read_evoked_besa_avr_incomplete(tmp_path):
    """Test reading incomplete BESA .avr files."""
    # Check old style .avr file without an .elp sidecar
    with open(f'{tmp_path}/missing.avr', 'w') as f:
        f.write('Npts= 1  TSB= 0  SB= 1.00  SC= 500.0  DI= 5\n0\n1\n2\n')
    ev = read_evoked_besa(f'{tmp_path}/missing.avr')
    assert ev.ch_names == ['CH01', 'CH02', 'CH03']

    # Create BESA file with missing header fields and verify things don't break
    with open(f'{tmp_path}/missing.avr', 'w') as f:
        f.write('DI= 5\n0\n')
    ev = read_evoked_besa(f'{tmp_path}/missing.avr')
    assert len(ev.ch_names) == len(ev.data) == 1
    assert ev.info['sfreq'] == 200
    assert ev.tmin == 0
    assert len(ev.times) == 1
    assert ev.ch_names == ['CH01']
    assert ev.comment == ''

    # The DI field (sample frequency) must exist
    with open(f'{tmp_path}/missing.avr', 'w') as f:
        f.write('Npts= 1  TSB= 0  SB= 1.00  SC= 500.0\n0\n')
    with pytest.raises(RuntimeError, match='No "DI" field present'):
        ev = read_evoked_besa(f'{tmp_path}/missing.avr')


def test_read_evoked_besa_mul_incomplete(tmp_path):
    """Test reading incomplete BESA .mul files."""
    # Create BESA file with missing header fields and verify things don't break
    with open(f'{tmp_path}/missing.mul', 'w') as f:
        f.write('SamplingInterval[ms]= 5\nCH1\n0\n')
    ev = read_evoked_besa(f'{tmp_path}/missing.mul')
    assert len(ev.ch_names) == len(ev.data) == 1
    assert ev.info['sfreq'] == 200
    assert ev.tmin == 0
    assert len(ev.times) == 1
    assert ev.ch_names == ['CH1']
    assert ev.comment == ''

    # The SamplingInterval[ms] field (sample frequency) must exist
    with open(f'{tmp_path}/missing.mul', 'w') as f:
        f.write('TimePoints= 1 Channels= 1\nCH1\n0\n')
    with pytest.raises(RuntimeError, match=r'No "SamplingInterval\[ms\]"'):
        ev = read_evoked_besa(f'{tmp_path}/missing.mul')
