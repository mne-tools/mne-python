from copy import deepcopy

from mne.datasets import testing
from mne.io import read_raw_fif

directory = testing.data_path() / "MEG" / "sample"
fname = directory / "sample_audvis_trunc_raw.fif"
raw = read_raw_fif(fname, preload=False)


def test_eq_ne():
    """Test == and != between projectors."""
    pca1 = deepcopy(raw.info["projs"][0])
    pca2 = deepcopy(raw.info["projs"][1])
    car = deepcopy(raw.info["projs"][3])

    assert pca1 != pca2
    assert pca1 != car
    assert pca2 != car
    assert pca1 == raw.info["projs"][0]
    assert pca2 == raw.info["projs"][1]
    assert car == raw.info["projs"][3]
