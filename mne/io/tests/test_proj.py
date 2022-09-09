from copy import deepcopy

from mne.datasets.testing import data_path, requires_testing_data
from mne.io import read_raw_fif

directory = data_path(download=False) / "MEG" / "sample"
fname = directory / "sample_audvis_trunc_raw.fif"


@requires_testing_data
def test_eq_ne():
    """Test == and != between projectors."""
    raw = read_raw_fif(fname, preload=False)

    pca1 = deepcopy(raw.info["projs"][0])
    pca2 = deepcopy(raw.info["projs"][1])
    car = deepcopy(raw.info["projs"][3])

    assert pca1 != pca2
    assert pca1 != car
    assert pca2 != car
    assert pca1 == raw.info["projs"][0]
    assert pca2 == raw.info["projs"][1]
    assert car == raw.info["projs"][3]
