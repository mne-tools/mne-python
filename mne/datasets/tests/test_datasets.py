from nose.tools import assert_true

from mne import datasets
from mne.externals.six import string_types


def test_datasets():
    """Test simple dataset functions
    """
    for dname in ('sample', 'somato', 'spm_face', 'testing'):
        dataset = getattr(datasets, dname)
    if dataset.data_path(download=False) != '':
        assert_true(isinstance(dataset.get_version(), string_types))
    else:
        assert_true(dataset.get_version() is None)
