from os import path as op
from nose.tools import assert_true, assert_equal, assert_raises

from mne import datasets
from mne.externals.six import string_types
from mne.utils import _TempDir, run_tests_if_main, requires_good_network


def test_datasets():
    """Test simple dataset functions
    """
    for dname in ('sample', 'somato', 'brainstorm', 'spm_face', 'testing'):
        dataset = getattr(datasets, dname)
    if dataset.data_path(download=False) != '':
        assert_true(isinstance(dataset.get_version(), string_types))
    else:
        assert_true(dataset.get_version() is None)
    # brainstorm tests
    assert_raises(ValueError, datasets.brainstorm.data_path, archive='foo')


@requires_good_network
def test_megsim():
    """Test MEGSIM URL handling
    """
    data_dir = _TempDir()
    paths = datasets.megsim.load_data(
        'index', 'text', 'text', path=data_dir, update_path=False)
    assert_equal(len(paths), 1)
    assert_true(paths[0].endswith('index.html'))


@requires_good_network
def test_downloads():
    """Test dataset URL handling
    """
    # Try actually downloading a dataset
    data_dir = _TempDir()
    path = datasets._fake.data_path(path=data_dir, update_path=False)
    assert_true(op.isfile(op.join(path, 'bar')))
    assert_true(datasets._fake.get_version() is None)


run_tests_if_main()
