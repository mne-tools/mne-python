import os
from os import path as op
from nose.tools import assert_true, assert_equal

from mne import datasets
from mne.externals.six import string_types
from mne.utils import _TempDir, run_tests_if_main, requires_good_network


def test_datasets():
    """Test simple dataset functions
    """
    for dname in ('sample', 'somato', 'spm_face', 'testing',
                  'bst_raw', 'bst_auditory', 'bst_resting',
                  'visual_92_categories'):
        if dname.startswith('bst'):
            dataset = getattr(datasets.brainstorm, dname)
        else:
            dataset = getattr(datasets, dname)
    if dataset.data_path(download=False) != '':
        assert_true(isinstance(dataset.get_version(), string_types))
    else:
        assert_true(dataset.get_version() is None)
    tempdir = _TempDir()
    # don't let it read from the config file to get the directory,
    # force it to look for the default
    os.environ['_MNE_FAKE_HOME_DIR'] = tempdir
    try:
        assert_equal(datasets.utils._get_path(None, 'foo', 'bar'),
                     op.join(tempdir, 'mne_data'))
    finally:
        del os.environ['_MNE_FAKE_HOME_DIR']


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
