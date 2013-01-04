# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
# License: BSD Style.

import os
import os.path as op
import shutil
from warnings import warn
from distutils.version import LooseVersion

import logging
logger = logging.getLogger('mne')

from ... import __version__ as mne_version
from ...utils import get_config


def _sample_version(path):
    """Get the version of the Sample dataset"""
    ver_fname = op.join(path, 'version.txt')
    if op.exists(ver_fname):
        fid = open(ver_fname, 'r')
        version = fid.readline().strip()  # version is on first line
        fid.close()
    else:
        # Sample dataset versioning was introduced after 0.3
        version = '0.3'

    return version


def data_path(path=None, force_update=False):
    """Get path to local copy of Sample dataset

    Parameters
    ----------
    path : None | str
        Location of where to look for the sample dataset.
        If None, the environment variable or config parameter
        MNE_DATASETS_PATH is used. If it doesn't exist, the current
        directory is used. If the path is not found, the data will be
        automatically downloaded to the specified folder.
    force_update : bool
        Force update of the sample dataset even if a local copy exists.
    """
    if path is None:
        # XXX use an intelligent guess
        def_path = '.'
        path = get_config('MNE_DATASETS_SAMPLE_PATH', def_path)

    if not isinstance(path, basestring):
        raise ValueError('path must be a string or None')

    archive_name = "MNE-sample-data-processed.tar.gz"
    url = "ftp://surfer.nmr.mgh.harvard.edu/pub/data/" + archive_name
    folder_name = "MNE-sample-data"
    folder_path = op.join(path, folder_name)
    rm_archive = False

    martinos_path = '/cluster/fusion/sample_data/' + archive_name
    neurospin_path = '/neurospin/tmp/gramfort/' + archive_name

    if not op.exists(folder_path) or force_update:
        # add confirmation that user wants to save data to this given path
        # XXX add

        # prompt user: by default, save given path to mne-python config file
        # if file is being downloaded
        # XXX add

        if op.exists(martinos_path):
            archive_name = martinos_path
        elif op.exists(neurospin_path):
            archive_name = neurospin_path
        else:
            archive_name = op.join(path, archive_name)
            rm_archive = True
            if op.exists(archive_name):
                msg = ("Archive already exists at %r. Overwrite it "
                       "(y/n)?" % archive_name)
                answer = raw_input(msg)
                if answer.lower() == 'y':
                    os.remove(archive_name)
                else:
                    err = ("Archive file already exists at target location "
                           "%r." % archive_name)
                    raise IOError(err)

            import urllib
            logger.info("Downloading data, please Wait (1.3 GB):")
            logger.info(url)
            opener = urllib.urlopen(url)
            open(archive_name, 'wb').write(opener.read())

        if op.exists(folder_path):
            shutil.rmtree(folder_path)

        import tarfile
        logger.info("Decompressiong the archive: " + archive_name)
        tarfile.open(archive_name, "r:gz").extractall(path=path)
        if rm_archive:
            os.remove(archive_name)

    path = op.join(path, folder_name)

    # compare the version of the Sample dataset and mne
    sample_version = _sample_version(path)
    if LooseVersion(sample_version) < LooseVersion(mne_version):
        warn('Sample dataset (version %s) is older than mne-python '
             '(version %s). If the examples fail, you may need to update '
             'the sample dataset by using force_update=True' % (sample_version,
             mne_version))

    return path
