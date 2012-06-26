# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
# License: BSD Style.


def _sample_version(path):
    """Get the version of the Sample dataset"""
    import os.path as op  # lazy import so it does not become visible outside
    ver_fname = op.join(path, 'version.txt')
    if op.exists(ver_fname):
        fid = open(ver_fname, 'r')
        version = fid.readline().strip()  # version is on first line
        fid.close()
    else:
        # Sample dataset versioning was introduced after 0.3
        version = '0.3'

    return version


def data_path(path='.', force_update=False):
    """Get path to local copy of Sample dataset

    Parameters
    ----------
    dir : string
        Location of where to look for the sample dataset.
        If not set. The data will be automatically downloaded in
        the local folder.
    force_update : bool
        Force update of the sample dataset even if a local copy exists.
    """
    # lazy import so things do not become visible outside
    import os.path as op
    from warnings import warn
    from distutils.version import LooseVersion

    from ... import __version__ as mne_version

    archive_name = "MNE-sample-data-processed.tar.gz"
    url = "ftp://surfer.nmr.mgh.harvard.edu/pub/data/" + archive_name
    folder_name = "MNE-sample-data"

    martinos_path = '/homes/6/gramfort/cluster/work/data/' + archive_name
    neurospin_path = '/neurospin/tmp/gramfort/' + archive_name

    if not op.exists(op.join(path, folder_name)) or force_update:
        if op.exists(martinos_path):
            archive_name = martinos_path
        elif op.exists(neurospin_path):
            archive_name = neurospin_path
        elif not op.exists(archive_name):
            import urllib
            print "Downloading data, please Wait (1.3 GB)..."
            print url
            opener = urllib.urlopen(url)
            open(archive_name, 'wb').write(opener.read())
            print

        import tarfile
        print "Decompressiong the archive: " + archive_name
        tarfile.open(archive_name, "r:gz").extractall(path=path)
        print

    path = op.join(path, folder_name)

    # compare the version of the Sample dataset and mne
    sample_version = _sample_version(path)
    if LooseVersion(sample_version) < LooseVersion(mne_version):
        warn('Sample dataset (version %s) is older than mne-python '
             '(version %s). If the examples fail, you may need to update '
             'the sample dataset by using force_update=True' % (sample_version,
             mne_version))

    return path

