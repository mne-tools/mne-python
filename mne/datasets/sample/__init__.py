# Author: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
# License: BSD Style.

import os
import os.path as op

def data_path(path='.'):
    """Get path to local copy of Sample dataset

    Parameters
    ----------
    dir : string
        Location of where to look for the sample dataset.
        If not set. The data will be automatically downloaded in
        the local folder.
    """
    archive_name = "MNE-sample-data-processed.tar.gz"
    url = "ftp://surfer.nmr.mgh.harvard.edu/pub/data/" + archive_name
    folder_name = "MNE-sample-data"

    martinos_path = '/homes/6/gramfort/cluster/work/data/MNE-sample-data-processed.tar.gz'

    if not os.path.exists(op.join(path, folder_name)):
        if os.path.exists(martinos_path):
            archive_name = martinos_path
        elif not os.path.exists(archive_name):
            import urllib
            print "Downloading data, please Wait (600 MB)..."
            print url
            opener = urllib.urlopen(url)
            open(archive_name, 'wb').write(opener.read())
            print

        import tarfile
        print "Decompressiong the archive: " + archive_name
        tarfile.open(archive_name, "r:gz").extractall(path=path)
        print

    path = op.join(path, folder_name)
    return path
