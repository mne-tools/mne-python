# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Denis Egnemann <d.engemann@fz-juelich.de>
# License: BSD Style.

from ..externals.six import string_types
import os
import os.path as op
import shutil
import tarfile
from warnings import warn

from .. import __version__ as mne_version
from ..utils import get_config, set_config, _fetch_file, logger


_doc = """Get path to local copy of {name} dataset

    Parameters
    ----------
    path : None | str
        Location of where to look for the {name} dataset.
        If None, the environment variable or config parameter
        {conf} is used. If it doesn't exist, the
        "mne-python/examples" directory is used. If the sample dataset
        is not found under the given path (e.g., as
        "mne-python/examples/MNE-{name}-data"), the data
        will be automatically downloaded to the specified folder.
    force_update : bool
        Force update of the sample dataset even if a local copy exists.
    update_path : bool | None
        If True, set the {conf} in mne-python
        config to the given path. If None, the user is prompted.
    download : bool
        If False and the {name} dataset has not been downloaded yet,
        it will not be downloaded and the path will be returned as
        '' (empty string). This is mostly used for debugging purposes
        and can be safely ignored by most users.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    path : str
        Path to {name} dataset directory.
"""


def _dataset_version(path, name):
    """Get the version of the dataset"""
    ver_fname = op.join(path, 'version.txt')
    if op.exists(ver_fname):
        fid = open(ver_fname, 'r')
        version = fid.readline().strip()  # version is on first line
        fid.close()
    else:
        # Sample dataset versioning was introduced after 0.3
        # SPM dataset was introduced with 0.7
        version = '0.3' if name == 'sample' else '0.7'

    return version


def _data_path(path=None, force_update=False, update_path=True,
               download=True, name=None, verbose=None):
    """Aux function
    """
    key = {'sample': 'MNE_DATASETS_SAMPLE_PATH',
           'spm': 'MNE_DATASETS_SPM_FACE_PATH'}[name]

    if path is None:
        # use an intelligent guess if it's not defined
        def_path = op.realpath(op.join(op.dirname(__file__),
                                       '..', '..', 'examples'))

        path = get_config(key, def_path)
        # use the same for all datasets
        if not os.path.exists(path):
            path = def_path

    if not isinstance(path, string_types):
        raise ValueError('path must be a string or None')

    if name == 'sample':
        archive_name = "MNE-sample-data-processed.tar.gz"
        url = "ftp://surfer.nmr.mgh.harvard.edu/pub/data/" + archive_name
        folder_name = "MNE-sample-data"
        folder_path = op.join(path, folder_name)
        rm_archive = False
    elif name == 'spm':
        archive_name = 'MNE-spm-face.tar.bz2'
        url = 'ftp://surfer.nmr.mgh.harvard.edu/pub/data/MNE/' + archive_name
        folder_name = "MNE-spm-face"
        folder_path = op.join(path, folder_name)
        rm_archive = False
    else:
        raise ValueError('Sorry, the dataset "%s" is not known.' % name)

    martinos_path = '/cluster/fusion/sample_data/' + archive_name
    neurospin_path = '/neurospin/tmp/gramfort/' + archive_name

    if not op.exists(folder_path) and not download:
        return ''

    if not op.exists(folder_path) or force_update:
        logger.info('Sample data archive %s not found at:\n%s\n'
                    'It will be downloaded and extracted at this location.'
                    % (archive_name, folder_path))

        if op.exists(martinos_path):
            archive_name = martinos_path
        elif op.exists(neurospin_path):
            archive_name = neurospin_path
        else:
            archive_name = op.join(path, archive_name)
            rm_archive = True
            if op.exists(archive_name):
                msg = ('Archive already exists at %r. Overwrite it '
                       '(y/[n])? ' % archive_name)
                answer = raw_input(msg)
                if answer.lower() == 'y':
                    os.remove(archive_name)
                else:
                    raise IOError('Archive file already exists at target '
                                  'location %r.' % archive_name)

            _fetch_file(url, archive_name, print_destination=False)

        if op.exists(folder_path):
            shutil.rmtree(folder_path)

        logger.info('Decompressiong the archive: ' + archive_name)
        logger.info('... please be patient, this can take some time')
        for ext in ['gz', 'bz2']:  # informed guess (and the only 2 options).
            try:
                tarfile.open(archive_name, 'r:%s' % ext).extractall(path=path)
            except tarfile.ReadError as err:
                logger.info('%s is %s trying "bz2"' % (archive_name, err))

        if rm_archive:
            os.remove(archive_name)

    path = op.abspath(path)
    if update_path is None:
        if get_config(key, '') != path:
            update_path = True
            msg = ('Do you want to set the path:\n    %s\nas the default '
                   'sample dataset path in the mne-python config [y]/n? '
                   % path)
            answer = raw_input(msg)
            if answer.lower() == 'n':
                update_path = False
        else:
            update_path = False

    if update_path is True:
        set_config(key, path)

    path = op.join(path, folder_name)

    # compare the version of the Sample dataset and mne
    data_version = _dataset_version(path, name)
    try:
        from distutils.version import LooseVersion as LV
    except:
        warn('Could not determine sample dataset version; dataset could\n'
             'be out of date. Please install the "distutils" package.')
    else:  # 0.7 < 0.7.git shoud be False, therefore strip
        if LV(data_version) < LV(mne_version.strip('.git')):
            warn('The {name} dataset (version {current}) is older than '
                 'the mne-python (version {newest}). If the examples fail, '
                 'you may need to update the {name} dataset by using'
                 'force_update=True'.format(name=name, current=data_version,
                                            newest=mne_version))

    return path


def has_dataset(name):
    """Helper for sample dataset presence"""
    endswith = {'sample': 'MNE-sample-data',
                'spm': 'MNE-spm-face'}[name]
    if _data_path(download=False, name=name).endswith(endswith):
        return True
    else:
        return False
        

def get_peak_evoked(data, times, tmin=None, tmax=None, use_abs=True):
    """Get feature-index and time of maximum signal from 2D array
    
    Note. This is a 'getter', not a 'finder'. For non-evoked type
    data and continuous signals, please use proper peak detection algorithms.
    
    Parameters
    ----------
    data : instance of numpy.ndarray (features, samples)
        The data, either evoked in sensor or source space
    times : instance of numpy.ndarray (samples)
        The times in seconds
    tmin : float | None
        The minimum point in time to be considered for peak getting.
    tmax : float | None
        The maximum point in time to be considered for peak getting.
    use_abs : bool
        Whether to consider absolute or signed data.
        
    Returns
    -------
    max_idx : int
        The index of the feature with the maximum value.
    latency : float
        The latency in seconds.
    """

    if tmin == None:
        tmin = times[0]
    if tmx == None:
        tmx = times[-1]
    
    if tmin < times.min():
        raise ValueError('The tmin value is out of bounds. It must be '
                         'within {0} and {1}'.format(times.min(), times.max()))
    if tmax < times.max():
        raise ValueError('The tmin value is out of bounds. It must be '
                         'within {0} and {1}'.format(times.min(), times.max()))
    if tmin >= tmax:
        raise ValueError('The tmin must be smaller than tma')
    
    maxs = enumerate(zip((np.abs(data) if take_abs else data).max(0), times))
    maxs = np.array(list(maxs))     
    time_win = (maxs[:, 2] >= tmin) & (maxs[:, 2] <= tmax)
    idx = maxs[:, 1, time_win].argmax()
    max_idx = maxs[idx][0]
    latency = maxs[idx][2]
    return int(max_idx)
    