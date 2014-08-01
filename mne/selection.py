# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

from os import path

from .utils import logger, verbose
from .externals import six


@verbose
def read_selection(name, fname=None, verbose=None):
    """Read channel selection from file

    By default, the selections used in mne_browse_raw are supported*.
    Additional selections can be added by specifying a selection file (e.g.
    produced using mne_browse_raw) using the fname parameter.

    The name parameter can be a string or a list of string. The returned
    selection will be the combination of all selections in the file where
    (at least) one element in name is a substring of the selection name in
    the file. For example, "name = ['temporal', 'Right-frontal']" will produce
    a comination of "Left-temporal", "Right-temporal", and "Right-frontal".

    * The included selections are: "Vertex", "Left-temporal", "Right-temporal",
    "Left-parietal", "Right-parietal", "Left-occipital", "Right-occipital",
    "Left-frontal", and "Right-frontal"

    Parameters
    ----------
    name : string or list of string
        Name of the selection. If is a list, the selections are combined.
    fname : string
        Filename of the selection file (if None, built-in selections are used).
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    sel : list of string
        List with channel names in the selection.
    """

    # convert name to list of string
    if isinstance(name, tuple):
        name = list(name)

    if not isinstance(name, list):
        name = [name]

    # use built-in selections by default
    if fname is None:
        fname = path.join(path.dirname(__file__), 'data', 'mne_analyze.sel')

    if not path.exists(fname):
        raise ValueError('The file %s does not exist.' % fname)

    # use this to make sure we find at least one match for each name
    name_found = {}
    for n in name:
        name_found[n] = False

    fid = open(fname, 'r')
    sel = []

    for line in fid:
        line = line.strip()

        # skip blank lines and comments
        if len(line) == 0 or line[0] == '#':
            continue

        # get the name of the selection in the file
        pos = line.find(':')
        if pos < 0:
            logger.info('":" delimiter not found in selections file, '
                        'skipping line')
            continue

        sel_name_file = line[:pos]

        # search for substring match with name provided
        for n in name:
            if sel_name_file.find(n) >= 0:
                sel.extend(line[pos + 1:].split('|'))
                name_found[n] = True
                break

    fid.close()

    # make sure we found at least one match for each name
    for n, found in six.iteritems(name_found):
        if not found:
            raise ValueError('No match for selection name "%s" found' % n)

    # make the selection a sorted list with unique elements
    sel = list(set(sel))
    sel.sort()

    return sel
