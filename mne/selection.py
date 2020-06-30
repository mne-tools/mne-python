# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Matti Hämäläinen <msh@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

from os import path
from collections import OrderedDict

import numpy as np

from .io.meas_info import Info
from .io.pick import _pick_data_channels, pick_types
from .utils import logger, verbose, _get_stim_channel

_SELECTIONS = ['Vertex', 'Left-temporal', 'Right-temporal', 'Left-parietal',
               'Right-parietal', 'Left-occipital', 'Right-occipital',
               'Left-frontal', 'Right-frontal']
_EEG_SELECTIONS = ['EEG 1-32', 'EEG 33-64', 'EEG 65-96', 'EEG 97-128']


@verbose
def read_selection(name, fname=None, info=None, verbose=None):
    """Read channel selection from file.

    By default, the selections used in ``mne_browse_raw`` are supported.
    Additional selections can be added by specifying a selection file (e.g.
    produced using ``mne_browse_raw``) using the ``fname`` parameter.

    The ``name`` parameter can be a string or a list of string. The returned
    selection will be the combination of all selections in the file where
    (at least) one element in name is a substring of the selection name in
    the file. For example, ``name=['temporal', 'Right-frontal']`` will produce
    a combination of ``'Left-temporal'``, ``'Right-temporal'``, and
    ``'Right-frontal'``.

    The included selections are:

        * ``'Vertex'``
        * ``'Left-temporal'``
        * ``'Right-temporal'``
        * ``'Left-parietal'``
        * ``'Right-parietal'``
        * ``'Left-occipital'``
        * ``'Right-occipital'``
        * ``'Left-frontal'``
        * ``'Right-frontal'``

    Parameters
    ----------
    name : str or list of str
        Name of the selection. If is a list, the selections are combined.
    fname : str
        Filename of the selection file (if None, built-in selections are used).
    info : instance of Info
        Measurement info file, which will be used to determine the spacing
        of channel names to return, e.g. ``'MEG 0111'`` for old Neuromag
        systems and ``'MEG0111'`` for new ones.
    %(verbose)s

    Returns
    -------
    sel : list of string
        List with channel names in the selection.
    """
    # convert name to list of string
    if not isinstance(name, (list, tuple)):
        name = [name]
    if isinstance(info, Info):
        picks = pick_types(info, meg=True, exclude=())
        if len(picks) > 0 and ' ' not in info['ch_names'][picks[0]]:
            spacing = 'new'
        else:
            spacing = 'old'
    elif info is not None:
        raise TypeError('info must be an instance of Info or None, not %s'
                        % (type(info),))
    else:  # info is None
        spacing = 'old'

    # use built-in selections by default
    if fname is None:
        fname = path.join(path.dirname(__file__), 'data', 'mne_analyze.sel')

    if not path.isfile(fname):
        raise ValueError('The file %s does not exist.' % fname)

    # use this to make sure we find at least one match for each name
    name_found = {n: False for n in name}
    with open(fname, 'r') as fid:
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

    # make sure we found at least one match for each name
    for n, found in name_found.items():
        if not found:
            raise ValueError('No match for selection name "%s" found' % n)

    # make the selection a sorted list with unique elements
    sel = list(set(sel))
    sel.sort()
    if spacing == 'new':  # "new" or "old" by now, "old" is default
        sel = [s.replace('MEG ', 'MEG') for s in sel]
    return sel


def _divide_to_regions(info, add_stim=True):
    """Divide channels to regions by positions."""
    from scipy.stats import zscore
    picks = _pick_data_channels(info, exclude=[])
    chs_in_lobe = len(picks) // 4
    pos = np.array([ch['loc'][:3] for ch in info['chs']])
    x, y, z = pos.T

    frontal = picks[np.argsort(y[picks])[-chs_in_lobe:]]
    picks = np.setdiff1d(picks, frontal)

    occipital = picks[np.argsort(y[picks])[:chs_in_lobe]]
    picks = np.setdiff1d(picks, occipital)

    temporal = picks[np.argsort(z[picks])[:chs_in_lobe]]
    picks = np.setdiff1d(picks, temporal)

    lt, rt = _divide_side(temporal, x)
    lf, rf = _divide_side(frontal, x)
    lo, ro = _divide_side(occipital, x)
    lp, rp = _divide_side(picks, x)  # Parietal lobe from the remaining picks.

    # Because of the way the sides are divided, there may be outliers in the
    # temporal lobes. Here we switch the sides for these outliers. For other
    # lobes it is not a big problem because of the vicinity of the lobes.
    with np.errstate(invalid='ignore'):  # invalid division, greater compare
        zs = np.abs(zscore(x[rt]))
        outliers = np.array(rt)[np.where(zs > 2.)[0]]
    rt = list(np.setdiff1d(rt, outliers))

    with np.errstate(invalid='ignore'):  # invalid division, greater compare
        zs = np.abs(zscore(x[lt]))
        outliers = np.append(outliers, (np.array(lt)[np.where(zs > 2.)[0]]))
    lt = list(np.setdiff1d(lt, outliers))

    l_mean = np.mean(x[lt])
    r_mean = np.mean(x[rt])
    for outlier in outliers:
        if abs(l_mean - x[outlier]) < abs(r_mean - x[outlier]):
            lt.append(outlier)
        else:
            rt.append(outlier)

    if add_stim:
        stim_ch = _get_stim_channel(None, info, raise_error=False)
        if len(stim_ch) > 0:
            for region in [lf, rf, lo, ro, lp, rp, lt, rt]:
                region.append(info['ch_names'].index(stim_ch[0]))
    return OrderedDict([('Left-frontal', lf), ('Right-frontal', rf),
                        ('Left-parietal', lp), ('Right-parietal', rp),
                        ('Left-occipital', lo), ('Right-occipital', ro),
                        ('Left-temporal', lt), ('Right-temporal', rt)])


def _divide_side(lobe, x):
    """Make a separation between left and right lobe evenly."""
    lobe = np.asarray(lobe)
    median = np.median(x[lobe])

    left = lobe[np.where(x[lobe] < median)[0]]
    right = lobe[np.where(x[lobe] > median)[0]]
    medians = np.where(x[lobe] == median)[0]

    left = np.sort(np.concatenate([left, lobe[medians[1::2]]]))
    right = np.sort(np.concatenate([right, lobe[medians[::2]]]))
    return list(left), list(right)
