# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Scott Burns <sburns@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import numpy as np
from scipy.optimize import fmin_powell

from .fiff.constants import FIFF


def fit_sphere_to_headshape(info):

    # get head digization points, excluding some frontal points (nose etc.)
    hsp = [p['r'] for p in info['dig'] if p['kind'] == FIFF.FIFFV_POINT_EXTRA
           and not (p['r'][2] < 0 and p['r'][1] > 0)]

    if len(hsp) == 0:
        raise ValueError('No head digitization points found')

    hsp = 1e3 * np.array(hsp)

    # initial guess for center and radius
    xradius = (np.max(hsp[:, 0]) - np.min(hsp[:, 0])) / 2
    yradius = (np.max(hsp[:, 1]) - np.min(hsp[:, 1])) / 2

    radius_init = (xradius + yradius) / 2
    center_init = np.array([0.0, 0.0, np.max(hsp[:, 2]) - radius_init])

    # optimization
    x0 = np.r_[center_init, radius_init]
    cost_fun = lambda x, hsp:\
        np.sum((np.sqrt(np.sum((hsp - x[:3]) ** 2, axis=1)) - x[3]) ** 2)

    x_opt = fmin_powell(cost_fun, x0, args=(hsp,))

    center = x_opt[:3]
    radius = x_opt[3]

    print ('Fitted sphere: r = %0.1f mm, center = %0.1f %0.1f %0.1f mm'
           % (radius, center[0], center[1], center[2]))

    return center, radius


def parse_config(fname):
    """Parse a config file (like .ave and .cov files)

    Parameters
    ----------
    fname : string
        config file name

    Returns
    -------
    conditions : list of dict
        Each condition is indexed by the event type.
        A condition contains as keys:
            tmin, tmax, name, grad_reject, mag_reject,
            eeg_reject, eog_reject
    """
    reject_params = read_reject_parameters(fname)

    try:
        with open(fname, 'r') as f:
            lines = f.readlines()
    except:
        raise ValueError("Error while reading %s" % fname)

    cat_ind = [i for i, x in enumerate(lines) if "category {" in x]
    event_dict = dict()
    for ind in cat_ind:
        for k in range(ind + 1, ind + 7):
            words = lines[k].split()
            if len(words) >= 2:
                key = words[0]
                if key == 'event':
                    event = int(words[1])
                    break
        else:
            raise ValueError('Could not find event id.')
        event_dict[event] = dict(**reject_params)
        for k in range(ind + 1, ind + 7):
            words = lines[k].split()
            if len(words) >= 2:
                key = words[0]
                if key == 'name':
                    name = ' '.join(words[1:])
                    if name[0] == '"':
                        name = name[1:]
                    if name[-1] == '"':
                        name = name[:-1]
                    event_dict[event]['name'] = name
                if key in ['tmin', 'tmax', 'basemin', 'basemax']:
                    event_dict[event][key] = float(words[1])
    return event_dict


def read_reject_parameters(fname):
    """Read rejection parameters from .cov or .ave config file"""

    try:
        with open(fname, 'r') as f:
            lines = f.readlines()
    except:
        raise ValueError("Error while reading %s" % fname)

    reject_names = ['gradReject', 'magReject', 'eegReject', 'eogReject',
                    'ecgReject']
    reject_pynames = ['grad', 'mag', 'eeg', 'eog', 'ecg']
    reject = dict()
    for line in lines:
        words = line.split()
        if words[0] in reject_names:
            reject[reject_pynames[reject_names.index(words[0])]] = \
                                                                float(words[1])

    return reject


def read_flat_parameters(fname):
    """Read flat channel rejection parameters from .cov or .ave config file"""

    try:
        with open(fname, 'r') as f:
            lines = f.readlines()
    except:
        raise ValueError("Error while reading %s" % fname)

    reject_names = ['gradFlat', 'magFlat', 'eegFlat', 'eogFlat', 'ecgFlat']
    reject_pynames = ['grad', 'mag', 'eeg', 'eog', 'ecg']
    flat = dict()
    for line in lines:
        words = line.split()
        if words[0] in reject_names:
            flat[reject_pynames[reject_names.index(words[0])]] = \
                                                                float(words[1])

    return flat
