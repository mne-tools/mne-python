# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Scott Burns <sburns@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)


def parse_config(fname):
    """Parse a config file (like .ave and .cov files).

    Parameters
    ----------
    fname : str
        Config file name.

    Returns
    -------
    conditions : list of dict
        Each condition is indexed by the event type.
        A condition contains as keys::

            tmin, tmax, name, grad_reject, mag_reject,
            eeg_reject, eog_reject
    """
    reject_params = read_reject_parameters(fname)

    with open(fname, 'r') as f:
        lines = f.readlines()

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
    """Read rejection parameters from .cov or .ave config file.

    Parameters
    ----------
    fname : str
        Filename to read.

    Returns
    -------
    params : dict
        The rejection parameters.
    """
    with open(fname, 'r') as f:
        lines = f.readlines()

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
    """Read flat channel rejection parameters from .cov or .ave config file."""
    with open(fname, 'r') as f:
        lines = f.readlines()

    reject_names = ['gradFlat', 'magFlat', 'eegFlat', 'eogFlat', 'ecgFlat']
    reject_pynames = ['grad', 'mag', 'eeg', 'eog', 'ecg']
    flat = dict()
    for line in lines:
        words = line.split()
        if words[0] in reject_names:
            flat[reject_pynames[reject_names.index(words[0])]] = \
                float(words[1])

    return flat
