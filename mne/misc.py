# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Scott Burns <sburns@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

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
    try:
        with open(fname, 'r') as f:
            ave_lines = f.readlines()
    except:
        print("Error while reading %s" % fname)

    reject_names = ['gradReject', 'magReject', 'eegReject', 'eogReject']
    reject_pynames = ['grad_reject', 'mag_reject', 'eeg_reject', 'eog_reject']
    reject_params = dict()
    for line in ave_lines:
        words = line.split()
        if words[0] in reject_names:
            reject_params[reject_pynames[reject_names.index(words[0])]] = \
                                                                float(words[1])

    cat_ind = [i for i, x in enumerate(ave_lines) if "category {" in x]
    event_dict = dict()
    for ind in cat_ind:
        for k in range(ind+1, ind+7):
            words = ave_lines[k].split()
            if len(words) >= 2:
                key = words[0]
                if key == 'event':
                    event = int(words[1])
                    break
        else:
            raise ValueError('Could not find event id.')
        event_dict[event] = dict(**reject_params)
        for k in range(ind+1, ind+7):
            words = ave_lines[k].split()
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

