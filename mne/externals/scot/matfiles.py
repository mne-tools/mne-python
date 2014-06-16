"""
Summary
-------
Routines for loading and saving Matlab's .mat files.
"""

from scipy.io import loadmat as sploadmat
from scipy.io import savemat as spsavemat
from scipy.io import matlab


def loadmat(filename):
    """This function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """
    data = sploadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


savemat = spsavemat


def _check_keys(dictionary):
    """
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    """
    for key in dictionary:
        if isinstance(dictionary[key], matlab.mio5_params.mat_struct):
            dictionary[key] = _todict(dictionary[key])
    return dictionary


def _todict(matobj):
    """
    a recursive function which constructs from matobjects nested dictionaries
    """
    dictionary = {}
    #noinspection PyProtectedMember
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, matlab.mio5_params.mat_struct):
            dictionary[strg] = _todict(elem)
        else:
            dictionary[strg] = elem
    return dictionary

