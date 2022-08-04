import numpy as np

from ...utils import _import_pymatreader_funcs


def _todict_from_np_struct(data):  # taken from pymatreader.utils
    data_dict = {}

    for cur_field_name in data.dtype.names:
        try:
            n_items = len(data[cur_field_name])
            cur_list = []

            for idx in np.arange(n_items):
                cur_value = data[cur_field_name].item(idx)
                cur_value = _check_for_scipy_mat_struct(cur_value)
                cur_list.append(cur_value)

            data_dict[cur_field_name] = cur_list
        except TypeError:
            cur_value = data[cur_field_name].item(0)
            cur_value = _check_for_scipy_mat_struct(cur_value)
            data_dict[cur_field_name] = cur_value

    return data_dict


def _handle_scipy_ndarray(data):  # taken from pymatreader.utils
    try:
        from scipy.io.matlab import MatlabFunction
    except ImportError:  # scipy < 1.8
        from scipy.io.matlab.mio5 import MatlabFunction

    if data.dtype == np.dtype('object') and not \
            isinstance(data, MatlabFunction):
        as_list = []
        for element in data:
            as_list.append(_check_for_scipy_mat_struct(element))
        data = as_list
    elif isinstance(data.dtype.names, tuple):
        data = _todict_from_np_struct(data)
        data = _check_for_scipy_mat_struct(data)

    if isinstance(data, np.ndarray):
        data = np.array(data)

    return data


def _check_for_scipy_mat_struct(data):  # taken from pymatreader.utils
    """Convert all scipy.io.matlab.mio5_params.mat_struct elements."""
    try:
        from scipy.io.matlab import MatlabOpaque
    except ImportError:  # scipy < 1.8
        from scipy.io.matlab.mio5_params import MatlabOpaque

    if isinstance(data, dict):
        for key in data:
            data[key] = _check_for_scipy_mat_struct(data[key])

    if isinstance(data, MatlabOpaque):
        try:
            if data[0][2] == b'string':
                return None
        except IndexError:
            pass

    if isinstance(data, np.ndarray):
        data = _handle_scipy_ndarray(data)

    return data


def _readmat(fname, uint16_codec=None):
    try:
        read_mat = _import_pymatreader_funcs('EEGLAB I/O')
    except RuntimeError:  # pymatreader not installed
        from scipy.io import loadmat
        eeg = loadmat(fname, squeeze_me=True, mat_dtype=False)
        return _check_for_scipy_mat_struct(eeg)
    else:
        return read_mat(fname, uint16_codec=uint16_codec)
