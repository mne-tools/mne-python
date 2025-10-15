# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np

try:
    from scipy.io.matlab import MatlabFunction, MatlabOpaque
except ImportError:  # scipy < 1.8
    from scipy.io.matlab.mio5 import MatlabFunction
    from scipy.io.matlab.mio5_params import MatlabOpaque
from scipy.io import loadmat, whosmat

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
    if data.dtype == np.dtype("object") and not isinstance(data, MatlabFunction):
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
    if isinstance(data, dict):
        for key in data:
            data[key] = _check_for_scipy_mat_struct(data[key])

    if isinstance(data, MatlabOpaque):
        try:
            if data[0][2] == b"string":
                return None
        except IndexError:
            pass

    if isinstance(data, np.ndarray):
        data = _handle_scipy_ndarray(data)

    return data


def _readmat(fname, uint16_codec=None, preload=False):
    try:
        read_mat = _import_pymatreader_funcs("EEGLAB I/O")
    except RuntimeError:  # pymatreader not installed
        if preload:
            eeg = loadmat(
                fname, squeeze_me=True, mat_dtype=False, uint16_codec=uint16_codec
            )
        else:
            # the files in eeglab are always the same field names
            # the the fields were taken from the eeglab sample reference
            # available at the eeglab github:
            # https://github.com/sccn/eeglab/blob/develop/sample_data/eeglab_data.set
            # The sample reference is the big reference for the field names
            # in eeglab files, and what is used in the eeglab tests.
            info_fields = """
                setname filename filepath subject group condition session comments
                nbchan trials pnts srate xmin xmax times icaact icawinv icasphere
                icaweights icachansind chanlocs urchanlocs chaninfo ref event
                urevent eventdescription epoch epochdescription reject stats
                specdata specicaact splinefile icasplinefile dipfit history saved
                etc
            """.split()

            eeg = loadmat(
                fname,
                variable_names=info_fields,
                squeeze_me=True,
                mat_dtype=False,
                uint16_codec=uint16_codec,
            )
            variables = whosmat(str(fname))
            for var in variables:
                if var[0] == "data":
                    numeric_types = """
                        int8 int16 int32
                        int64 uint8 uint16
                        uint32 uint64 single double
                    """.split()
                    if var[2] in numeric_types:
                        # in preload=False mode and data is in .set file
                        eeg["data"] = str(fname)
                    else:
                        eeg["data"] = loadmat(
                            fname,
                            variable_names=["data"],
                            squeeze_me=True,
                            mat_dtype=False,
                            uint16_codec=uint16_codec,
                        )
                    break
        return _check_for_scipy_mat_struct(eeg)
    else:
        return read_mat(fname, uint16_codec=uint16_codec)
