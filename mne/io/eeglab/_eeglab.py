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

from ...utils import _import_pymatreader_funcs, warn


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


def _scipy_reader(file_name, variable_names=None, uint16_codec=None):
    """Load with scipy and then run the check function."""
    mat_data = loadmat(
        file_name, squeeze_me=True, mat_dtype=False, uint16_codec=uint16_codec
    )
    return _check_for_scipy_mat_struct(mat_data)


def _readmat(fname, uint16_codec=None, *, preload=False):
    try:
        read_mat = _import_pymatreader_funcs("EEGLAB I/O")
    except RuntimeError:  # pymatreader not installed
        read_mat = _scipy_reader

    # First handle the preload=False case
    if not preload:
        # when preload is `False`, we need to be selective about what we load
        # and handle the 'data' field specially

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

        # We first load only the info fields that are not data
        # Then we check if 'data' is present and load it separately if needed
        mat_data = read_mat(
            fname,
            variable_names=info_fields,
            uint16_codec=uint16_codec,
        )

        # checking the variables in the .set file
        # to decide how to handle 'data' variable
        variables = whosmat(str(fname))

        is_possible_not_loaded = False

        for var in variables:
            # looking for 'data' variable
            if var[0] == "data":
                numeric_types = """
                    int8 int16 int32
                    int64 uint8 uint16
                    uint32 uint64 single double
                """.split()

                is_possible_not_loaded = var[2] in numeric_types

                if is_possible_not_loaded:
                    # in preload=False mode and data is in .set file
                    mat_data["data"] = str(fname)

        if is_possible_not_loaded:
            return _check_for_scipy_mat_struct(mat_data)
        else:
            warn(
                "The 'data' variable in the .set file appears to be numeric. "
                "In preload=False mode, the data is not loaded into memory. "
                "Instead, the filename is provided in mat_data['data']. "
                "To load the actual data, set preload=True."
            )
            preload = True

    # here is intended to be if and not else if
    if preload:
        return read_mat(fname, uint16_codec=uint16_codec)
