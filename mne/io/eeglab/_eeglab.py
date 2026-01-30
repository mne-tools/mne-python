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

from ...utils import _import_pymatreader_funcs, _soft_import, warn


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
        file_name,
        squeeze_me=True,
        mat_dtype=False,
        variable_names=variable_names,
        uint16_codec=uint16_codec,
    )
    return _check_for_scipy_mat_struct(mat_data)


def _whosmat_hdf5(fname: str):
    """List variables in a MATLAB v7.3 (HDF5) .mat file without loading data.

    This function provides similar functionality to :func:`scipy.io.whosmat` but
    for MATLAB v7.3 files stored in HDF5 format, which are not supported by SciPy.

    Parameters
    ----------
    fname : str | PathLike
        Path to the MATLAB v7.3 (.mat) file.

    Returns
    -------
    variables : list of tuple
        A list of (name, shape, class) tuples for each variable in the file.
        The name is a string, shape is a tuple of ints, and class is a string
        indicating the MATLAB data type (e.g., 'double', 'int32', 'struct').

    Notes
    -----
    This function only works with MATLAB v7.3 (HDF5) files. For earlier versions,
    use :func:`scipy.io.whosmat` instead.

    See Also
    --------
    scipy.io.whosmat : List variables in classic MATLAB files.
    """
    h5py = _soft_import("h5py", purpose="MATLAB v7.3 I/O", strict=False)
    if h5py is False:
        raise ModuleNotFoundError(
            "h5py is required to inspect MATLAB v7.3 files preload=`False` "
            "Please install h5py to use this functionality."
        )

    variables = []

    with h5py.File(str(fname), "r") as f:
        for name in f.keys():
            node = f[name]

            # Extract shape from HDF5 object
            if isinstance(node, h5py.Dataset):
                shape = tuple(int(x) for x in node.shape)
            else:
                shape = ()
                for attr_key in (
                    "MATLAB_shape",
                    "MATLAB_Size",
                    "MATLAB_size",
                    "dims",
                    "MATLAB_dims",
                ):
                    shp = node.attrs.get(attr_key)
                    if shp is not None:
                        try:
                            shape = tuple(int(x) for x in shp)
                            break
                        except Exception:
                            pass
                if not shape and "size" in node:
                    try:
                        shape = tuple(int(x) for x in node["size"][()])
                    except Exception:
                        pass

            # Infer MATLAB class from HDF5 object
            mcls = node.attrs.get("MATLAB_class", "").lower()
            if mcls:
                matlab_class = "char" if mcls == "string" else mcls
            elif isinstance(node, h5py.Dataset):
                dt = node.dtype
                # Handle complex numbers stored as {real, imag} struct
                if getattr(dt, "names", None) and {"real", "imag"} <= set(dt.names):
                    matlab_class = (
                        "double" if dt["real"].base.itemsize == 8 else "single"
                    )
                # Map NumPy dtype to MATLAB class
                elif (kind := dt.kind) == "f":
                    matlab_class = "double" if dt.itemsize == 8 else "single"
                elif kind == "i":
                    matlab_class = f"int{8 * dt.itemsize}"
                elif kind == "u":
                    matlab_class = f"uint{8 * dt.itemsize}"
                elif kind == "b":
                    matlab_class = "logical"
                elif kind in ("S", "U", "O"):
                    matlab_class = "char"
                else:
                    matlab_class = "unknown"
            # Check for sparse matrix structure
            elif {"ir", "jc", "data"}.issubset(set(node.keys())):
                matlab_class = "sparse"
            else:
                matlab_class = "unknown"

            variables.append((name, shape, matlab_class))

    return variables


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
        # the fields were taken from the eeglab sample reference
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
        try:
            variables = whosmat(str(fname))
        except NotImplementedError:
            try:
                variables = _whosmat_hdf5(str(fname))
            except ModuleNotFoundError:
                warn(
                    "pymatreader is required to preload=`False` for "
                    "Matlab files v7.3 files with HDF5 support. "
                    "Setting preload=True."
                )
                preload = True
                return read_mat(fname, uint16_codec=uint16_codec)

        is_possible_not_loaded = False

        numeric_types = """
            int8 int16 int32
            int64 uint8 uint16
            uint32 uint64 single double
        """.split()

        for var in variables:
            # looking for 'data' variable
            if var[0] != "data":
                continue

            # checking if 'data' variable is numeric
            is_numeric = var[2] in numeric_types

            # if any 'data' variable is numeric, mark as possibly not loaded
            if is_numeric:
                # set the 'data' field to the filename
                mat_data["data"] = str(fname)

            is_possible_not_loaded = is_possible_not_loaded or is_numeric

        if is_possible_not_loaded:
            return mat_data
        else:
            # "The 'data' variable in the .set file appears to be numeric. "
            # "In preload=False mode, the data is not loaded into memory. "
            # "Instead, the filename is provided in mat_data['data']. "
            # "To load the actual data, set preload=True."
            # this is case of single file .set with data inside
            preload = True

    # here is intended to be if and not else if
    if preload:
        return read_mat(fname, uint16_codec=uint16_codec)
