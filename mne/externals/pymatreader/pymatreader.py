# -*- coding: UTF-8 -*-
# Copyright (c) 2018, Dirk Gütlin & Thomas Hartmann
# All rights reserved.
#
# This file is part of the pymatreader Project, see:
# https://gitlab.com/obob/pymatreader
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


import sys
import numpy
import scipy.io
import types
import os

if sys.version_info <= (2, 7):
    chr = unichr  # noqa This is needed for python 2 and 3 compatibility

__all__ = 'read_mat'

"""
This is a small module intended to facilitate reading .mat files containing
large data structures into python, disregarding of the underlying .mat
file version.
"""


def _import_h5py():
    try:
        import h5py
    except Exception as exc:
        raise ImportError('h5py is required to read MATLAB files >= v7.3 '
                          '(%s)' % (exc,))
    return h5py


def read_mat(filename, variable_names=None, ignore_fields=None,
             uint16_codec=None):
    """This function reads .mat files of version <7.3 or 7.3 and returns the
    contained data structure as a dictionary of nested substructure similar
    to scipy.io.loadmat style.

    Parameters
    ----------
    filename: str
        Path and filename of the .mat file containing the data.
    variable_names: list of strings, optional
        Reads only the data contained in the specified dict key or
        variable name. Default is None.
    ignore_fields: list of strings, optional
        Ignores every dict key/variable name specified in the list within the
        entire structure. Only works for .mat files v 7.3. Default is [].
    uint16_codec : str | None
        If your file contains non-ascii characters, sometimes reading
        it may fail and give rise to error message stating that "buffer is
        too small". ``uint16_codec`` allows to specify what codec (for example:
        'latin1' or 'utf-8') should be used when reading character arrays and
        can therefore help you solve this problem.

    Returns
    -------
    dict
        A structure of nested dictionaries, with variable names as keys and
        variable data as values.
    """

    if not os.path.exists(filename):
        raise IOError('The file %s does not exist.' % (filename,))

    if ignore_fields is None:
        ignore_fields = []
    try:
        with open(filename, 'rb') as fid:  # avoid open file warnings on error
            raw_data = scipy.io.loadmat(fid, struct_as_record=True,
                                        squeeze_me=True, mat_dtype=True,
                                        variable_names=variable_names,
                                        uint16_codec=uint16_codec)
        data = _check_for_scipy_mat_struct(raw_data)
    except NotImplementedError:
        ignore_fields.append('#refs#')
        h5py = _import_h5py()
        with h5py.File(filename, 'r') as hdf5_file:
            data = _hdf5todict(hdf5_file, variable_names=variable_names,
                               ignore_fields=ignore_fields)
    return data


def _hdf5todict(hdf5_object, variable_names=None, ignore_fields=None):
    """
    Recursively converts a hdf5 object to a python dictionary,
    converting all types as well.

    Parameters
    ----------
    hdf5_object: Union[h5py.Group, h5py.Dataset]
        Object to convert. Can be a h5py File, Group or Dataset
    variable_names: iterable, optional
        Tuple or list of variables to include. If set to none, all
        variable are read.
    ignore_fields: iterable, optional
        Tuple or list of fields to ignore. If set to none, all fields will
        be read.

    Returns
    -------
    dict
        Python dictionary
    """

    h5py = _import_h5py()

    if isinstance(hdf5_object, h5py.Group):
        all_keys = set(hdf5_object.keys())
        if ignore_fields:
            all_keys = all_keys - set(ignore_fields)

        if variable_names:
            all_keys = all_keys & set(variable_names)

        return_dict = dict()

        for key in all_keys:
            return_dict[key] = _hdf5todict(hdf5_object[key],
                                           variable_names=None,
                                           ignore_fields=ignore_fields)

        return return_dict
    elif isinstance(hdf5_object, h5py.Dataset):
        if 'MATLAB_empty' in hdf5_object.attrs.keys():
            data = numpy.empty((0,))
        else:
            data = hdf5_object.value

        if isinstance(data, numpy.ndarray) and \
                data.dtype == numpy.dtype('object'):

            data = [hdf5_object.file[cur_data] for cur_data in data.flatten()]
            data = _hdf5todict(data)
            if isinstance(data, numpy.ndarray):
                data = numpy.squeeze(data).T

        return _assign_types(data)
    elif isinstance(hdf5_object, (list, types.GeneratorType)):
        return [_hdf5todict(item) for item in hdf5_object]
    else:
        raise TypeError('Unknown type in hdf5 file')


def _convert_string_hdf5(values):
    if values.size == 2 and numpy.all(
            values == [0, 0]):  # this is most probably the empty string
        assigned_values = u''
    elif values.size > 1:
        assigned_values = u''.join(chr(c) for c in values.flatten())
    else:
        assigned_values = chr(values)

    return assigned_values


def _assign_types(values):
    """private function, which assigns correct types to h5py extracted values
    from _browse_dataset()"""

    if type(values) == numpy.ndarray:
        values = numpy.squeeze(values).T
        if values.dtype in ("uint8", "uint16", "uint32", "uint64"):
            if values.ndim in (0, 1):
                assigned_values = _convert_string_hdf5(values)
            elif values.ndim == 2:
                assigned_values = [_convert_string_hdf5(cur_val)
                                   for cur_val in values]
            else:
                raise RuntimeError('String arrays with more than 2 dimensions'
                                   'are not supported at the moment.')
        else:
            assigned_values = values

        if isinstance(assigned_values, numpy.ndarray) and \
                assigned_values.size == 1:

            assigned_values = assigned_values.item()

    elif type(values) == numpy.float64:
        assigned_values = float(values)
    else:
        assigned_values = values
    return assigned_values


def _check_for_scipy_mat_struct(data):
    """
    Private function to check all entries of data for occurrences of
    scipy.io.matlab.mio5_params.mat_struct and convert them.

    Parameters
    ==========
    data: any
        data to be checked

    Returns
    =========
    object
        checked and converted data
    """
    if isinstance(data, dict):
        for key in data:
            data[key] = _check_for_scipy_mat_struct(data[key])

    if isinstance(data, numpy.ndarray) and \
            data.dtype == numpy.dtype('object') and not \
            isinstance(data, scipy.io.matlab.mio5.MatlabFunction):

        as_list = []
        for element in data:
            as_list.append(_check_for_scipy_mat_struct(element))
        data = as_list

    if isinstance(data, numpy.ndarray) and isinstance(data.dtype.names, tuple):
        data = _todict_from_np_struct(data)
        data = _check_for_scipy_mat_struct(data)

    if isinstance(data, numpy.ndarray):
        data = numpy.array(data)

    return data


def _todict_from_np_struct(data):
    data_dict = dict()

    for cur_field_name in data.dtype.names:
        try:
            n_items = len(data[cur_field_name])
            cur_list = list()

            for idx in numpy.arange(n_items):
                cur_value = data[cur_field_name].item(idx)
                cur_value = _check_for_scipy_mat_struct(cur_value)
                cur_list.append(cur_value)

            data_dict[cur_field_name] = cur_list
        except TypeError:
            cur_value = data[cur_field_name].item(0)
            cur_value = _check_for_scipy_mat_struct(cur_value)
            data_dict[cur_field_name] = cur_value

    return data_dict
