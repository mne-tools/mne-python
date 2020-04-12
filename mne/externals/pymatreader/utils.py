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
import types
import sys
import numpy
import scipy.io

if sys.version_info <= (2, 7):
    chr = unichr  # noqa This is needed for python 2 and 3 compatibility


def _import_h5py():
    try:
        import h5py
    except Exception as exc:
        raise ImportError('h5py is required to read MATLAB files >= v7.3 '
                          '(%s)' % (exc,))
    return h5py


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
        return _handle_hdf5_group(hdf5_object, variable_names=variable_names,
                                  ignore_fields=ignore_fields)

    elif isinstance(hdf5_object, h5py.Dataset):
        return _handle_hdf5_dataset(hdf5_object)
    elif isinstance(hdf5_object, (list, types.GeneratorType)):
        return [_hdf5todict(item) for item in hdf5_object]

    raise TypeError('Unknown type in hdf5 file')


def _handle_hdf5_group(hdf5_object, variable_names=None, ignore_fields=None):
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


def _handle_hdf5_dataset(hdf5_object):
    if 'MATLAB_empty' in hdf5_object.attrs.keys():
        data = numpy.empty((0,))
    else:
        # this used to be just hdf5_object.value, but this is deprecated
        data = hdf5_object[()]

    if isinstance(data, numpy.ndarray) and \
            data.dtype == numpy.dtype('object'):

        data = [hdf5_object.file[cur_data] for cur_data in data.flatten()]
        if len(data) == 1 and hdf5_object.attrs['MATLAB_class'] == b'cell':
            data = data[0]
            data = data[()]
            return _assign_types(data)

        data = _hdf5todict(data)

    return _assign_types(data)


def _convert_string_hdf5(values):
    if values.size > 1:
        assigned_values = u''.join(chr(c) for c in values.flatten())
    else:
        assigned_values = chr(values)

    return assigned_values


def _assign_types(values):
    """private function, which assigns correct types to h5py extracted values
    from _browse_dataset()"""
    if type(values) == numpy.ndarray:
        assigned_values = _handle_ndarray(values)
    elif type(values) == numpy.float64:
        assigned_values = float(values)
    else:
        assigned_values = values
    return assigned_values


def _handle_ndarray(values):
    """Handle conversion of ndarrays."""
    values = numpy.squeeze(values).T
    if values.dtype in ("uint8", "uint16", "uint32"):
        values = _handle_hdf5_strings(values)

    if isinstance(values, numpy.ndarray) and \
            values.size == 1:

        values = values.item()

    return values


def _handle_hdf5_strings(values):
    if values.ndim in (0, 1):
        values = _convert_string_hdf5(values)
    elif values.ndim == 2:
        values = [_convert_string_hdf5(cur_val)
                  for cur_val in values]
    else:
        raise RuntimeError('String arrays with more than 2 dimensions'
                           'are not supported at the moment.')

    return values


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

    if isinstance(data, numpy.ndarray):
        data = _handle_scipy_ndarray(data)

    return data


def _handle_scipy_ndarray(data):
    if data.dtype == numpy.dtype('object') and not \
            isinstance(data, scipy.io.matlab.mio5.MatlabFunction):
        as_list = []
        for element in data:
            as_list.append(_check_for_scipy_mat_struct(element))
        data = as_list
    elif isinstance(data.dtype.names, tuple):
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
