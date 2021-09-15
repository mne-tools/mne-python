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


import scipy.io
import os

from scipy.io.matlab.miobase import get_matfile_version

from .utils import _import_h5py, _hdf5todict, _check_for_scipy_mat_struct

__all__ = 'read_mat'

"""
This is a small module intended to facilitate reading .mat files containing
large data structures into python, disregarding of the underlying .mat
file version.
"""


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
            mjv, _ = get_matfile_version(fid)
            extra_kwargs = {}
            if mjv == 1:
                extra_kwargs['uint16_codec'] = uint16_codec

            raw_data = scipy.io.loadmat(fid, struct_as_record=True,
                                        squeeze_me=True, mat_dtype=False,
                                        variable_names=variable_names,
                                        **extra_kwargs)
        data = _check_for_scipy_mat_struct(raw_data)
    except NotImplementedError:
        ignore_fields.append('#refs#')
        h5py = _import_h5py()
        with h5py.File(filename, 'r') as hdf5_file:
            data = _hdf5todict(hdf5_file, variable_names=variable_names,
                               ignore_fields=ignore_fields)
    return data
