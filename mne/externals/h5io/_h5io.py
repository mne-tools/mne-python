# -*- coding: utf-8 -*-
# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import sys
import tempfile
from shutil import rmtree
from os import path as op

import numpy as np
try:
    from scipy import sparse
except ImportError:
    sparse = None

# Adapted from six
PY3 = sys.version_info[0] == 3
text_type = str if PY3 else unicode  # noqa
string_types = str if PY3 else basestring  # noqa

special_chars = {'{FWDSLASH}': '/'}


##############################################################################
# WRITING

def _check_h5py():
    """Helper to check if h5py is installed"""
    try:
        import h5py
    except ImportError:
        raise ImportError('the h5py module is required to use HDF5 I/O')
    return h5py


def _create_titled_group(root, key, title):
    """Helper to create a titled group in h5py"""
    out = root.create_group(key)
    out.attrs['TITLE'] = title
    return out


def _create_titled_dataset(root, key, title, data, comp_kw=None):
    """Helper to create a titled dataset in h5py"""
    comp_kw = {} if comp_kw is None else comp_kw
    out = root.create_dataset(key, data=data, **comp_kw)
    out.attrs['TITLE'] = title
    return out


def _create_pandas_dataset(fname, root, key, title, data):
    h5py = _check_h5py()
    rootpath = '/'.join([root, key])
    data.to_hdf(fname, rootpath)
    with h5py.File(fname, mode='a') as fid:
        fid[rootpath].attrs['TITLE'] = 'pd_dataframe'


def write_hdf5(fname, data, overwrite=False, compression=4,
               title='h5io', slash='error'):
    """Write python object to HDF5 format using h5py

    Parameters
    ----------
    fname : str
        Filename to use.
    data : object
        Object to write. Can be of any of these types:
            {ndarray, dict, list, tuple, int, float, str}
        Note that dict objects must only have ``str`` keys. It is recommended
        to use ndarrays where possible, as it is handled most efficiently.
    overwrite : True | False | 'update'
        If True, overwrite file (if it exists). If 'update', appends the title
        to the file (or replace value if title exists).
    compression : int
        Compression level to use (0-9) to compress data using gzip.
    title : str
        The top-level directory name to use. Typically it is useful to make
        this your package name, e.g. ``'mnepython'``.
    slash : 'error' | 'replace'
        Whether to replace forward-slashes ('/') in any key found nested within
        keys in data. This does not apply to the top level name (title).
        If 'error', '/' is not allowed in any lower-level keys.
    """
    h5py = _check_h5py()
    mode = 'w'
    if op.isfile(fname):
        if isinstance(overwrite, string_types):
            if overwrite != 'update':
                raise ValueError('overwrite must be "update" or a bool')
            mode = 'a'
        elif not overwrite:
            raise IOError('file "%s" exists, use overwrite=True to overwrite'
                          % fname)
    if not isinstance(title, string_types):
        raise ValueError('title must be a string')
    comp_kw = dict()
    if compression > 0:
        comp_kw = dict(compression='gzip', compression_opts=compression)
    with h5py.File(fname, mode=mode) as fid:
        if title in fid:
            del fid[title]
        cleanup_data = []
        _triage_write(title, data, fid, comp_kw, str(type(data)),
                      cleanup_data=cleanup_data, slash=slash, title=title)

    # Will not be empty if any extra data to be written
    for data in cleanup_data:
        # In case different extra I/O needs different inputs
        title = list(data.keys())[0]
        if title in ['pd_dataframe', 'pd_series']:
            rootname, key, value = data[title]
            _create_pandas_dataset(fname, rootname, key, title, value)


def _triage_write(key, value, root, comp_kw, where,
                  cleanup_data=[], slash='error', title=None):
    if key != title and '/' in key:
        if slash == 'error':
            raise ValueError('Found a key with "/", '
                             'this is not allowed if slash == error')
        elif slash == 'replace':
            # Auto-replace keys with proper values
            for key_spec, val_spec in special_chars.items():
                key = key.replace(val_spec, key_spec)
        else:
            raise ValueError("slash must be one of ['error', 'replace'")

    if isinstance(value, dict):
        sub_root = _create_titled_group(root, key, 'dict')
        for key, sub_value in value.items():
            if not isinstance(key, string_types):
                raise TypeError('All dict keys must be strings')
            _triage_write(
                'key_{0}'.format(key), sub_value, sub_root, comp_kw,
                where + '["%s"]' % key, cleanup_data=cleanup_data, slash=slash)
    elif isinstance(value, (list, tuple)):
        title = 'list' if isinstance(value, list) else 'tuple'
        sub_root = _create_titled_group(root, key, title)
        for vi, sub_value in enumerate(value):
            _triage_write(
                'idx_{0}'.format(vi), sub_value, sub_root, comp_kw,
                where + '[%s]' % vi, cleanup_data=cleanup_data, slash=slash)
    elif isinstance(value, type(None)):
        _create_titled_dataset(root, key, 'None', [False])
    elif isinstance(value, (int, float)):
        if isinstance(value, int):
            title = 'int'
        else:  # isinstance(value, float):
            title = 'float'
        _create_titled_dataset(root, key, title, np.atleast_1d(value))
    elif isinstance(value, np.bool_):
        _create_titled_dataset(root, key, 'np_bool_', np.atleast_1d(value))
    elif isinstance(value, string_types):
        if isinstance(value, text_type):  # unicode
            value = np.fromstring(value.encode('utf-8'), np.uint8)
            title = 'unicode'
        else:
            value = np.fromstring(value.encode('ASCII'), np.uint8)
            title = 'ascii'
        _create_titled_dataset(root, key, title, value, comp_kw)
    elif isinstance(value, np.ndarray):
        _create_titled_dataset(root, key, 'ndarray', value)
    elif sparse is not None and isinstance(value, sparse.csc_matrix):
        sub_root = _create_titled_group(root, key, 'csc_matrix')
        _triage_write('data', value.data, sub_root, comp_kw,
                      where + '.csc_matrix_data', cleanup_data=cleanup_data,
                      slash=slash)
        _triage_write('indices', value.indices, sub_root, comp_kw,
                      where + '.csc_matrix_indices', cleanup_data=cleanup_data,
                      slash=slash)
        _triage_write('indptr', value.indptr, sub_root, comp_kw,
                      where + '.csc_matrix_indptr', cleanup_data=cleanup_data,
                      slash=slash)
    elif sparse is not None and isinstance(value, sparse.csr_matrix):
        sub_root = _create_titled_group(root, key, 'csr_matrix')
        _triage_write('data', value.data, sub_root, comp_kw,
                      where + '.csr_matrix_data', cleanup_data=cleanup_data,
                      slash=slash)
        _triage_write('indices', value.indices, sub_root, comp_kw,
                      where + '.csr_matrix_indices', cleanup_data=cleanup_data,
                      slash=slash)
        _triage_write('indptr', value.indptr, sub_root, comp_kw,
                      where + '.csr_matrix_indptr', cleanup_data=cleanup_data,
                      slash=slash)
        _triage_write('shape', value.shape, sub_root, comp_kw,
                      where + '.csr_matrix_shape', cleanup_data=cleanup_data,
                      slash=slash)
    else:
        try:
            from pandas import DataFrame, Series
        except ImportError:
            pass
        else:
            if isinstance(value, (DataFrame, Series)):
                if isinstance(value, DataFrame):
                    title = 'pd_dataframe'
                else:
                    title = 'pd_series'
                rootname = root.name
                cleanup_data.append({title: (rootname, key, value)})
                return

        err_str = 'unsupported type %s (in %s)' % (type(value), where)
        raise TypeError(err_str)

##############################################################################
# READING


def read_hdf5(fname, title='h5io', slash='ignore'):
    """Read python object from HDF5 format using h5py

    Parameters
    ----------
    fname : str
        File to load.
    title : str
        The top-level directory name to use. Typically it is useful to make
        this your package name, e.g. ``'mnepython'``.
    slash : 'ignore' | 'replace'
        Whether to replace the string {FWDSLASH} with the value /. This does
        not apply to the top level name (title). If 'ignore', nothing will be
        replaced.

    Returns
    -------
    data : object
        The loaded data. Can be of any type supported by ``write_hdf5``.
    """
    h5py = _check_h5py()
    if not op.isfile(fname):
        raise IOError('file "%s" not found' % fname)
    if not isinstance(title, string_types):
        raise ValueError('title must be a string')
    with h5py.File(fname, mode='r') as fid:
        if title not in fid:
            raise ValueError('no "%s" data found' % title)
        if isinstance(fid[title], h5py.Group):
            if 'TITLE' not in fid[title].attrs:
                raise ValueError('no "%s" data found' % title)
        data = _triage_read(fid[title], slash=slash)
    return data


def _triage_read(node, slash='ignore'):
    if slash not in ['ignore', 'replace']:
        raise ValueError("slash must be one of 'replace', 'ignore'")
    h5py = _check_h5py()
    type_str = node.attrs['TITLE']
    if isinstance(type_str, bytes):
        type_str = type_str.decode()
    if isinstance(node, h5py.Group):
        if type_str == 'dict':
            data = dict()
            for key, subnode in node.items():
                if slash == 'replace':
                    for key_spec, val_spec in special_chars.items():
                        key = key.replace(key_spec, val_spec)
                data[key[4:]] = _triage_read(subnode, slash=slash)
        elif type_str in ['list', 'tuple']:
            data = list()
            ii = 0
            while True:
                subnode = node.get('idx_{0}'.format(ii), None)
                if subnode is None:
                    break
                data.append(_triage_read(subnode, slash=slash))
                ii += 1
            assert len(data) == ii
            data = tuple(data) if type_str == 'tuple' else data
            return data
        elif type_str == 'csc_matrix':
            if sparse is None:
                raise RuntimeError('scipy must be installed to read this data')
            data = sparse.csc_matrix((_triage_read(node['data'], slash=slash),
                                      _triage_read(node['indices'],
                                                   slash=slash),
                                      _triage_read(node['indptr'],
                                                   slash=slash)))
        elif type_str == 'csr_matrix':
            if sparse is None:
                raise RuntimeError('scipy must be installed to read this data')
            data = sparse.csr_matrix((_triage_read(node['data'], slash=slash),
                                      _triage_read(node['indices'],
                                                   slash=slash),
                                      _triage_read(node['indptr'],
                                                   slash=slash)),
                                     shape=_triage_read(node['shape']))
        elif type_str in ['pd_dataframe', 'pd_series']:
            from pandas import read_hdf
            rootname = node.name
            filename = node.file.filename
            data = read_hdf(filename, rootname, mode='r')
        else:
            raise NotImplementedError('Unknown group type: {0}'
                                      ''.format(type_str))
    elif type_str == 'ndarray':
        data = np.array(node)
    elif type_str in ('int', 'float'):
        cast = int if type_str == 'int' else float
        data = cast(np.array(node)[0])
    elif type_str == 'np_bool_':
        data = np.bool_(np.array(node)[0])
    elif type_str in ('unicode', 'ascii', 'str'):  # 'str' for backward compat
        decoder = 'utf-8' if type_str == 'unicode' else 'ASCII'
        cast = text_type if type_str == 'unicode' else str
        data = cast(np.array(node).tostring().decode(decoder))
    elif type_str == 'None':
        data = None
    else:
        raise TypeError('Unknown node type: {0}'.format(type_str))
    return data


# ############################################################################
# UTILITIES

def _sort_keys(x):
    """Sort and return keys of dict"""
    keys = list(x.keys())  # note: not thread-safe
    idx = np.argsort([str(k) for k in keys])
    keys = [keys[ii] for ii in idx]
    return keys


def object_diff(a, b, pre=''):
    """Compute all differences between two python variables

    Parameters
    ----------
    a : object
        Currently supported: dict, list, tuple, ndarray, int, str, bytes,
        float.
    b : object
        Must be same type as x1.
    pre : str
        String to prepend to each line.

    Returns
    -------
    diffs : str
        A string representation of the differences.
    """

    try:
        from pandas import DataFrame, Series
    except ImportError:
        DataFrame = Series = type(None)

    out = ''
    if type(a) != type(b):
        out += pre + ' type mismatch (%s, %s)\n' % (type(a), type(b))
    elif isinstance(a, dict):
        k1s = _sort_keys(a)
        k2s = _sort_keys(b)
        m1 = set(k2s) - set(k1s)
        if len(m1):
            out += pre + ' x1 missing keys %s\n' % (m1)
        for key in k1s:
            if key not in k2s:
                out += pre + ' x2 missing key %s\n' % key
            else:
                out += object_diff(a[key], b[key], pre + 'd1[%s]' % repr(key))
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            out += pre + ' length mismatch (%s, %s)\n' % (len(a), len(b))
        else:
            for xx1, xx2 in zip(a, b):
                out += object_diff(xx1, xx2, pre='')
    elif isinstance(a, (string_types, int, float, bytes)):
        if a != b:
            out += pre + ' value mismatch (%s, %s)\n' % (a, b)
    elif a is None:
        pass  # b must be None due to our type checking
    elif isinstance(a, np.ndarray):
        if not np.array_equal(a, b):
            out += pre + ' array mismatch\n'
    elif sparse is not None and sparse.isspmatrix(a):
        # sparsity and sparse type of b vs a already checked above by type()
        if b.shape != a.shape:
            out += pre + (' sparse matrix a and b shape mismatch'
                          '(%s vs %s)' % (a.shape, b.shape))
        else:
            c = a - b
            c.eliminate_zeros()
            if c.nnz > 0:
                out += pre + (' sparse matrix a and b differ on %s '
                              'elements' % c.nnz)
    elif isinstance(a, (DataFrame, Series)):
        if b.shape != a.shape:
            out += pre + (' pandas values a and b shape mismatch'
                          '(%s vs %s)' % (a.shape, b.shape))
        else:
            c = a.values - b.values
            nzeros = np.sum(c != 0)
            if nzeros > 0:
                out += pre + (' pandas values a and b differ on %s '
                              'elements' % nzeros)
    else:
        raise RuntimeError(pre + ': unsupported type %s (%s)' % (type(a), a))
    return out


class _TempDir(str):
    """Class for creating and auto-destroying temp dir

    This is designed to be used with testing modules. Instances should be
    defined inside test functions. Instances defined at module level can not
    guarantee proper destruction of the temporary directory.

    When used at module level, the current use of the __del__() method for
    cleanup can fail because the rmtree function may be cleaned up before this
    object (an alternative could be using the atexit module instead).
    """
    def __new__(self):
        new = str.__new__(self, tempfile.mkdtemp())
        return new

    def __init__(self):
        self._path = self.__str__()

    def __del__(self):
        rmtree(self._path, ignore_errors=True)
