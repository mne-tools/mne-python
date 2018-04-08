import os.path as op

import numpy as np
import sys
from collections import Mapping

from ..constants import Bunch
from ...utils import logger, warn


def _check_for_ascii_filename(eeg, input_fname):
    """Check to see if eeg.data is array of ascii values.

    The ascii values are assumed to represent a filename.
    This method does not check if extension is valid
    (or if it exists), since this is handled by _check_fname
    """
    if (isinstance(eeg.data, np.ndarray) and
       len(eeg.data.shape) == 1 and
       np.issubdtype(eeg.data.dtype, np.integer)):

        fname = ''.join([chr(x) for x in eeg.data])
        basedir = op.dirname(input_fname)
        data_fname = op.join(basedir, fname)
        if op.isfile(data_fname):
            return (True, fname)
        else:
            return (False, "")
    else:
        return (False, "")


def _bunchify(mapping, name='BU'):
    """Convert mappings to Bunches recursively.

    Based on https://gist.github.com/hangtwenty/5960435.
    """
    if isinstance(mapping, Mapping):
        for key, value in list(mapping.items()):
            mapping[key] = _bunchify(value)
        return _bunch_wrapper(name, **mapping)
    elif isinstance(mapping, list):
        return [_bunchify(item) for item in mapping]
    return mapping


def _bunch_wrapper(name, **kwargs):
    """Convert mappings to Bunches."""
    return Bunch(**kwargs)


def _bunch_data_2_strs(bunch_data, field, lower=True):
    """Convert ascii strings to human-readable strings.

    Takes string values stored as ascii values in numpy arrays
    in bunch objects and returns list of human-readable strings
    """
    str_list = [''.join([chr(x) for x in curr_label.__dict__[field]]).strip()
                for curr_label in bunch_data]
    return str_list


def _bunch_str_conversions(bunch_data, str_conversion_fields):
    """Convert selected fields.

    Fields are from bunch object from 1D array
    of ascii values of chars.
    """
    for curr_field in str_conversion_fields:
        c1 = (len(bunch_data) > 0)
        c2 = (c1 and (curr_field in bunch_data[0]))
        c3 = (c2 and
              not isinstance(bunch_data[0].__dict__[curr_field], str))
        if c3:
            str_data = _bunch_data_2_strs(bunch_data, curr_field)
            for ctr, curr_str in enumerate(str_data):
                bunch_data[ctr].__dict__[curr_field] = curr_str
    return bunch_data


def _bunch_derefs(orig, bunch_data, deref_fields):
    """Dereference h5py.h5r.Reference objects.

    Ensures that each field of bunch object with dereferenced
    objects stores them in a list, even if that list has only
    1 element.
    """
    import h5py

    for curr_field in deref_fields:
        bd = bunch_data[0].__dict__[curr_field]
        c1 = (len(bunch_data) > 0)
        c2 = (curr_field in bunch_data[0])
        c3 = (len(bd) > 0)
        c4 = (isinstance(bd[0], h5py.h5r.Reference))
        if (c1 and c2 and c3 and c4):
            for ctr in range(len(bunch_data)):
                bd = bunch_data[ctr].__dict__[curr_field]
                try:
                    # Ensure bunch_data[ctr].__dict__[curr_field] is iterable
                    # before attempting to iterate over it
                    iter(bd)
                except TypeError:
                    deref = [orig[bd].value.flatten()]
                else:
                    deref = [orig[x].value.flatten() for x in bd]

                bunch_data[ctr].__dict__[curr_field] = deref

    return bunch_data


def _get_hdf_eeg_data(input_fname):
    """Get data saved in hdf5 format.

    This is the entry method into the hdf-conversion methods,
    so it is the only one that needs the try/except block for
    import h5py.
    """
    try:
        import h5py
    except ImportError:
        raise RuntimeError('Reading v7+ MATLAB format .set',
                           'requires h5py, which could not',
                           'be imported')

    logger.info("Attempting to read Matlab style hdf file")
    with h5py.File(input_fname) as f:
        eeg_dict = hdf_2_dict(f, f['EEG'], parent=None)
    eeg = _bunchify(eeg_dict)
    ascii_check = _check_for_ascii_filename(eeg, input_fname)
    if ascii_check[0]:
        eeg.data = ascii_check[1]
    else:
        eeg.data = eeg.data.transpose()

    return eeg


def hdf_2_dict(orig, in_hdf, parent=None, indent=''):
    """Convert h5py obj to dict."""
    import h5py

    out_dict = {}
    variable_names = in_hdf.keys()
    indent_incr = '    '

    for curr in sorted(variable_names):
        if parent is None:
            curr_name = curr
        else:
            curr_name = '_'.join([parent, curr])

        msg = indent + "Converting " + curr_name
        if isinstance(in_hdf[curr], h5py.Dataset):
            suffix = " - Dataset"
            logger.debug(msg + suffix)
            temp = in_hdf[curr].value
            if 1 in temp.shape:
                temp = temp.flatten()

            if isinstance(temp[0], h5py.h5r.Reference):
                temp = np.array([orig[x].value.flatten()[0] for x in temp])

            if len(temp) == 1:
                temp = np.asscalar(temp[0])
                if isinstance(temp, float) and temp.is_integer():
                    temp = int(temp)

            out_dict[curr] = temp

        elif isinstance(in_hdf[curr], h5py.Group):
            suffix = " - Group"
            logger.debug(msg + suffix)

            if curr == 'chanlocs':
                temp = _hlGroup_2_bunch_list(orig, in_hdf[curr], curr,
                                             indent + indent_incr)
                # For some reason an empty chanloc field, which is stored as
                # [] <type 'numpy.ndarray'> in Matlab's original set file
                # becomes array([0, 0], dtype=uint64) when Matlab
                # stores as HDF5 (!?)
                # Since chanloc's values all appear to be scalars or strings,
                # each value of array[0,0] will be replaced by [].

                temp = [{curr_key: np.array([])
                         if np.array_equal(curr_dict[curr_key],
                                           np.array([0, 0], dtype=np.uint64))
                         else curr_dict[curr_key]
                         for curr_key in curr_dict}
                        for curr_dict in temp]

                # Rebunchify temp
                temp = [Bunch(**x) for x in temp]

                # TO DO add tests to know when to add
                # these (& other) string fields
                str_conversion_fields = ('type', 'labels')
                temp = _bunch_str_conversions(temp, str_conversion_fields)

            elif curr == 'event':
                temp = _hlGroup_2_bunch_list(orig, in_hdf[curr], curr,
                                             indent + indent_incr)

                # TO DO add tests to know when to add
                # these (& other) string fields
                str_conversion_fields = ('type', 'usertags')
                temp = _bunch_str_conversions(temp, str_conversion_fields)
                temp = np.asarray(temp)

            elif curr == 'epoch':
                temp = _hlGroup_2_bunch_list(orig, in_hdf[curr],
                                             curr_name, indent + indent_incr)

                deref_fields = ('eventtype', 'eventlatency', 'eventurevent',
                                'eventduration', 'eventvalue')
                temp = _bunch_derefs(orig, temp, deref_fields)

                for curr_elem in temp:
                    eventtype_str = [''.join([chr(x) for x in c_evt])
                                     for c_evt in curr_elem.eventtype]
                    curr_elem.eventtype = np.asarray(eventtype_str)

            else:
                temp = hdf_2_dict(orig, in_hdf[curr],
                                  curr_name, indent + indent_incr)
            out_dict[curr] = temp

        else:
            sys.exit("Unknown type")

    return out_dict


def _hlGroup_2_bunch_list(orig, in_hlGroup, tuple_name, indent):
    r"""Return list of Bunch objects.

    A Bunch object is a dictionary-like object that exposes its
    keys as attributes.
    ASSUMES: The group consists solely of arrays of HDF5 obj refs,
    and that these refs all reference 2D numpy arrays that need to
    be flattened to either 1D arrays or Scalars
    """
    import h5py

    try:
        # h5_values gives dict of 1D arrays of HDF obj references
        h5_values = {ct: in_hlGroup[ct].value.flatten() for ct in in_hlGroup}

        # derefs dereferences HDF obj references and cnverts arrays with
        # shapes = (1,) to scalars. Returns adict mapping keys to lists
        # of arrays and scalars
        derefs = {x: [orig[y].value.flatten()
                      if orig[y].value.flatten().shape != (1,)
                      else orig[y].value.flatten()[0]
                      for y in h5_values[x]]
                  if isinstance(h5_values[x], np.ndarray) and
                  isinstance(h5_values[x][0], h5py.Reference)
                  else h5_values[x]
                  for x in h5_values}

        for ct in in_hlGroup:
            msg = indent + "Converting " + tuple_name + '_' + ct
            logger.debug(msg)

    except IOError:
        derefs = {ct: [None] for ct in in_hlGroup}
        warn("Couldn't read", tuple_name, ". Assuming empty")

    sz = len(derefs[list(derefs.keys())[0]])
    bch_list = [Bunch(**{key: derefs[key][x] for key in derefs})
                for x in range(sz)]
    return bch_list
