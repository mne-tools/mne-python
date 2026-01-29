# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np

from ...epochs import EpochsArray
from ...evoked import EvokedArray
from ...utils import _check_fname, _import_pymatreader_funcs
from ..array._array import RawArray
from .utils import (
    _create_event_metadata,
    _create_events,
    _create_info,
    _set_tmin,
    _validate_ft_struct,
)


def read_raw_fieldtrip(fname, info, data_name="data") -> RawArray:
    """Load continuous (raw) data from a FieldTrip preprocessing structure.

    This function expects to find single trial raw data (FT_DATATYPE_RAW) in
    the structure data_name is pointing at.

    .. warning:: FieldTrip does not normally store the original information
                 concerning channel location, orientation, type etc. It is
                 therefore **highly recommended** to provide the info field.
                 This can be obtained by reading the original raw data file
                 with MNE functions (without preload). The returned object
                 contains the necessary info field.

    Parameters
    ----------
    fname : path-like
        Path and filename of the ``.mat`` file containing the data.
    info : dict or None
        The info dict of the raw data file corresponding to the data to import.
        If this is set to None, limited information is extracted from the
        FieldTrip structure.
    data_name : str
        Name of heading dict/variable name under which the data was originally
        saved in MATLAB.

    Returns
    -------
    raw : instance of RawArray
        A Raw Object containing the loaded data.
        See :class:`mne.io.Raw` for documentation of attributes and methods.

    See Also
    --------
    mne.io.Raw : Documentation of attributes and methods of RawArray.
    """
    read_mat = _import_pymatreader_funcs("FieldTrip I/O")
    fname = _check_fname(fname, overwrite="read", must_exist=True)

    ft_struct = read_mat(fname, ignore_fields=["previous"], variable_names=[data_name])

    # load data and set ft_struct to the heading dictionary
    ft_struct = ft_struct[data_name]

    _validate_ft_struct(ft_struct)

    info = _create_info(ft_struct, info)  # create info structure
    data = np.array(ft_struct["trial"])  # create the main data array

    if data.ndim > 2:
        data = np.squeeze(data)

    if data.ndim == 1:
        data = data[np.newaxis, ...]

    if data.ndim != 2:
        raise RuntimeError(
            "The data you are trying to load does not seem to be raw data"
        )

    raw = RawArray(data, info)  # create an MNE RawArray
    return raw


def read_epochs_fieldtrip(
    fname, info, data_name="data", trialinfo_column=0
) -> EpochsArray:
    """Load epoched data from a FieldTrip preprocessing structure.

    This function expects to find epoched data in the structure data_name is
    pointing at.

    .. warning:: Only epochs with the same amount of channels and samples are
                 supported!

    .. warning:: FieldTrip does not normally store the original information
                 concerning channel location, orientation, type etc. It is
                 therefore **highly recommended** to provide the info field.
                 This can be obtained by reading the original raw data file
                 with MNE functions (without preload). The returned object
                 contains the necessary info field.

    Parameters
    ----------
    fname : path-like
        Path and filename of the ``.mat`` file containing the data.
    info : dict or None
        The info dict of the raw data file corresponding to the data to import.
        If this is set to None, limited information is extracted from the
        FieldTrip structure.
    data_name : str
        Name of heading dict/ variable name under which the data was originally
        saved in MATLAB.
    trialinfo_column : int
        Column of the trialinfo matrix to use for the event codes.

    Returns
    -------
    epochs : instance of EpochsArray
        An EpochsArray containing the loaded data.
    """
    read_mat = _import_pymatreader_funcs("FieldTrip I/O")
    ft_struct = read_mat(fname, ignore_fields=["previous"], variable_names=[data_name])

    # load data and set ft_struct to the heading dictionary
    ft_struct = ft_struct[data_name]

    _validate_ft_struct(ft_struct)

    info = _create_info(ft_struct, info)  # create info structure
    data = np.array(ft_struct["trial"])  # create the epochs data array
    events = _create_events(ft_struct, trialinfo_column)
    if events is not None:
        metadata = _create_event_metadata(ft_struct)
    else:
        metadata = None
    tmin = _set_tmin(ft_struct)  # create start time

    epochs = EpochsArray(
        data=data, info=info, tmin=tmin, events=events, metadata=metadata, proj=False
    )
    return epochs


def read_evoked_fieldtrip(fname, info, comment=None, data_name="data"):
    """Load evoked data from a FieldTrip timelocked structure.

    This function expects to find timelocked data in the structure data_name is
    pointing at.

    .. warning:: FieldTrip does not normally store the original information
                 concerning channel location, orientation, type etc. It is
                 therefore **highly recommended** to provide the info field.
                 This can be obtained by reading the original raw data file
                 with MNE functions (without preload). The returned object
                 contains the necessary info field.

    Parameters
    ----------
    fname : path-like
        Path and filename of the ``.mat`` file containing the data.
    info : dict or None
        The info dict of the raw data file corresponding to the data to import.
        If this is set to None, limited information is extracted from the
        FieldTrip structure.
    comment : str
        Comment on dataset. Can be the condition.
    data_name : str
        Name of heading dict/ variable name under which the data was originally
        saved in MATLAB.

    Returns
    -------
    evoked : instance of EvokedArray
        An EvokedArray containing the loaded data.
    """
    read_mat = _import_pymatreader_funcs("FieldTrip I/O")
    ft_struct = read_mat(fname, ignore_fields=["previous"], variable_names=[data_name])
    ft_struct = ft_struct[data_name]

    _validate_ft_struct(ft_struct)

    info = _create_info(ft_struct, info)  # create info structure
    data_evoked = ft_struct["avg"]  # create evoked data

    evoked = EvokedArray(data_evoked, info, comment=comment)
    return evoked
