"""inst.to_data_frame() helper functions."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from inspect import signature

import numpy as np

from ..defaults import _handle_default
from ._logging import logger, verbose
from .check import check_version


@verbose
def _set_pandas_dtype(df, columns, dtype, verbose=None):
    """Try to set the right columns to dtype."""
    for column in columns:
        df[column] = df[column].astype(dtype)
        logger.info(f'Converting "{column}" to "{dtype}"...')


def _scale_dataframe_data(inst, data, picks, scalings):
    ch_types = inst.get_channel_types()
    ch_types_used = list()
    scalings = _handle_default("scalings", scalings)
    for tt in scalings.keys():
        if tt in ch_types:
            ch_types_used.append(tt)
    for tt in ch_types_used:
        scaling = scalings[tt]
        idx = [ii for ii in range(len(picks)) if ch_types[ii] == tt]
        if len(idx):
            data[:, idx] *= scaling
    return data


def _convert_times(times, time_format, meas_date=None, first_time=0):
    """Convert vector of time in seconds to ms, datetime, or timedelta."""
    # private function; pandas already checked in calling function
    from pandas import to_timedelta

    if time_format == "ms":
        times = np.round(times * 1e3).astype(np.int64)
    elif time_format == "timedelta":
        times = to_timedelta(times, unit="s")
    elif time_format == "datetime":
        times = to_timedelta(times + first_time, unit="s") + meas_date
    return times


def _inplace(df, method, **kwargs):
    # Handle transition: inplace=True (pandas <1.5) → copy=False (>=1.5)
    # and 3.0 warning:
    # E   DeprecationWarning: The copy keyword is deprecated and will be removed in a
    #     future version. Copy-on-Write is active in pandas since 3.0 which utilizes a
    #     lazy copy mechanism that defers copies until necessary. Use .copy() to make
    #     an eager copy if necessary.
    _meth = getattr(df, method)  # used for set_index() and rename()

    if check_version("pandas", "3.0"):
        return _meth(**kwargs)
    elif "copy" in signature(_meth).parameters:
        return _meth(**kwargs, copy=False)
    else:
        _meth(**kwargs, inplace=True)
        return df


@verbose
def _build_data_frame(
    inst,
    data,
    picks,
    long_format,
    mindex,
    index,
    default_index,
    col_names=None,
    col_kind="channel",
    verbose=None,
):
    """Build DataFrame from MNE-object-derived data array."""
    # private function; pandas already checked in calling function
    from pandas import DataFrame

    from ..source_estimate import _BaseSourceEstimate

    # build DataFrame
    if col_names is None:
        col_names = [inst.ch_names[p] for p in picks]
    df = DataFrame(data, columns=col_names)
    for i, (k, v) in enumerate(mindex):
        df.insert(i, k, v)
    # build Index
    if long_format:
        df = _inplace(df, "set_index", keys=default_index)
        df.columns.name = col_kind
    elif index is not None:
        df = _inplace(df, "set_index", keys=index)
        if set(index) == set(default_index):
            df.columns.name = col_kind
    # long format
    if long_format:
        df = df.stack().reset_index()
        df = _inplace(df, "rename", columns={0: "value"})
        # add column for channel types (as appropriate)
        ch_map = (
            None
            if isinstance(inst, _BaseSourceEstimate)
            else dict(
                zip(
                    np.array(inst.ch_names)[picks],
                    np.array(inst.get_channel_types())[picks],
                )
            )
        )
        if ch_map is not None:
            col_index = len(df.columns) - 1
            ch_type = df["channel"].map(ch_map)
            df.insert(col_index, "ch_type", ch_type)
        # restore index
        if index is not None:
            df = _inplace(df, "set_index", keys=index)
        # convert channel/vertex/ch_type columns to factors
        to_factor = [
            c for c in df.columns.tolist() if c not in ("freq", "time", "value")
        ]
        _set_pandas_dtype(df, to_factor, "category")
    return df
