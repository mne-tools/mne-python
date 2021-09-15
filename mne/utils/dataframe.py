# -*- coding: utf-8 -*-
"""inst.to_data_frame() helper functions."""
# Authors: Daniel McCloy <dan@mccloy.info>
#
# License: BSD-3-Clause

import numpy as np

from ._logging import logger
from ..defaults import _handle_default


def _set_pandas_dtype(df, columns, dtype):
    """Try to set the right columns to dtype."""
    for column in columns:
        df[column] = df[column].astype(dtype)
        logger.info('Converting "%s" to "%s"...' % (column, dtype))


def _scale_dataframe_data(inst, data, picks, scalings):
    ch_types = inst.get_channel_types()
    ch_types_used = list()
    scalings = _handle_default('scalings', scalings)
    for tt in scalings.keys():
        if tt in ch_types:
            ch_types_used.append(tt)
    for tt in ch_types_used:
        scaling = scalings[tt]
        idx = [ii for ii in range(len(picks)) if ch_types[ii] == tt]
        if len(idx):
            data[:, idx] *= scaling
    return data


def _convert_times(inst, times, time_format):
    """Convert vector of time in seconds to ms, datetime, or timedelta."""
    # private function; pandas already checked in calling function
    from pandas import to_timedelta
    if time_format == 'ms':
        times = np.round(times * 1e3).astype(np.int64)
    elif time_format == 'timedelta':
        times = to_timedelta(times, unit='s')
    elif time_format == 'datetime':
        times = (to_timedelta(times + inst.first_time, unit='s') +
                 inst.info['meas_date'])
    return times


def _build_data_frame(inst, data, picks, long_format, mindex, index,
                      default_index, col_names=None, col_kind='channel'):
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
        df.set_index(default_index, inplace=True)
        df.columns.name = col_kind
    elif index is not None:
        df.set_index(index, inplace=True)
        if set(index) == set(default_index):
            df.columns.name = col_kind
    # long format
    if long_format:
        df = df.stack().reset_index()
        df.rename(columns={0: 'value'}, inplace=True)
        # add column for channel types (as appropriate)
        ch_map = (None if isinstance(inst, _BaseSourceEstimate) else
                  dict(zip(np.array(inst.ch_names)[picks],
                           np.array(inst.get_channel_types())[picks])))
        if ch_map is not None:
            col_index = len(df.columns) - 1
            ch_type = df['channel'].map(ch_map)
            df.insert(col_index, 'ch_type', ch_type)
        # restore index
        if index is not None:
            df.set_index(index, inplace=True)
        # convert channel/vertex/ch_type columns to factors
        to_factor = [c for c in df.columns.tolist()
                     if c not in ('time', 'value')]
        _set_pandas_dtype(df, to_factor, 'category')
    return df
