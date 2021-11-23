# -*- coding: utf-8 -*-
# Authors: MNE Developers
#
# License: BSD-3-Clause

import os
import shutil
import datetime
import os.path as op

import numpy as np

from ..io.egi.egimff import _import_mffpy
from ..io.pick import pick_types, pick_channels
from ..utils import verbose, warn, _check_fname


@verbose
def export_evokeds_mff(fname, evoked, history=None, *, overwrite=False,
                       verbose=None):
    """Export evoked dataset to MFF.

    %(export_warning)s

    Parameters
    ----------
    %(export_params_fname)s
    evoked : list of Evoked instances
        List of evoked datasets to export to one file. Note that the
        measurement info from the first evoked instance is used, so be sure
        that information matches.
    history : None (default) | list of dict
        Optional list of history entries (dictionaries) to be written to
        history.xml. This must adhere to the format described in
        mffpy.xml_files.History.content. If None, no history.xml will be
        written.
    %(overwrite)s

        .. versionadded:: 0.24.1
    %(verbose)s

    Notes
    -----
    .. versionadded:: 0.24

    %(export_warning_note_evoked)s

    Only EEG channels are written to the output file.
    ``info['device_info']['type']`` must be a valid MFF recording device
    (e.g. 'HydroCel GSN 256 1.0'). This field is automatically populated when
    using MFF read functions.
    """
    mffpy = _import_mffpy('Export evokeds to MFF.')
    import pytz
    info = evoked[0].info
    if np.round(info['sfreq']) != info['sfreq']:
        raise ValueError('Sampling frequency must be a whole number. '
                         f'sfreq: {info["sfreq"]}')
    sampling_rate = int(info['sfreq'])

    # check for unapplied projectors
    if any(not proj['active'] for proj in evoked[0].info['projs']):
        warn('Evoked instance has unapplied projectors. Consider applying '
             'them before exporting with evoked.apply_proj().')

    # Initialize writer
    # Future changes: conditions based on version or mffpy requirement if
    # https://github.com/BEL-Public/mffpy/pull/92 is merged and released.
    fname = _check_fname(fname, overwrite=overwrite)
    if op.exists(fname):
        os.remove(fname) if op.isfile(fname) else shutil.rmtree(fname)
    writer = mffpy.Writer(fname)
    current_time = pytz.utc.localize(datetime.datetime.utcnow())
    writer.addxml('fileInfo', recordTime=current_time)
    try:
        device = info['device_info']['type']
    except (TypeError, KeyError):
        raise ValueError('No device type. Cannot determine sensor layout.')
    writer.add_coordinates_and_sensor_layout(device)

    # Add EEG data
    eeg_channels = pick_types(info, eeg=True, exclude=[])
    eeg_bin = mffpy.bin_writer.BinWriter(sampling_rate)
    for ave in evoked:
        # Signals are converted to µV
        block = (ave.data[eeg_channels] * 1e6).astype(np.float32)
        eeg_bin.add_block(block, offset_us=0)
    writer.addbin(eeg_bin)

    # Add categories
    categories_content = _categories_content_from_evokeds(evoked)
    writer.addxml('categories', categories=categories_content)

    # Add history
    if history:
        writer.addxml('historyEntries', entries=history)

    writer.write()


def _categories_content_from_evokeds(evoked):
    """Return categories.xml content for evoked dataset."""
    content = dict()
    begin_time = 0
    for ave in evoked:
        # Times are converted to microseconds
        sfreq = ave.info['sfreq']
        duration = np.round(len(ave.times) / sfreq * 1e6).astype(int)
        end_time = begin_time + duration
        event_time = begin_time - np.round(ave.tmin * 1e6).astype(int)
        eeg_bads = _get_bad_eeg_channels(ave.info)
        content[ave.comment] = [
            _build_segment_content(begin_time, end_time, event_time, eeg_bads,
                                   name='Average', nsegs=ave.nave)
        ]
        begin_time += duration
    return content


def _get_bad_eeg_channels(info):
    """Return a list of bad EEG channels formatted for categories.xml.

    Given a list of only the EEG channels in file, return the indices of this
    list (starting at 1) that correspond to bad channels.
    """
    if len(info['bads']) == 0:
        return []
    eeg_channels = pick_types(info, eeg=True, exclude=[])
    bad_channels = pick_channels(info['ch_names'], info['bads'])
    bads_elementwise = np.isin(eeg_channels, bad_channels)
    return list(np.flatnonzero(bads_elementwise) + 1)


def _build_segment_content(begin_time, end_time, event_time, eeg_bads,
                           status='unedited', name=None, pns_bads=None,
                           nsegs=None):
    """Build content for a single segment in categories.xml.

    Segments are sorted into categories in categories.xml. In a segmented MFF
    each category can contain multiple segments, but in an averaged MFF each
    category only contains one segment (the average).
    """
    channel_status = [{
        'signalBin': 1,
        'exclusion': 'badChannels',
        'channels': eeg_bads
    }]
    if pns_bads:
        channel_status.append({
            'signalBin': 2,
            'exclusion': 'badChannels',
            'channels': pns_bads
        })
    content = {
        'status': status,
        'beginTime': begin_time,
        'endTime': end_time,
        'evtBegin': event_time,
        'evtEnd': event_time,
        'channelStatus': channel_status,
    }
    if name:
        content['name'] = name
    if nsegs:
        content['keys'] = {
            '#seg': {
                'type': 'long',
                'data': nsegs
            }
        }
    return content
