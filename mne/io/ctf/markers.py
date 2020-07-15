# Author: Joan Massich <mailsik@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import os.path as op
from io import BytesIO

from ...annotations import Annotations
from .res4 import _read_res4
from .info import _convert_time


def _get_markers(fname):
    def consume(fid, predicate):  # just a consumer to move around conveniently
        while(predicate(fid.readline())):
            pass

    def parse_marker(string):  # XXX: there should be a nicer way to do that
        data = np.genfromtxt(
            BytesIO(string.encode()), dtype=[('trial', int), ('sync', float)])
        return int(data['trial']), float(data['sync'])

    markers = dict()
    with open(fname) as fid:
        consume(fid, lambda l: not l.startswith('NUMBER OF MARKERS:'))
        num_of_markers = int(fid.readline())

        for _ in range(num_of_markers):
            consume(fid, lambda l: not l.startswith('NAME:'))
            label = fid.readline().strip('\n')

            consume(fid, lambda l: not l.startswith('NUMBER OF SAMPLES:'))
            n_markers = int(fid.readline())

            consume(fid, lambda l: not l.startswith('LIST OF SAMPLES:'))
            next(fid)  # skip the samples header
            markers[label] = [
                parse_marker(next(fid)) for _ in range(n_markers)
            ]

    return markers


def _get_res4_info_needed_by_markers(directory):
    """Get required information from CTF res4 information file."""
    # we only need a few values from res4. Maybe we can read them directly
    # instead of parsing the entire res4 file.
    res4 = _read_res4(directory)

    total_offset_duration = res4['pre_trig_pts'] / res4['sfreq']
    trial_duration = res4['nsamp'] / res4['sfreq']

    meas_date = (_convert_time(res4['data_date'],
                               res4['data_time']), 0)
    return total_offset_duration, trial_duration, meas_date


def _read_annotations_ctf(directory):
    total_offset, trial_duration, meas_date \
        = _get_res4_info_needed_by_markers(directory)
    return _read_annotations_ctf_call(directory, total_offset, trial_duration,
                                      meas_date)


def _read_annotations_ctf_call(directory, total_offset, trial_duration,
                               meas_date):
    fname = op.join(directory, 'MarkerFile.mrk')
    if not op.exists(fname):
        return Annotations(list(), list(), list(), orig_time=meas_date)
    else:
        markers = _get_markers(fname)

        onset = [synctime + (trialnum * trial_duration) + total_offset
                 for _, m in markers.items() for (trialnum, synctime) in m]

        description = np.concatenate([
            np.repeat(label, len(m)) for label, m in markers.items()
        ])

        return Annotations(onset=onset, duration=np.zeros_like(onset),
                           description=description, orig_time=meas_date)
