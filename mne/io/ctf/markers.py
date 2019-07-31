import numpy as np
import os.path as op

from itertools.chain import from_iterable as _list_unpack

from ...annotations import Annotations
from .res4 import _read_res4


def _get_markers(fname):
    from io import StringIO

    def consume(fid, predicate):  # just a consumer to move around conveniently
        while(predicate(fid.readline())):
            pass

    def parse_marker(string):  # XXX: there should be a nicer way to do that
        data = np.genfromtxt(StringIO(string),
                             dtype=[('trial', int), ('sync', float)])
        return int(data['trial']), float(data['sync'])

    events = dict()
    with open(fname) as fid:
        consume(fid, lambda l: not l.startswith('NUMBER OF MARKERS:'))
        num_of_markers = int(fid.readline())

        for _ in range(num_of_markers):
            consume(fid, lambda l: not l.startswith('NAME:'))
            label = fid.readline().strip('\n')

            consume(fid, lambda l: not l.startswith('NUMBER OF SAMPLES:'))
            n_events = int(fid.readline())

            consume(fid, lambda l: not l.startswith('LIST OF SAMPLES:'))
            next(fid)  # skip the samples header
            events[label] = [parse_marker(next(fid)) for _ in range(n_events)]

    return events


def _get_res4_info_needed_by_markers(directory):
    """Get required information from CTF res4 information file."""
    # we only need 3 values from res4. Maybe we can read them directly instead
    # of parsing the entire res4 file.
    res4 = _read_res4(directory)

    total_offset_duration = res4['pre_trig_pts'] / res4['sfreq']
    trial_duration = res4['nsamp'] / res4['sfreq']

    return total_offset_duration, trial_duration


def _read_annotations_ctf(directory):
    total_offset, trial_duration = _get_res4_info_needed_by_markers(directory)
    markers = _get_markers(op.join(directory, 'MarkerFile.mrk'))

    onset = []
    for label in markers.keys():
        for trialnum, synctime in markers[label]:
            onset.append(synctime + (trialnum * trial_duration) + total_offset)

    description = list(
        _list_unpack([[label] * len(v) for label, v in markers])
    )

    return Annotations(onset=onset, duration=np.zeros_like(onset),
                       description=description, orig_time=None)
