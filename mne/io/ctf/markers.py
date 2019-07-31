import os
import itertools
import numpy as np
import os.path as op

from ...annotations import Annotations
from .res4 import _read_res4

class Markers:
    """Access to the markers of a CTF dataset. Each marker becomes a key
    that returns a list of (trial, time) pairs."""

    def __getitem__(self, key):
        return self.marks[key]

    #def __setitem__(self, key, value):
    #    self.marks[key] = value

    def get(self, key):
        return self.marks.get(key)

    def keys(self):
        return self.marks.keys()

    def __init__(self, dsname):
        self.marks = {}

        markerfilename = os.path.join(dsname, 'MarkerFile.mrk')
        print(markerfilename)
        try:
            f = open(markerfilename)
        except:
            return

        # Parse the file. Look at each line.

        START = 1
        MARK = 2
        NUM = 3
        LIST = 4
        state = START

        for l in f:
            s = l.split(':')
            if state == START:
                if s[0] == 'CLASSGROUPID':
                    state = MARK
            elif state == MARK:
                if s[0] == 'NAME':
                    name = next(f).split()[0]
                    state = NUM
            elif state == NUM:
                if s[0] == 'NUMBER OF SAMPLES':
                    num = int(next(f).split()[0])
                    state = LIST
            elif state == LIST:
                    next(f)
                    self._get_samples(f, name, num)
                    state = START
        f.close()

    def _get_samples(self, f, name, num):
        "Add all the samples for a marker to the marks dict."
        for x in range(num):
            l = next(f).split()
            trial = int(l[0])
            t = float(l[1])
            if not self.marks.get(name):
                self.marks[name] = []
            self.marks[name].append((trial, t))


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
    """Get required information from CTF res4 information file. """
    # we only need 3 values from res4. Maybe we can read them directly instead
    # of parsing the entire res4 file.
    res4 = _read_res4(directory)

    total_offset_duration = res4['pre_trig_pts'] / res4['sfreq']
    trial_duration = res4['nsamp'] / res4['sfreq']

    return total_offset_duration, trial_duration


def _read_annotations_ctf(directory):

    total_offset, trial_duration = _get_res4_info_needed_by_markers(directory)

    mm = _get_markers(op.join(directory, 'MarkerFile.mrk'))

    onset = []
    labels = []
    for current_marker_type in mm.keys():
        labels.append([current_marker_type] * len(mm[current_marker_type]))
        for trialnum, synctime in mm[current_marker_type]:
            onset.append(synctime + (trialnum * trial_duration) + total_offset)

    return Annotations(
        onset=onset,
        duration=np.zeros_like(onset),
        description=list(itertools.chain.from_iterable(labels)),
        orig_time=None
    )
