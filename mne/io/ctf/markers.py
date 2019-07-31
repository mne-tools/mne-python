import os
import numpy as np

from ...annotations import Annotations

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


def _read_annotations_ctf(directory):

    from mne.io.ctf.markers import Markers
    from mne.io.ctf.res4 import _read_res4
    import itertools

    mm = Markers(directory)
    res4 = _read_res4(directory)

    latencies = []
    labels = []
    for current_marker_type in mm.keys():
        labels.append([current_marker_type] * len(mm[current_marker_type]))
        for trialnum, synctime in mm[current_marker_type]:
            latencies.append(
                res4['pre_trig_pts'] +
                (trialnum * res4['nsamp']) +
                round(synctime * res4['sfreq'])
            )

    unshuffling = np.argsort(latencies)

    latencies = np.array(latencies)[unshuffling]
    _labels = list(itertools.chain.from_iterable(labels))
    labels = [_labels[x] for x in unshuffling]

    return Annotations(
        onset=latencies / res4['sfreq'],
        duration=np.zeros_like(latencies),
        description=labels,
        orig_time=None
    )
