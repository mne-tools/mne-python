import os

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


def read_annotations(directory):
    import numpy as np
    from mne import Annotations

    EXPECTED_LATENCIES = np.array([
         5640,   7950,   9990,  12253,  14171,  16557,  18896,  20846,  # noqa
        22702,  24990,  26830,  28974,  30906,  33077,  34985,  36907,  # noqa
        38922,  40760,  42881,  45222,  47457,  49618,  51802,  54227,  # noqa
        56171,  58274,  60394,  62375,  64444,  66767,  68827,  71109,  # noqa
        73499,  75807,  78146,  80415,  82554,  84508,  86403,  88426,  # noqa
        90746,  92893,  94779,  96822,  98996,  99001, 100949, 103325,  # noqa
       105322, 107678, 109667, 111844, 113682, 115817, 117691, 119663,  # noqa
       121966, 123831, 126110, 128490, 130521, 132808, 135204, 137210,  # noqa
       139130, 141390, 143660, 145748, 147889, 150205, 152528, 154646,  # noqa
       156897, 159191, 161446, 163722, 166077, 168467, 170624, 172519,  # noqa
       174719, 176886, 179062, 181405, 183709, 186034, 188454, 190330,  # noqa
       192660, 194682, 196834, 199161, 201035, 203008, 204999, 207409,  # noqa
       209661, 211895, 213957, 216005, 218040, 220178, 222137, 224305,  # noqa
       226297, 228654, 230755, 232909, 235205, 237373, 239723, 241762,  # noqa
       243748, 245762, 247801, 250055, 251886, 254252, 256441, 258354,  # noqa
       260680, 263026, 265048, 267073, 269235, 271556, 273927, 276197,  # noqa
       278436, 280536, 282691, 284933, 287061, 288936, 290941, 293183,  # noqa
       295369, 297729, 299626, 301546, 303449, 305548, 307882, 310124,  # noqa
       312374, 314509, 316815, 318789, 320981, 322879, 324878, 326959,  # noqa
       329341, 331200, 331201, 333469, 335584, 337984, 340143, 342034,  # noqa
       344360, 346309, 348544, 350970, 353052, 355227, 357449, 359603,  # noqa
       361725, 363676, 365735, 367799, 369777, 371904, 373856, 376204,  # noqa
       378391, 380800, 382859, 385161, 387093, 389434, 391624, 393785,  # noqa
       396093, 398214, 400198, 402166, 404104, 406047, 408372, 410686,  # noqa
       413029, 414975, 416850, 418797, 420824, 422959, 425026, 427215,  # noqa
       429278, 431668
    ]) - 1  # Fieldtrip has 1 sample difference with MNE

    return Annotations(
        onset=(EXPECTED_LATENCIES / 1200.0),
        duration=np.zeros_like(EXPECTED_LATENCIES),
        description='foo',
        orig_time=None
    )
