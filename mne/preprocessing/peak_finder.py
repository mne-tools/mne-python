import sys

from ..utils import copy_doc, deprecated
from ._peak_finder import peak_finder as _peak_finder


class peak_finder_wrapper(object):

    @deprecated('mne.preprocessing.peak_finder.peak_finder is deprecated and '
                'will be removed in 0.19, use mne.preprocessing.peak_finder'
                'directly instead. (For one cycle this will be a wrapper class'
                ' with a peak_finder method, and in 0.19 it will turn into a '
                'standard function.')
    @copy_doc(_peak_finder)
    def peak_finder(self, x0, thresh=None, extrema=1, verbose=None):
        return _peak_finder(x0, thresh, extrema, verbose)

    @copy_doc(_peak_finder)
    def __call__(self, x0, thresh=None, extrema=1, verbose=None):
        return _peak_finder(x0, thresh, extrema, verbose)


sys.modules[__name__] = peak_finder_wrapper()
