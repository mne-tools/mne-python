# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

from sklearn.decomposition import FastICA
import numpy as np
from inspect import getargspec
from ..cov import compute_whitener
from scipy.stats import kurtosis, skew
from scipy import linalg
from copy import copy
from ..fiff.raw import RawFromMerge


class ICA(object):
    """ MEG signal decomposition and denoising workflow
    Paramerters
    -----------
    raw : instance of Raw
        MEG raw data
    noise_cov : ndarray
        noise covariance used for whitening
    picks : array-like
        indices for channel selection as returned from mne.fiff.pick_channels
    n_components : integer
        number of components to extract. If none, no dimension
        reduciton will be applied.
    start : integer
        starting time slice
    stop : integer
        final time slice
    1exlude : list
        names of channels to exclude used for computing the whitener.

    Attributes
    ----------
    raw : instance of Raw
        raw object used for initializing the ica seed
    n_components : integer
        number of components to be extracted
    whitener : ndrarray
        whiter used for preprocessing
    seed : instance of FastICA
        FastICA instance used to perform decomposition
    source_ids : np.ndrarray
        array of source ids for sorting-book-keeping
    sorted_by : str
        flag informing about the active
    """
    def __init__(self, raw, picks, noise_cov=None, start=None, stop=None,
                 n_components=None, exclude=None):

        self.raw = raw
        self.n_components = n_components
        self._cov = True if noise_cov != None else False
        self.source_ids = np.arange(n_components)
        self.sorted_by = 'unsorted'
        self.picks = picks

        if raw._preloaded:
            self.raw_data = raw._data.copy()
        else:
            start = raw.first_samp if start == None else start
            stop = raw.last_samp if stop == None else stop
            self.raw_data = copy(raw[:, start:stop][0])

        if noise_cov != None:
            self.pre_whitener, _ = compute_whitener(noise_cov, raw.info,
                                                    picks, exclude)
            del _

            self.raw_data = np.dot(self.pre_whitener,
                                   self.raw_data[self.picks])

        elif noise_cov == None:  # use standardization as whitener
            std_chan = np.std(self.raw_data[self.picks], axis=1) ** -1
            self.pre_whitener = np.array([std_chan]).T
            self.raw_data[self.picks] *= self.pre_whitener

        self.seed = FastICA(n_components)

    def __repr__(self):
        out = 'ICA decomposition.\n    %i components' % self.n_components

        if hasattr(self, 'raw_sources'):
            n_samples = self.raw_sources.shape[1]
            out += ('\n    %i raw time slices' % n_samples)

        if self.sorted_by == 'unsorted':
            sorted_by = self.sorted_by

        else:
            sorted_by = '    sorted by %s' % self.sorted_by
        out += '\n    %s' % sorted_by

        return out

    def sort_sources(self, sources='raw', smethod='skew', inplace=True):
        """
        Paramerters
        -----------
        sources : str
            string for selecting the sources
        sort_method : string
            function used to sort the sources and the mixing matrix.
            If None, the output will be left unsorted
        """
        if smethod == 'skew':
            sort_func = skew
        elif smethod == 'kurtosis':
            sort_func = kurtosis
        elif smethod == 'back':
            sort_func = lambda x, y: self.source_ids
            sort_func.__name__ = 'unsorted'
        elif callable(smethod):
            args = getargspec(smethod).args
            if len(args) > 1:
                if args[:2] == ['a', 'axis']:
                    sort_func = smethod
            else:
                ValueError('%s is not a valid function.'
                           'The function needs an array and'
                           'an axis argument' % smethod.__name__)
        elif isinstance(smethod, str):
            ValueError('%s is not a valid sorting option' % smethod)

        self.sorted_by = sort_func.__name__

        if sources == 'raw':
            S = self.raw_sources
            A = self.raw_mixing
        else:
            # A = self.epochs_sources
            return NotImplemented  # not implemented

        sort_args = np.argsort(sort_func(S, 1))
        S = S[sort_args]
        A = A[sort_args]
        self.source_ids = self.source_ids[sort_args]

        if sources == 'raw':
            if inplace:
                self.raw_sources = S
                self.raw_mixing = A
        else:
            # A = self.epochs_sources
            return NotImplemented  # not implemented

        if not inplace:
            return S, A

    def fit_raw(self, smethod='skew'):
        """ Run the ica decomposition for raw data
        Paramerters
        -----------
        sort_method : string
            function used to sort the sources and the mixing matrix.
            If None, the output will be left unsorted

        Returns
        -------
        self : instance of ICA
            returns the instance for chaining
        """
        print ('\nComputing signal decomposition on raw data.'
               '\n    Please be patient. This may take some time')
        S = self.seed.fit_transform(self.raw_data[self.picks].T)
        A = self.seed.get_mixing_matrix()
        self.raw_sources = S.T
        self.raw_mixing = A.T
        self.sort_sources(sources='raw', smethod=smethod)

        return self

    def fit_epochs(self, epochs, picks):
        pass

    def denoise_raw(self, bads=[], make_raw=False):
        """ Recompose raw data

        Paramerters
        -----------
        bads : list-like
            Indices for transient component deselection
        make_raw : boolean
            Either return denoised data as nd array or newly instantiated
            Raw object.

        Returns
        -------
        out : depends on input arguments
            denoised raw data as ndarray or as instance of Raw
        """
        if self.sorted_by != 'unsorted':
            raw_sources, raw_mixing = self.sort_sources(smethod='back',
                                                        inplace=False)
        else:
            raw_sources = self.raw_sources.copy()
            raw_mixing = self.raw_mixing.copy()
        pre_whitener = self.pre_whitener
        if self._cov == False:  # revert standardization
            pre_whitener **= -1
            raw_mixing *= pre_whitener.T
        else:
            raw_mixing = np.dot(raw_mixing, linalg.pinv(pre_whitener))

        if bads != None:
            source_ids = self.source_ids.tolist()
            bads_idx = [source_ids.index(bad) for bad in bads]
            raw_sources[bads_idx, :] = 0

        out = np.dot(raw_sources.T, raw_mixing).T

        if make_raw == True:
            data = self.raw_data.copy()
            data[self.picks] = out
            return RawFromMerge(self.raw, data)

        return out

    def denoise_epochs(self, bads=[]):
        pass
