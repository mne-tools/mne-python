# Authors: Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

from sklearn.decomposition import FastICA
import copy
import numpy as np
from ..cov import compute_whitener
from ..fiff import pick_types


def _zscore_channels(data, info, picks, exclude):
    """ Compute zscores for channels
    """
    from scipy.stats import zscore
    pick_mag = pick_types(info, meg='mag', eeg=False, exclude=exclude)
    pick_grad = pick_types(info, meg='grad', eeg=False, exclude=exclude)
    pick_eeg = pick_types(info, meg=False, eeg=True, exclude=exclude)

    mag_idx = np.intersect1d(picks, pick_mag)
    grad_idx = np.intersect1d(picks, pick_grad)
    eeg_idx = np.intersect1d(picks, pick_eeg)

    ch_type_idx = [mag_idx, grad_idx, eeg_idx]
    # print mag_names, mag_idx, grad_names, grad_idx

    ch_chan_types = reduce(lambda x, y: list(x) + list(y), ch_type_idx)
    assert picks.tolist() == sorted(ch_chan_types)

    out = np.zeros(data.shape)

    for idxs in ch_type_idx:
        if idxs.shape[0]:
            out[idxs, :] = zscore(data[idxs], 0)

    out = out[picks]

    return out


def decompose_raw(raw, noise_cov, n_components=None, start=None, stop=None,
                  sort_method=None, meg=True, eeg=True, exclude=None,
                  *args, **kwargs):
    """ Run ICA decomposition on instance of Raw
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
    sort_method : string
        function used to sort the sources and the mixing matrix.
        If None, the output will be left unsorted
    meg : string, bool
        selection of meg channels, passed to pick_types
    eeg : bool
        selection of eeg channels, passed to pick_types
    exclude : list, string
        channel names to be passe to picks

    Return Values
    -------------
    latentent_sources : ndarray
        time slices x n_components array
    mix_matrix : ndarray
        n_components x n_components
    """
    start = raw.first_samp if start == None else start
    stop = raw.last_samp if stop == None else stop
    info = copy.deepcopy(raw.info)

    picks = pick_types(info, meg=meg, eeg=eeg, exclude=exclude)

    data, _ = raw[:, start:stop]
    del _

    if noise_cov != None:
        whitener, _ = compute_whitener(noise_cov, info, picks, exclude)
        del _
        data = np.dot(whitener, data[picks])

    elif noise_cov == None:
        data = _zscore_channels(data, info, picks, exclude)
        # data = data[picks]  #

    data = data.T  # sklearn expects column vectors / matrixes

    print ('\nCalculating signal decomposition.'
           '\n    Please be patient. This may take some time')
    ica = FastICA(n_components=n_components, whiten=True, *args, **kwargs)
    S = ica.fit(data).transform(data)
    A = ica.get_mixing_matrix()

    assert np.allclose(data, np.dot(S, A.T))

    if sort_method != None:
        from scipy.stats import kurtosis
        sort_functions = {np.var: 'var', kurtosis: 'kurtosis'}
        sort_func = sort_functions.get(sort_method, None)
        if sort_func != None:
            sorter = sort_func(A, 0)
            A = A[np.argsort(sorter), :]
            sorter = sort_func(S, 0)
            S = S[np.argsort(sorter), :]
        else:
            print '\n    This is not a valid sorting function'

    latent_sources = np.swapaxes(S, 0, 1)
    mixing_matrix = np.swapaxes(A, 0, 1)

    print '\nDone.'

    return latent_sources, mixing_matrix  # ?, whitener


def recompose_raw(raw, picks, noise_cov):
    pass
