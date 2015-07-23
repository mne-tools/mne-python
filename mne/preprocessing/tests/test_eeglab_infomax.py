import numpy as np

from scipy.linalg import svd

import mne
from mne.io import Raw
from mne import pick_types

import scipy.io as sio
from scipy.linalg import pinv
from mne.preprocessing.infomax_ import infomax
from numpy.testing import assert_almost_equal
from mne.utils import random_permutation
from mne.datasets import testing


def generate_data_for_comparing_against_eeglab_infomax(ch_type, random_state):

    data_path = testing.data_path()
    raw_fname = data_path + '/MEG/sample/sample_audvis_trunc_raw.fif'

    raw = Raw(raw_fname, preload=True)

    if ch_type == 'eeg':
        picks = pick_types(raw.info, meg=False, eeg=True, exclude='bads')
    else:
        picks = pick_types(raw.info, meg=ch_type,
                           eeg=False, exclude='bads')

    # select a small number of channels for the test
    number_of_channels_to_use = 5
    idx_perm = random_permutation(picks.shape[0], random_state)
    picks = picks[idx_perm[:number_of_channels_to_use]]

    raw.filter(1, 45, n_jobs=2)
    X = raw[picks, :][0][:, ::20]

    # Substract the mean
    mean_X = X.mean(axis=1)
    X -= mean_X[:, None]

    # pre_whitening: z-score
    X /= np.std(X)

    T = X.shape[1]
    cov_X = np.dot(X, X.T) / T

    # Let's whiten the data
    U, D, _ = svd(cov_X)
    W = np.dot(U, U.T / np.sqrt(D)[:, None])
    Y = np.dot(W, X)

    return Y


@testing.requires_testing_data
def test_mne_python_vs_eeglab():
    """ Test eeglab vs mne_python infomax code.
    """
    random_state = 42

    list_ch_types = ['eeg', 'mag']

    for ch_type in list_ch_types:

        if ch_type == 'eeg':
            eeglab_results_file = 'eeglab_infomax_results_eeg_data.mat'
        elif ch_type == 'mag':
            eeglab_results_file = 'eeglab_infomax_results_meg_data.mat'

        Y = generate_data_for_comparing_against_eeglab_infomax(ch_type,
                                                               random_state)
        N = Y.shape[0]
        T = Y.shape[1]

        # For comparasion against eeglab, make sure the folowing
        # parameter have the same value in mne_python and eeglab:
        #
        # - starting point
        # - random state
        # - learning rate
        # - block size
        # - blowup parameter
        # - blowup_fac parameter
        # - tolerance for stopping the algorithm
        # - number of iterations
        #
        # Notes:
        # * By default, eeglab whiten the data using the "sphering transform"
        #   instead of pca. The mne_python infomax code does not
        #   whiten the data. To make sure both mne_python and eeglab starts
        #   from the same point (i.e., the same matrix), we need to make sure
        #   to whiten the data outside, and pass these whiten data to
        #   mne_python and eeglab. Finally, we need to tell eeglab that
        #   the input data is already whiten, this can be done by calling
        #   eeglab with the following sintax:
        #
        #   [unmixing,sphere,meanvar,bias,signs,lrates,sources,y] = ...
        #       runica( Y, 'sphering', 'none');
        #
        #   By calling eeglab using the former code, we are using its default
        #   parameters, which are specified below in the section
        #   "EEGLAB default parameters".
        #
        # * eeglab does not expose a parameter for fixing the random state.
        #   Therefore, to accomplish this, we need to edit the runica.m
        #   file located at /path_to_eeglab/functions/sigprocfunc/runica.m
        #
        #   i) Comment the line related with the random number generator
        #      (line 812).
        #   ii) Then, add the following line just below line 812:
        #       rng(42); %use 42 as random seed.
        #
        # * eeglab does not have the parameter "n_small_angle",
        #   so we need to disable it for making a fair comparison.
        #
        # * Finally, we need to take the unmixing matrix estimated by the
        #   mne_python infomax implementation and order the components
        #   in the same way that eeglab does. This is done below in the section
        #   "Order the components in the same way that eeglab does".

        ###############################################################
        # EEGLAB default parameters
        ###############################################################
        l_rate_eeglab = 0.00065 / np.log(N)
        block_eeglab = int(np.ceil(np.min([5 * np.log(T), 0.3 * T])))
        blowup_eeglab = 1e9
        blowup_fac_eeglab = 0.8
        max_iter_eeglab = 512

        if N > 32:
            w_change_eeglab = 1e-7
        else:
            w_change_eeglab = 1e-6
        ###############################################################

        # Call mne_python infomax version using the following sintax
        # to obtain the same result than eeglab version
        unmixing = infomax(Y.T, extended=False,
                           random_state=random_state,
                           max_iter=max_iter_eeglab,
                           l_rate=l_rate_eeglab,
                           block=block_eeglab,
                           w_change=w_change_eeglab,
                           blowup=blowup_eeglab,
                           blowup_fac=blowup_fac_eeglab,
                           n_small_angle=None,
                           )

        #######################################################################
        # Order the components in the same way that eeglab does
        #######################################################################

        sources = np.dot(unmixing, Y)
        mixing = pinv(unmixing)

        mvar = np.sum(mixing ** 2, axis=0) * \
            np.sum(sources ** 2, axis=1) / (N * T - 1)
        windex = np.argsort(mvar)[::-1]

        unmixing_ordered = unmixing[windex, :]
        #######################################################################

        #######################################################################
        # Load the eeglab results, then compare the unmixing matrices estimated
        # by mne_python and eeglab. To make the comparison use the
        # \ell_inf norm:
        # ||unmixing_mne_python - unmixing_eeglab||_inf
        #######################################################################

        eeglab_data = sio.loadmat(mne.__path__[0] +
                                  '/preprocessing/tests/data/' +
                                  eeglab_results_file)
        unmixing_eeglab = eeglab_data['unmixing_eeglab']

        maximum_difference = np.max(np.abs(unmixing_ordered - unmixing_eeglab))

        assert_almost_equal(maximum_difference, 1e-12, decimal=10)
