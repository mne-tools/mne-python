import numpy as np
import mne
from scipy.signal import filtfilt, detrend
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from fit_ecgTemplate import fit_ecgTemplate
import math
import h5py


def PCA_OBS(data, **kwargs):

    # Declare class to hold pca information
    class PCAInfo():
        def __init__(self):
            pass

    # Instantiate class
    pca_info = PCAInfo()

    # Check all necessary arguments sent in
    required_kws = ["debug_mode", "qrs", "filter_coords", "sr", "savename", "ch_names", "sub_nr", "condition",
                    "current_channel"]
    assert all([kw in kwargs.keys() for kw in required_kws]), "Error. Some KWs not passed into PCA_OBS."

    # Extract all kwargs
    debug_mode = kwargs['debug_mode']
    qrs = kwargs['qrs']
    filter_coords = kwargs['filter_coords']
    sr = kwargs['sr']
    ch_names = kwargs['ch_names']
    sub_nr = kwargs['sub_nr']
    condition = kwargs['condition']
    if debug_mode:  # Only need current channel and saving if we're debugging
        current_channel = kwargs['current_channel']
        savename = kwargs['savename']

    fs = sr

    # Standard delay between QRS peak and artifact
    delay = 0

    Gwindow = 2
    GHW = np.floor(Gwindow / 2).astype('int')
    rcount = 0
    firstplot = 1

    # set to baseline
    data = data.reshape(-1, 1)
    data = data.T
    data = data - np.mean(data, axis=1)

    # Allocate memory
    fitted_art = np.zeros(data.shape)
    peakplot = np.zeros(data.shape)

    # Extract QRS events
    for idx in qrs[0]:
        if idx < len(peakplot[0, :]):
            peakplot[0, idx] = 1  # logical indexed locations of qrs events
    # sh = np.zeros((1, delay))
    # np1 = len(peakplot)
    # peakplot = [sh, peakplot[0:np1 - delay]] # shifts indexed array by the delay - skipped here since delay=0

    peak_idx = np.nonzero(peakplot)[1]  # Selecting indices along columns
    peak_idx = peak_idx.reshape(-1, 1)
    peak_count = len(peak_idx)

    ################################################################
    # Preparatory work - reserving memory, configure sizes, de-trend
    ################################################################
    print('Pulse artifact subtraction in progress...Please wait!')

    # define peak range based on RR
    RR = np.diff(peak_idx[:, 0])
    mRR = np.median(RR)
    peak_range = round(mRR/2)  # Rounds to an integer
    midP = peak_range + 1
    baseline_range = [0, round(peak_range/8)]
    n_samples_fit = round(peak_range/8)  # sample fit for interpolation between fitted artifact windows

    # make sure array is long enough for PArange (if not cut off  last ECG peak)
    pa = peak_count  # Number of QRS complexes detected
    while peak_idx[pa-1, 0] + peak_range > len(data[0]):
        pa = pa - 1
    steps = 1 * pa
    peak_count = pa

    # Filter channel
    eegchan = filtfilt(filter_coords, 1, data)

    # build PCA matrix(heart-beat-epochs x window-length)
    pcamat = np.zeros((peak_count - 1, 2*peak_range+1))  # [epoch x time]
    # picking out heartbeat epochs
    for p in range(1, peak_count):
        pcamat[p-1, :] = eegchan[0, peak_idx[p, 0] - peak_range: peak_idx[p, 0] + peak_range+1]

    # detrending matrix(twice)
    pcamat = detrend(pcamat, type='constant', axis=1)  # [epoch x time] - detrended along the epoch
    mean_effect = np.mean(pcamat, axis=0)  # [1 x time], contains the mean over all epochs
    std_effect = np.std(pcamat, axis=0)  # want mean and std of each column
    dpcamat = detrend(pcamat, type='constant', axis=1)  # [time x epoch]

    ###################################################################
    # Perform PCA with sklearn
    ###################################################################
    # run PCA(performs SVD(singular value decomposition))
    pca = PCA(svd_solver="full")
    pca.fit(dpcamat)
    eigen_vectors = pca.components_
    eigen_values = pca.explained_variance_
    factor_loadings = pca.components_.T*np.sqrt(pca.explained_variance_)
    pca_info.eigen_vectors = eigen_vectors
    pca_info.factor_loadings = factor_loadings
    pca_info.eigen_values = eigen_values
    pca_info.expl_var = pca.explained_variance_ratio_

    # define selected number of  components using profile likelihood
    pca_info.nComponents = 4

    # Creates plots
    if debug_mode:
        # plot pca variables figure
        comp2plot = pca_info.nComponents
        fig, axs = plt.subplots(2, 2)
        for a in np.arange(comp2plot):
            axs[0, 0].plot(pca_info.eigen_vectors[:, a], label=f"Data {a}")
            axs[1, 1].plot(pca_info.factor_loadings[:, a], label=f"Data {a}")
        axs[0, 0].set_title('Evec')
        axs[1, 1].set_title('Factor Loadings')
        axs[1, 1].set(xlabel='time')

        axs[0, 1].plot(np.arange(len(pca_info.expl_var)), pca_info.expl_var, 'r*')
        axs[0, 1].set(xlabel='components', ylabel='var explained (%)')
        cum_explained = np.cumsum(pca_info.expl_var)
        axs[0, 1].set_title(f"first {pca_info.nComponents} comp, {cum_explained[pca_info.nComponents]} % var")

        axs[1, 0].plot(pca_info.eigen_values)
        axs[1, 0].set_title('eigenvalues')
        axs[1, 0].set(xlabel='components')

        fig.suptitle(f"{sub_nr} thresholds PCA vars channel {current_channel}")
        plt.tight_layout()
        fig.savefig(f"{savename}_{condition}.jpg")

    if debug_mode:
        pca_info.chan = current_channel
    pca_info.meanEffect = mean_effect.T
    nComponents = pca_info.nComponents

    #######################################################################
    # Make template of the ECG artefact
    #######################################################################
    mean_effect = mean_effect.reshape(-1, 1)
    pca_template = np.c_[mean_effect, factor_loadings[:, 0:nComponents]]

    # Plot template vars
    if debug_mode:
        # plot template vars
        fig = plt.figure()
        pcatime = (np.arange(-peak_range, peak_range+1))/fs
        pcatime = pcatime.reshape(-1)
        plt.plot(pcatime, std_effect)
        plt.plot(pcatime, mean_effect)
        plt.plot(pcatime, factor_loadings[:, 0: nComponents])
        plt.legend(['std effect', 'mean effect', 'PCA_1', 'PCA_2', 'PCA_3', 'PCA_4'])
        fig.suptitle(f"{sub_nr} papc channel {current_channel}")
        plt.tight_layout()
        fig.savefig(f"{savename}_templateVars_{condition}.jpg")

    ###################################################################################
    # Data Fitting
    ###################################################################################
    window_start_idx = []
    window_end_idx = []
    for p in range(0, peak_count):
        # Deals with start portion of data
        if p == 0:
            pre_range = peak_range
            post_range = math.floor((peak_idx[p + 1] - peak_idx[p])/2)
            if post_range > peak_range:
                post_range = peak_range
            try:
                post_idx_nextPeak = []
                fitted_art, post_idx_nextPeak = fit_ecgTemplate(data, pca_template, peak_idx[p], peak_range,
                                                                pre_range, post_range, baseline_range, midP,
                                                                fitted_art, post_idx_nextPeak, n_samples_fit)
                # Appending to list instead of using counter
                window_start_idx.append(peak_idx[p] - peak_range)
                window_end_idx.append(peak_idx[p] + peak_range)
            except Exception as e:
                print(f'Cannot fit first ECG epoch. Reason: {e}')

        # Deals with last edge of data
        elif p == peak_count:
            print('On last section - almost there!')
            try:
                pre_range = math.floor((peak_idx[p] - peak_idx[p - 1]) / 2)
                post_range = peak_range
                if pre_range > peak_range:
                    pre_range = peak_range
                fitted_art, _ = fit_ecgTemplate(data, pca_template, peak_idx(p), peak_range, pre_range, post_range,
                                                baseline_range, midP, fitted_art, post_idx_nextPeak, n_samples_fit)
                window_start_idx.append(peak_idx[p] - peak_range)
                window_end_idx.append(peak_idx[p] + peak_range)
            except Exception as e:
                print(f'Cannot fit last ECG epoch. Reason: {e}')

        # Deals with middle portion of data
        else:
            try:
                # ---------------- Processing of central data - --------------------
                # cycle through peak artifacts identified by peakplot
                pre_range = math.floor((peak_idx[p] - peak_idx[p - 1]) / 2)
                post_range = math.floor((peak_idx[p + 1] - peak_idx[p]) / 2)
                if pre_range >= peak_range:
                    pre_range = peak_range
                if post_range > peak_range:
                    post_range = peak_range

                aTemplate = pca_template[midP - peak_range-1:midP + peak_range+1, :]
                fitted_art, post_idx_nextPeak = fit_ecgTemplate(data, aTemplate, peak_idx[p], peak_range, pre_range,
                                                                post_range, baseline_range, midP, fitted_art,
                                                                post_idx_nextPeak, n_samples_fit)
                window_start_idx.append(peak_idx[p] - peak_range)
                window_end_idx.append(peak_idx[p] + peak_range)
            except Exception as e:
                print(f"Cannot fit middle section of data. Reason: {e}")

    # Plot some channels
    if debug_mode:
        # check with plot what has been done
        # First check if this channel is one we want to plot for debugging
        plotChannel = 0
        for ii in range(0,len(ch_names)):
            if current_channel == ch_names[ii]:
                plotChannel = 1

        if plotChannel == 1:
            fig = plt.figure()
            plt.plot((np.arange(0, len(fitted_art[0, :]))/fs).reshape(-1, 1), data[:].T, zorder=0)
            plt.plot((np.arange(0, len(fitted_art[0, :]))/fs).reshape(-1, 1), eegchan[:].T, 'r', zorder=5)
            plt.plot((np.arange(0, len(fitted_art[0, :]))/fs).reshape(-1, 1), fitted_art[:].T, 'g', zorder=10)
            plt.plot((np.arange(0, len(fitted_art[0, :]))/fs).reshape(-1, 1), (np.subtract(data[:], fitted_art[:])).T, 'm', zorder=15)
            plt.legend(['raw data', 'filtered', 'fitted_art', 'clean'], loc='upper right').set_zorder(20)
            plt.xlabel('time [s]')
            plt.ylabel('amplitude [V]')
            plt.title('Subject ' + sub_nr + ', channel ' + current_channel)
            fig.savefig(f"{savename}_compareresults_{condition}.jpg")

    # Actually subtract the artefact, return needs to be the same shape as input data
    # One sample shift purely due to the fact the r-peaks are currently detected in MATLAB
    data = data.reshape(-1)
    fitted_art = fitted_art.reshape(-1)

    # data -= fitted_art

    data_ = np.zeros(len(data))
    data_[0] = data[0]
    data_[1:] = data[1:] - fitted_art[:-1]
    data = data_

    # Original code is this:
    # data -= fitted_art
    # data = data.T.reshape(-1)

    # Can't add annotations for window start and end (sample number added) to mne raw structure here
    # Add it to the pca info class and store it that way
    # Can then access in the main rm_heart_artefact and create the annotations
    # Only save the pca vars if we're in debug mode
    if debug_mode:
        pca_info.window_start_idx = window_start_idx
        pca_info.window_end_idx = window_end_idx
        dataset_keywords = [a for a in dir(pca_info) if not a.startswith('__')]
        fn = f"{savename}_{condition}_pca_info.h5"
        with h5py.File(fn, "w") as outfile:
            for keyword in dataset_keywords:
                outfile.create_dataset(keyword, data=getattr(pca_info, keyword))

    # Can only return data
    return data
