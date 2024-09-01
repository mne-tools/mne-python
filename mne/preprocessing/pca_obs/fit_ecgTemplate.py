import numpy as np
from scipy.signal import detrend
from scipy.interpolate import PchipInterpolator as pchip
import h5py


def fit_ecgTemplate(data, pca_template, aPeak_idx, peak_range, pre_range, post_range, baseline_range, midP, fitted_art, post_idx_previousPeak, n_samples_fit):
    # Declare class to hold ecg fit information
    class fitECG():
        def __init__(self):
            pass

    # Instantiate class
    fitecg = fitECG()

    # post_idx_nextpeak is passed in in PCA_OBS, used here as post_idx_previouspeak
    # Then nextpeak is returned at the end and the process repeats
    # select window of template
    template = pca_template[midP - peak_range-1: midP + peak_range+1, :]

    # select window of data and detrend it
    slice = data[0, aPeak_idx[0] - peak_range:aPeak_idx[0] + peak_range+1]
    detrended_data = detrend(slice.reshape(-1), type='constant')

    # maps data on template and then maps it again back to the sensor space
    least_square = np.linalg.lstsq(template, detrended_data, rcond=None)
    pad_fit = np.dot(template, least_square[0])

    # fit artifact, I already loop through externally channel to channel
    fitted_art[0, aPeak_idx[0] - pre_range-1: aPeak_idx[0] + post_range] = pad_fit[midP - pre_range-1: midP + post_range].T

    fitecg.fitted_art = fitted_art
    fitecg.template = template
    fitecg.detrended_data = detrended_data
    fitecg.pad_fit = pad_fit
    fitecg.aPeak_idx = aPeak_idx
    fitecg.midP = midP
    fitecg.peak_range = peak_range
    fitecg.data = data

    post_idx_nextPeak = [aPeak_idx[0] + post_range]

    # Check it's not empty
    if len(post_idx_previousPeak) != 0:
        # interpolate time between peaks
        intpol_window = np.ceil([post_idx_previousPeak[0], aPeak_idx[0] - pre_range]).astype('int') # interpolation window
        fitecg.intpol_window = intpol_window

        if intpol_window[0] < intpol_window[1]:
            # Piecewise Cubic Hermite Interpolating Polynomial(PCHIP) + replace EEG data

            # You have x_fit which is two slices on either side of the interpolation window endpoints
            # You have y_fit which is the y vals corresponding to x values above
            # You have x_interpol which is the time points between the two slices in x_fit that you want to interpolate
            # You have y_interpol which is values from pchip at the time points specified in x_interpol
            x_interpol = np.arange(intpol_window[0], intpol_window[1] + 1, 1)  # points to be interpolated in pt - the gap between the endpoints of the window
            x_fit = np.concatenate([np.arange(intpol_window[0] - n_samples_fit, intpol_window[0] + 1, 1),
                                    np.arange(intpol_window[1], intpol_window[1] + n_samples_fit + 1, 1)])  # Entire range of x values in this step (taking some number of samples before and after the window)
            y_fit = fitted_art[0, x_fit]
            y_interpol = pchip(x_fit, y_fit)(x_interpol)  # perform interpolation

            # Then make fitted artefact in the desired range equal to the completed fit above
            fitted_art[0, post_idx_previousPeak[0]: aPeak_idx[0] - pre_range+1] = y_interpol

            fitecg.x_fit = x_fit
            fitecg.y_fit = y_fit
            fitecg.x_interpol = x_interpol
            fitecg.y_interpol = y_interpol
            fitecg.fitted_art = fitted_art  # Reassign if we've gone into this loop

    # # Save to file to compare to matlab - only for debugging
    # dataset_keywords = [a for a in dir(fitecg) if not a.startswith('__')]
    # fn = f"/data/pt_02569/tmp_data/test_data/fitecg_test_{aPeak_idx[0]}_sub-001_tibial_S35.h5"
    # with h5py.File(fn, "w") as outfile:
    #     for keyword in dataset_keywords:
    #         outfile.create_dataset(keyword, data=getattr(fitecg, keyword))

    return fitted_art, post_idx_nextPeak
