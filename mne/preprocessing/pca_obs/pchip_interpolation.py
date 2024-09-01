# Function to interpolate based on PCHIP rather than MNE inbuilt linear option

import mne
import numpy as np
from scipy.interpolate import PchipInterpolator as pchip
import matplotlib.pyplot as plt


def PCHIP_interpolation(data, **kwargs):
    # Check all necessary arguments sent in
    required_kws = ["trigger_indices", "interpol_window_sec", "fs", "debug_mode"]
    assert all([kw in kwargs.keys() for kw in required_kws]), "Error. Some KWs not passed into PCA_OBS."

    # Extract all kwargs - more elegant ways to do this
    fs = kwargs['fs']
    interpol_window_sec = kwargs['interpol_window_sec']
    trigger_indices = kwargs['trigger_indices']
    debug_mode = kwargs['debug_mode']

    if debug_mode:
        plt.figure()
        # plot signal with artifact
        plot_range = [-50, 100]
        test_trial = 100
        xx = (np.arange(plot_range[0], plot_range[1])) / fs * 1000
        plt.plot(xx, data[trigger_indices[test_trial] + plot_range[0]:trigger_indices[test_trial] + plot_range[1]])

    # Convert intpol window to msec then convert to samples
    pre_window = round((interpol_window_sec[0]*1000) * fs / 1000)  # in samples
    post_window = round((interpol_window_sec[1]*1000) * fs / 1000)  # in samples
    intpol_window = np.ceil([pre_window, post_window]).astype(int)  # interpolation window

    n_samples_fit = 5  # number of samples before and after cut used for interpolation fit

    x_fit_raw = np.concatenate([np.arange(intpol_window[0]-n_samples_fit-1, intpol_window[0], 1),
                                np.arange(intpol_window[1]+1, intpol_window[1]+n_samples_fit+2, 1)])
    x_interpol_raw = np.arange(intpol_window[0], intpol_window[1]+1, 1)  # points to be interpolated; in pt

    for ii in np.arange(0, len(trigger_indices)):  # loop through all stimulation events
        x_fit = trigger_indices[ii] + x_fit_raw  # fit point latencies for this event
        x_interpol = trigger_indices[ii] + x_interpol_raw  # latencies for to-be-interpolated data points

        # Data is just a string of values
        y_fit = data[x_fit]  # y values to be fitted
        y_interpol = pchip(x_fit, y_fit)(x_interpol)  # perform interpolation
        data[x_interpol] = y_interpol  # replace in data

        if np.mod(ii, 100) == 0:  # talk to the operator every 100th trial
            print(f'stimulation event {ii} \n')

    if debug_mode:
        # plot signal with interpolated artifact
        plt.figure()
        plt.plot(xx, data[trigger_indices[test_trial] + plot_range[0]: trigger_indices[test_trial] + plot_range[1]])
        plt.title('After Correction')

    plt.show()

    return data
