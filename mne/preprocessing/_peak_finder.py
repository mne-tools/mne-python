import numpy as np

from .. utils import logger, verbose, _pl


@verbose
def peak_finder(x0, thresh=None, extrema=1, verbose=None):
    """Noise-tolerant fast peak-finding algorithm.

    Parameters
    ----------
    x0 : 1d array
        A real vector from the maxima will be found (required).
    thresh : float
        The amount above surrounding data for a peak to be
        identified (default = (max(x0)-min(x0))/4). Larger values mean
        the algorithm is more selective in finding peaks.
    extrema : {-1, 1}
        1 if maxima are desired, -1 if minima are desired
        (default = maxima, 1).
    %(verbose)s

    Returns
    -------
    peak_loc : array
        The indices of the identified peaks in x0.
    peak_mag : array
        The magnitude of the identified peaks.

    Notes
    -----
    If repeated values are found the first is identified as the peak.
    Conversion from initial Matlab code from:
    Nathanael C. Yoder (ncyoder@purdue.edu)

    Examples
    --------
    >>> import numpy as np
    >>> from mne.preprocessing import peak_finder
    >>> t = np.arange(0, 3, 0.01)
    >>> x = np.sin(np.pi*t) - np.sin(0.5*np.pi*t)
    >>> peak_locs, peak_mags = peak_finder(x) # doctest: +SKIP
    >>> peak_locs # doctest: +SKIP
    array([36, 260]) # doctest: +SKIP
    >>> peak_mags # doctest: +SKIP
    array([0.36900026, 1.76007351]) # doctest: +SKIP
    """
    x0 = np.asanyarray(x0)
    s = x0.size

    if x0.ndim >= 2 or s == 0:
        raise ValueError('The input data must be a non empty 1D vector')

    if thresh is None:
        thresh = (np.max(x0) - np.min(x0)) / 4
        logger.debug('Peak finder automatic threshold: %0.2g' % (thresh,))

    assert extrema in [-1, 1]

    if extrema == -1:
        x0 = extrema * x0  # Make it so we are finding maxima regardless

    dx0 = np.diff(x0)  # Find derivative
    # This is so we find the first of repeated values
    dx0[dx0 == 0] = -np.finfo(float).eps
    # Find where the derivative changes sign
    ind = np.where(dx0[:-1:] * dx0[1::] < 0)[0] + 1

    # Include endpoints in potential peaks and valleys
    x = np.concatenate((x0[:1], x0[ind], x0[-1:]))
    ind = np.concatenate(([0], ind, [s - 1]))
    del x0

    #  x only has the peaks, valleys, and endpoints
    length = x.size
    min_mag = np.min(x)

    if length > 2:  # Function with peaks and valleys

        # Set initial parameters for loop
        temp_mag = min_mag
        found_peak = False
        left_min = min_mag

        # Deal with first point a little differently since tacked it on
        # Calculate the sign of the derivative since we taked the first point
        # on it does not necessarily alternate like the rest.
        signDx = np.sign(np.diff(x[:3]))
        if signDx[0] <= 0:  # The first point is larger or equal to the second
            ii = -1
            if signDx[0] == signDx[1]:  # Want alternating signs
                x = np.concatenate((x[:1], x[2:]))
                ind = np.concatenate((ind[:1], ind[2:]))
                length -= 1

        else:  # First point is smaller than the second
            ii = 0
            if signDx[0] == signDx[1]:  # Want alternating signs
                x = x[1:]
                ind = ind[1:]
                length -= 1

        # Preallocate max number of maxima
        maxPeaks = int(np.ceil(length / 2.0))
        peak_loc = np.zeros(maxPeaks, dtype=np.int)
        peak_mag = np.zeros(maxPeaks)
        c_ind = 0
        # Loop through extrema which should be peaks and then valleys
        while ii < (length - 1):
            ii += 1  # This is a peak
            # Reset peak finding if we had a peak and the next peak is bigger
            # than the last or the left min was small enough to reset.
            if found_peak and ((x[ii] > peak_mag[-1]) or
                               (left_min < peak_mag[-1] - thresh)):
                temp_mag = min_mag
                found_peak = False

            # Make sure we don't iterate past the length of our vector
            if ii == length - 1:
                break  # We assign the last point differently out of the loop

            # Found new peak that was lager than temp mag and threshold larger
            # than the minimum to its left.
            if (x[ii] > temp_mag) and (x[ii] > left_min + thresh):
                temp_loc = ii
                temp_mag = x[ii]

            ii += 1  # Move onto the valley
            # Come down at least thresh from peak
            if not found_peak and (temp_mag > (thresh + x[ii])):
                found_peak = True  # We have found a peak
                left_min = x[ii]
                peak_loc[c_ind] = temp_loc  # Add peak to index
                peak_mag[c_ind] = temp_mag
                c_ind += 1
            elif x[ii] < left_min:  # New left minima
                left_min = x[ii]

        # Check end point
        if (x[-1] > temp_mag) and (x[-1] > (left_min + thresh)):
            peak_loc[c_ind] = length - 1
            peak_mag[c_ind] = x[-1]
            c_ind += 1
        elif not found_peak and temp_mag > min_mag:
            # Check if we still need to add the last point
            peak_loc[c_ind] = temp_loc
            peak_mag[c_ind] = temp_mag
            c_ind += 1

        # Create output
        peak_inds = ind[peak_loc[:c_ind]]
        peak_mags = peak_mag[:c_ind]
    else:  # This is a monotone function where an endpoint is the only peak
        x_ind = np.argmax(x)
        peak_mags = x[x_ind]
        if peak_mags > (min_mag + thresh):
            peak_inds = ind[x_ind]
        else:
            peak_mags = []
            peak_inds = []

    # Change sign of data if was finding minima
    if extrema < 0:
        peak_mags *= -1.0

    # ensure output type array
    if not isinstance(peak_inds, np.ndarray):
        peak_inds = np.atleast_1d(peak_inds).astype('int64')

    if not isinstance(peak_mags, np.ndarray):
        peak_mags = np.atleast_1d(peak_mags).astype('float64')

    # Plot if no output desired
    if len(peak_inds) == 0:
        logger.info('No significant peaks found')
    else:
        logger.info('Found %d significant peak%s'
                    % (len(peak_inds), _pl(peak_inds)))

    return peak_inds, peak_mags
