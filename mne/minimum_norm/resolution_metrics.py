# -*- coding: utf-8 -*-
# Authors: Olaf Hauk <olaf.hauk@mrc-cbu.cam.ac.uk>
#
# License: BSD (3-clause)
"""Compute resolution metrics from resolution matrix.

Resolution metrics: localisation error, spatial extent, relative amplitude.
Metrics can be computed for point-spread and cross-talk functions (PSFs/CTFs).
"""
import numpy as np

from mne import SourceEstimate
from mne.utils import logger, verbose


@verbose
def resolution_metrics(resmat, src, function, kind, metric):
    """Compute spatial resolution metrics for linear solvers.

    Parameters
    ----------
    resmat : array, shape (n_orient*n_locations, n_locations)
        The resolution matrix.
        If not a square matrix and if the number of rows is a multiple of
        number of columns (e.g. free or loose orientations), then the Euclidean
        length per source location is computed (e.g. if inverse operator with
        free orientations was applied to foward solution with fixed
        orientations).
    src : instance of SourceSpaces
        Source space object from forward or inverse operator.
    function: str
        'psf'|'ctf'
        Whether to compute metrics for columns (point-spread functions, PSFs)
        or rows (cross-talk functions, CTFs) of the resolution matrix.
    kind : str
        What kind of resolution feature to consider.
        Options are 'localization_error' | 'spatial_extent' | 'amplitude'.
    metric : str
        The exact resolution metric for the kind of resolution feature.
        This must match with 'kind'. The allowed options are:
        kind='localization_error':
            'peak': Peak localization error (PLE), Euclidean distance between
                    peak
                    and true source location.
            'cog': Centre-of-gravity localisation error (CoG), Euclidean
                   distance between CoG and true source location.
        kind='spatial_extent':
            'sd': spatial deviation (e.g. Molins et al.).
            'maxrad': maximum radius to 50% of max amplitude.
        kind='relative_amplitude':
            'peak': Ratio between absolute maximum amplitudes of peaks per
                    location and maximum peak across locations.
            'sum': Ratio between sums of absolute amplitudes.

    Returns
    -------
    resolution_metric : instance of SourceEstimate
        The source estimate contains the resolution metric as an array with
        shape (n_locations,)
    """
    # Check if input options are valid
    if kind == 'localization_error':
        if metric not in ['peak', 'cog']:
            print('Not an allowed metric for localization_error: %s' % metric)

    elif kind == 'spatial_extent':
        if metric not in ['sd', 'maxrad']:
            print('Not an allowed metric for spatial_extent: %s' % metric)

    elif kind == 'relative_amplitude':
        if metric not in ['peak', 'sum']:
            print('Not an allowed metric for amplitude: %s' % metric)

    else:
        print('Not a recognised kind of resolution feature: %s.' % kind)

        return

    if function not in ['psf', 'ctf']:
        print('Not a recognised resolution function: %s.' % function)
        return

    if kind == 'localization_error':

        resolution_metric = _localisation_error(resmat, src, function=function,
                                                metric=metric)

    elif kind == 'spatial_extent':

        resolution_metric = _spatial_extent(resmat, src, function=function,
                                            metric=metric)

    elif kind == 'relative_amplitude':

        resolution_metric = _relative_amplitude(resmat, src, function=function,
                                                metric=metric)

    # get vertices from source space
    vertno_lh = src[0]['vertno']
    vertno_rh = src[1]['vertno']
    vertno = [vertno_lh, vertno_rh]

    # Convert array to source estimate
    resolution_metric = SourceEstimate(resolution_metric, vertno, tmin=0.,
                                       tstep=1.)

    return resolution_metric


def _localisation_error(resmat, src, function, metric):
    """Compute localisation error metrics for resolution matrix.

    Parameters
    ----------
    resmat: array, shape (n_orient*n_locations, n_locations)
        The resolution matrix.
        If not a square matrix and if the number of rows is a multiple of
        number of columns (i.e. n_orient>1), then the Euclidean length per
        source location is computed (e.g. if inverse operator with free
        orientations was applied to foward solution with fixed orientations).
    src: Source Space
        Source space object from forward or inverse operator.
    function: str
        'psf'|'ctf'
        Whether to compute metrics for columns (point-spread functions, PSFs)
        or rows (cross-talk functions, CTFs).
    metric: str
        What type of localisation error to compute.
        'peak': Peak localisation error (PLE), Euclidean distance between peak
                and true source location, in centimeters.
        'cog': Centre-of-gravity localisation error (CoG), Euclidean distance
               between CoG and true source location, in centimeters.

    Returns
    -------
        locerr: array, shape (n_locations,)
        Localisation error per location (in cm).
    """
    # ensure resolution matrix is square
    # combine rows (Euclidean length) if necessary
    resmat = _rectify_resolution_matrix(resmat)

    # locations used in forward and inverse operator
    locations = _get_src_locations(src)

    # convert to cm (more common)
    locations = 100. * locations

    # only use absolute values
    resmat = np.absolute(resmat)

    # The below will operate on columns
    if function == 'ctf':

        resmat = resmat.T

    # Euclidean distance between true location and maximum
    if metric.lower() == 'peak':

        # find indices of maxima along columns
        resmax = resmat.argmax(axis=0)

        # locations of maxima
        maxloc = locations[resmax, :]

        # difference between locations of maxima and true locations
        diffloc = locations - maxloc

        # Euclidean distance
        locerr = np.sqrt(np.sum(diffloc**2, 1))

    # centre of gravity
    elif metric.lower() == 'cog':

        # initialise result array
        locerr = np.empty(locations.shape[0])

        for (ii, rr) in enumerate(locations):

            # corresponding column of resmat
            resvec = resmat[:, ii].T

            # centre of gravity
            cog = resvec.dot(locations) / np.sum(resvec)

            # centre of gravity
            locerr[ii] = np.sqrt(np.sum((rr - cog)**2))

    else:

        print('Not a valid metric for localisation error: %s.' % metric)

    return locerr


def _spatial_extent(resmat, src, function, metric):
    """Compute spatial width metrics for resolution matrix.

    Parameters
    ----------
    resmat : array, shape (n_orient*n_dipoles, n_dipoles)
        The resolution matrix.
        If not a square matrix and if the number of rows is a multiple of
        number of columns (i.e. n_orient>1), then the Euclidean length per
        source location is computed (e.g. if inverse operator with free
        orientations was applied to foward solution with fixed orientations).
    src : Source Space
        Source space object from forward or inverse operator.
    function : str
        'psf'|'ctf'
        Whether to compute metrics for columns (PSFs) or rows (CTFs).
    metric : string ('sd' | 'rad')
        What type of width metric to compute.
        'sd': spatial deviation (e.g. Molins et al.), in centimeters.
        'maxrad': maximum radius to 50% of max amplitude, in centimeters.

    Returns
    -------
    width : array, shape (n_dipoles,)
        Spatial width metric per location.
    """
    # locations used in forward and inverse operator
    locations = _get_src_locations(src)

    # convert to cm (more common)
    locations = 100. * locations

    # only use absolute values
    resmat = np.absolute(resmat)

    # The below will operate on columns
    if function == 'ctf':

        resmat = resmat.T

    # find indices of maxima along rows
    resmax = resmat.argmax(axis=0)

    # initialise output array
    width = np.empty(len(resmax))

    # spatial deviation as in Molins et al.
    if metric.lower() == 'sd':

        for ii in range(0, locations.shape[0]):

            # locations relative to true source
            diffloc = locations - locations[ii, :]

            # squared Euclidean distances to true source
            locerr = np.sum(diffloc**2, 1)

            # pick current row
            resvec = resmat[:, ii]**2

            # spatial deviation (Molins et al, NI 2008, eq. 12)
            width[ii] = np.sqrt(np.sum(np.multiply(locerr, resvec)) /
                                np.sum(resvec))

    # maximum radius to 50% of max amplitude
    elif metric.lower() == 'maxrad':

        # peak amplitudes per location across columns
        maxamp = resmat.max(axis=0)

        for (ii, aa) in enumerate(maxamp):  # for all locations

            # pick current column
            resvec = resmat[:, ii]

            # indices of elements with values larger than 50% of peak amplitude
            amps50idx = np.where(resvec > 0.5 * aa)[0]

            # get distances for those indices from true source position
            locs50 = locations[amps50idx, :] - locations[ii, :]

            # get maximum distance
            width[ii] = np.sqrt(np.sum(locs50**2, 1).max())

    else:

        print('Not a valid metric for spatial width: %s.' % metric)

    return width


def _relative_amplitude(resmat, src, function, metric):
    """Compute relative amplitude metrics for resolution matrix.

    Parameters
    ----------
    resmat : array, shape (n_orient*n_dipoles, n_dipoles)
        The resolution matrix.
        If not a square matrix and if the number of rows is a multiple of
        number of columns (i.e. n_orient>1), then the Euclidean length per
        source location is computed (e.g. if inverse operator with free
        orientations was applied to foward solution with fixed orientations).
    src : Source Space
        Source space object from forward or inverse operator.
    function : str
        'psf'|'ctf'
        Whether to compute metrics for columns (PSFs) or rows (CTFs).
    metric : str
        Which amplitudes to use.
        'peak': Ratio between absolute maximum amplitudes of peaks per location
                and maximum peak across locations.
        'sum': Ratio between sums of absolute amplitudes.

    Returns
    -------
    relamp: array, shape (n_dipoles,)
        Relative amplitude metric per location.
    """
    # The below will operate on columns
    if function == 'ctf':

        resmat = resmat.T

    # only use absolute values
    resmat = np.absolute(resmat)

    # Ratio between amplitude at peak and global peak maximum
    if metric.lower() == 'peak':

        # maximum amplitudes per column
        maxamps = resmat.max(axis=0)

        # global absolute maximum
        maxmaxamps = maxamps.max()

        relamp = maxamps / maxmaxamps

    # ratio between sums of absolute amplitudes
    elif metric.lower() == 'sum':

        # sum of amplitudes per column
        sumamps = np.sum(resmat, axis=0)

        # maximum of summed amplitudes
        sumampsmax = sumamps.max()

        relamp = sumamps / sumampsmax

    else:

        print('Not a valid metric for relative amplitude: %s.' % metric)

    return relamp


def _get_src_locations(src):
    """Helper to get source positions from src object."""
    # vertices used in forward and inverse operator
    vertno_lh = src[0]['vertno']
    vertno_rh = src[1]['vertno']

    # locations corresponding to vertices for both hemispheres
    locations_lh = src[0]['rr'][vertno_lh, :]
    locations_rh = src[1]['rr'][vertno_rh, :]
    locations = np.vstack([locations_lh, locations_rh])

    return locations


def _rectify_resolution_matrix(resmat):
    """
    Helper to ensure resolution matrix is square matrix.

    If resmat is not a square matrix, it is assumed that the inverse operator
    had free or loose orientation constraint, i.e. multiple values per source
    location. The Euclidean length for values at each location is computed to
    make resmat a square matrix.
    """
    shape = resmat.shape
    if not shape[0] == shape[1]:
        if shape[0] < shape[1]:
            raise ValueError('Number of target sources (%d) cannot be lower '
                             'than number of input sources (%d)' % shape[0],
                             shape[1])

        if np.mod(shape[0], shape[1]):  # if ratio not integer
            raise ValueError('Number of target sources (%d) must be a '
                             'multiple of the number of input sources (%d)'
                             % shape[0], shape[1])

        ns = int(shape[0] / shape[1])  # number of source components per vertex

        # Combine rows of resolution matrix
        resmatl = [np.sqrt((resmat[ns * i:ns * (i + 1), :]**2).sum(axis=0))
                   for i in np.arange(0, shape[1], dtype=int)]

        resmat = np.array(resmatl)

        logger.info('Rectified resolution matrix from (%d, %d) to (%d, %d).' %
                    (shape[0], shape[1], resmat.shape[0], resmat.shape[1]))

    return resmat
