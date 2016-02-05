"""Functions to make simple plots with M/EEG data
"""
from __future__ import print_function

# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Cathy Nangini <cnangini@gmail.com>
#          Mainak Jas <mainak@neuro.hut.fi>
#
# License: Simplified BSD

import copy
from glob import glob
from itertools import cycle
import os.path as op

import numpy as np
from scipy import linalg

from ..surface import read_surface
from ..io.proj import make_projector
from ..utils import logger, verbose, get_subjects_dir, warn
from ..io.pick import pick_types
from .utils import tight_layout, COLORS, _prepare_trellis, plt_show


@verbose
def plot_cov(cov, info, exclude=[], colorbar=True, proj=False, show_svd=True,
             show=True, verbose=None):
    """Plot Covariance data

    Parameters
    ----------
    cov : instance of Covariance
        The covariance matrix.
    info: dict
        Measurement info.
    exclude : list of string | str
        List of channels to exclude. If empty do not exclude any channel.
        If 'bads', exclude info['bads'].
    colorbar : bool
        Show colorbar or not.
    proj : bool
        Apply projections or not.
    show_svd : bool
        Plot also singular values of the noise covariance for each sensor
        type. We show square roots ie. standard deviations.
    show : bool
        Show figure if True.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).

    Returns
    -------
    fig_cov : instance of matplotlib.pyplot.Figure
        The covariance plot.
    fig_svd : instance of matplotlib.pyplot.Figure | None
        The SVD spectra plot of the covariance.
    """
    if exclude == 'bads':
        exclude = info['bads']
    ch_names = [n for n in cov.ch_names if n not in exclude]
    ch_idx = [cov.ch_names.index(n) for n in ch_names]
    info_ch_names = info['ch_names']
    sel_eeg = pick_types(info, meg=False, eeg=True, ref_meg=False,
                         exclude=exclude)
    sel_mag = pick_types(info, meg='mag', eeg=False, ref_meg=False,
                         exclude=exclude)
    sel_grad = pick_types(info, meg='grad', eeg=False, ref_meg=False,
                          exclude=exclude)
    idx_eeg = [ch_names.index(info_ch_names[c])
               for c in sel_eeg if info_ch_names[c] in ch_names]
    idx_mag = [ch_names.index(info_ch_names[c])
               for c in sel_mag if info_ch_names[c] in ch_names]
    idx_grad = [ch_names.index(info_ch_names[c])
                for c in sel_grad if info_ch_names[c] in ch_names]

    idx_names = [(idx_eeg, 'EEG covariance', 'uV', 1e6),
                 (idx_grad, 'Gradiometers', 'fT/cm', 1e13),
                 (idx_mag, 'Magnetometers', 'fT', 1e15)]
    idx_names = [(idx, name, unit, scaling)
                 for idx, name, unit, scaling in idx_names if len(idx) > 0]

    C = cov.data[ch_idx][:, ch_idx]

    if proj:
        projs = copy.deepcopy(info['projs'])

        #   Activate the projection items
        for p in projs:
            p['active'] = True

        P, ncomp, _ = make_projector(projs, ch_names)
        if ncomp > 0:
            logger.info('    Created an SSP operator (subspace dimension'
                        ' = %d)' % ncomp)
            C = np.dot(P, np.dot(C, P.T))
        else:
            logger.info('    The projection vectors do not apply to these '
                        'channels.')

    import matplotlib.pyplot as plt

    fig_cov = plt.figure(figsize=(2.5 * len(idx_names), 2.7))
    for k, (idx, name, _, _) in enumerate(idx_names):
        plt.subplot(1, len(idx_names), k + 1)
        plt.imshow(C[idx][:, idx], interpolation="nearest", cmap='RdBu_r')
        plt.title(name)
    plt.subplots_adjust(0.04, 0.0, 0.98, 0.94, 0.2, 0.26)
    tight_layout(fig=fig_cov)

    fig_svd = None
    if show_svd:
        fig_svd = plt.figure()
        for k, (idx, name, unit, scaling) in enumerate(idx_names):
            s = linalg.svd(C[idx][:, idx], compute_uv=False)
            plt.subplot(1, len(idx_names), k + 1)
            plt.ylabel('Noise std (%s)' % unit)
            plt.xlabel('Eigenvalue index')
            plt.semilogy(np.sqrt(s) * scaling)
            plt.title(name)
            tight_layout(fig=fig_svd)

    plt_show(show)
    return fig_cov, fig_svd


def plot_source_spectrogram(stcs, freq_bins, tmin=None, tmax=None,
                            source_index=None, colorbar=False, show=True):
    """Plot source power in time-freqency grid.

    Parameters
    ----------
    stcs : list of SourceEstimate
        Source power for consecutive time windows, one SourceEstimate object
        should be provided for each frequency bin.
    freq_bins : list of tuples of float
        Start and end points of frequency bins of interest.
    tmin : float
        Minimum time instant to show.
    tmax : float
        Maximum time instant to show.
    source_index : int | None
        Index of source for which the spectrogram will be plotted. If None,
        the source with the largest activation will be selected.
    colorbar : bool
        If true, a colorbar will be added to the plot.
    show : bool
        Show figure if True.
    """
    import matplotlib.pyplot as plt

    # Input checks
    if len(stcs) == 0:
        raise ValueError('cannot plot spectrogram if len(stcs) == 0')

    stc = stcs[0]
    if tmin is not None and tmin < stc.times[0]:
        raise ValueError('tmin cannot be smaller than the first time point '
                         'provided in stcs')
    if tmax is not None and tmax > stc.times[-1] + stc.tstep:
        raise ValueError('tmax cannot be larger than the sum of the last time '
                         'point and the time step, which are provided in stcs')

    # Preparing time-frequency cell boundaries for plotting
    if tmin is None:
        tmin = stc.times[0]
    if tmax is None:
        tmax = stc.times[-1] + stc.tstep
    time_bounds = np.arange(tmin, tmax + stc.tstep, stc.tstep)
    freq_bounds = sorted(set(np.ravel(freq_bins)))
    freq_ticks = copy.deepcopy(freq_bounds)

    # Reject time points that will not be plotted and gather results
    source_power = []
    for stc in stcs:
        stc = stc.copy()  # copy since crop modifies inplace
        stc.crop(tmin, tmax - stc.tstep)
        source_power.append(stc.data)
    source_power = np.array(source_power)

    # Finding the source with maximum source power
    if source_index is None:
        source_index = np.unravel_index(source_power.argmax(),
                                        source_power.shape)[1]

    # If there is a gap in the frequency bins record its locations so that it
    # can be covered with a gray horizontal bar
    gap_bounds = []
    for i in range(len(freq_bins) - 1):
        lower_bound = freq_bins[i][1]
        upper_bound = freq_bins[i + 1][0]
        if lower_bound != upper_bound:
            freq_bounds.remove(lower_bound)
            gap_bounds.append((lower_bound, upper_bound))

    # Preparing time-frequency grid for plotting
    time_grid, freq_grid = np.meshgrid(time_bounds, freq_bounds)

    # Plotting the results
    fig = plt.figure(figsize=(9, 6))
    plt.pcolor(time_grid, freq_grid, source_power[:, source_index, :],
               cmap='Reds')
    ax = plt.gca()

    plt.title('Time-frequency source power')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    time_tick_labels = [str(np.round(t, 2)) for t in time_bounds]
    n_skip = 1 + len(time_bounds) // 10
    for i in range(len(time_bounds)):
        if i % n_skip != 0:
            time_tick_labels[i] = ''

    ax.set_xticks(time_bounds)
    ax.set_xticklabels(time_tick_labels)
    plt.xlim(time_bounds[0], time_bounds[-1])
    plt.yscale('log')
    ax.set_yticks(freq_ticks)
    ax.set_yticklabels([np.round(freq, 2) for freq in freq_ticks])
    plt.ylim(freq_bounds[0], freq_bounds[-1])

    plt.grid(True, ls='-')
    if colorbar:
        plt.colorbar()
    tight_layout(fig=fig)

    # Covering frequency gaps with horizontal bars
    for lower_bound, upper_bound in gap_bounds:
        plt.barh(lower_bound, time_bounds[-1] - time_bounds[0], upper_bound -
                 lower_bound, time_bounds[0], color='#666666')

    plt_show(show)
    return fig


def _plot_mri_contours(mri_fname, surf_fnames, orientation='coronal',
                       slices=None, show=True):
    """Plot BEM contours on anatomical slices.

    Parameters
    ----------
    mri_fname : str
        The name of the file containing anatomical data.
    surf_fnames : list of str
        The filenames for the BEM surfaces in the format
        ['inner_skull.surf', 'outer_skull.surf', 'outer_skin.surf'].
    orientation : str
        'coronal' or 'axial' or 'sagittal'
    slices : list of int
        Slice indices.
    show : bool
        Show figure if True.

    Returns
    -------
    fig : Instance of matplotlib.figure.Figure
        The figure.
    """
    import matplotlib.pyplot as plt
    import nibabel as nib

    if orientation not in ['coronal', 'axial', 'sagittal']:
        raise ValueError("Orientation must be 'coronal', 'axial' or "
                         "'sagittal'. Got %s." % orientation)

    # Load the T1 data
    nim = nib.load(mri_fname)
    data = nim.get_data()
    affine = nim.get_affine()

    n_sag, n_axi, n_cor = data.shape
    orientation_name2axis = dict(sagittal=0, axial=1, coronal=2)
    orientation_axis = orientation_name2axis[orientation]

    if slices is None:
        n_slices = data.shape[orientation_axis]
        slices = np.linspace(0, n_slices, 12, endpoint=False).astype(np.int)

    # create of list of surfaces
    surfs = list()

    trans = linalg.inv(affine)
    # XXX : next line is a hack don't ask why
    trans[:3, -1] = [n_sag // 2, n_axi // 2, n_cor // 2]

    for surf_fname in surf_fnames:
        surf = dict()
        surf['rr'], surf['tris'] = read_surface(surf_fname)
        # move back surface to MRI coordinate system
        surf['rr'] = nib.affines.apply_affine(trans, surf['rr'])
        surfs.append(surf)

    fig, axs = _prepare_trellis(len(slices), 4)

    for ax, sl in zip(axs, slices):

        # adjust the orientations for good view
        if orientation == 'coronal':
            dat = data[:, :, sl].transpose()
        elif orientation == 'axial':
            dat = data[:, sl, :]
        elif orientation == 'sagittal':
            dat = data[sl, :, :]

        # First plot the anatomical data
        ax.imshow(dat, cmap=plt.cm.gray)
        ax.axis('off')

        # and then plot the contours on top
        for surf in surfs:
            if orientation == 'coronal':
                ax.tricontour(surf['rr'][:, 0], surf['rr'][:, 1],
                              surf['tris'], surf['rr'][:, 2],
                              levels=[sl], colors='yellow', linewidths=2.0)
            elif orientation == 'axial':
                ax.tricontour(surf['rr'][:, 2], surf['rr'][:, 0],
                              surf['tris'], surf['rr'][:, 1],
                              levels=[sl], colors='yellow', linewidths=2.0)
            elif orientation == 'sagittal':
                ax.tricontour(surf['rr'][:, 2], surf['rr'][:, 1],
                              surf['tris'], surf['rr'][:, 0],
                              levels=[sl], colors='yellow', linewidths=2.0)

    plt.subplots_adjust(left=0., bottom=0., right=1., top=1., wspace=0.,
                        hspace=0.)
    plt_show(show)
    return fig


def plot_bem(subject=None, subjects_dir=None, orientation='coronal',
             slices=None, show=True):
    """Plot BEM contours on anatomical slices.

    Parameters
    ----------
    subject : str
        Subject name.
    subjects_dir : str | None
        Path to the SUBJECTS_DIR. If None, the path is obtained by using
        the environment variable SUBJECTS_DIR.
    orientation : str
        'coronal' or 'axial' or 'sagittal'.
    slices : list of int
        Slice indices.
    show : bool
        Show figure if True.

    Returns
    -------
    fig : Instance of matplotlib.figure.Figure
        The figure.
    """
    subjects_dir = get_subjects_dir(subjects_dir, raise_error=True)

    # Get the MRI filename
    mri_fname = op.join(subjects_dir, subject, 'mri', 'T1.mgz')
    if not op.isfile(mri_fname):
        raise IOError('MRI file "%s" does not exist' % mri_fname)

    # Get the BEM surface filenames
    bem_path = op.join(subjects_dir, subject, 'bem')

    if not op.isdir(bem_path):
        raise IOError('Subject bem directory "%s" does not exist' % bem_path)

    surf_fnames = []
    for surf_name in ['*inner_skull', '*outer_skull', '*outer_skin']:
        surf_fname = glob(op.join(bem_path, surf_name + '.surf'))
        if len(surf_fname) > 0:
            surf_fname = surf_fname[0]
            logger.info("Using surface: %s" % surf_fname)
            surf_fnames.append(surf_fname)

    if len(surf_fnames) == 0:
        raise IOError('No surface files found. Surface files must end with '
                      'inner_skull.surf, outer_skull.surf or outer_skin.surf')

    # Plot the contours
    return _plot_mri_contours(mri_fname, surf_fnames, orientation=orientation,
                              slices=slices, show=show)


def plot_events(events, sfreq=None, first_samp=0, color=None, event_id=None,
                axes=None, equal_spacing=True, show=True):
    """Plot events to get a visual display of the paradigm

    Parameters
    ----------
    events : array, shape (n_events, 3)
        The events.
    sfreq : float | None
        The sample frequency. If None, data will be displayed in samples (not
        seconds).
    first_samp : int
        The index of the first sample. Typically the raw.first_samp
        attribute. It is needed for recordings on a Neuromag
        system as the events are defined relative to the system
        start and not to the beginning of the recording.
    color : dict | None
        Dictionary of event_id value and its associated color. If None,
        colors are automatically drawn from a default list (cycled through if
        number of events longer than list of default colors).
    event_id : dict | None
        Dictionary of event label (e.g. 'aud_l') and its associated
        event_id value. Label used to plot a legend. If None, no legend is
        drawn.
    axes : instance of matplotlib.axes.AxesSubplot
       The subplot handle.
    equal_spacing : bool
        Use equal spacing between events in y-axis.
    show : bool
        Show figure if True.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.

    Notes
    -----
    .. versionadded:: 0.9.0
    """

    if sfreq is None:
        sfreq = 1.0
        xlabel = 'samples'
    else:
        xlabel = 'Time (s)'

    events = np.asarray(events)
    unique_events = np.unique(events[:, 2])

    if event_id is not None:
        # get labels and unique event ids from event_id dict,
        # sorted by value
        event_id_rev = dict((v, k) for k, v in event_id.items())
        conditions, unique_events_id = zip(*sorted(event_id.items(),
                                                   key=lambda x: x[1]))

        for this_event in unique_events_id:
            if this_event not in unique_events:
                raise ValueError('%s from event_id is not present in events.'
                                 % this_event)

        for this_event in unique_events:
            if this_event not in unique_events_id:
                warn('event %s missing from event_id will be ignored'
                     % this_event)
    else:
        unique_events_id = unique_events

    if color is None:
        if len(unique_events) > len(COLORS):
            warn('More events than colors available. You should pass a list '
                 'of unique colors.')
        colors = cycle(COLORS)
        color = dict()
        for this_event, this_color in zip(unique_events_id, colors):
            color[this_event] = this_color
    else:
        for this_event in color:
            if this_event not in unique_events_id:
                raise ValueError('%s from color is not present in events '
                                 'or event_id.' % this_event)

        for this_event in unique_events_id:
            if this_event not in color:
                warn('Color is not available for event %d. Default colors '
                     'will be used.' % this_event)

    import matplotlib.pyplot as plt

    fig = None
    if axes is None:
        fig = plt.figure()
    ax = axes if axes else plt.gca()

    unique_events_id = np.array(unique_events_id)
    min_event = np.min(unique_events_id)
    max_event = np.max(unique_events_id)

    for idx, ev in enumerate(unique_events_id):
        ev_mask = events[:, 2] == ev
        kwargs = {}
        if event_id is not None:
            event_label = '{0} ({1})'.format(event_id_rev[ev],
                                             np.sum(ev_mask))
            kwargs['label'] = event_label
        if ev in color:
            kwargs['color'] = color[ev]
        if equal_spacing:
            ax.plot((events[ev_mask, 0] - first_samp) / sfreq,
                    (idx + 1) * np.ones(ev_mask.sum()), '.', **kwargs)
        else:
            ax.plot((events[ev_mask, 0] - first_samp) / sfreq,
                    events[ev_mask, 2], '.', **kwargs)

    if equal_spacing:
        ax.set_ylim(0, unique_events_id.size + 1)
        ax.set_yticks(1 + np.arange(unique_events_id.size))
        ax.set_yticklabels(unique_events_id)
    else:
        ax.set_ylim([min_event - 1, max_event + 1])

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Events id')

    ax.grid('on')

    fig = fig if fig is not None else plt.gcf()
    if event_id is not None:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        fig.canvas.draw()
    plt_show(show)
    return fig


def _get_presser(fig):
    """Helper to get our press callback"""
    callbacks = fig.canvas.callbacks.callbacks['button_press_event']
    func = None
    for key, val in callbacks.items():
        if val.func.__class__.__name__ == 'partial':
            func = val.func
            break
    assert func is not None
    return func


def plot_dipole_amplitudes(dipoles, colors=None, show=True):
    """Plot the amplitude traces of a set of dipoles

    Parameters
    ----------
    dipoles : list of instance of Dipoles
        The dipoles whose amplitudes should be shown.
    colors: list of colors | None
        Color to plot with each dipole. If None default colors are used.
    show : bool
        Show figure if True.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot.

    Notes
    -----
    .. versionadded:: 0.9.0
    """
    import matplotlib.pyplot as plt
    if colors is None:
        colors = cycle(COLORS)
    fig, ax = plt.subplots(1, 1)
    xlim = [np.inf, -np.inf]
    for dip, color in zip(dipoles, colors):
        ax.plot(dip.times, dip.amplitude, color=color, linewidth=1.5)
        xlim[0] = min(xlim[0], dip.times[0])
        xlim[1] = max(xlim[1], dip.times[-1])
    ax.set_xlim(xlim)
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Amplitude (nAm)')
    if show:
        fig.show(warn=False)
    return fig
