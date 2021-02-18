# -*- coding: utf-8 -*-
"""The documentation functions."""
# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

from copy import deepcopy
import inspect
import os
import os.path as op
import sys
import warnings
import webbrowser

from .config import get_config
from ..defaults import HEAD_SIZE_DEFAULT
from ..externals.doccer import filldoc, unindent_dict
from .check import _check_option


##############################################################################
# Define our standard documentation entries

docdict = dict()

# Verbose
docdict['verbose'] = """
verbose : bool, str, int, or None
    If not None, override default verbose level (see :func:`mne.verbose`
    and :ref:`Logging documentation <tut_logging>` for more).
    If used, it should be passed as a keyword-argument only."""
docdict['verbose_meth'] = (docdict['verbose'] + ' Defaults to self.verbose.')

# Preload
docdict['preload'] = """
preload : bool or str (default False)
    Preload data into memory for data manipulation and faster indexing.
    If True, the data will be preloaded into memory (fast, requires
    large amount of memory). If preload is a string, preload is the
    file name of a memory-mapped file which is used to store the data
    on the hard drive (slower, requires less memory)."""
docdict['preload_concatenate'] = """
preload : bool, str, or None (default None)
    Preload data into memory for data manipulation and faster indexing.
    If True, the data will be preloaded into memory (fast, requires
    large amount of memory). If preload is a string, preload is the
    file name of a memory-mapped file which is used to store the data
    on the hard drive (slower, requires less memory). If preload is
    None, preload=True or False is inferred using the preload status
    of the instances passed in.
"""

# Raw
_on_missing_base = """\
Can be ``'raise'`` (default) to raise an error, ``'warn'`` to emit a
    warning, or ``'ignore'`` to ignore when"""
docdict['on_split_missing'] = """
on_split_missing : str
    Can be ``'raise'`` to raise an error, ``'warn'`` (default) to emit a
    warning, or ``'ignore'`` to ignore when a split file is missing.
    The default will change from ``'warn'`` to ``'raise'`` in 0.23, set the
    value explicitly to avoid deprecation warnings.

    .. versionadded:: 0.22
"""  # after deprecation period, this can use _on_missing_base

# Cropping
docdict['include_tmax'] = """
include_tmax : bool
    If True (default), include tmax. If False, exclude tmax (similar to how
    Python indexing typically works).

    .. versionadded:: 0.19
"""
docdict['notes_tmax_included_by_default'] = """
Unlike Python slices, MNE time intervals by default include **both**
their end points; ``crop(tmin, tmax)`` returns the interval
``tmin <= t <= tmax``. Pass ``include_tmax=False`` to specify the half-open
interval ``tmin <= t < tmax`` instead.
"""
docdict['raw_tmin'] = """
tmin : float
    Start time of the raw data to use in seconds (must be >= 0).
"""
docdict['raw_tmax'] = """
tmax : float
    End time of the raw data to use in seconds (cannot exceed data duration).
"""

# Raw
docdict['standardize_names'] = """
standardize_names : bool
    If True, standardize MEG and EEG channel names to be
    ``"MEG ###"`` and ``"EEG ###"``. If False (default), native
    channel names in the file will be used when possible.
"""

docdict['event_color'] = """
event_color : color object | dict | None
    Color(s) to use for events. To show all events in the same color, pass any
    matplotlib-compatible color. To color events differently, pass a `dict`
    that maps event names or integer event numbers to colors (must include
    entries for *all* events, or include a "fallback" entry with key ``-1``).
    If ``None``, colors are chosen from the current Matplotlib color cycle.
"""

docdict['browse_group_by'] = """
group_by : str
    How to group channels. ``'type'`` groups by channel type,
    ``'original'`` plots in the order of ch_names, ``'selection'`` uses
    Elekta's channel groupings (only works for Neuromag data),
    ``'position'`` groups the channels by the positions of the sensors.
    ``'selection'`` and ``'position'`` modes allow custom selections by
    using a lasso selector on the topomap. In butterfly mode, ``'type'``
    and ``'original'`` group the channels by type, whereas ``'selection'``
    and ``'position'`` use regional grouping. ``'type'`` and ``'original'``
    modes are ignored when ``order`` is not ``None``. Defaults to ``'type'``.
"""


# Epochs
docdict['proj_epochs'] = """
proj : bool | 'delayed'
    Apply SSP projection vectors. If proj is 'delayed' and reject is not
    None the single epochs will be projected before the rejection
    decision, but used in unprojected state if they are kept.
    This way deciding which projection vectors are good can be postponed
    to the evoked stage without resulting in lower epoch counts and
    without producing results different from early SSP application
    given comparable parameters. Note that in this case baselining,
    detrending and temporal decimation will be postponed.
    If proj is False no projections will be applied which is the
    recommended value if SSPs are not used for cleaning the data.
"""

# Reject by annotation
docdict['reject_by_annotation_all'] = """
reject_by_annotation : bool
    Whether to omit bad segments from the data before fitting. If ``True``
    (default), annotated segments whose description begins with ``'bad'`` are
    omitted. If ``False``, no rejection based on annotations is performed.
"""
docdict['reject_by_annotation_epochs'] = """
reject_by_annotation : bool
    Whether to reject based on annotations. If ``True`` (default), epochs
    overlapping with segments whose description begins with ``'bad'`` are
    rejected. If ``False``, no rejection based on annotations is performed.
"""
docdict['reject_by_annotation_raw'] = docdict['reject_by_annotation_all'] + """
    Has no effect if ``inst`` is not a :class:`mne.io.Raw` object.
"""

# General plotting
docdict["show"] = """
show : bool
    Show figure if True.
"""
docdict["title_None"] = """
title : str | None
    Title. If None (default), no title is displayed.
"""
docdict["plot_proj"] = """
proj : bool | 'interactive' | 'reconstruct'
    If true SSP projections are applied before display. If 'interactive',
    a check box for reversible selection of SSP projection vectors will
    be shown. If 'reconstruct', projection vectors will be applied and then
    M/EEG data will be reconstructed via field mapping to reduce the signal
    bias caused by projection.

    .. versionchanged:: 0.21
       Support for 'reconstruct' was added.
"""
docdict["topomap_ch_type"] = """
ch_type : 'mag' | 'grad' | 'planar1' | 'planar2' | 'eeg' | None
    The channel type to plot. For 'grad', the gradiometers are collected in
    pairs and the RMS for each pair is plotted.
    If None, then channels are chosen in the order given above.
"""
docdict["topomap_vmin_vmax"] = """
vmin, vmax : float | callable | None
    Lower and upper bounds of the colormap, in the same units as the data.
    If ``vmin`` and ``vmax`` are both ``None``, they are set at ± the
    maximum absolute value of the data (yielding a colormap with midpoint
    at 0). If only one of ``vmin``, ``vmax`` is ``None``, will use
    ``min(data)`` or ``max(data)``, respectively. If callable, should
    accept a :class:`NumPy array <numpy.ndarray>` of data and return a
    float.
"""
docdict["topomap_cmap"] = """
cmap : matplotlib colormap | (colormap, bool) | 'interactive' | None
    Colormap to use. If tuple, the first value indicates the colormap to
    use and the second value is a boolean defining interactivity. In
    interactive mode the colors are adjustable by clicking and dragging the
    colorbar with left and right mouse button. Left mouse button moves the
    scale up and down and right mouse button adjusts the range (zoom).
    The mouse scroll can also be used to adjust the range. Hitting space
    bar resets the range. Up and down arrows can be used to change the
    colormap. If None (default), 'Reds' is used for all positive data,
    otherwise defaults to 'RdBu_r'. If 'interactive', translates to
    (None, True).

    .. warning::  Interactive mode works smoothly only for a small amount
        of topomaps. Interactive mode is disabled by default for more than
        2 topomaps.
"""
docdict["topomap_sensors"] = """
sensors : bool | str
    Add markers for sensor locations to the plot. Accepts matplotlib plot
    format string (e.g., 'r+' for red plusses). If True (default),
    circles will be used.
"""
docdict["topomap_colorbar"] = """
colorbar : bool
    Plot a colorbar in the rightmost column of the figure.
"""
docdict["topomap_scalings"] = """
scalings : dict | float | None
    The scalings of the channel types to be applied for plotting.
    If None, defaults to ``dict(eeg=1e6, grad=1e13, mag=1e15)``.
"""
docdict["topomap_units"] = """
units : dict | str | None
    The unit of the channel type used for colorbar label. If
    scale is None the unit is automatically determined.
"""
docdict["topomap_res"] = """
res : int
    The resolution of the topomap image (n pixels along each side).
"""
docdict["topomap_size"] = """
size : float
    Side length per topomap in inches.
"""
docdict["topomap_cbar_fmt"] = """
cbar_fmt : str
    String format for colorbar values.
"""
docdict["topomap_mask"] = """
mask : ndarray of bool, shape (n_channels, n_times) | None
    The channels to be marked as significant at a given time point.
    Indices set to ``True`` will be considered. Defaults to ``None``.
"""
docdict["topomap_mask_params"] = """
mask_params : dict | None
    Additional plotting parameters for plotting significant sensors.
    Default (None) equals::

        dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                linewidth=0, markersize=4)
"""
docdict['topomap_outlines'] = """
outlines : 'head' | 'skirt' | dict | None
    The outlines to be drawn. If 'head', the default head scheme will be
    drawn. If 'skirt' the head scheme will be drawn, but sensors are
    allowed to be plotted outside of the head circle. If dict, each key
    refers to a tuple of x and y positions, the values in 'mask_pos' will
    serve as image mask.
    Alternatively, a matplotlib patch object can be passed for advanced
    masking options, either directly or as a function that returns patches
    (required for multi-axis plots). If None, nothing will be drawn.
    Defaults to 'head'.
"""
docdict['topomap_contours'] = """
contours : int | array of float
    The number of contour lines to draw. If 0, no contours will be drawn.
    When an integer, matplotlib ticker locator is used to find suitable
    values for the contour thresholds (may sometimes be inaccurate, use
    array for accuracy). If an array, the values represent the levels for
    the contours. The values are in µV for EEG, fT for magnetometers and
    fT/m for gradiometers. If colorbar=True, the ticks in colorbar
    correspond to the contour levels. Defaults to 6.
"""
docdict['topomap_image_interp'] = """
image_interp : str
    The image interpolation to be used. All matplotlib options are
    accepted.
"""
docdict['topomap_average'] = """
average : float | None
    The time window around a given time to be used for averaging (seconds).
    For example, 0.01 would translate into window that starts 5 ms before
    and ends 5 ms after a given time point. Defaults to None, which means
    no averaging.
"""
docdict['topomap_axes'] = """
axes : instance of Axes | list | None
    The axes to plot to. If list, the list must be a list of Axes of the
    same length as ``times`` (unless ``times`` is None). If instance of
    Axes, ``times`` must be a float or a list of one float.
    Defaults to None.
"""
docdict['topomap_extrapolate'] = """
extrapolate : str
    Options:

    - ``'box'``
        Extrapolate to four points placed to form a square encompassing all
        data points, where each side of the square is three times the range
        of the data in the respective dimension.
    - ``'local'`` (default)
        Extrapolate only to nearby points (approximately to points closer than
        median inter-electrode distance). This will also set the
        mask to be polygonal based on the convex hull of the sensors.
    - ``'head'``
        Extrapolate out to the edges of the clipping circle. This will be on
        the head circle when the sensors are contained within the head circle,
        but it can extend beyond the head when sensors are plotted outside
        the head circle.

    .. versionchanged:: 0.21

       - The default was changed to ``'local'``
       - ``'local'`` was changed to use a convex hull mask
       - ``'head'`` was changed to extrapolate out to the clipping circle.
"""
docdict['topomap_border'] = """
border : float | 'mean'
    Value to extrapolate to on the topomap borders. If ``'mean'`` (default),
    then each extrapolated point has the average value of its neighbours.

    .. versionadded:: 0.20
"""
docdict['topomap_sphere'] = """
sphere : float | array-like | instance of ConductorModel
    The sphere parameters to use for the cartoon head.
    Can be array-like of shape (4,) to give the X/Y/Z origin and radius in
    meters, or a single float to give the radius (origin assumed 0, 0, 0).
    Can also be a spherical ConductorModel, which will use the origin and
    radius. Can also be None (default) which is an alias for %s.
    Currently the head radius does not affect plotting.

    .. versionadded:: 0.20
""" % (HEAD_SIZE_DEFAULT,)
docdict['topomap_sphere_auto'] = """
sphere : float | array-like | str | None
    The sphere parameters to use for the cartoon head.
    Can be array-like of shape (4,) to give the X/Y/Z origin and radius in
    meters, or a single float to give the radius (origin assumed 0, 0, 0).
    Can also be a spherical ConductorModel, which will use the origin and
    radius. Can be "auto" to use a digitization-based fit.
    Can also be None (default) to use 'auto' when enough extra digitization
    points are available, and %s otherwise.
    Currently the head radius does not affect plotting.

    .. versionadded:: 0.20
""" % (HEAD_SIZE_DEFAULT,)
docdict['topomap_ch_type'] = """
ch_type : str
    The channel type being plotted. Determines the ``'auto'``
    extrapolation mode.

    .. versionadded:: 0.21
"""
docdict['topomap_show_names'] = """
show_names : bool | callable
    If True, show channel names on top of the map. If a callable is
    passed, channel names will be formatted using the callable; e.g., to
    delete the prefix 'MEG ' from all channel names, pass the function
    ``lambda x: x.replace('MEG ', '')``. If ``mask`` is not None, only
    significant sensors will be shown.
"""

# PSD topomaps
docdict["psd_topo_vlim_joint"] = """
vlim : tuple of length 2 | 'joint'
    Colormap limits to use. If a :class:`tuple` of floats, specifies the
    lower and upper bounds of the colormap (in that order); providing
    ``None`` for either entry will set the corresponding boundary at the
    min/max of the data (separately for each topomap). Elements of the
    :class:`tuple` may also be callable functions which take in a
    :class:`NumPy array <numpy.ndarray>` and return a scalar.
    If ``vlim='joint'``, will compute the colormap limits jointly across
    all topomaps of the same channel type, using the min/max of the data.
    Defaults to ``(None, None)``.

    .. versionadded:: 0.21
"""
docdict['psd_topo_agg_fun'] = """
agg_fun : callable
    The function used to aggregate over frequencies. Defaults to
    :func:`numpy.sum` if ``normalize=True``, else :func:`numpy.mean`.
"""
docdict['psd_topo_dB'] = """
dB : bool
    If ``True``, transform data to decibels (with ``10 * np.log10(data)``)
    following the application of ``agg_fun``. Ignored if ``normalize=True``.
"""
docdict['psd_topo_cmap'] = """
cmap : matplotlib colormap | (colormap, bool) | 'interactive' | None
    Colormap to use. If :class:`tuple`, the first value indicates the colormap
    to use and the second value is a boolean defining interactivity. In
    interactive mode the colors are adjustable by clicking and dragging the
    colorbar with left and right mouse button. Left mouse button moves the
    scale up and down and right mouse button adjusts the range. Hitting
    space bar resets the range. Up and down arrows can be used to change
    the colormap. If ``None``, ``'Reds'`` is used for data that is either
    all-positive or all-negative, and ``'RdBu_r'`` is used otherwise.
    ``'interactive'`` is equivalent to ``(None, True)``. Defaults to ``None``.
"""
docdict['psd_topo_cbar_fmt'] = """
cbar_fmt : str
    Format string for the colorbar tick labels. If ``'auto'``, is equivalent
    to '%0.3f' if ``dB=False`` and '%0.1f' if ``dB=True``. Defaults to
    ``'auto'``.
"""
docdict['psd_topo_normalize'] = """
normalize : bool
    If True, each band will be divided by the total power. Defaults to
    False.
"""
docdict['psd_topo_bands'] = """
bands : list of tuple | None
    The frequencies or frequency ranges to plot. Length-2 tuples specify
    a single frequency and a subplot title (e.g.,
    ``(6.5, 'presentation rate')``); length-3 tuples specify lower and
    upper band edges and a subplot title. If ``None`` (the default),
    expands to::

        bands = [(0, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'),
                 (12, 30, 'Beta'), (30, 45, 'Gamma')]

    In bands where a single frequency is provided, the topomap will reflect
    the single frequency bin that is closest to the provided value.
"""
docdict['psd_topo_axes'] = """
axes : list of Axes | None
    List of axes to plot consecutive topographies to. If ``None`` the axes
    will be created automatically. Defaults to ``None``.
"""

# Picks
picks_header = 'picks : str | list | slice | None'
picks_intro = ('Channels to include. Slices and lists of integers will be '
               'interpreted as channel indices.')
_reminder = ("Note that channels in ``info['bads']`` *will be included* if "
             "their {}indices are explicitly provided.\n")
reminder = _reminder.format('names or ')
reminder_nostr = _reminder.format('')
noref = f'(excluding reference MEG channels). {reminder}'
picks_base = f"""{picks_header}
    {picks_intro} In lists, channel *type* strings
    (e.g., ``['meg', 'eeg']``) will pick channels of those
    types, channel *name* strings (e.g., ``['MEG0111', 'MEG2623']``
    will pick the given channels. Can also be the string values
    "all" to pick all channels, or "data" to pick :term:`data channels`.
    None (default) will pick"""
docdict['picks_header'] = picks_header  # these get reused as stubs in a
docdict['picks_base'] = picks_base      # couple places (e.g., BaseEpochs)
docdict['picks_all'] = f'{picks_base} all channels. {reminder}'
docdict['picks_all_data'] = f'{picks_base} all data channels. {reminder}'
docdict['picks_good_data'] = f'{picks_base} good data channels. {reminder}'
docdict['picks_all_data_noref'] = f'{picks_base} all data channels {noref}'
docdict['picks_good_data_noref'] = f'{picks_base} good data channels {noref}'
docdict['picks_nostr'] = f"""picks : list | slice | None
    {picks_intro} None (default) will pick all channels. {reminder_nostr}"""

# Filtering
docdict['l_freq'] = """
l_freq : float | None
    For FIR filters, the lower pass-band edge; for IIR filters, the lower
    cutoff frequency. If None the data are only low-passed.
"""
docdict['h_freq'] = """
h_freq : float | None
    For FIR filters, the upper pass-band edge; for IIR filters, the upper
    cutoff frequency. If None the data are only high-passed.
"""
docdict['filter_length'] = """
filter_length : str | int
    Length of the FIR filter to use (if applicable):

    * **'auto' (default)**: The filter length is chosen based
      on the size of the transition regions (6.6 times the reciprocal
      of the shortest transition band for fir_window='hamming'
      and fir_design="firwin2", and half that for "firwin").
    * **str**: A human-readable time in
      units of "s" or "ms" (e.g., "10s" or "5500ms") will be
      converted to that number of samples if ``phase="zero"``, or
      the shortest power-of-two length at least that duration for
      ``phase="zero-double"``.
    * **int**: Specified length in samples. For fir_design="firwin",
      this should not be used.
"""
docdict['filter_length_notch'] = docdict['filter_length'] + """
    When ``method=='spectrum_fit'``, this sets the effective window duration
    over which fits are computed. See :func:`mne.filter.create_filter`
    for options. Longer window lengths will give more stable frequency
    estimates, but require (potentially much) more processing and are not able
    to adapt as well to non-stationarities.

    The default in 0.21 is None, but this will change to ``'10s'`` in 0.22.
"""
docdict['l_trans_bandwidth'] = """
l_trans_bandwidth : float | str
    Width of the transition band at the low cut-off frequency in Hz
    (high pass or cutoff 1 in bandpass). Can be "auto"
    (default) to use a multiple of ``l_freq``::

        min(max(l_freq * 0.25, 2), l_freq)

    Only used for ``method='fir'``.
"""
docdict['h_trans_bandwidth'] = """
h_trans_bandwidth : float | str
    Width of the transition band at the high cut-off frequency in Hz
    (low pass or cutoff 2 in bandpass). Can be "auto"
    (default in 0.14) to use a multiple of ``h_freq``::

        min(max(h_freq * 0.25, 2.), info['sfreq'] / 2. - h_freq)

    Only used for ``method='fir'``.
"""
docdict['phase'] = """
phase : str
    Phase of the filter, only used if ``method='fir'``.
    Symmetric linear-phase FIR filters are constructed, and if ``phase='zero'``
    (default), the delay of this filter is compensated for, making it
    non-causal. If ``phase=='zero-double'``,
    then this filter is applied twice, once forward, and once backward
    (also making it non-causal). If 'minimum', then a minimum-phase filter will
    be constricted and applied, which is causal but has weaker stop-band
    suppression.

    .. versionadded:: 0.13
"""
docdict['fir_design'] = """
fir_design : str
    Can be "firwin" (default) to use :func:`scipy.signal.firwin`,
    or "firwin2" to use :func:`scipy.signal.firwin2`. "firwin" uses
    a time-domain design technique that generally gives improved
    attenuation using fewer samples than "firwin2".

    .. versionadded:: 0.15
"""
docdict['fir_window'] = """
fir_window : str
    The window to use in FIR design, can be "hamming" (default),
    "hann" (default in 0.13), or "blackman".

    .. versionadded:: 0.15
"""
docdict['pad-fir'] = """
pad : str
    The type of padding to use. Supports all :func:`numpy.pad` ``mode``
    options. Can also be "reflect_limited", which pads with a
    reflected version of each vector mirrored on the first and last
    values of the vector, followed by zeros. Only used for ``method='fir'``.
"""
docdict['method-fir'] = """
method : str
    'fir' will use overlap-add FIR filtering, 'iir' will use IIR
    forward-backward filtering (via filtfilt).
"""
docdict['n_jobs-fir'] = """
n_jobs : int | str
    Number of jobs to run in parallel. Can be 'cuda' if ``cupy``
    is installed properly and method='fir'.
"""
docdict['n_jobs-cuda'] = """
n_jobs : int | str
    Number of jobs to run in parallel. Can be 'cuda' if ``cupy``
    is installed properly.
"""
docdict['iir_params'] = """
iir_params : dict | None
    Dictionary of parameters to use for IIR filtering.
    If iir_params is None and method="iir", 4th order Butterworth will be used.
    For more information, see :func:`mne.filter.construct_iir_filter`.
"""
docdict['npad'] = """
npad : int | str
    Amount to pad the start and end of the data.
    Can also be "auto" to use a padding that will result in
    a power-of-two size (can be much faster).
"""
docdict['window-resample'] = """
window : str | tuple
    Frequency-domain window to use in resampling.
    See :func:`scipy.signal.resample`.
"""
docdict['average-psd'] = """
average : str | None
    How to average the segments. If ``mean`` (default), calculate the
    arithmetic mean. If ``median``, calculate the median, corrected for
    its bias relative to the mean. If ``None``, returns the unaggregated
    segments.
"""
docdict['window-psd'] = """
window : str | float | tuple
    Windowing function to use. See :func:`scipy.signal.get_window`.
"""
docdict['decim'] = """
decim : int
    Factor by which to subsample the data.

    .. warning:: Low-pass filtering is not performed, this simply selects
                 every Nth sample (where N is the value passed to
                 ``decim``), i.e., it compresses the signal (see Notes).
                 If the data are not properly filtered, aliasing artifacts
                 may occur.
"""
docdict['decim_offset'] = """
offset : int
    Apply an offset to where the decimation starts relative to the
    sample corresponding to t=0. The offset is in samples at the
    current sampling rate.

    .. versionadded:: 0.12
"""
docdict['decim_notes'] = """
For historical reasons, ``decim`` / "decimation" refers to simply subselecting
samples from a given signal. This contrasts with the broader signal processing
literature, where decimation is defined as (quoting
:footcite:`OppenheimEtAl1999`, p. 172; which cites
:footcite:`CrochiereRabiner1983`):

    "... a general system for downsampling by a factor of M is the one shown
    in Figure 4.23. Such a system is called a decimator, and downsampling
    by lowpass filtering followed by compression [i.e, subselecting samples]
    has been termed decimation (Crochiere and Rabiner, 1983)."

Hence "decimation" in MNE is what is considered "compression" in the signal
processing community.

Decimation can be done multiple times. For example,
``inst.decimate(2).decimate(2)`` will be the same as
``inst.decimate(4)``.
"""

# cHPI
docdict['chpi_t_window'] = """
t_window : float
    Time window to use to estimate the amplitudes, default is
    0.2 (200 ms).
"""
docdict['chpi_ext_order'] = """
ext_order : int
    The external order for SSS-like interfence suppression.
    The SSS bases are used as projection vectors during fitting.

    .. versionchanged:: 0.20
        Added ``ext_order=1`` by default, which should improve
        detection of true HPI signals.
"""
docdict['chpi_adjust_dig'] = """
adjust_dig : bool
    If True, adjust the digitization locations used for fitting based on
    the positions localized at the start of the file.
"""
docdict['chpi_amplitudes'] = """
chpi_amplitudes : dict
    The time-varying cHPI coil amplitudes, with entries
    "times", "proj", and "slopes".
"""
docdict['chpi_locs'] = """
chpi_locs : dict
    The time-varying cHPI coils locations, with entries
    "times", "rrs", "moments", and "gofs".
"""

# EEG reference: set_eeg_reference
docdict['set_eeg_reference_ref_channels'] = """
ref_channels : list of str | str
    Can be:

    - The name(s) of the channel(s) used to construct the reference.
    - ``'average'`` to apply an average reference (default)
    - ``'REST'`` to use the Reference Electrode Standardization Technique
      infinity reference :footcite:`Yao2001`.
    - An empty list, in which case MNE will not attempt any re-referencing of
      the data
"""
docdict['set_eeg_reference_projection'] = """
projection : bool
    If ``ref_channels='average'`` this argument specifies if the
    average reference should be computed as a projection (True) or not
    (False; default). If ``projection=True``, the average reference is
    added as a projection and is not applied to the data (it can be
    applied afterwards with the ``apply_proj`` method). If
    ``projection=False``, the average reference is directly applied to
    the data. If ``ref_channels`` is not ``'average'``, ``projection``
    must be set to ``False`` (the default in this case).
"""
docdict['set_eeg_reference_ch_type'] = """
ch_type : 'auto' | 'eeg' | 'ecog' | 'seeg'
    The name of the channel type to apply the reference to. If 'auto',
    the first channel type of eeg, ecog or seeg that is found (in that
    order) will be selected.

    .. versionadded:: 0.19
"""
docdict['set_eeg_reference_forward'] = """
forward : instance of Forward | None
    Forward solution to use. Only used with ``ref_channels='REST'``.

    .. versionadded:: 0.21
"""
docdict['set_eeg_reference_see_also_notes'] = """
See Also
--------
mne.set_bipolar_reference : Convenience function for creating bipolar
                        references.

Notes
-----
Some common referencing schemes and the corresponding value for the
``ref_channels`` parameter:

- Average reference:
    A new virtual reference electrode is created by averaging the current
    EEG signal by setting ``ref_channels='average'``. Bad EEG channels are
    automatically excluded if they are properly set in ``info['bads']``.

- A single electrode:
    Set ``ref_channels`` to a list containing the name of the channel that
    will act as the new reference, for example ``ref_channels=['Cz']``.

- The mean of multiple electrodes:
    A new virtual reference electrode is created by computing the average
    of the current EEG signal recorded from two or more selected channels.
    Set ``ref_channels`` to a list of channel names, indicating which
    channels to use. For example, to apply an average mastoid reference,
    when using the 10-20 naming scheme, set ``ref_channels=['M1', 'M2']``.

- REST
    The given EEG electrodes are referenced to a point at infinity using the
    lead fields in ``forward``, which helps standardize the signals.

1. If a reference is requested that is not the average reference, this
   function removes any pre-existing average reference projections.

2. During source localization, the EEG signal should have an average
   reference.

3. In order to apply a reference, the data must be preloaded. This is not
   necessary if ``ref_channels='average'`` and ``projection=True``.

4. For an average or REST reference, bad EEG channels are automatically
   excluded if they are properly set in ``info['bads']``.

.. versionadded:: 0.9.0

References
----------
.. footbibliography::
"""

# ICA
docdict['n_pca_components_apply'] = """
n_pca_components : int | float | None
    The number of PCA components to be kept, either absolute (int)
    or fraction of the explained variance (float). If None (default),
    the ``ica.n_pca_components`` from initialization will be used in 0.22;
    in 0.23 all components will be used.
"""

# Maxwell filtering
docdict['maxwell_origin'] = """
origin : array-like, shape (3,) | str
    Origin of internal and external multipolar moment space in meters.
    The default is ``'auto'``, which means ``(0., 0., 0.)`` when
    ``coord_frame='meg'``, and a head-digitization-based
    origin fit using :func:`~mne.bem.fit_sphere_to_headshape`
    when ``coord_frame='head'``. If automatic fitting fails (e.g., due
    to having too few digitization points),
    consider separately calling the fitting function with different
    options or specifying the origin manually.
"""
docdict['maxwell_int'] = """
int_order : int
    Order of internal component of spherical expansion.
"""
docdict['maxwell_ext'] = """
ext_order : int
    Order of external component of spherical expansion.
"""
docdict['maxwell_cal'] = """
calibration : str | None
    Path to the ``'.dat'`` file with fine calibration coefficients.
    File can have 1D or 3D gradiometer imbalance correction.
    This file is machine/site-specific.
"""
docdict['maxwell_cross'] = """
cross_talk : str | None
    Path to the FIF file with cross-talk correction information.
"""
docdict['maxwell_coord'] = """
coord_frame : str
    The coordinate frame that the ``origin`` is specified in, either
    ``'meg'`` or ``'head'``. For empty-room recordings that do not have
    a head<->meg transform ``info['dev_head_t']``, the MEG coordinate
    frame should be used.
"""
docdict['maxwell_reg'] = """
regularize : str | None
    Basis regularization type, must be "in" or None.
    "in" is the same algorithm as the "-regularize in" option in
    MaxFilter™.
"""
docdict['maxwell_ref'] = """
ignore_ref : bool
    If True, do not include reference channels in compensation. This
    option should be True for KIT files, since Maxwell filtering
    with reference channels is not currently supported.
"""
docdict['maxwell_cond'] = """
bad_condition : str
    How to deal with ill-conditioned SSS matrices. Can be "error"
    (default), "warning", "info", or "ignore".
"""
docdict['maxwell_pos'] = """
head_pos : array | None
    If array, movement compensation will be performed.
    The array should be of shape (N, 10), holding the position
    parameters as returned by e.g. ``read_head_pos``.
"""
docdict['maxwell_dest'] = """
destination : str | array-like, shape (3,) | None
    The destination location for the head. Can be ``None``, which
    will not change the head position, or a string path to a FIF file
    containing a MEG device<->head transformation, or a 3-element array
    giving the coordinates to translate to (with no rotations).
    For example, ``destination=(0, 0, 0.04)`` would translate the bases
    as ``--trans default`` would in MaxFilter™ (i.e., to the default
    head location).
"""
docdict['maxwell_st_fixed_only'] = """
st_fixed : bool
    If True (default), do tSSS using the median head position during the
    ``st_duration`` window. This is the default behavior of MaxFilter
    and has been most extensively tested.

    .. versionadded:: 0.12
st_only : bool
    If True, only tSSS (temporal) projection of MEG data will be
    performed on the output data. The non-tSSS parameters (e.g.,
    ``int_order``, ``calibration``, ``head_pos``, etc.) will still be
    used to form the SSS bases used to calculate temporal projectors,
    but the output MEG data will *only* have temporal projections
    performed. Noise reduction from SSS basis multiplication,
    cross-talk cancellation, movement compensation, and so forth
    will not be applied to the data. This is useful, for example, when
    evoked movement compensation will be performed with
    :func:`~mne.epochs.average_movements`.

    .. versionadded:: 0.12
"""
docdict['maxwell_mag'] = """
mag_scale : float | str
    The magenetometer scale-factor used to bring the magnetometers
    to approximately the same order of magnitude as the gradiometers
    (default 100.), as they have different units (T vs T/m).
    Can be ``'auto'`` to use the reciprocal of the physical distance
    between the gradiometer pickup loops (e.g., 0.0168 m yields
    59.5 for VectorView).
"""
docdict['maxwell_skip'] = """
skip_by_annotation : str | list of str
    If a string (or list of str), any annotation segment that begins
    with the given string will not be included in filtering, and
    segments on either side of the given excluded annotated segment
    will be filtered separately (i.e., as independent signals).
    The default ``('edge', 'bad_acq_skip')`` will separately filter
    any segments that were concatenated by :func:`mne.concatenate_raws`
    or :meth:`mne.io.Raw.append`, or separated during acquisition.
    To disable, provide an empty list.
"""
docdict['maxwell_extended'] = """
extended_proj : list
    The empty-room projection vectors used to extend the external
    SSS basis (i.e., use eSSS).

    .. versionadded:: 0.21
"""

# Rank
docdict['rank'] = """
rank : None | 'info' | 'full' | dict
    This controls the rank computation that can be read from the
    measurement info or estimated from the data.

    :data:`python:None`
        The rank will be estimated from the data after proper scaling of
        different channel types.
    ``'info'``
        The rank is inferred from ``info``. If data have been processed
        with Maxwell filtering, the Maxwell filtering header is used.
        Otherwise, the channel counts themselves are used.
        In both cases, the number of projectors is subtracted from
        the (effective) number of channels in the data.
        For example, if Maxwell filtering reduces the rank to 68, with
        two projectors the returned value will be 66.
    ``'full'``
        The rank is assumed to be full, i.e. equal to the
        number of good channels. If a `~mne.Covariance` is passed, this can
        make sense if it has been (possibly improperly) regularized without
        taking into account the true data rank.
    :class:`dict`
        Calculate the rank only for a subset of channel types, and explicitly
        specify the rank for the remaining channel types. This can be
        extremely useful if you already **know** the rank of (part of) your
        data, for instance in case you have calculated it earlier.

        This parameter must be a dictionary whose **keys** correspond to
        channel types in the data (e.g. ``'meg'``, ``'mag'``, ``'grad'``,
        ``'eeg'``), and whose **values** are integers representing the
        respective ranks. For example, ``{'mag': 90, 'eeg': 45}`` will assume
        a rank of ``90`` and ``45`` for magnetometer data and EEG data,
        respectively.

        The ranks for all channel types present in the data, but
        **not** specified in the dictionary will be estimated empirically.
        That is, if you passed a dataset containing magnetometer, gradiometer,
        and EEG data together with the dictionary from the previous example,
        only the gradiometer rank would be determined, while the specified
        magnetometer and EEG ranks would be taken for granted.
"""
docdict['rank_None'] = docdict['rank'] + "\n    The default is ``None``."
docdict['rank_info'] = docdict['rank'] + "\n    The default is ``'info'``."
docdict['rank_tol'] = """
tol : float | 'auto'
    Tolerance for singular values to consider non-zero in
    calculating the rank. The singular values are calculated
    in this method such that independent data are expected to
    have singular value around one. Can be 'auto' to use the
    same thresholding as :func:`scipy.linalg.orth`.
"""
docdict['rank_tol_kind'] = """
tol_kind : str
    Can be: "absolute" (default) or "relative". Only used if ``tol`` is a
    float, because when ``tol`` is a string the mode is implicitly relative.
    After applying the chosen scale factors / normalization to the data,
    the singular values are computed, and the rank is then taken as:

    - ``'absolute'``
        The number of singular values ``s`` greater than ``tol``.
        This mode can fail if your data do not adhere to typical
        data scalings.
    - ``'relative'``
        The number of singular values ``s`` greater than ``tol * s.max()``.
        This mode can fail if you have one or more large components in the
        data (e.g., artifacts).

    .. versionadded:: 0.21.0
"""

# Inverses
docdict['depth'] = """
depth : None | float | dict
    How to weight (or normalize) the forward using a depth prior.
    If float (default 0.8), it acts as the depth weighting exponent (``exp``)
    to use, which must be between 0 and 1. None is equivalent to 0, meaning
    no depth weighting is performed. It can also be a :class:`dict`
    containing keyword arguments to pass to
    :func:`mne.forward.compute_depth_prior` (see docstring for details and
    defaults). This is effectively ignored when ``method='eLORETA'``.

    .. versionchanged:: 0.20
       Depth bias ignored for ``method='eLORETA'``.
"""
docdict['loose'] = """
loose : float | 'auto' | dict
    Value that weights the source variances of the dipole components
    that are parallel (tangential) to the cortical surface. Can be:

    - float between 0 and 1 (inclusive)
        If 0, then the solution is computed with fixed orientation.
        If 1, it corresponds to free orientations.
    - ``'auto'`` (default)
        Uses 0.2 for surface source spaces (unless ``fixed`` is True) and
        1.0 for other source spaces (volume or mixed).
    - dict
        Mapping from the key for a given source space type (surface, volume,
        discrete) to the loose value. Useful mostly for mixed source spaces.
"""
_pick_ori_novec = """
    Options:

    - ``None``
        Pooling is performed by taking the norm of loose/free
        orientations. In case of a fixed source space no norm is computed
        leading to signed source activity.
    - ``"normal"``
        Only the normal to the cortical surface is kept. This is only
        implemented when working with loose orientations.
"""
docdict['pick_ori-novec'] = """
pick_ori : None | "normal"
""" + _pick_ori_novec
docdict['pick_ori'] = """
pick_ori : None | "normal" | "vector"
""" + _pick_ori_novec + """
    - ``"vector"``
        No pooling of the orientations is done, and the vector result
        will be returned in the form of a :class:`mne.VectorSourceEstimate`
        object.
"""
docdict['reduce_rank'] = """
reduce_rank : bool
    If True, the rank of the denominator of the beamformer formula (i.e.,
    during pseudo-inversion) will be reduced by one for each spatial location.
    Setting ``reduce_rank=True`` is typically necessary if you use a single
    sphere model with MEG data.

    .. versionchanged:: 0.20
        Support for reducing rank in all modes (previously only supported
        ``pick='max_power'`` with weight normalization).
"""
docdict['weight_norm'] = """
weight_norm : str | None
    Can be:

    - ``None``
        The unit-gain LCMV beamformer :footcite:`SekiharaNagarajan2008` will be
        computed.
    - ``'unit-noise-gain'``
        The unit-noise gain minimum variance beamformer will be computed
        (Borgiotti-Kaplan beamformer) :footcite:`SekiharaNagarajan2008`,
        which is not rotation invariant when ``pick_ori='vector'``.
        This should be combined with
        :meth:`stc.project('pca') <mne.VectorSourceEstimate.project>` to follow
        the definition in :footcite:`SekiharaNagarajan2008`.
    - ``'nai'``
        The Neural Activity Index :footcite:`VanVeenEtAl1997` will be computed,
        which simply scales all values from ``'unit-noise-gain'`` by a fixed
        value.
    - ``'unit-noise-gain-invariante'``
        Compute a rotation-invariant normalization using the matrix square
        root. This differs from ``'unit-noise-gain'`` only when
        ``pick_ori='vector'``, creating a solution that:

        1. Is rotation invariant (``'unit-noise-gain'`` is not);
        2. Satisfies the first requirement from
           :footcite:`SekiharaNagarajan2008` that ``w @ w.conj().T == I``,
           whereas ``'unit-noise-gain'`` has non-zero off-diagonals; but
        3. Does not satisfy the second requirement that ``w @ G.T = θI``,
           which arguably does not make sense for a rotation-invariant
           solution.
"""
docdict['bf_pick_ori'] = """
pick_ori : None | str
    For forward solutions with fixed orientation, None (default) must be
    used and a scalar beamformer is computed. For free-orientation forward
    solutions, a vector beamformer is computed and:

    - ``None``
        Orientations are pooled after computing a vector beamformer (Default).
    - ``'normal'``
        Filters are computed for the orientation tangential to the
        cortical surface.
    - ``'max-power'``
        Filters are computed for the orientation that maximizes power.
"""
docdict['bf_inversion'] = """
inversion : 'single' | 'matrix'
    This determines how the beamformer deals with source spaces in "free"
    orientation. Such source spaces define three orthogonal dipoles at each
    source point. When ``inversion='single'``, each dipole is considered
    as an individual source and the corresponding spatial filter is
    computed for each dipole separately. When ``inversion='matrix'``, all
    three dipoles at a source vertex are considered as a group and the
    spatial filters are computed jointly using a matrix inversion. While
    ``inversion='single'`` is more stable, ``inversion='matrix'`` is more
    precise. See section 5 of :footcite:`vanVlietEtAl2018`.
    Defaults to ``'matrix'``.
"""
docdict['use_cps'] = """
use_cps : bool
    Whether to use cortical patch statistics to define normal orientations for
    surfaces (default True).
"""
docdict['use_cps_restricted'] = docdict['use_cps'] + """
    Only used when the inverse is free orientation (``loose=1.``),
    not in surface orientation, and ``pick_ori='normal'``.
"""
docdict['pctf_mode'] = """
mode : None | 'mean' | 'max' | 'svd'
    Compute summary of PSFs/CTFs across all indices specified in 'idx'.
    Can be:

    * None : Output individual PSFs/CTFs for each specific vertex
      (Default).
    * 'mean' : Mean of PSFs/CTFs across vertices.
    * 'max' : PSFs/CTFs with maximum norm across vertices. Returns the
      n_comp largest PSFs/CTFs.
    * 'svd' : SVD components across PSFs/CTFs across vertices. Returns the
      n_comp first SVD components.
"""
docdict['pctf_idx'] = """
idx : list of int | list of Label
    Source for indices for which to compute PSFs or CTFs. If mode is None,
    PSFs/CTFs will be returned for all indices. If mode is not None, the
    corresponding summary measure will be computed across all PSFs/CTFs
    available from idx.
    Can be:

    * list of integers : Compute PSFs/CTFs for all indices to source space
      vertices specified in idx.
    * list of Label : Compute PSFs/CTFs for source space vertices in
      specified labels.
"""
docdict['pctf_n_comp'] = """
n_comp : int
    Number of PSF/CTF components to return for mode='max' or mode='svd'.
    Default n_comp=1.
"""
docdict['pctf_norm'] = """
norm : None | 'max' | 'norm'
    Whether and how to normalise the PSFs and CTFs. This will be applied
    before computing summaries as specified in 'mode'.
    Can be:

    * None : Use un-normalized PSFs/CTFs (Default).
    * 'max' : Normalize to maximum absolute value across all PSFs/CTFs.
    * 'norm' : Normalize to maximum norm across all PSFs/CTFs.
"""
docdict['pctf_return_pca_vars'] = """
return_pca_vars : bool
    Whether or not to return the explained variances across the specified
    vertices for individual SVD components. This is only valid if
    mode='svd'.
    Default return_pca_vars=False.
"""
docdict['pctf_pca_vars'] = """
pca_vars : array, shape (n_comp,) | list of array
    The explained variances of the first n_comp SVD components across the
    PSFs/CTFs for the specified vertices. Arrays for multiple labels are
    returned as list. Only returned if mode='svd' and return_pca_vars=True.
"""
docdict['pctf_stcs'] = """
stcs : instance of SourceEstimate | list of instances of SourceEstimate
    PSFs or CTFs as STC objects.
    All PSFs/CTFs will be returned as successive samples in STC objects,
    in the order they are specified in idx. STCs for different labels will
    be returned as a list.
"""

# Forward
docdict['on_missing_fwd'] = """
on_missing : str
    %s ``stc`` has vertices that are not in ``fwd``.
""" % (_on_missing_base,)
docdict['dig_kinds'] = """
dig_kinds : list of str | str
    Kind of digitization points to use in the fitting. These can be any
    combination of ('cardinal', 'hpi', 'eeg', 'extra'). Can also
    be 'auto' (default), which will use only the 'extra' points if
    enough (more than 4) are available, and if not, uses 'extra' and
    'eeg' points.
"""
docdict['exclude_frontal'] = """
exclude_frontal : bool
    If True, exclude points that have both negative Z values
    (below the nasion) and positivy Y values (in front of the LPA/RPA).
"""
_trans_base = """\
If str, the path to the head<->MRI transform ``*-trans.fif`` file produced
    during coregistration. Can also be ``'fsaverage'`` to use the built-in
    fsaverage transformation."""
docdict['trans_not_none'] = """
trans : str | dict | instance of Transform
    %s
""" % (_trans_base,)
docdict['trans_deprecated'] = """
trans : str | dict | instance of Transform
    Deprecated and will be removed in 0.23, do not pass this argument.
"""
docdict['trans'] = """
trans : str | dict | instance of Transform | None
    %s
    If trans is None, an identity matrix is assumed.

    .. versionchanged:: 0.19
       Support for 'fsaverage' argument.
""" % (_trans_base,)
docdict['subjects_dir'] = """
subjects_dir : str | None
    The path to the FreeSurfer subjects reconstructions.
    It corresponds to FreeSurfer environment variable ``SUBJECTS_DIR``.
"""
docdict['subject'] = """
subject : str
    The FreeSurfer subject name.
"""

# Simulation
docdict['interp'] = """
interp : str
    Either 'hann', 'cos2' (default), 'linear', or 'zero', the type of
    forward-solution interpolation to use between forward solutions
    at different head positions.
"""
docdict['head_pos'] = """
head_pos : None | str | dict | tuple | array
    Name of the position estimates file. Should be in the format of
    the files produced by MaxFilter. If dict, keys should
    be the time points and entries should be 4x4 ``dev_head_t``
    matrices. If None, the original head position (from
    ``info['dev_head_t']``) will be used. If tuple, should have the
    same format as data returned by ``head_pos_to_trans_rot_t``.
    If array, should be of the form returned by
    :func:`mne.chpi.read_head_pos`.
"""
docdict['n_jobs'] = """
n_jobs : int
    The number of jobs to run in parallel (default 1).
    Requires the joblib package.
"""

# Random state
docdict['random_state'] = """
random_state : None | int | instance of ~numpy.random.RandomState
    If ``random_state`` is an :class:`int`, it will be used as a seed for
    :class:`~numpy.random.RandomState`. If ``None``, the seed will be
    obtained from the operating system (see
    :class:`~numpy.random.RandomState` for details). Default is
    ``None``.
"""

docdict['seed'] = """
seed : None | int | instance of ~numpy.random.RandomState
    If ``seed`` is an :class:`int`, it will be used as a seed for
    :class:`~numpy.random.RandomState`. If ``None``, the seed will be
    obtained from the operating system (see
    :class:`~numpy.random.RandomState` for details). Default is
    ``None``.
"""

# Visualization
docdict['combine'] = """
combine : None | str | callable
    How to combine information across channels. If a :class:`str`, must be
    one of 'mean', 'median', 'std' (standard deviation) or 'gfp' (global
    field power).
"""

docdict['show_scrollbars'] = """
show_scrollbars : bool
    Whether to show scrollbars when the plot is initialized. Can be toggled
    after initialization by pressing :kbd:`z` ("zen mode") while the plot
    window is focused. Default is ``True``.

    .. versionadded:: 0.19.0
"""

# PSD plotting
docdict["plot_psd_doc"] = """
Plot the power spectral density across channels.

Different channel types are drawn in sub-plots. When the data have been
processed with a bandpass, lowpass or highpass filter, dashed lines (╎)
indicate the boundaries of the filter. The line noise frequency is
also indicated with a dashed line (⋮)
"""
docdict['plot_psd_picks_good_data'] = docdict['picks_good_data'][:-2] + """
    Cannot be None if ``ax`` is supplied.If both ``picks`` and ``ax`` are None
    separate subplots will be created for each standard channel type
    (``mag``, ``grad``, and ``eeg``).
"""
docdict["plot_psd_color"] = """
color : str | tuple
    A matplotlib-compatible color to use. Has no effect when
    spatial_colors=True.
"""
docdict["plot_psd_xscale"] = """
xscale : str
    Can be 'linear' (default) or 'log'.
"""
docdict["plot_psd_area_mode"] = """
area_mode : str | None
    Mode for plotting area. If 'std', the mean +/- 1 STD (across channels)
    will be plotted. If 'range', the min and max (across channels) will be
    plotted. Bad channels will be excluded from these calculations.
    If None, no area will be plotted. If average=False, no area is plotted.
"""
docdict["plot_psd_area_alpha"] = """
area_alpha : float
    Alpha for the area.
"""
docdict["plot_psd_dB"] = """
dB : bool
    Plot Power Spectral Density (PSD), in units (amplitude**2/Hz (dB)) if
    ``dB=True``, and ``estimate='power'`` or ``estimate='auto'``. Plot PSD
    in units (amplitude**2/Hz) if ``dB=False`` and,
    ``estimate='power'``. Plot Amplitude Spectral Density (ASD), in units
    (amplitude/sqrt(Hz)), if ``dB=False`` and ``estimate='amplitude'`` or
    ``estimate='auto'``. Plot ASD, in units (amplitude/sqrt(Hz) (db)), if
    ``dB=True`` and ``estimate='amplitude'``.
"""
docdict["plot_psd_estimate"] = """
estimate : str, {'auto', 'power', 'amplitude'}
    Can be "power" for power spectral density (PSD), "amplitude" for
    amplitude spectrum density (ASD), or "auto" (default), which uses
    "power" when dB is True and "amplitude" otherwise.
"""
docdict["plot_psd_average"] = """
average : bool
    If False, the PSDs of all channels is displayed. No averaging
    is done and parameters area_mode and area_alpha are ignored. When
    False, it is possible to paint an area (hold left mouse button and
    drag) to plot a topomap.
"""
docdict["plot_psd_line_alpha"] = """
line_alpha : float | None
    Alpha for the PSD line. Can be None (default) to use 1.0 when
    ``average=True`` and 0.1 when ``average=False``.
"""
docdict["plot_psd_spatial_colors"] = """
spatial_colors : bool
    Whether to use spatial colors. Only used when ``average=False``.
"""

# plot_projs_topomap
docdict["proj_topomap_kwargs"] = """
cmap : matplotlib colormap | (colormap, bool) | 'interactive' | None
    Colormap to use. If tuple, the first value indicates the colormap to
    use and the second value is a boolean defining interactivity. In
    interactive mode (only works if ``colorbar=True``) the colors are
    adjustable by clicking and dragging the colorbar with left and right
    mouse button. Left mouse button moves the scale up and down and right
    mouse button adjusts the range. Hitting space bar resets the range. Up
    and down arrows can be used to change the colormap. If None (default),
    'Reds' is used for all positive data, otherwise defaults to 'RdBu_r'.
    If 'interactive', translates to (None, True).
sensors : bool | str
    Add markers for sensor locations to the plot. Accepts matplotlib plot
    format string (e.g., 'r+' for red plusses). If True, a circle will be
    used (via .add_artist). Defaults to True.
colorbar : bool
    Plot a colorbar.
res : int
    The resolution of the topomap image (n pixels along each side).
size : scalar
    Side length of the topomaps in inches (only applies when plotting
    multiple topomaps at a time).
show : bool
    Show figure if True.
%(topomap_outlines)s
contours : int | array of float
    The number of contour lines to draw. If 0, no contours will be drawn.
    When an integer, matplotlib ticker locator is used to find suitable
    values for the contour thresholds (may sometimes be inaccurate, use
    array for accuracy). If an array, the values represent the levels for
    the contours. Defaults to 6.
image_interp : str
    The image interpolation to be used. All matplotlib options are
    accepted.
axes : instance of Axes | list | None
    The axes to plot to. If list, the list must be a list of Axes of
    the same length as the number of projectors. If instance of Axes,
    there must be only one projector. Defaults to None.
vlim : tuple of length 2 | 'joint'
    Colormap limits to use. If :class:`tuple`, specifies the lower and
    upper bounds of the colormap (in that order); providing ``None`` for
    either of these will set the corresponding boundary at the min/max of
    the data (separately for each projector). The keyword value ``'joint'``
    will compute the colormap limits jointly across all provided
    projectors of the same channel type, using the min/max of the projector
    data. If vlim is ``'joint'``, ``info`` must not be ``None``. Defaults
    to ``(None, None)``.
""" % docdict

# Montage
docdict["montage"] = """
montage : None | str | DigMontage
    A montage containing channel positions. If str or DigMontage is
    specified, the channel info will be updated with the channel
    positions. Default is None. See also the documentation of
    :class:`mne.channels.DigMontage` for more information.
"""
docdict["match_case"] = """
match_case : bool
    If True (default), channel name matching will be case sensitive.

    .. versionadded:: 0.20
"""
docdict['on_header_missing'] = """
on_header_missing : str
    %s the FastSCAN header is missing.

    .. versionadded:: 0.22
""" % (_on_missing_base,)
docdict['on_missing_events'] = """
on_missing : str
    %s event numbers from ``event_id`` are missing from ``events``.
    When numbers from ``events`` are missing from ``event_id`` they will be
    ignored and a warning emitted; consider using ``verbose='error'`` in
    this case.

    .. versionadded:: 0.21
""" % (_on_missing_base,)
docdict['on_missing_montage'] = """
on_missing : str
    %s channels have missing coordinates.

    .. versionadded:: 0.20.1
""" % (_on_missing_base,)
docdict['rename_channels_mapping'] = """
mapping : dict | callable
    A dictionary mapping the old channel to a new channel name
    e.g. {'EEG061' : 'EEG161'}. Can also be a callable function
    that takes and returns a string.

    .. versionchanged:: 0.10.0
       Support for a callable function.
"""

# Brain plotting
docdict["clim"] = """
clim : str | dict
    Colorbar properties specification. If 'auto', set clim automatically
    based on data percentiles. If dict, should contain:

        ``kind`` : 'value' | 'percent'
            Flag to specify type of limits.
        ``lims`` : list | np.ndarray | tuple of float, 3 elements
            Lower, middle, and upper bounds for colormap.
        ``pos_lims`` : list | np.ndarray | tuple of float, 3 elements
            Lower, middle, and upper bound for colormap. Positive values
            will be mirrored directly across zero during colormap
            construction to obtain negative control points.

    .. note:: Only one of ``lims`` or ``pos_lims`` should be provided.
              Only sequential colormaps should be used with ``lims``, and
              only divergent colormaps should be used with ``pos_lims``.
"""
docdict["clim_onesided"] = """
clim : str | dict
    Colorbar properties specification. If 'auto', set clim automatically
    based on data percentiles. If dict, should contain:

        ``kind`` : 'value' | 'percent'
            Flag to specify type of limits.
        ``lims`` : list | np.ndarray | tuple of float, 3 elements
            Lower, middle, and upper bound for colormap.

    Unlike :meth:`stc.plot <mne.SourceEstimate.plot>`, it cannot use
    ``pos_lims``, as the surface plot must show the magnitude.
"""
docdict["colormap"] = """
colormap : str | np.ndarray of float, shape(n_colors, 3 | 4)
    Name of colormap to use or a custom look up table. If array, must
    be (n x 3) or (n x 4) array for with RGB or RGBA values between
    0 and 255.
"""
docdict["transparent"] = """
transparent : bool | None
    If True: use a linear transparency between fmin and fmid
    and make values below fmin fully transparent (symmetrically for
    divergent colormaps). None will choose automatically based on colormap
    type.
"""
docdict["brain_time_interpolation"] = """
interpolation : str | None
    Interpolation method (:class:`scipy.interpolate.interp1d` parameter).
    Must be one of 'linear', 'nearest', 'zero', 'slinear', 'quadratic',
    or 'cubic'.
"""
docdict["brain_screenshot_time_viewer"] = """
time_viewer : bool
    If True, include time viewer traces. Only used if
    ``time_viewer=True`` and ``separate_canvas=False``.
"""
docdict["show_traces"] = """
show_traces : bool | str | float
    If True, enable interactive picking of a point on the surface of the
    brain and plot its time course.
    This feature is only available with the PyVista 3d backend, and requires
    ``time_viewer=True``. Defaults to 'auto', which will use True if and
    only if ``time_viewer=True``, the backend is PyVista, and there is more
    than one time point. If float (between zero and one), it specifies what
    proportion of the total window should be devoted to traces (True is
    equivalent to 0.25, i.e., it will occupy the bottom 1/4 of the figure).

    .. versionadded:: 0.20.0
"""
docdict["time_label"] = """
time_label : str | callable | None
    Format of the time label (a format string, a function that maps
    floating point time values to strings, or None for no label). The
    default is ``'auto'``, which will use ``time=%0.2f ms`` if there
    is more than one time point.
"""
docdict["fmin_fmid_fmax"] = """
fmin : float
    Minimum value in colormap (uses real fmin if None).
fmid : float
    Intermediate value in colormap (fmid between fmin and
    fmax if None).
fmax : float
    Maximum value in colormap (uses real max if None).
"""
docdict["thresh"] = """
thresh : None or float
    Not supported yet.
    If not None, values below thresh will not be visible.
"""
docdict["center"] = """
center : float or None
    If not None, center of a divergent colormap, changes the meaning of
    fmin, fmax and fmid.
"""
docdict["src_volume_options"] = """
src : instance of SourceSpaces | None
    The source space corresponding to the source estimate. Only necessary
    if the STC is a volume or mixed source estimate.
volume_options : float | dict | None
    Options for volumetric source estimate plotting, with key/value pairs:

    - ``'resolution'`` : float | None
        Resolution (in mm) of volume rendering. Smaller (e.g., 1.) looks
        better at the cost of speed. None (default) uses the volume source
        space resolution, which is often something like 7 or 5 mm,
        without resampling.
    - ``'blending'`` : str
        Can be "mip" (default) for :term:`maximum intensity projection` or
        "composite" for composite blending using alpha values.
    - ``'alpha'`` : float | None
        Alpha for the volumetric rendering. Defaults are 0.4 for vector source
        estimates and 1.0 for scalar source estimates.
    - ``'surface_alpha'`` : float | None
        Alpha for the surface enclosing the volume(s). None (default) will use
        half the volume alpha. Set to zero to avoid plotting the surface.
    - ``'silhouette_alpha'`` : float | None
        Alpha for a silhouette along the outside of the volume. None (default)
        will use ``0.25 * surface_alpha``.
    - ``'silhouette_linewidth'`` : float
        The line width to use for the silhouette. Default is 2.

    A float input (default 1.) or None will be used for the ``'resolution'``
    entry.
"""
docdict['view_layout'] = """
view_layout : str
    Can be "vertical" (default) or "horizontal". When using "horizontal" mode,
    the PyVista backend must be used and hemi cannot be "split".
"""
docdict['add_data_kwargs'] = """
add_data_kwargs : dict | None
    Additional arguments to brain.add_data (e.g.,
    ``dict(time_label_size=10)``).
"""
docdict['views'] = """
views : str | list
    View to use. Can be any of::

        ['lateral', 'medial', 'rostral', 'caudal', 'dorsal', 'ventral',
         'frontal', 'parietal', 'axial', 'sagittal', 'coronal']

    Three letter abbreviations (e.g., ``'lat'``) are also supported.
    Using multiple views (list) is not supported for mpl backend.
"""

# STC label time course
docdict['eltc_labels'] = """
labels : Label | BiHemiLabel | list | tuple | str
    If using a surface or mixed source space, this should be the
    :class:`~mne.Label`'s for which to extract the time course.
    If working with whole-brain volume source estimates, this must be one of:

    - a string path to a FreeSurfer atlas for the subject (e.g., their
      'aparc.a2009s+aseg.mgz') to extract time courses for all volumes in the
      atlas
    - a two-element list or tuple, the first element being a path to an atlas,
      and the second being a list or dict of ``volume_labels`` to extract
      (see :func:`mne.setup_volume_source_space` for details).

    .. versionchanged:: 0.21.0
       Support for volume source estimates.
"""
docdict['eltc_src'] = """
src : instance of SourceSpaces
    The source spaces for the source time courses.
"""
docdict['eltc_mode'] = """
mode : str
    Extraction mode, see Notes.
"""
docdict['eltc_allow_empty'] = """
allow_empty : bool | str
    ``False`` (default) will emit an error if there are labels that have no
    vertices in the source estimate. ``True`` and ``'ignore'`` will return
    all-zero time courses for labels that do not have any vertices in the
    source estimate, and True will emit a warning while and "ignore" will
    just log a message.

    .. versionchanged:: 0.21.0
       Support for "ignore".
"""
docdict
docdict['eltc_mri_resolution'] = """
mri_resolution : bool
    If True (default), the volume source space will be upsampled to the
    original MRI resolution via trilinear interpolation before the atlas values
    are extracted. This ensnures that each atlas label will contain source
    activations. When False, only the original source space points are used,
    and some atlas labels thus may not contain any source space vertices.

    .. versionadded:: 0.21.0
"""
docdict['eltc_returns'] = """
label_tc : array | list (or generator) of array, shape (n_labels[, n_orient], n_times)
    Extracted time course for each label and source estimate.
"""  # noqa: E501
docdict['eltc_mode_notes'] = """
Valid values for ``mode`` are:

- ``'max'``
    Maximum value across vertices at each time point within each label.
- ``'mean'``
    Average across vertices at each time point within each label. Ignores
    orientation of sources for standard source estimates, which varies
    across the cortical surface, which can lead to cancellation.
    Vector source estimates are always in XYZ / RAS orientation, and are thus
    already geometrically aligned.
- ``'mean_flip'``
    Finds the dominant direction of source space normal vector orientations
    within each label, applies a sign-flip to time series at vertices whose
    orientation is more than 180° different from the dominant direction, and
    then averages across vertices at each time point within each label.
- ``'pca_flip'``
    Applies singular value decomposition to the time courses within each label,
    and uses the first right-singular vector as the representative label time
    course. This signal is scaled so that its power matches the average
    (per-vertex) power within the label, and sign-flipped by multiplying by
    ``np.sign(u @ flip)``, where ``u`` is the first left-singular vector and
    ``flip`` is the same sign-flip vector used when ``mode='mean_flip'``. This
    sign-flip ensures that extracting time courses from the same label in
    similar STCs does not result in 180° direction/phase changes.
- ``'auto'`` (default)
    Uses ``'mean_flip'`` when a standard source estimate is applied, and
    ``'mean'`` when a vector source estimate is supplied.

    .. versionadded:: 0.21
       Support for ``'auto'``, vector, and volume source estimates.

The only modes that work for vector and volume source estimates are ``'mean'``,
``'max'``, and ``'auto'``.
"""
docdict['get_peak_parameters'] = """
tmin : float | None
    The minimum point in time to be considered for peak getting.
tmax : float | None
    The maximum point in time to be considered for peak getting.
mode : {'pos', 'neg', 'abs'}
    How to deal with the sign of the data. If 'pos' only positive
    values will be considered. If 'neg' only negative values will
    be considered. If 'abs' absolute values will be considered.
    Defaults to 'abs'.
vert_as_index : bool
    Whether to return the vertex index (True) instead of of its ID
    (False, default).
time_as_index : bool
    Whether to return the time index (True) instead of the latency
    (False, default).
"""

# Clustering
docdict['clust_thresh'] = """
threshold : float | dict | None
    If numeric, vertices with data values more extreme than ``threshold`` will
    be used to form clusters. If threshold is ``None``, {} will be chosen
    automatically that corresponds to a p-value of 0.05 for the given number of
    observations (only valid when using {}). If ``threshold`` is a
    :class:`dict` (with keys ``'start'`` and ``'step'``) then threshold-free
    cluster enhancement (TFCE) will be used (see the
    :ref:`TFCE example <tfce_example>` and :footcite:`SmithNichols2009`).
"""
f_test = ('an F-threshold', 'an F-statistic')
t_test = ('a t-threshold', 'a t-statistic')
docdict['clust_thresh_f'] = docdict['clust_thresh'].format(*f_test)
docdict['clust_thresh_t'] = docdict['clust_thresh'].format(*t_test)
docdict['clust_nperm'] = """
n_permutations : int{}
    The number of permutations to compute.{}
"""
nperm_all = (" | 'all'", " Can be 'all' to perform an exact test.")
docdict['clust_nperm_all'] = docdict['clust_nperm'].format(*nperm_all)
docdict['clust_nperm_int'] = docdict['clust_nperm'].format('', '')
docdict['clust_tail'] = """
tail : int
    If tail is 1, the statistic is thresholded above threshold.
    If tail is -1, the statistic is thresholded below threshold.
    If tail is 0, the statistic is thresholded on both sides of
    the distribution.
"""
docdict['clust_stat'] = """
stat_fun : callable | None
    Function called to calculate the test statistic. Must accept 1D-array as
    input and return a 1D array. If ``None`` (the default), uses
    :func:`mne.stats.{}`.
"""
docdict['clust_stat_f'] = docdict['clust_stat'].format('f_oneway')
docdict['clust_stat_t'] = docdict['clust_stat'].format('ttest_1samp_no_p')
docdict['clust_adj'] = """
adjacency : scipy.sparse.spmatrix | None | False
    Defines adjacency between locations in the data, where "locations" can
    be spatial vertices, frequency bins, etc. If ``False``, assumes no
    adjacency (each location is treated as independent and unconnected).
    If ``None``, a regular lattice adjacency is assumed, connecting
    each {sp} location to its neighbor(s) along the last dimension
    of {{eachgrp}} ``{{x}}``{lastdim}.
    If ``adjacency`` is a matrix, it is assumed to be symmetric (only the
    upper triangular half is used) and must be square with dimension equal to
    ``{{x}}.shape[-1]`` {parone} or ``{{x}}.shape[-1] * {{x}}.shape[-2]``
    {partwo}.{memory}
"""
mem = (' If spatial adjacency is uniform in time, it is recommended to use '
       'a square matrix with dimension ``{x}.shape[-1]`` (n_vertices) to save '
       'memory and computation, and to use ``max_step`` to define the extent '
       'of temporal adjacency to consider when clustering.')
st = dict(sp='spatial', lastdim='', parone='(n_vertices)',
          partwo='(n_times * n_vertices)', memory=mem)
tf = dict(sp='', lastdim=' (or the last two dimensions if ``{x}`` is 2D)',
          parone='', partwo='', memory='')
nogroups = dict(eachgrp='', x='X')
groups = dict(eachgrp='each group ', x='X[k]')
docdict['clust_adj_st1'] = docdict['clust_adj'].format(**st).format(**nogroups)
docdict['clust_adj_stn'] = docdict['clust_adj'].format(**st).format(**groups)
docdict['clust_adj_1'] = docdict['clust_adj'].format(**tf).format(**nogroups)
docdict['clust_adj_n'] = docdict['clust_adj'].format(**tf).format(**groups)
docdict['clust_maxstep'] = """
max_step : int
    Maximum distance along the second dimension (typically this is the "time"
    axis) between samples that are considered "connected". Only used
    when ``connectivity`` has shape (n_vertices, n_vertices).
"""
docdict['clust_stepdown'] = """
step_down_p : float
    To perform a step-down-in-jumps test, pass a p-value for clusters to
    exclude from each successive iteration. Default is zero, perform no
    step-down test (since no clusters will be smaller than this value).
    Setting this to a reasonable value, e.g. 0.05, can increase sensitivity
    but costs computation time.
"""
docdict['clust_power'] = """
t_power : float
    Power to raise the statistical values (usually {}-values) by before
    summing (sign will be retained). Note that ``t_power=0`` will give a
    count of locations in each cluster, ``t_power=1`` will weight each location
    by its statistical score.
"""
docdict['clust_power_t'] = docdict['clust_power'].format('t')
docdict['clust_power_f'] = docdict['clust_power'].format('F')
docdict['clust_out'] = """
out_type : 'mask' | 'indices'
    Output format of clusters. If ``'mask'``, returns boolean arrays the same
    shape as the input data, with ``True`` values indicating locations that are
    part of a cluster. If ``'indices'``, returns a list of lists, where each
    sublist contains the indices of locations that together form a cluster.
    Note that for large datasets, ``'indices'`` may use far less memory than
    ``'mask'``. Default is ``'indices'``.
"""
docdict['clust_out_none'] = """
out_type : 'mask' | 'indices'
    Output format of clusters. If ``'mask'``, returns boolean arrays the same
    shape as the input data, with ``True`` values indicating locations that are
    part of a cluster. If ``'indices'``, returns a list of lists, where each
    sublist contains the indices of locations that together form a cluster.
    Note that for large datasets, ``'indices'`` may use far less memory than
    ``'mask'``. The default translates to ``'mask'`` in version 0.21 but will
    change to ``'indices'`` in version 0.22.
"""
docdict['clust_disjoint'] = """
check_disjoint : bool
    Whether to check if the connectivity matrix can be separated into disjoint
    sets before clustering. This may lead to faster clustering, especially if
    the second dimension of ``X`` (usually the "time" dimension) is large.
"""
docdict['clust_buffer'] = """
buffer_size : int | None
    Block size to use when computing test statistics. This can significantly
    reduce memory usage when n_jobs > 1 and memory sharing between processes is
    enabled (see :func:`mne.set_cache_dir`), because ``X`` will be shared
    between processes and each process only needs to allocate space for a small
    block of locations at a time.
"""

# DataFrames
docdict['df_index'] = """
index : {} | None
    Kind of index to use for the DataFrame. If ``None``, a sequential
    integer index (:class:`pandas.RangeIndex`) will be used. If ``'time'``, a
    :class:`pandas.Float64Index`, :class:`pandas.Int64Index`, {}or
    :class:`pandas.TimedeltaIndex` will be used
    (depending on the value of ``time_format``). {}
"""
datetime = ':class:`pandas.DatetimeIndex`, '
multiindex = ('If a list of two or more string values, a '
              ':class:`pandas.MultiIndex` will be created. ')
raw = ("'time'", datetime, '')
epo = ('str | list of str', '', multiindex)
evk = ("'time'", '', '')
docdict['df_index_raw'] = docdict['df_index'].format(*raw)
docdict['df_index_epo'] = docdict['df_index'].format(*epo)
docdict['df_index_evk'] = docdict['df_index'].format(*evk)
docdict['df_tf'] = """
time_format : str | None
    Desired time format. If ``None``, no conversion is applied, and time values
    remain as float values in seconds. If ``'ms'``, time values will be rounded
    to the nearest millisecond and converted to integers. If ``'timedelta'``,
    time values will be converted to :class:`pandas.Timedelta` values. {}
    Default is ``'ms'`` in version 0.22, and will change to ``None`` in
    version 0.23.
"""  # XXX make sure we deal with this deprecation in 0.23
raw_tf = ("If ``'datetime'``, time values will be converted to "
          ":class:`pandas.Timestamp` values, relative to "
          "``raw.info['meas_date']`` and offset by ``raw.first_samp``. ")
docdict['df_time_format_raw'] = docdict['df_tf'].format(raw_tf)
docdict['df_time_format'] = docdict['df_tf'].format('')
docdict['df_scalings'] = """
scalings : dict | None
    Scaling factor applied to the channels picked. If ``None``, defaults to
    ``dict(eeg=1e6, mag=1e15, grad=1e13)`` — i.e., converts EEG to µV,
    magnetometers to fT, and gradiometers to fT/cm.
"""
docdict['df_copy'] = """
copy : bool
    If ``True``, data will be copied. Otherwise data may be modified in place.
    Defaults to ``True``.
"""
docdict['df_longform'] = """
long_format : bool
    If True, the DataFrame is returned in long format where each row is one
    observation of the signal at a unique combination of time point{}.
    {}Defaults to ``False``.
"""
ch_type = ('For convenience, a ``ch_type`` column is added to facilitate '
           'subsetting the resulting DataFrame. ')
raw = (' and channel', ch_type)
epo = (', channel, epoch number, and condition', ch_type)
stc = (' and vertex', '')
docdict['df_longform_raw'] = docdict['df_longform'].format(*raw)
docdict['df_longform_epo'] = docdict['df_longform'].format(*epo)
docdict['df_longform_stc'] = docdict['df_longform'].format(*stc)
docdict['df_return'] = """
df : instance of pandas.DataFrame
    A dataframe suitable for usage with other statistical/plotting/analysis
    packages.
"""

# Dipole
docdict['dipole_locs_fig_title'] = """
title : str | None
    The title of the figure if ``mode='orthoview'`` (ignored for all other
    modes). If ``None``, dipole number and its properties (amplitude,
    orientation etc.) will be shown. Defaults to ``None``.
"""

# TFRs
docdict['tfr_average'] = """
average : bool, default True
    If ``False`` return an `EpochsTFR` containing separate TFRs for each
    epoch. If ``True`` return an `AverageTFR` containing the average of all
    TFRs across epochs.

    .. note::
        Using ``average=True`` is functionally equivalent to using
        ``average=False`` followed by ``EpochsTFR.average()``, but is
        more memory efficient.

    .. versionadded:: 0.13.0
"""

# Anonymization
docdict['anonymize_info_parameters'] = """
daysback : int | None
    Number of days to subtract from all dates.
    If ``None`` (default), the acquisition date, ``info['meas_date']``,
    will be set to ``January 1ˢᵗ, 2000``. This parameter is ignored if
    ``info['meas_date']`` is ``None`` (i.e., no acquisition date has been set).
keep_his : bool
    If ``True``, ``his_id`` of ``subject_info`` will **not** be overwritten.
    Defaults to ``False``.

    .. warning:: This could mean that ``info`` is not fully
                 anonymized. Use with caution.
"""
docdict['anonymize_info_notes'] = """
Removes potentially identifying information if it exists in ``info``.
Specifically for each of the following we use:

- meas_date, file_id, meas_id
        A default value, or as specified by ``daysback``.
- subject_info
        Default values, except for 'birthday' which is adjusted
        to maintain the subject age.
- experimenter, proj_name, description
        Default strings.
- utc_offset
        ``None``.
- proj_id
        Zeros.
- proc_history
        Dates use the ``meas_date`` logic, and experimenter a default string.
- helium_info, device_info
        Dates use the ``meas_date`` logic, meta info uses defaults.

If ``info['meas_date']`` is ``None``, it will remain ``None`` during processing
the above fields.

Operates in place.
"""

# Baseline
docdict['rescale_baseline'] = """
baseline : None | tuple of length 2
    The time interval to consider as "baseline" when applying baseline
    correction. If ``None``, do not apply baseline correction.
    If a tuple ``(a, b)``, the interval is between ``a`` and ``b``
    (in seconds), including the endpoints.
    If ``a`` is ``None``, the **beginning** of the data is used; and if ``b``
    is ``None``, it is set to the **end** of the interval.
    If ``(None, None)``, the entire time interval is used.

    .. note:: The baseline ``(a, b)`` includes both endpoints, i.e. all
                timepoints ``t`` such that ``a <= t <= b``.
"""
docdict['baseline_epochs'] = """%(rescale_baseline)s
    Correction is applied **to each epoch and channel individually** in the
    following way:

    1. Calculate the mean signal of the baseline period.
    2. Subtract this mean from the **entire** epoch.

""" % docdict
docdict['baseline_evoked'] = """%(rescale_baseline)s
    Correction is applied **to each channel individually** in the following
    way:

    1. Calculate the mean signal of the baseline period.
    2. Subtract this mean from the **entire** ``Evoked``.

""" % docdict
docdict['baseline_stc'] = """%(rescale_baseline)s
    Correction is applied **to each source individually** in the following
    way:

    1. Calculate the mean signal of the baseline period.
    2. Subtract this mean from the **entire** source estimate data.

    .. note:: Baseline correction is appropriate when signal and noise are
              approximately additive, and the noise level can be estimated from
              the baseline interval. This can be the case for non-normalized
              source activities (e.g. signed and unsigned MNE), but it is not
              the case for normalized estimates (e.g. signal-to-noise ratios,
              dSPM, sLORETA).

""" % docdict
docdict['baseline_report'] = """%(rescale_baseline)s
    Correction is applied in the following way **to each channel:**

    1. Calculate the mean signal of the baseline period.
    2. Subtract this mean from the **entire** time period.

    For `~mne.Epochs`, this algorithm is run **on each epoch individually.**
""" % docdict

# Epochs
reject_common = """
    Reject epochs based on peak-to-peak signal amplitude (PTP), i.e. the
    absolute difference between the lowest and the highest signal value. In
    each individual epoch, the PTP is calculated for every channel. If the
    PTP of any one channel exceeds the rejection threshold, the respective
    epoch will be dropped.

    The dictionary keys correspond to the different channel types; valid
    keys are: ``'grad'``, ``'mag'``, ``'eeg'``, ``'eog'``, and ``'ecg'``.

    Example::

        reject = dict(grad=4000e-13,  # unit: T / m (gradiometers)
                      mag=4e-12,      # unit: T (magnetometers)
                      eeg=40e-6,      # unit: V (EEG channels)
                      eog=250e-6      # unit: V (EOG channels)
                      )

    .. note:: Since rejection is based on a signal **difference**
              calculated for each channel separately, applying baseline
              correction does not affect the rejection procedure, as the
              difference will be preserved.
"""
docdict['reject_epochs'] = f"""
reject : dict | None
{reject_common}
    If ``reject`` is ``None`` (default), no rejection is performed.
"""
docdict['reject_drop_bad'] = f"""
reject : dict | str | None
{reject_common}
    If ``reject`` is ``None``, no rejection is performed. If ``'existing'``
    (default), then the rejection parameters set at instantiation are used.
"""
flat_common = """
    Rejection parameters based on flatness of signal.
    Valid **keys** are ``'grad'``, ``'mag'``, ``'eeg'``, ``'eog'``, ``'ecg'``.
    The **values** are floats that set the minimum acceptable peak-to-peak
    amplitude (PTP). If the PTP is smaller than this threshold, the epoch will
    be dropped. If ``None`` then no rejection is performed based on flatness
    of the signal."""
docdict['flat'] = f"""
flat : dict | None
{flat_common}
"""
docdict['flat_drop_bad'] = f"""
flat : dict | str | None
{flat_common}
    If ``'existing'``, then the flat parameters set during epoch creation are
    used.
"""

# ECG detection
docdict['ecg_event_id'] = """
event_id : int
    The index to assign to found ECG events.
"""
docdict['ecg_ch_name'] = """
ch_name : None | str
    The name of the channel to use for ECG peak detection.
    If ``None`` (default), ECG channel is used if present. If ``None`` and
    **no** ECG channel is present, a synthetic ECG channel is created from
    the cross-channel average. This synthetic channel can only be created from
    MEG channels.
"""
docdict['ecg_filter_freqs'] = """
l_freq : float
    Low pass frequency to apply to the ECG channel while finding events.
h_freq : float
    High pass frequency to apply to the ECG channel while finding events.
"""
docdict['ecg_filter_length'] = """
filter_length : str | int | None
    Number of taps to use for filtering.
"""
docdict['ecg_tstart'] = """
tstart : float
    Start ECG detection after ``tstart`` seconds. Useful when the beginning
    of the run is noisy.
"""
docdict['create_ecg_epochs'] = """This function will:

#. Filter the ECG data channel.

#. Find ECG R wave peaks using :func:`mne.preprocessing.find_ecg_events`.

#. Filter the raw data.

#. Create `~mne.Epochs` around the R wave peaks, capturing the heartbeats.
"""

# EOG detection
docdict['create_eog_epochs'] = """This function will:

#. Filter the EOG data channel.

#. Find the peaks of eyeblinks in the EOG data using
   :func:`mne.preprocessing.find_eog_events`.

#. Filter the raw data.

#. Create `~mne.Epochs` around the eyeblinks.
"""

# SSP
docdict['compute_ssp'] = """This function aims to find those SSP vectors that
will project out the ``n`` most prominent signals from the data for each
specified sensor type. Consequently, if the provided input data contains high
levels of noise, the produced SSP vectors can then be used to eliminate that
noise from the data.
"""
compute_proj_common = """
#. Optionally average the `~mne.Epochs` to produce an `~mne.Evoked` if
   ``average=True`` was passed (default).

#. Calculate SSP projection vectors on that data to capture the artifacts."""
docdict['compute_proj_ecg'] = f"""%(create_ecg_epochs)s {compute_proj_common}
""" % docdict
docdict['compute_proj_eog'] = f"""%(create_eog_epochs)s {compute_proj_common}
""" % docdict

# Other
docdict['accept'] = """
accept : bool
    If True (default False), accept the license terms of this dataset.
"""

# Finalize
docdict = unindent_dict(docdict)
fill_doc = filldoc(docdict, unindent_params=False)


##############################################################################
# Utilities for docstring manipulation.

def copy_doc(source):
    """Copy the docstring from another function (decorator).

    The docstring of the source function is prepepended to the docstring of the
    function wrapped by this decorator.

    This is useful when inheriting from a class and overloading a method. This
    decorator can be used to copy the docstring of the original method.

    Parameters
    ----------
    source : function
        Function to copy the docstring from

    Returns
    -------
    wrapper : function
        The decorated function

    Examples
    --------
    >>> class A:
    ...     def m1():
    ...         '''Docstring for m1'''
    ...         pass
    >>> class B (A):
    ...     @copy_doc(A.m1)
    ...     def m1():
    ...         ''' this gets appended'''
    ...         pass
    >>> print(B.m1.__doc__)
    Docstring for m1 this gets appended
    """
    def wrapper(func):
        if source.__doc__ is None or len(source.__doc__) == 0:
            raise ValueError('Cannot copy docstring: docstring was empty.')
        doc = source.__doc__
        if func.__doc__ is not None:
            doc += func.__doc__
        func.__doc__ = doc
        return func
    return wrapper


def copy_function_doc_to_method_doc(source):
    """Use the docstring from a function as docstring for a method.

    The docstring of the source function is prepepended to the docstring of the
    function wrapped by this decorator. Additionally, the first parameter
    specified in the docstring of the source function is removed in the new
    docstring.

    This decorator is useful when implementing a method that just calls a
    function.  This pattern is prevalent in for example the plotting functions
    of MNE.

    Parameters
    ----------
    source : function
        Function to copy the docstring from.

    Returns
    -------
    wrapper : function
        The decorated method.

    Notes
    -----
    The parsing performed is very basic and will break easily on docstrings
    that are not formatted exactly according to the ``numpydoc`` standard.
    Always inspect the resulting docstring when using this decorator.

    Examples
    --------
    >>> def plot_function(object, a, b):
    ...     '''Docstring for plotting function.
    ...
    ...     Parameters
    ...     ----------
    ...     object : instance of object
    ...         The object to plot
    ...     a : int
    ...         Some parameter
    ...     b : int
    ...         Some parameter
    ...     '''
    ...     pass
    ...
    >>> class A:
    ...     @copy_function_doc_to_method_doc(plot_function)
    ...     def plot(self, a, b):
    ...         '''
    ...         Notes
    ...         -----
    ...         .. versionadded:: 0.13.0
    ...         '''
    ...         plot_function(self, a, b)
    >>> print(A.plot.__doc__)
    Docstring for plotting function.
    <BLANKLINE>
        Parameters
        ----------
        a : int
            Some parameter
        b : int
            Some parameter
    <BLANKLINE>
            Notes
            -----
            .. versionadded:: 0.13.0
    <BLANKLINE>
    """
    def wrapper(func):
        doc = source.__doc__.split('\n')
        if len(doc) == 1:
            doc = doc[0]
            if func.__doc__ is not None:
                doc += func.__doc__
            func.__doc__ = doc
            return func

        # Find parameter block
        for line, text in enumerate(doc[:-2]):
            if (text.strip() == 'Parameters' and
                    doc[line + 1].strip() == '----------'):
                parameter_block = line
                break
        else:
            # No parameter block found
            raise ValueError('Cannot copy function docstring: no parameter '
                             'block found. To simply copy the docstring, use '
                             'the @copy_doc decorator instead.')

        # Find first parameter
        for line, text in enumerate(doc[parameter_block:], parameter_block):
            if ':' in text:
                first_parameter = line
                parameter_indentation = len(text) - len(text.lstrip(' '))
                break
        else:
            raise ValueError('Cannot copy function docstring: no parameters '
                             'found. To simply copy the docstring, use the '
                             '@copy_doc decorator instead.')

        # Find end of first parameter
        for line, text in enumerate(doc[first_parameter + 1:],
                                    first_parameter + 1):
            # Ignore empty lines
            if len(text.strip()) == 0:
                continue

            line_indentation = len(text) - len(text.lstrip(' '))
            if line_indentation <= parameter_indentation:
                # Reach end of first parameter
                first_parameter_end = line

                # Of only one parameter is defined, remove the Parameters
                # heading as well
                if ':' not in text:
                    first_parameter = parameter_block

                break
        else:
            # End of docstring reached
            first_parameter_end = line
            first_parameter = parameter_block

        # Copy the docstring, but remove the first parameter
        doc = ('\n'.join(doc[:first_parameter]) + '\n' +
               '\n'.join(doc[first_parameter_end:]))
        if func.__doc__ is not None:
            doc += func.__doc__
        func.__doc__ = doc
        return func
    return wrapper


def copy_base_doc_to_subclass_doc(subclass):
    """Use the docstring from a parent class methods in derived class.

    The docstring of a parent class method is prepended to the
    docstring of the method of the class wrapped by this decorator.

    Parameters
    ----------
    subclass : wrapped class
        Class to copy the docstring to.

    Returns
    -------
    subclass : Derived class
        The decorated class with copied docstrings.
    """
    ancestors = subclass.mro()[1:-1]

    for source in ancestors:
        methodList = [method for method in dir(source)
                      if callable(getattr(source, method))]
        for method_name in methodList:
            # discard private methods
            if method_name[0] == '_':
                continue
            base_method = getattr(source, method_name)
            sub_method = getattr(subclass, method_name)
            if base_method is not None and sub_method is not None:
                doc = base_method.__doc__
                if sub_method.__doc__ is not None:
                    doc += '\n' + sub_method.__doc__
                sub_method.__doc__ = doc

    return subclass


def linkcode_resolve(domain, info):
    """Determine the URL corresponding to a Python object.

    Parameters
    ----------
    domain : str
        Only useful when 'py'.
    info : dict
        With keys "module" and "fullname".

    Returns
    -------
    url : str
        The code URL.

    Notes
    -----
    This has been adapted to deal with our "verbose" decorator.

    Adapted from SciPy (doc/source/conf.py).
    """
    import mne
    if domain != 'py':
        return None

    modname = info['module']
    fullname = info['fullname']

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split('.'):
        try:
            obj = getattr(obj, part)
        except Exception:
            return None
    # deal with our decorators properly
    while hasattr(obj, '__wrapped__'):
        obj = obj.__wrapped__

    try:
        fn = inspect.getsourcefile(obj)
    except Exception:
        fn = None
    if not fn:
        try:
            fn = inspect.getsourcefile(sys.modules[obj.__module__])
        except Exception:
            fn = None
    if not fn:
        return None
    fn = op.relpath(fn, start=op.dirname(mne.__file__))
    fn = '/'.join(op.normpath(fn).split(os.sep))  # in case on Windows

    try:
        source, lineno = inspect.getsourcelines(obj)
    except Exception:
        lineno = None

    if lineno:
        linespec = "#L%d-L%d" % (lineno, lineno + len(source) - 1)
    else:
        linespec = ""

    if 'dev' in mne.__version__:
        kind = 'main'
    else:
        kind = 'maint/%s' % ('.'.join(mne.__version__.split('.')[:2]))
    return "http://github.com/mne-tools/mne-python/blob/%s/mne/%s%s" % (
        kind, fn, linespec)


def open_docs(kind=None, version=None):
    """Launch a new web browser tab with the MNE documentation.

    Parameters
    ----------
    kind : str | None
        Can be "api" (default), "tutorials", or "examples".
        The default can be changed by setting the configuration value
        MNE_DOCS_KIND.
    version : str | None
        Can be "stable" (default) or "dev".
        The default can be changed by setting the configuration value
        MNE_DOCS_VERSION.
    """
    if kind is None:
        kind = get_config('MNE_DOCS_KIND', 'api')
    help_dict = dict(api='python_reference.html', tutorials='tutorials.html',
                     examples='auto_examples/index.html')
    _check_option('kind', kind, sorted(help_dict.keys()))
    kind = help_dict[kind]
    if version is None:
        version = get_config('MNE_DOCS_VERSION', 'stable')
    _check_option('version', version, ['stable', 'dev'])
    webbrowser.open_new_tab('https://mne.tools/%s/%s' % (version, kind))


# Following deprecated class copied from scikit-learn

# force show of DeprecationWarning even on python 2.7
warnings.filterwarnings('always', category=DeprecationWarning, module='mne')


class deprecated(object):
    """Mark a function or class as deprecated (decorator).

    Issue a warning when the function is called/the class is instantiated and
    adds a warning to the docstring.

    The optional extra argument will be appended to the deprecation message
    and the docstring. Note: to use this with the default value for extra, put
    in an empty of parentheses::

        >>> from mne.utils import deprecated
        >>> deprecated() # doctest: +ELLIPSIS
        <mne.utils.docs.deprecated object at ...>

        >>> @deprecated()
        ... def some_function(): pass


    Parameters
    ----------
    extra: string
        To be added to the deprecation messages.
    """

    # Adapted from http://wiki.python.org/moin/PythonDecoratorLibrary,
    # but with many changes.

    def __init__(self, extra=''):  # noqa: D102
        self.extra = extra

    def __call__(self, obj):  # noqa: D105
        """Call.

        Parameters
        ----------
        obj : object
            Object to call.
        """
        if isinstance(obj, type):
            return self._decorate_class(obj)
        else:
            return self._decorate_fun(obj)

    def _decorate_class(self, cls):
        msg = "Class %s is deprecated" % cls.__name__
        if self.extra:
            msg += "; %s" % self.extra

        # FIXME: we should probably reset __new__ for full generality
        init = cls.__init__

        def deprecation_wrapped(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning)
            return init(*args, **kwargs)
        cls.__init__ = deprecation_wrapped

        deprecation_wrapped.__name__ = '__init__'
        deprecation_wrapped.__doc__ = self._update_doc(init.__doc__)
        deprecation_wrapped.deprecated_original = init

        return cls

    def _decorate_fun(self, fun):
        """Decorate function fun."""
        msg = "Function %s is deprecated" % fun.__name__
        if self.extra:
            msg += "; %s" % self.extra

        def deprecation_wrapped(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning)
            return fun(*args, **kwargs)

        deprecation_wrapped.__name__ = fun.__name__
        deprecation_wrapped.__dict__ = fun.__dict__
        deprecation_wrapped.__doc__ = self._update_doc(fun.__doc__)

        return deprecation_wrapped

    def _update_doc(self, olddoc):
        newdoc = ".. warning:: DEPRECATED"
        if self.extra:
            newdoc = "%s: %s" % (newdoc, self.extra)
        if olddoc:
            # Get the spacing right to avoid sphinx warnings
            n_space = 4
            for li, line in enumerate(olddoc.split('\n')):
                if li > 0 and len(line.strip()):
                    n_space = len(line) - len(line.lstrip())
                    break
            newdoc = "%s\n\n%s%s" % (newdoc, ' ' * n_space, olddoc)

        return newdoc


def deprecated_alias(dep_name, func, removed_in=None):
    """Inject a deprecated alias into the namespace."""
    if removed_in is None:
        from .._version import __version__
        removed_in = __version__.split('.')[:2]
        removed_in[1] = str(int(removed_in[1]) + 1)
        removed_in = '.'.join(removed_in)
    # Inject a deprecated version into the namespace
    inspect.currentframe().f_back.f_globals[dep_name] = deprecated(
        f'{dep_name} has been deprecated in favor of {func.__name__} and will '
        f'be removed in {removed_in}'
    )(deepcopy(func))
