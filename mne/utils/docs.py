# -*- coding: utf-8 -*-
"""The documentation functions."""
# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause

from copy import deepcopy
import inspect
import os
import os.path as op
import sys
import warnings
import webbrowser

from ..defaults import HEAD_SIZE_DEFAULT
from ..externals.doccer import indentcount_lines
from ..externals.decorator import FunctionMaker


##############################################################################
# Define our standard documentation entries

docdict = dict()

# Verbose
docdict['verbose'] = """
verbose : bool | str | int | None
    Control verbosity of the logging output. If ``None``, use the default
    verbosity level. See the :ref:`logging documentation <tut-logging>` and
    :func:`mne.verbose` for details. Should only be passed as a keyword
    argument."""
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
    %s split file is missing.

    .. versionadded:: 0.22
""" % (_on_missing_base,)
docdict['on_info_mismatch'] = f"""
on_mismatch : 'raise' | 'warn' | 'ignore'
    {_on_missing_base} the device-to-head transformation differs between
    instances.

    .. versionadded:: 0.24
"""
docdict['saturated'] = """\
saturated : str
    Replace saturated segments of data with NaNs, can be:

    ``"ignore"``
        The measured data is returned, even if it contains measurements
        while the amplifier was saturated.
    ``"nan"``
        The returned data will contain NaNs during time segments
        when the amplifier was saturated.
    ``"annotate"`` (default)
        The returned data will contain annotations specifying
        sections the saturate segments.

    This argument will only be used if there is no .nosatflags file
    (only if a NIRSport device is used and saturation occurred).

    .. versionadded:: 0.24
"""
docdict['nirx_notes'] = """\
This function has only been tested with NIRScout and NIRSport1 devices.

The NIRSport device can detect if the amplifier is saturated.
Starting from NIRStar 14.2, those saturated values are replaced by NaNs
in the standard .wlX files.
The raw unmodified measured values are stored in another file
called .nosatflags_wlX. As NaN values can cause unexpected behaviour with
mathematical functions the default behaviour is to return the
saturated data.
"""
docdict['hitachi_notes'] = """\
Hitachi does not encode their channel positions, so you will need to
create a suitable mapping using :func:`mne.channels.make_standard_montage`
or :func:`mne.channels.make_dig_montage` like (for a 3x5/ETG-7000 example):

>>> mon = mne.channels.make_standard_montage('standard_1020')
>>> need = 'S1 D1 S2 D2 S3 D3 S4 D4 S5 D5 S6 D6 S7 D7 S8'.split()
>>> have = 'F3 FC3 C3 CP3 P3 F5 FC5 C5 CP5 P5 F7 FT7 T7 TP7 P7'.split()
>>> mon.rename_channels(dict(zip(have, need)))
>>> raw.set_montage(mon)  # doctest: +SKIP

The 3x3 (ETG-100) is laid out as two separate layouts::

    S1--D1--S2    S6--D6--S7
    |   |   |     |   |   |
    D2--S3--D3    D7--S8--D8
    |   |   |     |   |   |
    S4--D4--S5    S9--D9--S10

The 3x5 (ETG-7000) is laid out as::

    S1--D1--S2--D2--S3
    |   |   |   |   |
    D3--S4--D4--S5--D5
    |   |   |   |   |
    S6--D6--S7--D7--S8

The 4x4 (ETG-7000) is laid out as::

    S1--D1--S2--D2
    |   |   |   |
    D3--S3--D4--S4
    |   |   |   |
    S5--D5--S6--D6
    |   |   |   |
    D7--S7--D8--S8

The 3x11 (ETG-4000) is laid out as::

    S1--D1--S2--D2--S3--D3--S4--D4--S5--D5--S6
    |   |   |   |   |   |   |   |   |   |   |
    D6--S7--D7--S8--D8--S9--D9--S10-D10-S11-D11
    |   |   |   |   |   |   |   |   |   |   |
    S12-D12-S13-D13-S14-D14-S16-D16-S17-D17-S18

For each layout, the channels come from the (left-to-right) neighboring
source-detector pairs in the first row, then between the first and second row,
then the second row, etc.

.. versionadded:: 0.24
"""

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

# raw/epochs/evoked apply_function method
# apply_function method summary
applyfun_summary = """\
The function ``fun`` is applied to the channels defined in ``picks``.
The {} object's data is modified in-place. If the function returns a different
data type (e.g. :py:obj:`numpy.complex128`) it must be specified
using the ``dtype`` parameter, which causes the data type of **all** the data
to change (even if the function is only applied to channels in ``picks``).{}

.. note:: If ``n_jobs`` > 1, more memory is required as
          ``len(picks) * n_times`` additional time points need to
          be temporarily stored in memory.
.. note:: If the data type changes (``dtype != None``), more memory is
          required since the original and the converted data needs
          to be stored in memory.
"""
applyfun_preload = (' The object has to have the data loaded e.g. with '
                    '``preload=True`` or ``self.load_data()``.')
docdict['applyfun_summary_raw'] = \
    applyfun_summary.format('raw', applyfun_preload)
docdict['applyfun_summary_epochs'] = \
    applyfun_summary.format('epochs', applyfun_preload)
docdict['applyfun_summary_evoked'] = \
    applyfun_summary.format('evoked', '')
# apply_function params: fun
applyfun_fun = """
fun : callable
    A function to be applied to the channels. The first argument of
    fun has to be a timeseries (:class:`numpy.ndarray`). The function must
    operate on an array of shape ``(n_times,)`` {}.
    The function must return an :class:`~numpy.ndarray` shaped like its input.
"""
docdict['applyfun_fun'] = applyfun_fun.format(
    ' if ``channel_wise=True`` and ``(len(picks), n_times)`` otherwise')
docdict['applyfun_fun_evoked'] = applyfun_fun.format(
    ' because it will apply channel-wise')
docdict['applyfun_dtype'] = """
dtype : numpy.dtype
    Data type to use after applying the function. If None
    (default) the data type is not modified.
"""
chwise = """
channel_wise : bool
    Whether to apply the function to each channel {}individually. If ``False``,
    the function will be applied to all {}channels at once. Default ``True``.
"""
docdict['applyfun_chwise'] = chwise.format('', '')
docdict['applyfun_chwise_epo'] = chwise.format('in each epoch ', 'epochs and ')
docdict['kwarg_fun'] = """
**kwargs : dict
    Additional keyword arguments to pass to ``fun``.
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
docdict['epochs_fname'] = """
fname : path-like | file-like
    The epochs to load. If a filename, should end with ``-epo.fif`` or
    ``-epo.fif.gz``. If a file-like object, preloading must be used.
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
docdict['annot_ch_names'] = """
ch_names : list | None
    List of lists of channel names associated with the annotations.
    Empty entries are assumed to be associated with no specific channel,
    i.e., with all channels or with the time slice itself. None (default) is
    the same as passing all empty lists. For example, this creates three
    annotations, associating the first with the time interval itself, the
    second with two channels, and the third with a single channel::

        Annotations(onset=[0, 3, 10], duration=[1, 0.25, 0.5],
                    description=['Start', 'BAD_flux', 'BAD_noise'],
                    ch_names=[[], ['MEG0111', 'MEG2563'], ['MEG1443']])
"""

# General plotting
docdict["show"] = """
show : bool
    Show the figure if ``True``.
"""
docdict["title_None"] = """
title : str | None
    The title of the generated figure. If ``None`` (default), no title is
    displayed.
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
docdict["evoked_topomap_ch_type"] = """
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
mask_base = """
mask : ndarray of bool, shape {shape} | None
    Array indicating channel{shape_appendix} to highlight with a distinct
    plotting style{example}. Array elements set to ``True`` will be plotted
    with the parameters given in ``mask_params``. Defaults to ``None``,
    equivalent to an array of all ``False`` elements.
"""
docdict['topomap_mask'] = mask_base.format(
    shape='(n_channels,)', shape_appendix='(s)', example='')
docdict['evoked_topomap_mask'] = mask_base.format(
    shape='(n_channels, n_times)', shape_appendix='-time combinations',
    example=' (useful for, e.g. marking which channels at which times a '
            'statistical test of the data reaches significance)')
docdict['patterns_topomap_mask'] = mask_base.format(
    shape='(n_channels, n_patterns)', shape_appendix='-pattern combinations',
    example='')
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
docdict['picks_ica'] = """
picks : int | list of int | slice | None
    Indices of the ICA components to visualize.
"""

# Units
docdict['units'] = """
units : str | dict | None
    Specify the unit(s) that the data should be returned in. If
    ``None`` (default), the data is returned in the
    channel-type-specific default units, which are SI units (see
    :ref:`units` and :term:`data channels`). If a string, must be a
    sub-multiple of SI units that will be used to scale the data from
    all channels of the type associated with that unit. This only works
    if the data contains one channel type that has a unit (unitless
    channel types are left unchanged). For example if there are only
    EEG and STIM channels, ``units='uV'`` will scale EEG channels to
    micro-Volts while STIM channels will be unchanged. Finally, if a
    dictionary is provided, keys must be channel types, and values must
    be units to scale the data of that channel type to. For example
    ``dict(grad='fT/cm', mag='fT')`` will scale the corresponding types
    accordingly, but all other channel types will remain in their
    channel-type-specific default unit.
"""

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
docdict['chpi_on_missing'] = f"""
on_missing : 'raise' | 'warn' | 'ignore'
    {_on_missing_base} no cHPI information can be found. If ``'ignore'`` or
    ``'warn'``, all return values will be empty arrays or ``None``. If
    ``'raise'``, an exception will be raised.
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
ch_type : list of str | str
    The name of the channel type to apply the reference to.
    Valid channel types are ``'auto'``, ``'eeg'``, ``'ecog'``, ``'seeg'``,
    ``'dbs'``. If ``'auto'``, the first channel type of eeg, ecog, seeg or dbs
    that is found (in that order) will be selected.

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
docdict['measure'] = """
measure : 'zscore' | 'correlation'
    Which method to use for finding outliers among the components:

    - ``'zscore'`` (default) is the iterative z-scoring method. This method
      computes the z-score of the component's scores and masks the components
      with a z-score above threshold. This process is repeated until no
      supra-threshold component remains.
    - ``'correlation'`` is an absolute raw correlation threshold ranging from 0
      to 1.

    .. versionadded:: 0.21"""

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
    measurement info or estimated from the data. When a noise covariance
    is used for whitening, this should reflect the rank of that covariance,
    otherwise amplification of noise components can occur in whitening (e.g.,
    often during source localization).

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
    to use None is equivalent to 0, meaning no depth weighting is performed.
    It can also be a :class:`dict` containing keyword arguments to pass to
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
docdict['on_rank_mismatch'] = """
on_rank_mismatch : str
    If an explicit MEG value is passed, what to do when it does not match
    an empirically computed rank (only used for covariances).
    Can be 'raise' to raise an error, 'warn' (default) to emit a warning, or
    'ignore' to ignore.

    .. versionadded:: 0.23
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
on_missing : 'raise' | 'warn' | 'ignore'
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
docdict['trans'] = """
trans : str | dict | instance of Transform | None
    %s
    If trans is None, an identity matrix is assumed.

    .. versionchanged:: 0.19
       Support for 'fsaverage' argument.
""" % (_trans_base,)
docdict['subjects_dir'] = """
subjects_dir : str | pathlib.Path | None
    The path to the directory containing the FreeSurfer subjects
    reconstructions. If ``None``, defaults to the ``SUBJECTS_DIR`` environment
    variable.
"""
_info_base = ('The :class:`mne.Info` object with information about the '
              'sensors and methods of measurement.')
docdict['info_not_none'] = f"""
info : mne.Info
    {_info_base}
"""
docdict['info'] = f"""
info : mne.Info | None
    {_info_base}
"""
docdict['info_str'] = f"""
info : mne.Info | str
    {_info_base} If ``str``, then it should be a filepath to a file with
    measurement information (e.g. :class:`mne.io.Raw`).
"""
docdict['subject'] = """
subject : str
    The FreeSurfer subject name.
"""
docdict['subject_none'] = """
subject : str | None
    The FreeSurfer subject name.
"""
docdict['label_subject'] = """\
subject : str | None
    Subject which this label belongs to. Should only be specified if it is not
    specified in the label.
"""
docdict['surface'] = """\
surface : str
    The surface along which to do the computations, defaults to ``'white'``
    (the gray-white matter boundary).
"""


# Freesurfer
docdict["aseg"] = """
aseg : str
    The anatomical segmentation file. Default ``aparc+aseg``. This may
    be any anatomical segmentation file in the mri subdirectory of the
    Freesurfer subject directory.
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
    The number of jobs to run in parallel (default ``1``). If ``-1``, it is set
    to the number of CPU cores. Requires the ``joblib`` package.
"""

# Random state
random_state_common = """\
    A seed for the NumPy random number generator (RNG). If ``None`` (default),
    the seed will be  obtained from the operating system
    (see  :class:`~numpy.random.RandomState` for details), meaning it will most
    likely produce different output every time this function or method is run.
    To achieve reproducible results, pass a value here to explicitly initialize
    the RNG with a defined state.\
"""
docdict['random_state'] = f"""
random_state : None | int | instance of ~numpy.random.RandomState
{random_state_common}
"""
docdict['seed'] = f"""
seed : None | int | instance of ~numpy.random.RandomState
{random_state_common}
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

docdict['show_scalebars'] = """
show_scalebars : bool
    Whether to show scale bars when the plot is initialized. Can be toggled
    after initialization by pressing :kbd:`s` while the plot window is focused.
    Default is ``True``.
"""

docdict['time_format'] = """
time_format : 'float' | 'clock'
    Style of time labels on the horizontal axis. If ``'float'``, labels will be
    number of seconds from the start of the recording. If ``'clock'``,
    labels will show "clock time" (hours/minutes/seconds) inferred from
    ``raw.info['meas_date']``. Default is ``'float'``.

    .. versionadded:: 0.24
"""

# Visualization with pyqtgraph
docdict['precompute'] = """
precompute : bool | str
    Whether to load all data (not just the visible portion) into RAM and
    apply preprocessing (e.g., projectors) to the full data array in a separate
    processor thread, instead of window-by-window during scrolling. The default
    ``'auto'`` compares available RAM space to the expected size of the
    precomputed data, and precomputes only if enough RAM is available. ``True``
    and ``'auto'`` only work if using the PyQtGraph backend.

    .. versionadded:: 0.24
"""

docdict['use_opengl'] = """
use_opengl : bool | None
    Whether to use OpenGL when rendering the plot (requires ``pyopengl``).
    May increase performance, but effect is dependent on system CPU and
    graphics hardware. Only works if using the PyQtGraph backend. Default is
    None, which will use False unless the user configuration variable
    ``MNE_BROWSER_USE_OPENGL`` is set to ``'true'``,
    see :func:`mne.set_config`.

    .. versionadded:: 0.24
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
    positions. Default is None. For valid :class:`str` values see documentation
    of :func:`mne.channels.make_standard_montage`. See also the documentation
    of :class:`mne.channels.DigMontage` for more information.
"""
docdict["montage_types"] = """EEG/sEEG/ECoG/DBS/fNIRS"""
docdict["match_case"] = """
match_case : bool
    If True (default), channel name matching will be case sensitive.

    .. versionadded:: 0.20
"""
docdict["match_alias"] = """
match_alias : bool | dict
    Whether to use a lookup table to match unrecognized channel location names
    to their known aliases. If True, uses the mapping in
    ``mne.io.constants.CHANNEL_LOC_ALIASES``. If a :class:`dict` is passed, it
    will be used instead, and should map from non-standard channel names to
    names in the specified ``montage``. Default is ``False``.

    .. versionadded:: 0.23
"""
docdict['on_header_missing'] = """
on_header_missing : str
    %s the FastSCAN header is missing.

    .. versionadded:: 0.22
""" % (_on_missing_base,)
docdict['on_missing_events'] = """
on_missing : 'raise' | 'warn' | 'ignore'
    %s event numbers from ``event_id`` are missing from ``events``.
    When numbers from ``events`` are missing from ``event_id`` they will be
    ignored and a warning emitted; consider using ``verbose='error'`` in
    this case.

    .. versionadded:: 0.21
""" % (_on_missing_base,)
docdict['on_missing_montage'] = """
on_missing : 'raise' | 'warn' | 'ignore'
    %s channels have missing coordinates.

    .. versionadded:: 0.20.1
""" % (_on_missing_base,)
docdict['on_missing_ch_names'] = """
on_missing : 'raise' | 'warn' | 'ignore'
    %s entries in ch_names are not present in the raw instance.

    .. versionadded:: 0.23.0
""" % (_on_missing_base,)
docdict['rename_channels_mapping_duplicates'] = """
mapping : dict | callable
    A dictionary mapping the old channel to a new channel name
    e.g. {'EEG061' : 'EEG161'}. Can also be a callable function
    that takes and returns a string.

    .. versionchanged:: 0.10.0
       Support for a callable function.
allow_duplicates : bool
    If True (default False), allow duplicates, which will automatically
    be renamed with ``-N`` at the end.

    .. versionadded:: 0.22.0
"""

# Brain plotting
docdict["view"] = """
view : str
    The name of the view to show (e.g. "lateral"). Other arguments
    take precedence and modify the camera starting from the ``view``.
"""
docdict["roll"] = """
roll : float | None
    The roll of the camera rendering the view in degrees.
"""
docdict["distance"] = """
distance : float | None
    The distance from the camera rendering the view to the focalpoint
    in plot units (either m or mm).
"""
docdict["azimuth"] = """
azimuth : float
    The azimuthal angle of the camera rendering the view in degrees.
"""
docdict["elevation"] = """
elevation : float
    The The zenith angle of the camera rendering the view in degrees.
"""
docdict["focalpoint"] = """
focalpoint : tuple, shape (3,) | None
    The focal point of the camera rendering the view: (x, y, z) in
    plot units (either m or mm).
"""
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
docdict["smooth"] = """
smooth : float in [0, 1)
    The smoothing factor to be applied. Default 0 is no smoothing.
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
docdict['brain_kwargs'] = """
brain_kwargs : dict | None
    Additional arguments to the :class:`mne.viz.Brain` constructor (e.g.,
    ``dict(silhouette=True)``).
"""
docdict['views'] = """
views : str | list
    View to use. Can be any of::

        ['lateral', 'medial', 'rostral', 'caudal', 'dorsal', 'ventral',
         'frontal', 'parietal', 'axial', 'sagittal', 'coronal']

    Three letter abbreviations (e.g., ``'lat'``) are also supported.
    Using multiple views (list) is not supported for mpl backend.
"""

# Coregistration
scene_interaction_common = """\
    How interactions with the scene via an input device (e.g., mouse or
    trackpad) modify the camera position. If ``'terrain'``, one axis is
    fixed, enabling "turntable-style" rotations. If ``'trackball'``,
    movement along all axes is possible, which provides more freedom of
    movement, but you may incidentally perform unintentional rotations along
    some axes.\
"""
docdict['scene_interaction'] = f"""
interaction : 'trackball' | 'terrain'
{scene_interaction_common}
"""
docdict['scene_interaction_None'] = f"""
interaction : 'trackball' | 'terrain' | None
{scene_interaction_common}
    If ``None``, the setting stored in the MNE-Python configuration file is
    used.
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
    `mne.stats.{}`.
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
comb = ' The function `mne.stats.combine_adjacency` may be useful for 4D data.'
st = dict(sp='spatial', lastdim='', parone='(n_vertices)',
          partwo='(n_times * n_vertices)', memory=mem)
tf = dict(sp='', lastdim=' (or the last two dimensions if ``{x}`` is 2D)',
          parone='(for 3D data)', partwo='(for 4D data)', memory=comb)
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
    Output format of clusters within a list.
    If ``'mask'``, returns a list of boolean arrays,
    each with the same shape as the input data (or slices if the shape is 1D
    and adjacency is None), with ``True`` values indicating locations that are
    part of a cluster. If ``'indices'``, returns a list of tuple of ndarray,
    where each ndarray contains the indices of locations that together form the
    given cluster along the given dimension. Note that for large datasets,
    ``'indices'`` may use far less memory than ``'mask'``.
    Default is ``'indices'``.
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
docdict['report_image_format'] = """
image_format : 'png' | 'svg' | 'gif' | None
    The image format to be used for the report, can be ``'png'``,
    ``'svg'``, or ``'gif'``.
    None (default) will use the default specified during `~mne.Report`
    instantiation.
"""
docdict['report_tags'] = """
tags : collection of str
    Tags to add for later interactive filtering.
"""
docdict['report_replace'] = """
replace : bool
    If ``True``, content already present that has the same ``title`` will be
    replaced. Defaults to ``False``, which will cause duplicate entries in the
    table of contents if an entry for ``title`` already exists.
"""
docdict['report_projs'] = """
projs : bool | None
    Whether to add SSP projector plots if projectors are present in
    the data. If ``None``, use ``projs`` from `~mne.Report` creation.
"""
docdict['report_stc_plot_kwargs'] = """
stc_plot_kwargs : dict
    Dictionary of keyword arguments to pass to
    :class:`mne.SourceEstimate.plot`. Only used when plotting in 3D
    mode.
"""
docdict['topomap_kwargs'] = """
topomap_kwargs : dict | None
    Keyword arguments to pass to topomap functions (
    :func:`mne.viz.plot_evoked_topomap`, :func:`mne.viz.plot_projs_topomap`,
    etc.).
"""

# Epochs
docdict['epochs_tmin_tmax'] = """
tmin, tmax : float
    Start and end time of the epochs in seconds, relative to the time-locked
    event. Defaults to -0.2 and 0.5, respectively.
"""
docdict['epochs_reject_tmin_tmax'] = """
reject_tmin, reject_tmax : float | None
    Start and end of the time window used to reject epochs based on
    peak-to-peak (PTP) amplitudes as specified via ``reject`` and ``flat``.
    The default ``None`` corresponds to the first and last time points of the
    epochs, respectively.

    .. note:: This parameter controls the time period used in conjunction with
              both, ``reject`` and ``flat``.
"""
docdict['epochs_events_event_id'] = """
events : array of int, shape (n_events, 3)
    The events typically returned by the read_events function.
    If some events don't match the events of interest as specified
    by event_id, they will be marked as 'IGNORED' in the drop log.
event_id : int | list of int | dict | None
    The id of the event to consider. If dict,
    the keys can later be used to access associated events. Example:
    dict(auditory=1, visual=3). If int, a dict will be created with
    the id as string. If a list, all events with the IDs specified
    in the list are used. If None, all events will be used with
    and a dict is created with string integer names corresponding
    to the event id integers.
"""
docdict['epochs_preload'] = """
    Load all epochs from disk when creating the object
    or wait before accessing each epoch (more memory
    efficient but can be slower).
"""
docdict['epochs_detrend'] = """
detrend : int | None
    If 0 or 1, the data channels (MEG and EEG) will be detrended when
    loaded. 0 is a constant (DC) detrend, 1 is a linear detrend. None
    is no detrending. Note that detrending is performed before baseline
    correction. If no DC offset is preferred (zeroth order detrending),
    either turn off baseline correction, as this may introduce a DC
    shift, or set baseline correction to use the entire time interval
    (will yield equivalent results but be slower).
"""
docdict['epochs_metadata'] = """
metadata : instance of pandas.DataFrame | None
    A :class:`pandas.DataFrame` specifying metadata about each epoch.
    If given, ``len(metadata)`` must equal ``len(events)``. The DataFrame
    may only contain values of type (str | int | float | bool).
    If metadata is given, then pandas-style queries may be used to select
    subsets of data, see :meth:`mne.Epochs.__getitem__`.
    When a subset of the epochs is created in this (or any other
    supported) manner, the metadata object is subsetted accordingly, and
    the row indices will be modified to match ``epochs.selection``.

    .. versionadded:: 0.16
"""
docdict['epochs_event_repeated'] = """
event_repeated : str
    How to handle duplicates in ``events[:, 0]``. Can be ``'error'``
    (default), to raise an error, 'drop' to only retain the row occurring
    first in the ``events``, or ``'merge'`` to combine the coinciding
    events (=duplicates) into a new event (see Notes for details).

    .. versionadded:: 0.19
"""
docdict['by_event_type'] = """
by_event_type : bool
    When ``False`` (the default) all epochs are processed together and a single
    :class:`~mne.Evoked` object is returned. When ``True``, epochs are first
    grouped by event type (as specified using the ``event_id`` parameter) and a
    list is returned containing a separate :class:`~mne.Evoked` object for each
    event type. The ``.comment`` attribute is set to the label of the event
    type.

    .. versionadded:: 0.24.0
"""
_by_event_type_return_base = """\
When ``by_event_type=True`` was specified, a list is returned containing a
    separate :class:`~mne.Evoked` object for each event type. The list has the
    same order as the event types as specified in the ``event_id``
    dictionary."""
docdict['by_event_type_returns_average'] = f"""
evoked : instance of Evoked | list of Evoked
    The averaged epochs.
    {_by_event_type_return_base}
"""
docdict['by_event_type_returns_stderr'] = f"""
std_err : instance of Evoked | list of Evoked
    The standard error over epochs.
    {_by_event_type_return_base}
"""
docdict['epochs_raw'] = """
raw : Raw object
    An instance of `~mne.io.Raw`.
"""
docdict['epochs_on_missing'] = """
on_missing : 'raise' | 'warn' | 'ignore'
    What to do if one or several event ids are not found in the recording.
    Valid keys are 'raise' | 'warn' | 'ignore'
    Default is 'raise'. If on_missing is 'warn' it will proceed but
    warn, if 'ignore' it will proceed silently. Note.
    If none of the event ids are found in the data, an error will be
    automatically generated irrespective of this parameter.
"""
reject_common = """\
    Reject epochs based on **maximum** peak-to-peak signal amplitude (PTP),
    i.e. the absolute difference between the lowest and the highest signal
    value. In each individual epoch, the PTP is calculated for every channel.
    If the PTP of any one channel exceeds the rejection threshold, the
    respective epoch will be dropped.

    The dictionary keys correspond to the different channel types; valid
    **keys** can be any channel type present in the object.

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
    .. note:: To constrain the time period used for estimation of signal
              quality, pass the ``reject_tmin`` and ``reject_tmax`` parameters.

    If ``reject`` is ``None`` (default), no rejection is performed.
"""
docdict['reject_drop_bad'] = f"""
reject : dict | str | None
{reject_common}
    If ``reject`` is ``None``, no rejection is performed. If ``'existing'``
    (default), then the rejection parameters set at instantiation are used.
"""
flat_common = """\
    Reject epochs based on **minimum** peak-to-peak signal amplitude (PTP).
    Valid **keys** can be any channel type present in the object. The
    **values** are floats that set the minimum acceptable PTP. If the PTP
    is smaller than this threshold, the epoch will be dropped. If ``None``
    then no rejection is performed based on flatness of the signal."""
docdict['flat'] = f"""
flat : dict | None
{flat_common}

    .. note:: To constrain the time period used for estimation of signal
              quality, pass the ``reject_tmin`` and ``reject_tmax`` parameters.
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

ecg_epoch_or_proj = """This function will:

#. Filter the ECG data channel.

#. Find ECG R wave peaks using :func:`mne.preprocessing.find_ecg_events`.
{filter_step}
#. Create `~mne.Epochs` around the R wave peaks, capturing the heartbeats.
{extra_steps}"""

docdict['create_ecg_epochs'] = ecg_epoch_or_proj.format(filter_step='',
                                                        extra_steps='')

# EOG detection
eog_epoch_or_proj = """This function will:

#. Filter the EOG data channel.

#. Find the peaks of eyeblinks in the EOG data using
   :func:`mne.preprocessing.find_eog_events`.
{filter_step}
#. Create `~mne.Epochs` around the eyeblinks.
{extra_steps}"""

docdict['create_eog_epochs'] = eog_epoch_or_proj.format(filter_step='',
                                                        extra_steps='')

docdict['eog_ch_name'] = """
ch_name : str | list of str | None
    The name of the channel(s) to use for EOG peak detection. If a string,
    can be an arbitrary channel. This doesn't have to be a channel of
    ``eog`` type; it could, for example, also be an ordinary EEG channel
    that was placed close to the eyes, like ``Fp1`` or ``Fp2``.

    Multiple channel names can be passed as a list of strings.

    If ``None`` (default), use the channel(s) in ``raw`` with type ``eog``.
"""

# SSP
docdict['compute_ssp'] = """This function aims to find those SSP vectors that
will project out the ``n`` most prominent signals from the data for each
specified sensor type. Consequently, if the provided input data contains high
levels of noise, the produced SSP vectors can then be used to eliminate that
noise from the data.
"""
proj_filter_step = """
#. Filter the raw data.
"""

proj_extra_steps = """
#. Optionally average the `~mne.Epochs` to produce an `~mne.Evoked` if
   ``average=True`` was passed (default).

#. Calculate SSP projection vectors on that data to capture the artifacts."""

docdict['compute_proj_ecg'] = ecg_epoch_or_proj.format(
    filter_step=proj_filter_step, extra_steps=proj_extra_steps)

docdict['compute_proj_eog'] = eog_epoch_or_proj.format(
    filter_step=proj_filter_step, extra_steps=proj_extra_steps)

# BEM
docdict['on_defects'] = f"""
on_defects : 'raise' | 'warn' | 'ignore'
    What to do if the surface is found to have topological defects.
    {_on_missing_base} one or more defects are found.
    Note that a lot of computations in MNE-Python assume the surfaces to be
    topologically correct, topological defects may still make other
    computations (e.g., `mne.make_bem_model` and `mne.make_bem_solution`)
    fail irrespective of this parameter.
"""

# Export
docdict['export_warning'] = """\
.. warning::
    Since we are exporting to external formats, there's no guarantee that all
    the info will be preserved in the external format. See Notes for details.
"""
docdict['export_params_fname'] = """
fname : str
    Name of the output file.
"""
docdict['export_params_fmt'] = """
fmt : 'auto' | 'eeglab' | 'edf'
    Format of the export. Defaults to ``'auto'``, which will infer the format
    from the filename extension. See supported formats above for more
    information.
"""
docdict['export_params_physical_range'] = """
physical_range : str | tuple
    The physical range of the data. If 'auto' (default), then
    it will infer the physical min and max from the data itself,
    taking the minimum and maximum values per channel type.
    If it is a 2-tuple of minimum and maximum limit, then those
    physical ranges will be used. Only used for exporting EDF files.
"""
docdict['export_params_add_ch_type'] = """
add_ch_type : bool
    Whether to incorporate the channel type into the signal label (e.g. whether
    to store channel "Fz" as "EEG Fz"). Only used for EDF format. Default is
    ``False``.
"""
docdict['export_warning_note'] = """\
Export to external format may not preserve all the information from the
instance. To save in native MNE format (``.fif``) without information loss,
use :meth:`mne.{0}.save` instead.
Export does not apply projector(s). Unapplied projector(s) will be lost.
Consider applying projector(s) before exporting with
:meth:`mne.{0}.apply_proj`."""
docdict['export_warning_note_raw'] = \
    docdict['export_warning_note'].format('io.Raw')
docdict['export_warning_note_epochs'] = \
    docdict['export_warning_note'].format('Epochs')
docdict['export_warning_note_evoked'] = \
    docdict['export_warning_note'].format('Evoked')
docdict['export_eeglab_note'] = """
For EEGLAB exports, channel locations are expanded to full EEGLAB format.
For more details see :func:`eeglabio.utils.cart_to_eeglab`.
"""
docdict['export_edf_note'] = """
For EDF exports, only channels measured in Volts are allowed; in MNE-Python
this means channel types 'eeg', 'ecog', 'seeg', 'emg', 'eog', 'ecg', 'dbs',
'bio', and 'misc'. 'stim' channels are dropped. Although this function
supports storing channel types in the signal label (e.g. ``EEG Fz`` or
``MISC E``), other software may not support this (optional) feature of
the EDF standard.

If ``add_ch_type`` is True, then channel types are written based on what
they are currently set in MNE-Python. One should double check that all
their channels are set correctly. You can call
:attr:`raw.set_channel_types <mne.io.Raw.set_channel_types>` to set
channel types.

In addition, EDF does not support storing a montage. You will need
to store the montage separately and call :attr:`raw.set_montage()
<mne.io.Raw.set_montage>`.
"""

# Other
docdict['accept'] = """
accept : bool
    If True (default False), accept the license terms of this dataset.
"""
docdict['overwrite'] = """
overwrite : bool
    If True (default False), overwrite the destination file if it
    exists.
"""
docdict['split_naming'] = """
split_naming : 'neuromag' | 'bids'
    When splitting files, append a filename partition with the appropriate
    naming schema: for ``'neuromag'``, a split file ``fname.fif`` will be named
    ``fname.fif``, ``fname-1.fif``, ``fname-2.fif`` etc.; while for ``'bids'``,
    it will be named ``fname_split-01.fif``, ``fname_split-02.fif``, etc.
"""
docdict['ref_channels'] = """
ref_channels : str | list of str
    Name of the electrode(s) which served as the reference in the
    recording. If a name is provided, a corresponding channel is added
    and its data is set to 0. This is useful for later re-referencing.
"""

# Morphing
docdict['reg_affine'] = """
reg_affine : ndarray of float, shape (4, 4)
    The affine that registers one volume to another.
"""
docdict['sdr_morph'] = """
sdr_morph : instance of dipy.align.DiffeomorphicMap
    The class that applies the the symmetric diffeomorphic registration
    (SDR) morph.
"""
docdict['moving'] = """
moving : instance of SpatialImage
    The image to morph ("from" volume).
"""
docdict['static'] = """
static : instance of SpatialImage
    The image to align with ("to" volume).
"""
docdict['niter'] = """
niter : dict | tuple | None
    For each phase of the volume registration, ``niter`` is the number of
    iterations per successive stage of optimization. If a tuple is
    provided, it will be used for all steps (except center of mass, which does
    not iterate). It should have length 3 to
    correspond to ``sigmas=[3.0, 1.0, 0.0]`` and ``factors=[4, 2, 1]`` in
    the pipeline (see :func:`dipy.align.affine_registration
    <dipy.align._public.affine_registration>` for details).
    If a dictionary is provided, number of iterations can be set for each
    step as a key. Steps not in the dictionary will use the default value.
    The default (None) is equivalent to:

        niter=dict(translation=(100, 100, 10),
                   rigid=(100, 100, 10),
                   affine=(100, 100, 10),
                   sdr=(5, 5, 3))
"""
docdict['pipeline'] = """
pipeline : str | tuple
    The volume registration steps to perform (a ``str`` for a single step,
    or ``tuple`` for a set of sequential steps). The following steps can be
    performed, and do so by matching mutual information between the images
    (unless otherwise noted):

    ``'translation'``
        Translation.

    ``'rigid'``
        Rigid-body, i.e., rotation and translation.

    ``'affine'``
        A full affine transformation, which includes translation, rotation,
        scaling, and shear.

    ``'sdr'``
        Symmetric diffeomorphic registration :footcite:`AvantsEtAl2008`, a
        non-linear similarity-matching algorithm.

    The following string shortcuts can also be used:

    ``'all'`` (default)
        All steps will be performed above in the order above, i.e.,
        ``('translation', 'rigid', 'affine', 'sdr')``.

    ``'rigids'``
        The rigid steps (first two) will be performed, which registers
        the volume without distorting its underlying structure, i.e.,
        ``('translation', 'rigid')``. This is useful for
        example when registering images from the same subject, such as
        CT and MR images.

    ``'affines'``
        The affine steps (first three) will be performed, i.e., omitting
        the SDR step.
"""

# 3D viewing
docdict['meg'] = """
meg : str | list | bool | None
    Can be "helmet", "sensors" or "ref" to show the MEG helmet, sensors or
    reference sensors respectively, or a combination like
    ``('helmet', 'sensors')`` (same as None, default). True translates to
    ``('helmet', 'sensors', 'ref')``.
"""
docdict['eeg'] = """
eeg : bool | str | list
    String options are:

    - "original" (default; equivalent to ``True``)
        Shows EEG sensors using their digitized locations (after
        transformation to the chosen ``coord_frame``)
    - "projected"
        The EEG locations projected onto the scalp, as is done in
        forward modeling

    Can also be a list of these options, or an empty list (``[]``,
    equivalent of ``False``).
"""
docdict['fnirs'] = """
fnirs : str | list | bool | None
    Can be "channels", "pairs", "detectors", and/or "sources" to show the
    fNIRS channel locations, optode locations, or line between
    source-detector pairs, or a combination like ``('pairs', 'channels')``.
    True translates to ``('pairs',)``.
"""
docdict['ecog'] = """
ecog : bool
    If True (default), show ECoG sensors.
"""
docdict['seeg'] = """
seeg : bool
    If True (default), show sEEG electrodes.
"""
docdict['dbs'] = """
dbs : bool
    If True (default), show DBS (deep brain stimulation) electrodes.
"""

# Decoding
docdict['scoring'] = """
scoring : callable | str | None
    Score function (or loss function) with signature
    ``score_func(y, y_pred, **kwargs)``.
    Note that the "predict" method is automatically identified if scoring is
    a string (e.g. ``scoring='roc_auc'`` calls ``predict_proba``), but is
    **not**  automatically set if ``scoring`` is a callable (e.g.
    ``scoring=sklearn.metrics.roc_auc_score``).
"""
docdict['base_estimator'] = """
base_estimator : object
    The base estimator to iteratively fit on a subset of the dataset.
"""

docdict_indented = {}


def fill_doc(f):
    """Fill a docstring with docdict entries.

    Parameters
    ----------
    f : callable
        The function to fill the docstring of. Will be modified in place.

    Returns
    -------
    f : callable
        The function, potentially with an updated ``__doc__``.
    """
    docstring = f.__doc__
    if not docstring:
        return f
    lines = docstring.splitlines()
    # Find the minimum indent of the main docstring, after first line
    if len(lines) < 2:
        icount = 0
    else:
        icount = indentcount_lines(lines[1:])
    # Insert this indent to dictionary docstrings
    try:
        indented = docdict_indented[icount]
    except KeyError:
        indent = ' ' * icount
        docdict_indented[icount] = indented = {}
        for name, dstr in docdict.items():
            lines = dstr.splitlines()
            try:
                newlines = [lines[0]]
                for line in lines[1:]:
                    newlines.append(indent + line)
                indented[name] = '\n'.join(newlines)
            except IndexError:
                indented[name] = dstr
    try:
        f.__doc__ = docstring % indented
    except (TypeError, ValueError, KeyError) as exp:
        funcname = f.__name__
        funcname = docstring.split('\n')[0] if funcname is None else funcname
        raise RuntimeError('Error documenting %s:\n%s'
                           % (funcname, str(exp)))
    return f


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
    from .check import _check_option
    from .config import get_config
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


class deprecated:
    """Mark a function, class, or method as deprecated (decorator).

    Originally adapted from sklearn and
    http://wiki.python.org/moin/PythonDecoratorLibrary, then modified to make
    arguments populate properly following our verbose decorator methods based
    on externals.decorator.

    Parameters
    ----------
    extra : str
        Extra information beyond just saying the class/function/method
        is deprecated.
    """

    def __init__(self, extra=''):  # noqa: D102
        self.extra = extra

    def __call__(self, obj):  # noqa: D105
        """Call.

        Parameters
        ----------
        obj : object
            Object to call.

        Returns
        -------
        obj : object
            The modified object.
        """
        if isinstance(obj, type):
            return self._decorate_class(obj)
        else:
            return self._decorate_fun(obj)

    def _decorate_class(self, cls):
        msg = f"Class {cls.__name__} is deprecated"
        cls.__init__ = self._make_fun(cls.__init__, msg)
        return cls

    def _decorate_fun(self, fun):
        """Decorate function fun."""
        msg = f"Function {fun.__name__} is deprecated"
        return self._make_fun(fun, msg)

    def _make_fun(self, function, msg):
        if self.extra:
            msg += "; %s" % self.extra

        body = f"""\
def %(name)s(%(signature)s):\n
    import warnings
    warnings.warn({repr(msg)}, category=DeprecationWarning)
    return _function_(%(shortsignature)s)"""
        evaldict = dict(_function_=function)
        fm = FunctionMaker(
            function, None, None, None, None, function.__module__)
        attrs = dict(__wrapped__=function, __qualname__=function.__qualname__,
                     __globals__=function.__globals__)
        dep = fm.make(body, evaldict, addsource=True, **attrs)
        dep.__doc__ = self._update_doc(dep.__doc__)
        dep._deprecated_original = function
        return dep

    def _update_doc(self, olddoc):
        newdoc = ".. warning:: DEPRECATED"
        if self.extra:
            newdoc = "%s: %s" % (newdoc, self.extra)
        newdoc += '.'
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
        f'be removed in {removed_in}.'
    )(deepcopy(func))
