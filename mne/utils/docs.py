# -*- coding: utf-8 -*-
"""The documentation functions."""
# Authors: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

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
    and :ref:`Logging documentation <tut_logging>` for more)."""
docdict['verbose_meth'] = (docdict['verbose'] + ' Defaults to self.verbose.')

# Preload
docdict['preload'] = """
preload : bool or str (default False)
    Preload data into memory for data manipulation and faster indexing.
    If True, the data will be preloaded into memory (fast, requires
    large amount of memory). If preload is a string, preload is the
    file name of a memory-mapped file which is used to store the data
    on the hard drive (slower, requires less memory)."""

# Cropping
docdict['include_tmax'] = """
include_tmax : bool
    If True (default), include tmax. If False, exclude tmax (similar to how
    Python indexing typically works).

    .. versionadded:: 0.19
"""
docdict['raw_tmin'] = """
tmin : float
    Start time of the raw data to use in seconds (must be >= 0).
"""
docdict['raw_tmax'] = """
tmax : float
    End time of the raw data to use in seconds (cannot exceed data duration).
"""

# General plotting
docdict["show"] = """
show : bool
    Show figure if True."""
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
docdict['topomap_extrapolate'] = """
extrapolate : str
    Options:

    - 'box' (default)
        Extrapolate to four points placed to form a square encompassing all
        data points, where each side of the square is three times the range
        of the data in the respective dimension.
    - 'local'
        Extrapolate only to nearby points (approximately to points closer than
        median inter-electrode distance).
    - 'head'
        Extrapolate to the edges of the head circle (does not work well
        with sensors outside the head circle).
"""
docdict['topomap_border'] = """
border : float | 'mean'
    Value to extrapolate to on the topomap borders. If ``'mean'`` then each
    extrapolated point has the average value of its neighbours.

    .. versionadded:: 0.20
"""
docdict['topomap_head_pos'] = """
head_pos : dict | None
    Deprecated and will be removed in 0.21. Use ``sphere`` instead.
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
docdict['layout_dep'] = """
layout : None
    Deprecated and will be removed in 0.21. Use ``sphere`` to control
    head-sensor relationship instead.
"""

# Picks
docdict['picks_header'] = 'picks : str | list | slice | None'
docdict['picks_base'] = docdict['picks_header'] + """
    Channels to include. Slices and lists of integers will be
    interpreted as channel indices. In lists, channel *type* strings
    (e.g., ``['meg', 'eeg']``) will pick channels of those
    types, channel *name* strings (e.g., ``['MEG0111', 'MEG2623']``
    will pick the given channels. Can also be the string values
    "all" to pick all channels, or "data" to pick :term:`data channels`.
    None (default) will pick """
docdict['picks_all'] = docdict['picks_base'] + 'all channels.\n'
docdict['picks_all_data'] = docdict['picks_base'] + 'all data channels.\n'
docdict['picks_all_data_noref'] = (docdict['picks_all_data'][:-2] +
                                   '(excluding reference MEG channels).\n')
docdict['picks_good_data'] = docdict['picks_base'] + 'good data channels.\n'
docdict['picks_good_data_noref'] = (docdict['picks_good_data'][:-2] +
                                    '(excluding reference MEG channels).\n')
docdict['picks_nostr'] = """
picks : list | slice | None
    Channels to include. Slices and lists of integers will be
    interpreted as channel indices. None (default) will pick all channels.
"""

# Filtering
docdict['l_freq'] = """
l_freq : float | None
    For FIR filters, the lower pass-band edge; for IIR filters, the upper
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

1. If a reference is requested that is not the average reference, this
   function removes any pre-existing average reference projections.

2. During source localization, the EEG signal should have an average
   reference.

3. In order to apply a reference, the data must be preloaded. This is not
   necessary if ``ref_channels='average'`` and ``projection=True``.

4. For an average reference, bad EEG channels are automatically excluded if
   they are properly set in ``info['bads']``.

.. versionadded:: 0.9.0
"""


# Maxwell filtering
docdict['maxwell_origin_int_ext_calibration_cross'] = """
origin : array-like, shape (3,) | str
    Origin of internal and external multipolar moment space in meters.
    The default is ``'auto'``, which means ``(0., 0., 0.)`` when
    ``coord_frame='meg'``, and a head-digitization-based
    origin fit using :func:`~mne.bem.fit_sphere_to_headshape`
    when ``coord_frame='head'``. If automatic fitting fails (e.g., due
    to having too few digitization points),
    consider separately calling the fitting function with different
    options or specifying the origin manually.
int_order : int
    Order of internal component of spherical expansion.
ext_order : int
    Order of external component of spherical expansion.
calibration : str | None
    Path to the ``'.dat'`` file with fine calibration coefficients.
    File can have 1D or 3D gradiometer imbalance correction.
    This file is machine/site-specific.
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
docdict['maxwell_reg_ref_cond_pos'] = """
regularize : str | None
    Basis regularization type, must be "in" or None.
    "in" is the same algorithm as the "-regularize in" option in
    MaxFilter™.
ignore_ref : bool
    If True, do not include reference channels in compensation. This
    option should be True for KIT files, since Maxwell filtering
    with reference channels is not currently supported.
bad_condition : str
    How to deal with ill-conditioned SSS matrices. Can be "error"
    (default), "warning", "info", or "ignore".
head_pos : array | None
    If array, movement compensation will be performed.
    The array should be of shape (N, 10), holding the position
    parameters as returned by e.g. `read_head_pos`.
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

# Rank
docdict['rank'] = """
rank : None | dict | 'info' | 'full'
    This controls the rank computation that can be read from the
    measurement info or estimated from the data. See ``Notes``
    of :func:`mne.compute_rank` for details."""
docdict['rank_None'] = docdict['rank'] + 'The default is None.'
docdict['rank_info'] = docdict['rank'] + 'The default is "info".'

# Inverses
docdict['depth'] = """
depth : None | float | dict
    How to weight (or normalize) the forward using a depth prior.
    If float (default 0.8), it acts as the depth weighting exponent (``exp``)
    to use, which must be between 0 and 1. None is equivalent to 0, meaning
    no depth weighting is performed. It can also be a `dict` containing
    keyword arguments to pass to :func:`mne.forward.compute_depth_prior`
    (see docstring for details and defaults). This is effectively ignored
    when ``method='eLORETA'``.

    .. versionchanged:: 0.20
       Depth bias ignored for ``method='eLORETA'``.
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
        object. This is only implemented when working with loose orientations.
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
docdict['use_cps'] = """
use_cps : bool
    Whether to use cortical patch statistics to define normal orientations for
    surfaces (default True).
"""
docdict['use_cps_restricted'] = docdict['use_cps'] + """
    Only used when the inverse is free orientation (``loose=1.``),
    not in surface orientation, and ``pick_ori='normal'``.
"""

# Forward
docdict['on_missing'] = """
on_missing : str
    Behavior when ``stc`` has vertices that are not in ``fwd``.
    Can be "ignore", "warn"", or "raise"."""
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
docdict['trans'] = """
trans : str | dict | instance of Transform | None
    If str, the path to the head<->MRI transform ``*-trans.fif`` file produced
    during coregistration. Can also be ``'fsaverage'`` to use the built-in
    fsaverage transformation. If trans is None, an identity matrix is assumed.

    .. versionchanged:: 0.19
       Support for 'fsaverage' argument.
"""
docdict['subjects_dir'] = """
subjects_dir : str | None
    The path to the freesurfer subjects reconstructions.
    It corresponds to Freesurfer environment variable SUBJECTS_DIR.
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
    same format as data returned by `head_pos_to_trans_rot_t`.
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
processed with a bandpass, lowpass or highpass filter, dashed lines
indicate the boundaries of the filter (--). The line noise frequency is
also indicated with a dashed line (-.)
"""
docdict['plot_psd_picks_good_data'] = docdict['picks_good_data'][:-2] + """
    Cannot be None if `ax` is supplied.If both `picks` and `ax` are None
    separate subplots will be created for each standard channel type
    (`mag`, `grad`, and `eeg`).
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
layout : None
    Deprecated, will be removed in 0.20. Use ``info`` instead.
""" % docdict

# Montage
docdict["montage_deprecated"] = """
montage : str | None | instance of Montage
    Path or instance of montage containing electrode positions.
    If None, sensor locations are (0,0,0). See the documentation of
    :func:`mne.channels.read_montage` for more information.

    DEPRECATED in version 0.19
    Use the `set_montage` method.
"""
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
    If True, use a linear transparency between fmin and fmid.
    None will choose automatically based on colormap type.
"""
docdict["brain_time_interpolation"] = """
interpolation : str | None
    Interpolation method (:func:`scipy.interpolate.interp1d` parameter).
    Must be one of 'linear', 'nearest', 'zero', 'slinear', 'quadratic',
    or 'cubic'.
"""
docdict["show_traces"] = """
show_traces : bool | str
    If True, enable interactive picking of a point on the surface of the
    brain and plot it's time course using the bottom 1/3 of the figure.
    This feature is only available with the PyVista 3d backend when
    ``time_viewer=True``. Defaults to 'auto', which will use True if and
    only if ``time_viewer=True``, the backend is PyVista, and there is more
    than one time point.

    .. versionadded:: 0.20.0
"""
docdict["time_label"] = """
time_label : str | callable | None
    Format of the time label (a format string, a function that maps
    floating point time values to strings, or None for no label). The
    default is ``'auto'``, which will use ``time=%%0.2f ms`` if there
    is more than one time point.
"""

# STC label time course
docdict['eltc_labels'] = """
labels : Label | BiHemiLabel | list of Label or BiHemiLabel
    The labels for which to extract the time course.
"""
docdict['eltc_src'] = """
src : list
    Source spaces for left and right hemisphere.
"""
docdict['eltc_mode'] = """
mode : str
    Extraction mode, see Notes.
"""
docdict['eltc_allow_empty'] = """
allow_empty : bool
    Instead of emitting an error, return all-zero time courses for labels
    that do not have any vertices in the source estimate. Default is ``False``.
"""
docdict['eltc_mode_notes'] = """
Valid values for ``mode`` are:

- ``'max'``
    Maximum value across vertices at each time point within each label.
- ``'mean'``
    Average across vertices at each time point within each label. Ignores
    orientation of sources.
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
docdict['clust_con'] = """
connectivity : scipy.sparse.spmatrix | None | False
    Defines connectivity between locations in the data, where "locations" can
    be spatial vertices, frequency bins, etc. If ``False``, assumes no
    connectivity (each location is treated as independent and unconnected).
    If ``None``, a regular lattice connectivity is assumed, connecting
    each {sp} location to its neighbor(s) along the last dimension
    of {{eachgrp}} ``{{x}}``{lastdim}.
    If ``connectivity`` is a matrix, it is assumed to be symmetric (only the
    upper triangular half is used) and must be square with dimension equal to
    ``{{x}}.shape[-1]`` {parone} or ``{{x}}.shape[-1] * {{x}}.shape[-2]``
    {partwo}.{memory}
"""
mem = (' If spatial connectivity is uniform in time, it is recommended to use '
       'a square matrix with dimension ``{x}.shape[-1]`` (n_vertices) to save '
       'memory and computation, and to use ``max_step`` to define the extent '
       'of temporal adjacency to consider when clustering.')
st = dict(sp='spatial', lastdim='', parone='(n_vertices)',
          partwo='(n_times * n_vertices)', memory=mem)
tf = dict(sp='', lastdim=' (or the last two dimensions if ``{x}`` is 2D)',
          parone='', partwo='', memory='')
nogroups = dict(eachgrp='', x='X')
groups = dict(eachgrp='each group ', x='X[k]')
docdict['clust_con_st1'] = docdict['clust_con'].format(**st).format(**nogroups)
docdict['clust_con_stn'] = docdict['clust_con'].format(**st).format(**groups)
docdict['clust_con_1'] = docdict['clust_con'].format(**tf).format(**nogroups)
docdict['clust_con_n'] = docdict['clust_con'].format(**tf).format(**groups)
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
    ``'mask'``.
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
docdict['df_scaling_time_deprecated'] = """
scaling_time : None
    Deprecated; use ``time_format`` instead. If you need to scale time values
    by a factor other than 1000 (seconds → milliseconds), create the DataFrame
    first, then scale its ``time`` column afterwards.
"""
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
    Defaults to ``'ms'``.
"""
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
        kind = 'master'
    else:
        kind = 'maint/%s' % ('.'.join(mne.__version__.split('.')[:2]))
    return "http://github.com/mne-tools/mne-python/blob/%s/mne/%s%s" % (  # noqa
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
