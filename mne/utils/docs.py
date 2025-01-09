"""The documentation functions."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import inspect
import os
import os.path as op
import re
import sys
import webbrowser
from copy import deepcopy

from decorator import FunctionMaker

from ..defaults import HEAD_SIZE_DEFAULT
from ._bunch import BunchConst

# # # WARNING # # #
# This list must also be updated in doc/_templates/autosummary/class.rst if it
# is changed here!

_doc_special_members = (
    "__contains__",
    "__getitem__",
    "__iter__",
    "__len__",
    "__add__",
    "__sub__",
    "__mul__",
    "__div__",
    "__neg__",
)


def _reflow_param_docstring(docstring, has_first_line=True, width=75):
    """Reflow text to a nice width for terminals.

    WARNING: does not handle gracefully things like .. versionadded::
    """
    maxsplit = docstring.count("\n") - 1 if has_first_line else -1
    merged = " ".join(
        line.strip() for line in docstring.rsplit("\n", maxsplit=maxsplit)
    )
    reflowed = "\n    ".join(re.findall(rf".{{1,{width}}}(?:\s+|$)", merged))
    if has_first_line:
        reflowed = reflowed.replace("\n    \n", "\n", 1)
    return reflowed


##############################################################################
# Define our standard documentation entries
#
# To reduce redundancy across functions, please standardize the format to
# ``argument_optional_keywords``. For example ``tmin_raw`` for an entry that
# is specific to ``raw`` and since ``tmin`` is used other places, needs to
# be disambiguated. This way the entries will be easy to find since they
# are alphabetized (you can look up by the name of the argument). This way
# the same ``docdict`` entries are easier to reuse.

docdict = BunchConst()

# %%
# A

tfr_arithmetics_return_template = """
Returns
-------
tfr : instance of RawTFR | instance of EpochsTFR | instance of AverageTFR
    {}
"""

tfr_add_sub_template = """
Parameters
----------
other : instance of RawTFR | instance of EpochsTFR | instance of AverageTFR
    The TFR instance to {}. Must have the same type as ``self``, and matching
    ``.times`` and ``.freqs`` attributes.

{}
"""

tfr_mul_truediv_template = """
Parameters
----------
num : int | float
    The number to {} by.

{}
"""

tfr_arithmetics_return = tfr_arithmetics_return_template.format(
    "A new TFR instance, of the same type as ``self``."
)
tfr_inplace_arithmetics_return = tfr_arithmetics_return_template.format(
    "The modified TFR instance."
)

docdict["__add__tfr"] = tfr_add_sub_template.format("add", tfr_arithmetics_return)
docdict["__iadd__tfr"] = tfr_add_sub_template.format(
    "add", tfr_inplace_arithmetics_return
)
docdict["__imul__tfr"] = tfr_mul_truediv_template.format(
    "multiply", tfr_inplace_arithmetics_return
)
docdict["__isub__tfr"] = tfr_add_sub_template.format(
    "subtract", tfr_inplace_arithmetics_return
)
docdict["__itruediv__tfr"] = tfr_mul_truediv_template.format(
    "divide", tfr_inplace_arithmetics_return
)
docdict["__mul__tfr"] = tfr_mul_truediv_template.format(
    "multiply", tfr_arithmetics_return
)
docdict["__sub__tfr"] = tfr_add_sub_template.format("subtract", tfr_arithmetics_return)
docdict["__truediv__tfr"] = tfr_mul_truediv_template.format(
    "divide", tfr_arithmetics_return
)


docdict["accept"] = """
accept : bool
    If True (default False), accept the license terms of this dataset.
"""

docdict["add_ch_type_export_params"] = """
add_ch_type : bool
    Whether to incorporate the channel type into the signal label (e.g. whether
    to store channel "Fz" as "EEG Fz"). Only used for EDF format. Default is
    ``False``.
"""

docdict["add_data_kwargs"] = """
add_data_kwargs : dict | None
    Additional arguments to brain.add_data (e.g.,
    ``dict(time_label_size=10)``).
"""

docdict["add_frames"] = """
add_frames : int | None
    If int, enable (>=1) or disable (0) the printing of stack frame
    information using formatting. Default (None) does not change the
    formatting. This can add overhead so is meant only for debugging.
"""

docdict["adjacency_clust"] = """
adjacency : scipy.sparse.spmatrix | None | False
    Defines adjacency between locations in the data, where "locations" can be
    spatial vertices, frequency bins, time points, etc. For spatial vertices
    (i.e. sensor space data), see :func:`mne.channels.find_ch_adjacency` or
    :func:`mne.spatial_inter_hemi_adjacency`. For source space data, see
    :func:`mne.spatial_src_adjacency` or
    :func:`mne.spatio_temporal_src_adjacency`. If ``False``, assumes
    no adjacency (each location is treated as independent and unconnected).
    If ``None``, a regular lattice adjacency is assumed, connecting
    each {sp} location to its neighbor(s) along the last dimension
    of {{eachgrp}} ``{{x}}``{lastdim}.
    If ``adjacency`` is a matrix, it is assumed to be symmetric (only the
    upper triangular half is used) and must be square with dimension equal to
    ``{{x}}.shape[-1]`` {parone} or ``{{x}}.shape[-1] * {{x}}.shape[-2]``
    {partwo} or (optionally)
    ``{{x}}.shape[-1] * {{x}}.shape[-2] * {{x}}.shape[-3]``
    {parthree}.{memory}
"""

mem = (
    " If spatial adjacency is uniform in time, it is recommended to use "
    "a square matrix with dimension ``{x}.shape[-1]`` (n_vertices) to save "
    "memory and computation, and to use ``max_step`` to define the extent "
    "of temporal adjacency to consider when clustering."
)
comb = " The function `mne.stats.combine_adjacency` may be useful for 4D data."
st = dict(
    sp="spatial",
    lastdim="",
    parone="(n_vertices)",
    partwo="(n_times * n_vertices)",
    parthree="(n_times * n_freqs * n_vertices)",
    memory=mem,
)
tf = dict(
    sp="",
    lastdim=" (or the last two dimensions if ``{x}`` is 2D)",
    parone="(for 2D data)",
    partwo="(for 3D data)",
    parthree="(for 4D data)",
    memory=comb,
)
nogroups = dict(eachgrp="", x="X")
groups = dict(eachgrp="each group ", x="X[k]")
docdict["adjacency_clust_1"] = (
    docdict["adjacency_clust"].format(**tf).format(**nogroups)
)
docdict["adjacency_clust_n"] = docdict["adjacency_clust"].format(**tf).format(**groups)
docdict["adjacency_clust_st1"] = (
    docdict["adjacency_clust"].format(**st).format(**nogroups)
)
docdict["adjacency_clust_stn"] = (
    docdict["adjacency_clust"].format(**st).format(**groups)
)

docdict["adjust_dig_chpi"] = """
adjust_dig : bool
    If True, adjust the digitization locations used for fitting based on
    the positions localized at the start of the file.
"""

docdict["agg_fun_psd_topo"] = """
agg_fun : callable
    The function used to aggregate over frequencies. Defaults to
    :func:`numpy.sum` if ``normalize=True``, else :func:`numpy.mean`.
"""

docdict["align_view"] = """
align : bool
    If True, consider view arguments relative to canonical MRI
    directions (closest to MNI for the subject) rather than native MRI
    space. This helps when MRIs are not in standard orientation (e.g.,
    have large rotations).
"""

docdict["allow_2d"] = """
allow_2d : bool
    If True, allow 2D data as input (i.e. n_samples, n_features).
"""

docdict["allow_empty_eltc"] = """
allow_empty : bool | str
    ``False`` (default) will emit an error if there are labels that have no
    vertices in the source estimate. ``True`` and ``'ignore'`` will return
    all-zero time courses for labels that do not have any vertices in the
    source estimate, and True will emit a warning while and "ignore" will
    just log a message.

    .. versionchanged:: 0.21.0
       Support for "ignore".
"""

docdict["alpha"] = """
alpha : float in [0, 1]
    Alpha level to control opacity.
"""

docdict["anonymize_info_notes"] = """
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

# raw/epochs/evoked apply_function method
# apply_function method summary
applyfun_summary = """\
The function ``fun`` is applied to the {applies_to} defined in ``picks``.
The {data_type} object's data is modified in-place. If the function returns a different
data type (e.g. :py:obj:`numpy.complex128`) it must be specified
using the ``dtype`` parameter, which causes the data type of **all** the data
to change (even if the function is only applied to {applies_to} in
``picks``).{preload}

.. note:: If ``n_jobs`` > 1, more memory is required as
          ``len(picks) * n_times`` additional time points need to
          be temporarily stored in memory.
.. note:: If the data type changes (``dtype != None``), more memory is
          required since the original and the converted data needs
          to be stored in memory.
"""
applyfun_preload = (
    " The object has to have the data loaded e.g. with "
    "``preload=True`` or ``self.load_data()``."
)
docdict["applyfun_summary_epochs"] = applyfun_summary.format(
    applies_to="channels", data_type="epochs", preload=applyfun_preload
)
docdict["applyfun_summary_evoked"] = applyfun_summary.format(
    applies_to="channels", data_type="evoked", preload=""
)
docdict["applyfun_summary_raw"] = applyfun_summary.format(
    applies_to="channels", data_type="raw", preload=applyfun_preload
)
docdict["applyfun_summary_stc"] = applyfun_summary.format(
    applies_to="vertices", data_type="source estimate", preload=""
)

docdict["area_alpha_plot_psd"] = """\
area_alpha : float
    Alpha for the area.
"""

docdict["area_mode_plot_psd"] = """\
area_mode : str | None
    Mode for plotting area. If 'std', the mean +/- 1 STD (across channels)
    will be plotted. If 'range', the min and max (across channels) will be
    plotted. Bad channels will be excluded from these calculations.
    If None, no area will be plotted. If average=False, no area is plotted.
"""

docdict["aseg"] = """
aseg : str
    The anatomical segmentation file. Default ``auto`` uses ``aparc+aseg``
    if available and ``wmparc`` if not. This may be any anatomical
    segmentation file in the mri subdirectory of the Freesurfer subject
    directory.

    .. versionchanged:: 1.8
       Added support for the new default ``'auto'``.
"""

docdict["average_plot_evoked_topomap"] = """
average : float | array-like of float, shape (n_times,) | None
    The time window (in seconds) around a given time point to be used for
    averaging. For example, 0.2 would translate into a time window that
    starts 0.1 s before and ends 0.1 s after the given time point. If the
    time window exceeds the duration of the data, it will be clipped.
    Different time windows (one per time point) can be provided by
    passing an ``array-like`` object (e.g., ``[0.1, 0.2, 0.3]``). If
    ``None`` (default), no averaging will take place.

    .. versionchanged:: 1.1
       Support for ``array-like`` input.
"""

docdict["average_plot_psd"] = """\
average : bool
    If False, the PSDs of all channels is displayed. No averaging
    is done and parameters area_mode and area_alpha are ignored. When
    False, it is possible to paint an area (hold left mouse button and
    drag) to plot a topomap.
"""

docdict["average_psd"] = """\
average : str | None
    How to average the segments. If ``mean`` (default), calculate the
    arithmetic mean. If ``median``, calculate the median, corrected for
    its bias relative to the mean. If ``None``, returns the unaggregated
    segments.
"""

docdict["average_tfr"] = """
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

_axes_base = """\
{param} : instance of Axes | {allowed}None
    The axes to plot into. If ``None``, a new :class:`~matplotlib.figure.Figure`
    will be created{created}. {list_extra}{extra}Default is ``None``.
"""
_axes_list = _axes_base.format(
    param="{param}",
    allowed="list of Axes | ",
    created=" with the correct number of axes",
    list_extra="""If :class:`~matplotlib.axes.Axes`
    are provided (either as a single instance or a :class:`list` of axes),
    the number of axes provided must {must}. """,
    extra="{extra}",
)
_match_chtypes_present_in = "match the number of channel types present in the {}object."
docdict["ax_plot_psd"] = _axes_list.format(
    param="ax", must=_match_chtypes_present_in.format(""), extra=""
)
docdict["axes_cov_plot_topomap"] = _axes_list.format(
    param="axes", must="be length 1", extra=""
)
docdict["axes_evoked_plot_topomap"] = _axes_list.format(
    param="axes",
    must="match the number of ``times`` provided (unless ``times`` is ``None``)",
    extra="",
)
docdict["axes_montage"] = """
axes : instance of Axes | instance of Axes3D | None
    Axes to draw the sensors to. If ``kind='3d'``, axes must be an instance
    of Axes3D. If None (default), a new axes will be created.
"""
docdict["axes_plot_projs_topomap"] = _axes_list.format(
    param="axes",
    must="match the number of projectors",
    extra="",
)
docdict["axes_plot_topomap"] = _axes_base.format(
    param="axes",
    allowed="",
    created="",
    list_extra="",
    extra="",
)
docdict["axes_spectrum_plot"] = _axes_list.format(
    param="axes",
    must=_match_chtypes_present_in.format(":class:`~mne.time_frequency.Spectrum` "),
    extra="",
)
docdict["axes_spectrum_plot_topo"] = _axes_list.format(
    param="axes",
    must="be length 1 (for efficiency, subplots for each channel are simulated "
    "within a single :class:`~matplotlib.axes.Axes` object)",
    extra="",
)
docdict["axes_spectrum_plot_topomap"] = _axes_list.format(
    param="axes", must="match the length of ``bands``", extra=""
)
docdict["axes_tfr_plot"] = _axes_list.format(
    param="axes",
    must="match the number of picks",
    extra="""If ``combine`` is not None,
    ``axes`` must either be an instance of Axes, or a list of length 1. """,
)

docdict["axis_facecolor"] = """\
axis_facecolor : str | tuple
    A matplotlib-compatible color to use for the axis background.
    Defaults to black.
"""

docdict["azimuth"] = """
azimuth : float
    The azimuthal angle of the camera rendering the view in degrees.
"""

# %%
# B

docdict["bad_condition_maxwell_cond"] = """
bad_condition : str
    How to deal with ill-conditioned SSS matrices. Can be ``"error"``
    (default), ``"warning"``, ``"info"``, or ``"ignore"``.
"""

docdict["bands_psd_topo"] = """
bands : None | dict | list of tuple
    The frequencies or frequency ranges to plot. If a :class:`dict`, keys will
    be used as subplot titles and values should be either a single frequency
    (e.g., ``{'presentation rate': 6.5}``) or a length-two sequence of lower
    and upper frequency band edges (e.g., ``{'theta': (4, 8)}``). If a single
    frequency is provided, the plot will show the frequency bin that is closest
    to the requested value. If ``None`` (the default), expands to::

        bands = {'Delta (0-4 Hz)': (0, 4), 'Theta (4-8 Hz)': (4, 8),
                 'Alpha (8-12 Hz)': (8, 12), 'Beta (12-30 Hz)': (12, 30),
                 'Gamma (30-45 Hz)': (30, 45)}

    .. note::
       For backwards compatibility, :class:`tuples<tuple>` of length 2 or 3 are
       also accepted, where the last element of the tuple is the subplot title
       and the other entries are frequency values (a single value or band
       edges). New code should use :class:`dict` or ``None``.

    .. versionchanged:: 1.2
       Allow passing a dict and discourage passing tuples.
"""

docdict["base_estimator"] = """
base_estimator : object
    The base estimator to iteratively fit on a subset of the dataset.
"""

_baseline_rescale_base = """
baseline : None | tuple of length 2
    The time interval to consider as "baseline" when applying baseline
    correction. If ``None``, do not apply baseline correction.
    If a tuple ``(a, b)``, the interval is between ``a`` and ``b``
    (in seconds), including the endpoints.
    If ``a`` is ``None``, the **beginning** of the data is used; and if ``b``
    is ``None``, it is set to the **end** of the data.
    If ``(None, None)``, the entire time interval is used.

    .. note::
        The baseline ``(a, b)`` includes both endpoints, i.e. all timepoints ``t``
        such that ``a <= t <= b``.
"""

docdict["baseline_epochs"] = f"""{_baseline_rescale_base}
    Correction is applied **to each epoch and channel individually** in the
    following way:

    1. Calculate the mean signal of the baseline period.
    2. Subtract this mean from the **entire** epoch.

"""

docdict["baseline_evoked"] = f"""{_baseline_rescale_base}
    Correction is applied **to each channel individually** in the following
    way:

    1. Calculate the mean signal of the baseline period.
    2. Subtract this mean from the **entire** ``Evoked``.

"""

docdict["baseline_report"] = f"""{_baseline_rescale_base}
    Correction is applied in the following way **to each channel:**

    1. Calculate the mean signal of the baseline period.
    2. Subtract this mean from the **entire** time period.

    For `~mne.Epochs`, this algorithm is run **on each epoch individually.**
"""

docdict["baseline_rescale"] = _baseline_rescale_base

docdict["baseline_stc"] = f"""{_baseline_rescale_base}
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

"""

docdict["baseline_tfr_attr"] = """
baseline : array-like, shape (2,)
    The start and end times of the baseline period, in seconds."""


docdict["block"] = """\
block : bool
    Whether to halt program execution until the figure is closed.
    May not work on all systems / platforms. Defaults to ``False``.
"""

docdict["border_topo"] = """
border : str
    Matplotlib border style to be used for each sensor plot.
"""
docdict["border_topomap"] = """
border : float | 'mean'
    Value to extrapolate to on the topomap borders. If ``'mean'`` (default),
    then each extrapolated point has the average value of its neighbours.
"""

docdict["brain_kwargs"] = """
brain_kwargs : dict | None
    Additional arguments to the :class:`mne.viz.Brain` constructor (e.g.,
    ``dict(silhouette=True)``).
"""

docdict["brain_update"] = """
update : bool
    Force an update of the plot. Defaults to True.
"""

docdict["browser"] = """
fig : matplotlib.figure.Figure | mne_qt_browser.figure.MNEQtBrowser
    Browser instance.
"""

docdict["buffer_size_clust"] = """
buffer_size : int | None
    Block size to use when computing test statistics. This can significantly
    reduce memory usage when ``n_jobs > 1`` and memory sharing between
    processes is enabled (see :func:`mne.set_cache_dir`), because ``X`` will be
    shared between processes and each process only needs to allocate space for
    a small block of locations at a time.
"""

docdict["by_event_type"] = """
by_event_type : bool
    When ``False`` (the default) all epochs are processed together and a single
    :class:`~mne.Evoked` object is returned. When ``True``, epochs are first
    grouped by event type (as specified using the ``event_id`` parameter) and a
    list is returned containing a separate :class:`~mne.Evoked` object for each
    event type. The ``.comment`` attribute is set to the label of the event
    type.

    .. versionadded:: 0.24.0
"""

# %%
# C

docdict["calibration_maxwell_cal"] = """
calibration : str | None
    Path to the ``'.dat'`` file with fine calibration coefficients.
    File can have 1D or 3D gradiometer imbalance correction.
    This file is machine/site-specific.
"""

docdict["cbar_fmt_topomap"] = """\
cbar_fmt : str
    Formatting string for colorbar tick labels. See :ref:`formatspec` for
    details.
"""
docdict["cbar_fmt_topomap_psd"] = (
    docdict["cbar_fmt_topomap"]
    + """\
    If ``'auto'``, is equivalent to '%0.3f' if ``dB=False`` and '%0.1f' if
    ``dB=True``. Defaults to ``'auto'``.
"""
)

docdict["center"] = """
center : float or None
    If not None, center of a divergent colormap, changes the meaning of
    fmin, fmax and fmid.
"""

docdict["ch_name_ecg"] = """
ch_name : None | str
    The name of the channel to use for ECG peak detection.
    If ``None`` (default), ECG channel is used if present. If ``None`` and
    **no** ECG channel is present, a synthetic ECG channel is created from
    the cross-channel average. This synthetic channel can only be created from
    MEG channels.
"""

docdict["ch_name_eog"] = """
ch_name : str | list of str | None
    The name of the channel(s) to use for EOG peak detection. If a string,
    can be an arbitrary channel. This doesn't have to be a channel of
    ``eog`` type; it could, for example, also be an ordinary EEG channel
    that was placed close to the eyes, like ``Fp1`` or ``Fp2``.

    Multiple channel names can be passed as a list of strings.

    If ``None`` (default), use the channel(s) in ``raw`` with type ``eog``.
"""

docdict["ch_names_annot"] = """
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
docdict["ch_names_tfr_attr"] = """
ch_names : list
    The channel names."""

docdict["ch_type_set_eeg_reference"] = """
ch_type : list of str | str
    The name of the channel type to apply the reference to.
    Valid channel types are ``'auto'``, ``'eeg'``, ``'ecog'``, ``'seeg'``,
    ``'dbs'``. If ``'auto'``, the first channel type of eeg, ecog, seeg or dbs
    that is found (in that order) will be selected.

    .. versionadded:: 0.19
    .. versionchanged:: 1.2
       ``list-of-str`` is now supported with ``projection=True``.
"""

_ch_type_topomap_base = """\
ch_type : 'mag' | 'grad' | 'planar1' | 'planar2' | 'eeg' | None{}
    The channel type to plot. For ``'grad'``, the gradiometers are
    collected in pairs and the {} for each pair is plotted. If
    ``None`` {}. {}Defaults to ``None``.
"""
_ch_type_topomap = _ch_type_topomap_base.format(
    "{}", "{}", "the first available channel type from order shown above is used", "{}"
)
docdict["ch_type_topomap"] = _ch_type_topomap.format("", "RMS", "")
docdict["ch_type_topomap_proj"] = _ch_type_topomap_base.format(
    " | list",
    "RMS",
    "it will return all channel types present.",
    "If a list of ch_types is provided, it will return multiple figures. ",
)
docdict["ch_type_topomap_psd"] = _ch_type_topomap.format("", "mean", "")

chwise = """
channel_wise : bool
    Whether to apply the function to each channel {}individually. If ``False``,
    the function will be applied to all {}channels at once. Default ``True``.
"""
docdict["channel_wise_applyfun"] = chwise.format("", "")
docdict["channel_wise_applyfun_epo"] = chwise.format("in each epoch ", "epochs and ")


docdict["check_disjoint_clust"] = """
check_disjoint : bool
    Whether to check if the connectivity matrix can be separated into disjoint
    sets before clustering. This may lead to faster clustering, especially if
    the second dimension of ``X`` (usually the "time" dimension) is large.
"""

docdict["chpi_amplitudes"] = """
chpi_amplitudes : dict
    The time-varying cHPI coil amplitudes, with entries
    "times", "proj", and "slopes".
"""

docdict["chpi_locs"] = """
chpi_locs : dict
    The time-varying cHPI coils locations, with entries
    "times", "rrs", "moments", and "gofs".
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

_cmap_template = """
cmap : matplotlib colormap | str{allowed}
        The :class:`~matplotlib.colors.Colormap` to use. If a :class:`str`, must be a
        valid Matplotlib colormap name. Default is {default}.
"""
docdict["cmap"] = _cmap_template.format(
    allowed=" | None",
    default="``None``, which will use the Matplotlib default colormap",
)
docdict["cmap_tfr_plot_topo"] = _cmap_template.format(
    allowed="", default='``"RdBu_r"``'
)
docdict["cmap_topomap"] = """\
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

    .. warning::  Interactive mode works smoothly only for a small amount
        of topomaps. Interactive mode is disabled by default for more than
        2 topomaps.
"""

docdict["cmap_topomap_simple"] = """
cmap : matplotlib colormap | None
    Colormap to use. If None, 'Reds' is used for all positive data,
    otherwise defaults to 'RdBu_r'.
"""

docdict["cnorm"] = """
cnorm : matplotlib.colors.Normalize | None
    How to normalize the colormap. If ``None``, standard linear normalization
    is performed. If not ``None``, ``vmin`` and ``vmax`` will be ignored.
    See :ref:`Matplotlib docs <matplotlib:colormapnorms>`
    for more details on colormap normalization, and
    :ref:`the ERDs example<cnorm-example>` for an example of its use.
"""

docdict["color_matplotlib"] = """
color : color
    A list of anything matplotlib accepts: string, RGB, hex, etc.
"""

docdict["color_plot_psd"] = """\
color : str | tuple
    A matplotlib-compatible color to use. Has no effect when
    spatial_colors=True.
"""

docdict["color_spectrum_plot_topo"] = """\
color : str | tuple
    A matplotlib-compatible color to use for the curves. Defaults to
    white.
"""

docdict["colorbar"] = """\
colorbar : bool
    Whether to add a colorbar to the plot. Default is ``True``.
"""
docdict["colorbar_tfr_plot_joint"] = """
colorbar : bool
    Whether to add a colorbar to the plot (for the topomap annotations). Not compatible
    with user-defined ``axes``. Default is ``True``.
"""
docdict["colorbar_topomap"] = """
colorbar : bool
    Plot a colorbar in the rightmost column of the figure.
"""

docdict["colormap"] = """
colormap : str | np.ndarray of float, shape(n_colors, 3 | 4)
    Name of colormap to use or a custom look up table. If array, must
    be (n x 3) or (n x 4) array for with RGB or RGBA values between
    0 and 255.
"""

_combine_template = """
combine : 'mean' | {literals} | callable{none}
    How to aggregate across channels. {none_sentence}If a string,
    ``"mean"`` uses :func:`numpy.mean`, {other_string}.
    If :func:`callable`, it must operate on an :class:`array <numpy.ndarray>`
    of shape ``({shape})`` and return an array of shape
    ``({return_shape})``. {example}{notes}Defaults to {default}.
"""
_example = """For example::

        combine = lambda data: np.median(data, axis=1)

    """  # ← the 4 trailing spaces are intentional here!
_median_std_gfp = """``"median"`` computes the `marginal median
    <https://en.wikipedia.org/wiki/Median#Marginal_median>`__, ``"std"``
    uses :func:`numpy.std`, and ``"gfp"`` computes global field power
    for EEG channels and RMS amplitude for MEG channels"""
_none_default = dict(none=" | None", default="``None``")
docdict["combine_plot_compare_evokeds"] = _combine_template.format(
    literals="'median' | 'std' | 'gfp'",
    **_none_default,
    none_sentence="""If ``None``, channels are combined by
    computing GFP/RMS, unless ``picks`` is a single channel (not channel type)
    or ``axes="topo"``, in which cases no combining is performed. """,
    other_string=_median_std_gfp,
    shape="n_evokeds, n_channels, n_times",
    return_shape="n_evokeds, n_times",
    example=_example,
    notes="",
)
docdict["combine_plot_epochs_image"] = _combine_template.format(
    literals="'median' | 'std' | 'gfp'",
    **_none_default,
    none_sentence="""If ``None``, channels are combined by
    computing GFP/RMS, unless ``group_by`` is also ``None`` and ``picks`` is a
    list of specific channels (not channel types), in which case no combining
    is performed and each channel gets its own figure. """,
    other_string=_median_std_gfp,
    shape="n_epochs, n_channels, n_times",
    return_shape="n_epochs, n_times",
    example=_example,
    notes="See Notes for further details. ",
)
docdict["combine_tfr_plot"] = _combine_template.format(
    literals="'rms'",
    **_none_default,
    none_sentence="If ``None``, plot one figure per selected channel. ",
    shape="n_channels, n_freqs, n_times",
    return_shape="n_freqs, n_times",
    other_string='``"rms"`` computes the root-mean-square',
    example="",
    notes="",
)
docdict["combine_tfr_plot_joint"] = _combine_template.format(
    literals="'rms'",
    none="",
    none_sentence="",
    shape="n_channels, n_freqs, n_times",
    return_shape="n_freqs, n_times",
    other_string='``"rms"`` computes the root-mean-square',
    example="",
    notes="",
    default='``"mean"``',
)

_comment_template = """
comment : str{or_none}
    Comment on the data, e.g., the experimental condition(s){avgd}.{extra}"""
docdict["comment_averagetfr"] = _comment_template.format(
    or_none=" | None",
    avgd="averaged",
    extra="""Default is ``None``
    which is replaced with ``inst.comment`` (for :class:`~mne.Evoked` instances)
    or a comma-separated string representation of the keys in ``inst.event_id``
    (for :class:`~mne.Epochs` instances).""",
)
docdict["comment_averagetfr_attr"] = _comment_template.format(
    or_none="", avgd=" averaged", extra=""
)
docdict["comment_tfr_attr"] = _comment_template.format(or_none="", avgd="", extra="")

docdict["compute_proj_ecg"] = """This function will:

#. Filter the ECG data channel.

#. Find ECG R wave peaks using :func:`mne.preprocessing.find_ecg_events`.

#. Filter the raw data.

#. Create `~mne.Epochs` around the R wave peaks, capturing the heartbeats.

#. Optionally average the `~mne.Epochs` to produce an `~mne.Evoked` if
   ``average=True`` was passed (default).

#. Calculate SSP projection vectors on that data to capture the artifacts."""

docdict["compute_proj_eog"] = """This function will:

#. Filter the EOG data channel.

#. Find the peaks of eyeblinks in the EOG data using
   :func:`mne.preprocessing.find_eog_events`.

#. Filter the raw data.

#. Create `~mne.Epochs` around the eyeblinks.

#. Optionally average the `~mne.Epochs` to produce an `~mne.Evoked` if
   ``average=True`` was passed (default).

#. Calculate SSP projection vectors on that data to capture the artifacts."""

docdict["compute_ssp"] = """This function aims to find those SSP vectors that
will project out the ``n`` most prominent signals from the data for each
specified sensor type. Consequently, if the provided input data contains high
levels of noise, the produced SSP vectors can then be used to eliminate that
noise from the data.
"""

docdict["contours_topomap"] = """
contours : int | array-like
    The number of contour lines to draw. If ``0``, no contours will be drawn.
    If a positive integer, that number of contour levels are chosen using the
    matplotlib tick locator (may sometimes be inaccurate, use array for
    accuracy). If array-like, the array values are used as the contour levels.
    The values should be in µV for EEG, fT for magnetometers and fT/m for
    gradiometers. If ``colorbar=True``, the colorbar will have ticks
    corresponding to the contour levels. Default is ``6``.
"""

docdict["coord_frame_maxwell"] = """
coord_frame : str
    The coordinate frame that the ``origin`` is specified in, either
    ``'meg'`` or ``'head'``. For empty-room recordings that do not have
    a head<->meg transform ``info['dev_head_t']``, the MEG coordinate
    frame should be used.
"""

docdict["copy_df"] = """
copy : bool
    If ``True``, data will be copied. Otherwise data may be modified in place.
    Defaults to ``True``.
"""

docdict["create_ecg_epochs"] = """This function will:

#. Filter the ECG data channel.

#. Find ECG R wave peaks using :func:`mne.preprocessing.find_ecg_events`.

#. Create `~mne.Epochs` around the R wave peaks, capturing the heartbeats.
"""

docdict["create_eog_epochs"] = """This function will:

#. Filter the EOG data channel.

#. Find the peaks of eyeblinks in the EOG data using
   :func:`mne.preprocessing.find_eog_events`.

#. Create `~mne.Epochs` around the eyeblinks.
"""

docdict["cross_talk_maxwell"] = """
cross_talk : str | None
    Path to the FIF file with cross-talk correction information.
"""

# %%
# D

_dB = """
dB : bool
    Whether to plot on a decibel-like scale. If ``True``, plots
    10 × log₁₀({quantity}){caveat}.{extra}
"""
_ignored_if_normalize = " Ignored if ``normalize=True``."
_psd = "spectral power"

docdict["dB_plot_psd"] = """\
dB : bool
    Plot Power Spectral Density (PSD), in units (amplitude**2/Hz (dB)) if
    ``dB=True``, and ``estimate='power'`` or ``estimate='auto'``. Plot PSD
    in units (amplitude**2/Hz) if ``dB=False`` and,
    ``estimate='power'``. Plot Amplitude Spectral Density (ASD), in units
    (amplitude/sqrt(Hz)), if ``dB=False`` and ``estimate='amplitude'`` or
    ``estimate='auto'``. Plot ASD, in units (amplitude/sqrt(Hz) (dB)), if
    ``dB=True`` and ``estimate='amplitude'``.
"""
docdict["dB_plot_topomap"] = _dB.format(
    quantity=_psd,
    caveat=" following the application of ``agg_fun``",
    extra=_ignored_if_normalize,
)
docdict["dB_spectrum_plot"] = _dB.format(quantity=_psd, caveat="", extra="")
docdict["dB_spectrum_plot_topo"] = _dB.format(
    quantity=_psd, caveat="", extra=_ignored_if_normalize
)
docdict["dB_tfr_plot_topo"] = _dB.format(quantity="data", caveat="", extra="")

_data_template = """
data : ndarray, shape ({})
    The data.
"""
docdict["data_tfr"] = _data_template.format("n_channels, n_freqs, n_times")

docdict["daysback_anonymize_info"] = """
daysback : int | None
    Number of days to subtract from all dates.
    If ``None`` (default), the acquisition date, ``info['meas_date']``,
    will be set to ``January 1ˢᵗ, 2000``. This parameter is ignored if
    ``info['meas_date']`` is ``None`` (i.e., no acquisition date has been set).
"""

docdict["dbs"] = """
dbs : bool
    If True (default), show DBS (deep brain stimulation) electrodes.
"""
docdict["decim"] = """
decim : int
    Factor by which to subsample the data.

    .. warning:: Low-pass filtering is not performed, this simply selects
                 every Nth sample (where N is the value passed to
                 ``decim``), i.e., it compresses the signal (see Notes).
                 If the data are not properly filtered, aliasing artifacts
                 may occur.
                 See :ref:`resampling-and-decimating` for more information.
"""

docdict["decim_notes"] = """
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

docdict["decim_tfr"] = """
decim : int | slice
    Decimation factor, applied *after* time-frequency decomposition.

    - if :class:`int`, returns ``tfr[..., ::decim]`` (keep only every Nth
      sample along the time axis).
    - if :class:`slice`, returns ``tfr[..., decim]`` (keep only the specified
      slice along the time axis).

    .. note::
        Decimation is done after convolutions and may create aliasing
        artifacts.
"""

docdict["depth"] = """
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

docdict["destination_maxwell_dest"] = """
destination : path-like | array-like, shape (3,) | instance of Transform | None
    The destination location for the head. Can be:

    ``None``
      Will not change the head position.
    :class:`~mne.transforms.Transform`
      A MEG device<->head transformation, e.g. ``info["dev_head_t"]``.
    :class:`numpy.ndarray`
      A 3-element array giving the coordinates to translate to (with no rotations).
      For example, ``destination=(0, 0, 0.04)`` would translate the bases
      as ``--trans default`` would in MaxFilter™ (i.e., to the default
      head location).
    ``path-like``
      A path to a FIF file containing the destination MEG device<->head transformation.
"""

docdict["detrend_epochs"] = """
detrend : int | None
    If 0 or 1, the data channels (MEG and EEG) will be detrended when
    loaded. 0 is a constant (DC) detrend, 1 is a linear detrend. None
    is no detrending. Note that detrending is performed before baseline
    correction. If no DC offset is preferred (zeroth order detrending),
    either turn off baseline correction, as this may introduce a DC
    shift, or set baseline correction to use the entire time interval
    (will yield equivalent results but be slower).
"""

docdict["df_return"] = """
df : instance of pandas.DataFrame
    A dataframe suitable for usage with other statistical/plotting/analysis
    packages.
"""

docdict["dig_kinds"] = """
dig_kinds : list of str | str
    Kind of digitization points to use in the fitting. These can be any
    combination of ('cardinal', 'hpi', 'eeg', 'extra'). Can also
    be 'auto' (default), which will use only the 'extra' points if
    enough (more than 4) are available, and if not, uses 'extra' and
    'eeg' points.
"""

docdict["dipole"] = """
dipole : instance of Dipole | list of Dipole
    Dipole object containing position, orientation and amplitude of
    one or more dipoles. Multiple simultaneous dipoles may be defined by
    assigning them identical times. Alternatively, multiple simultaneous
    dipoles may also be specified as a list of Dipole objects.

    .. versionchanged:: 1.1
        Added support for a list of :class:`mne.Dipole` instances.
"""

docdict["distance"] = """
distance : float | "auto" | None
    The distance from the camera rendering the view to the focalpoint
    in plot units (either m or mm). If "auto", the bounds of visible objects will be
    used to set a reasonable distance.

    .. versionchanged:: 1.6
       ``None`` will no longer change the distance, use ``"auto"`` instead.
"""

docdict["drop_log"] = """
drop_log : tuple | None
    Tuple of tuple of strings indicating which epochs have been marked to
    be ignored."""

docdict["dtype_applyfun"] = """
dtype : numpy.dtype
    Data type to use after applying the function. If None
    (default) the data type is not modified.
"""

# %%
# E

docdict["ecog"] = """
ecog : bool
    If True (default), show ECoG sensors.
"""

docdict["edf_resamp_note"] = """
:class:`mne.io.Raw` only stores signals with matching sampling frequencies.
Therefore, if mixed sampling frequency signals are requested, all signals
are upsampled to the highest loaded sampling frequency. In this case, using
preload=True is recommended, as otherwise, edge artifacts appear when
slices of the signal are requested.
"""

docdict["eeg"] = """
eeg : bool | str | list | dict
    String options are:

    - "original" (default; equivalent to ``True``)
        Shows EEG sensors using their digitized locations (after
        transformation to the chosen ``coord_frame``)
    - "projected"
        The EEG locations projected onto the scalp, as is done in
        forward modeling

    Can also be a list of these options, or a dict to specify the alpha values
    to use, e.g. ``dict(original=0.2, projected=0.8)``.

    .. versionchanged:: 1.6
       Added support for specifying alpha values as a dict.
"""

docdict["elevation"] = """
elevation : float
    The The zenith angle of the camera rendering the view in degrees.
"""

docdict["eltc_mode_notes"] = """
Valid values for ``mode`` are:

- ``'max'``
    Maximum absolute value across vertices at each time point within each label.
- ``'mean'``
    Average across vertices at each time point within each label. Ignores
    orientation of sources for standard source estimates, which varies
    across the cortical surface, which can lead to cancellation.
    Vector source estimates are always in XYZ / RAS orientation, and are thus
    already geometrically aligned.
- ``'mean_flip'``
    Finds the dominant direction of source space normal vector orientations
    within each label, applies a sign-flip to time series at vertices whose
    orientation is more than 90° different from the dominant direction, and
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
- ``None``
    No aggregation is performed, and an array of shape ``(n_vertices, n_times)`` is
    returned.

    .. versionadded:: 0.21
       Support for ``'auto'``, vector, and volume source estimates.

The only modes that work for vector and volume source estimates are ``'mean'``,
``'max'``, and ``'auto'``.
"""

docdict["emit_warning"] = """
emit_warning : bool
    Whether to emit warnings when cropping or omitting annotations.
"""

docdict["encoding_edf"] = """
encoding : str
    Encoding of annotations channel(s). Default is "utf8" (the only correct
    encoding according to the EDF+ standard).
"""

docdict["encoding_nirx"] = """
encoding : str
    Text encoding of the NIRX header file. See :ref:`standard-encodings`.
"""

docdict["epochs_preload"] = """
    Load all epochs from disk when creating the object
    or wait before accessing each epoch (more memory
    efficient but can be slower).
"""

docdict["epochs_reject_tmin_tmax"] = """
reject_tmin, reject_tmax : float | None
    Start and end of the time window used to reject epochs based on
    peak-to-peak (PTP) amplitudes as specified via ``reject`` and ``flat``.
    The default ``None`` corresponds to the first and last time points of the
    epochs, respectively.

    .. note:: This parameter controls the time period used in conjunction with
              both, ``reject`` and ``flat``.
"""

docdict["epochs_tmin_tmax"] = """
tmin, tmax : float
    Start and end time of the epochs in seconds, relative to the time-locked
    event. The closest or matching samples corresponding to the start and end
    time are included. Defaults to ``-0.2`` and ``0.5``, respectively.
"""

docdict["equalize_events_method"] = """
method : ``'truncate'`` | ``'mintime'`` | ``'random'``
    If ``'truncate'``, events will be truncated from the end of each event
    list. If ``'mintime'``, timing differences between each event list will be
    minimized. If ``'random'``, events will be randomly selected from each event
    list.

    .. versionadded:: 1.8
"""

docdict["estimate_plot_psd"] = """\
estimate : str, {'power', 'amplitude'}
    Can be "power" for power spectral density (PSD; default), "amplitude" for
    amplitude spectrum density (ASD).
"""

docdict["event_color"] = """
event_color : color object | dict | None
    Color(s) to use for :term:`events`. To show all :term:`events` in the same
    color, pass any matplotlib-compatible color. To color events differently,
    pass a `dict` that maps event names or integer event numbers to colors
    (must include entries for *all* events, or include a "fallback" entry with
    key ``-1``). If ``None``, colors are chosen from the current Matplotlib
    color cycle.
"""

docdict["event_id"] = """
event_id : int | list of int | dict | str | list of str | None
    The id of the :term:`events` to consider. If dict, the keys can later be
    used to access associated :term:`events`. Example:
    dict(auditory=1, visual=3). If int, a dict will be created with the id as
    string. If a list of int, all :term:`events` with the IDs specified in the list
    are used. If a str or list of str, ``events`` must be ``None`` to use annotations
    and then the IDs must be the name(s) of the annotations to use.
    If None, all :term:`events` will be used and a dict is created
    with string integer names corresponding to the event id integers."""
_event_id_template = """
event_id : dict{or_none}
    Mapping from condition descriptions (strings) to integer event codes.{extra}"""
docdict["event_id_attr"] = _event_id_template.format(or_none="", extra="")
docdict["event_id_ecg"] = """
event_id : int
    The index to assign to found ECG events.
"""
docdict["event_id_epochstfr"] = _event_id_template.format(
    or_none=" | None",
    extra="""If ``None``,
    all events in ``events`` will be included, and the ``event_id`` attribute
    will be a :class:`dict` mapping a string version of each integer event ID
    to the corresponding integer.""",
)

docdict["event_repeated_epochs"] = """
event_repeated : str
    How to handle duplicates in ``events[:, 0]``. Can be ``'error'``
    (default), to raise an error, 'drop' to only retain the row occurring
    first in the :term:`events`, or ``'merge'`` to combine the coinciding
    events (=duplicates) into a new event (see Notes for details).

    .. versionadded:: 0.19
"""

_events_template = """
events : ndarray of int, shape (n_events, 3){or_none}
    The identity and timing of experimental events, around which the epochs were
    created. See :term:`events` for more information.{extra}
"""
docdict["events"] = _events_template.format(or_none="", extra="")
docdict["events_attr"] = """
events : ndarray of int, shape (n_events, 3)
    The events array."""
docdict["events_epochs"] = _events_template.format(
    or_none="",
    extra="""Events that don't match
    the events of interest as specified by ``event_id`` will be marked as
    ``IGNORED`` in the drop log.""",
)
docdict["events_epochstfr"] = _events_template.format(
    or_none=" | None",
    extra="""If ``None``, all integer
    event codes are set to ``1`` (i.e., all epochs are assumed to be of the same
    type) and their corresponding sample numbers are set as arbitrary, equally
    spaced sample numbers with a step size of ``len(times)``.""",
)

docdict["evoked_by_event_type_returns"] = """
evoked : instance of Evoked | list of Evoked
    The averaged epochs.
    When ``by_event_type=True`` was specified, a list is returned containing a
    separate :class:`~mne.Evoked` object for each event type. The list has the
    same order as the event types as specified in the ``event_id``
    dictionary.
"""

docdict["evoked_ylim_plot"] = """
ylim : dict | None
    Y-axis limits for plots (after scaling has been applied). :class:`dict`
    keys should match channel types; valid keys are for instance ``eeg``, ``mag``,
    ``grad``, ``misc``, ``csd``, .. (example: ``ylim=dict(eeg=[-20, 20])``). If
    ``None``, the y-axis limits will be set automatically by matplotlib. Defaults to
    ``None``."""

docdict["exclude_after_unique"] = """
exclude_after_unique : bool
    If True, exclude channels are searched for after they have been made
    unique. This is useful to choose channels that have been made unique
    by adding a suffix.  If False, the original names are checked.

    .. versionchanged:: 1.7
"""

docdict["exclude_clust"] = """
exclude : bool array or None
    Mask to apply to the data to exclude certain points from clustering
    (e.g., medial wall vertices). Should be the same shape as ``X``.
    If ``None``, no points are excluded.
"""

docdict["exclude_frontal"] = """
exclude_frontal : bool
    If True, exclude points that have both negative Z values
    (below the nasion) and positive Y values (in front of the LPA/RPA).
"""

_exclude_spectrum = """\
exclude : list of str | 'bads'
    Channel names to exclude{}. If ``'bads'``, channels
    in ``{}info['bads']`` are excluded; pass an empty list to
    include all channels (including "bad" channels, if any).
"""

docdict["exclude_psd"] = _exclude_spectrum.format("", "")
docdict["exclude_spectrum_get_data"] = _exclude_spectrum.format("", "spectrum.")
docdict["exclude_spectrum_plot"] = _exclude_spectrum.format(
    " from being drawn", "spectrum."
)

docdict["export_edf_note"] = """
Although this function supports storing channel types in the signal label (e.g.
``EEG Fz`` or ``MISC E``), other software may not support this (optional) feature of the
EDF standard.

If ``add_ch_type`` is True, then channel types are written based on what they are
currently set in MNE-Python. One should double check that all their channels are set
correctly. You can call :meth:`mne.io.Raw.set_channel_types` to set channel types.

In addition, EDF does not support storing a montage. You will need to store the montage
separately and call :meth:`mne.io.Raw.set_montage`.

The physical range of the signals is determined by signal type by default
(``physical_range="auto"``). However, if individual channel ranges vary significantly
due to the presence of e.g. drifts/offsets/biases, setting
``physical_range="channelwise"`` might be more appropriate. This will ensure a maximum
resolution for each individual channel, but some tools might not be able to handle this
appropriately (even though channel-wise ranges are covered by the EDF standard).
"""

docdict["export_eeglab_note"] = """
For EEGLAB exports, channel locations are expanded to full EEGLAB format.
For more details see :func:`eeglabio.utils.cart_to_eeglab`.
"""

_export_fmt_params_base = """\
Format of the export. Defaults to ``'auto'``, which will infer the format
    from the filename extension. See supported formats above for more
    information."""

docdict["export_fmt_params_epochs"] = f"""
fmt : 'auto' | 'eeglab'
    {_export_fmt_params_base}
"""

docdict["export_fmt_params_evoked"] = f"""
fmt : 'auto' | 'mff'
    {_export_fmt_params_base}
"""

docdict["export_fmt_params_raw"] = f"""
fmt : 'auto' | 'brainvision' | 'edf' | 'eeglab'
    {_export_fmt_params_base}
"""

docdict["export_fmt_support_epochs"] = """\
Supported formats:
    - EEGLAB (``.set``, uses :mod:`eeglabio`)
"""

docdict["export_fmt_support_evoked"] = """\
Supported formats:
    - MFF (``.mff``, uses :func:`mne.export.export_evokeds_mff`)
"""

docdict["export_fmt_support_raw"] = """\
Supported formats:
    - BrainVision (``.vhdr``, ``.vmrk``, ``.eeg``, uses `pybv <https://github.com/bids-standard/pybv>`_)
    - EEGLAB (``.set``, uses :mod:`eeglabio`)
    - EDF (``.edf``, uses `edfio <https://github.com/the-siesta-group/edfio>`_)
"""  # noqa: E501

docdict["export_warning"] = """\
.. warning::
    Since we are exporting to external formats, there's no guarantee that all
    the info will be preserved in the external format. See Notes for details.
"""

_export_warning_note_base = """\
Export to external format may not preserve all the information from the
instance. To save in native MNE format (``.fif``) without information loss,
use :meth:`mne.{0}.save` instead.
Export does not apply projector(s). Unapplied projector(s) will be lost.
Consider applying projector(s) before exporting with
:meth:`mne.{0}.apply_proj`."""

docdict["export_warning_note_epochs"] = _export_warning_note_base.format("Epochs")

docdict["export_warning_note_evoked"] = _export_warning_note_base.format("Evoked")

docdict["export_warning_note_raw"] = _export_warning_note_base.format("io.Raw")

docdict["ext_order_chpi"] = """
ext_order : int
    The external order for SSS-like interfence suppression.
    The SSS bases are used as projection vectors during fitting.

    .. versionchanged:: 0.20
        Added ``ext_order=1`` by default, which should improve
        detection of true HPI signals.
"""

docdict["ext_order_maxwell"] = """
ext_order : int
    Order of external component of spherical expansion.
"""

docdict["extended_proj_maxwell"] = """
extended_proj : list
    The empty-room projection vectors used to extend the external
    SSS basis (i.e., use eSSS).

    .. versionadded:: 0.21
"""

docdict["extrapolate_topomap"] = """
extrapolate : str
    Options:

    - ``'box'``
        Extrapolate to four points placed to form a square encompassing all
        data points, where each side of the square is three times the range
        of the data in the respective dimension.
    - ``'local'`` (default for MEG sensors)
        Extrapolate only to nearby points (approximately to points closer than
        median inter-electrode distance). This will also set the
        mask to be polygonal based on the convex hull of the sensors.
    - ``'head'`` (default for non-MEG sensors)
        Extrapolate out to the edges of the clipping circle. This will be on
        the head circle when the sensors are contained within the head circle,
        but it can extend beyond the head when sensors are plotted outside
        the head circle.
"""

docdict["eyelink_apply_offsets"] = """
apply_offsets : bool (default False)
    Adjusts the onset time of the :class:`~mne.Annotations` created from Eyelink
    experiment messages, if offset values exist in the ASCII file. If False, any
    offset-like values will be prepended to the annotation description.
"""

docdict["eyelink_create_annotations"] = """
create_annotations : bool | list (default True)
    Whether to create :class:`~mne.Annotations` from occular events
    (blinks, fixations, saccades) and experiment messages. If a list, must
    contain one or more of ``['fixations', 'saccades',' blinks', messages']``.
    If True, creates :class:`~mne.Annotations` for both occular events and
    experiment messages.
"""

docdict["eyelink_find_overlaps"] = """
find_overlaps : bool (default False)
    Combine left and right eye :class:`mne.Annotations` (blinks, fixations,
    saccades) if their start times and their stop times are both not
    separated by more than overlap_threshold.
"""

docdict["eyelink_fname"] = """
fname : path-like
    Path to the eyelink file (``.asc``)."""

docdict["eyelink_overlap_threshold"] = """
overlap_threshold : float (default 0.05)
    Time in seconds. Threshold of allowable time-gap between both the start and
    stop times of the left and right eyes. If the gap is larger than the threshold,
    the :class:`mne.Annotations` will be kept separate (i.e. ``"blink_L"``,
    ``"blink_R"``). If the gap is smaller than the threshold, the
    :class:`mne.Annotations` will be merged and labeled as ``"blink_both"``.
    Defaults to ``0.05`` seconds (50 ms), meaning that if the blink start times of
    the left and right eyes are separated by less than 50 ms, and the blink stop
    times of the left and right eyes are separated by less than 50 ms, then the
    blink will be merged into a single :class:`mne.Annotations`.
"""

# %%
# F

docdict["f_power_clust"] = """
t_power : float
    Power to raise the statistical values (usually F-values) by before
    summing (sign will be retained). Note that ``t_power=0`` will give a
    count of locations in each cluster, ``t_power=1`` will weight each location
    by its statistical score.
"""

docdict["fiducials"] = """
fiducials : list | dict | str
    The fiducials given in the MRI (surface RAS) coordinate
    system. If a dictionary is provided, it must contain the **keys**
    ``'lpa'``, ``'rpa'``, and ``'nasion'``, with **values** being the
    respective coordinates in meters.
    If a list, it must be a list of ``DigPoint`` instances as returned by the
    :func:`mne.io.read_fiducials` function.
    If ``'estimated'``, the fiducials are derived from the ``fsaverage``
    template. If ``'auto'`` (default), tries to find the fiducials
    in a file with the canonical name
    (``{subjects_dir}/{subject}/bem/{subject}-fiducials.fif``)
    and if absent, falls back to ``'estimated'``.
"""

docdict["fig_background"] = """
fig_background : None | array
    A background image for the figure. This must be a valid input to
    :func:`matplotlib.pyplot.imshow`. Defaults to ``None``.
"""
docdict["fig_facecolor"] = """
fig_facecolor : str | tuple
    A matplotlib-compatible color to use for the figure background. Defaults to black.
"""

docdict["filter_length"] = """
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

docdict["filter_length_ecg"] = """
filter_length : str | int | None
    Number of taps to use for filtering.
"""

docdict["filter_length_notch"] = """
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

    When ``method=='spectrum_fit'``, this sets the effective window duration
    over which fits are computed. See :func:`mne.filter.create_filter`
    for options. Longer window lengths will give more stable frequency
    estimates, but require (potentially much) more processing and are not able
    to adapt as well to non-stationarities.

    The default in 0.21 is None, but this will change to ``'10s'`` in 0.22.
"""

docdict["fir_design"] = """
fir_design : str
    Can be "firwin" (default) to use :func:`scipy.signal.firwin`,
    or "firwin2" to use :func:`scipy.signal.firwin2`. "firwin" uses
    a time-domain design technique that generally gives improved
    attenuation using fewer samples than "firwin2".

    .. versionadded:: 0.15
"""

docdict["fir_window"] = """
fir_window : str
    The window to use in FIR design, can be "hamming" (default),
    "hann" (default in 0.13), or "blackman".

    .. versionadded:: 0.15
"""

_flat_common = """\
    Reject epochs based on **minimum** peak-to-peak signal amplitude (PTP).
    Valid **keys** can be any channel type present in the object. The
    **values** are floats that set the minimum acceptable PTP. If the PTP
    is smaller than this threshold, the epoch will be dropped. If ``None``
    then no rejection is performed based on flatness of the signal."""

docdict["flat"] = f"""
flat : dict | None
{_flat_common}

    .. note:: To constrain the time period used for estimation of signal
              quality, pass the ``reject_tmin`` and ``reject_tmax`` parameters.
"""

docdict["flat_drop_bad"] = """
flat : dict | str | None
    Reject epochs based on **minimum** peak-to-peak signal amplitude (PTP)
    or a custom function. Valid **keys** can be any channel type present
    in the object. If using PTP, **values** are floats that set the minimum
    acceptable PTP. If the PTP is smaller than this threshold, the epoch
    will be dropped. If ``None`` then no rejection is performed based on
    flatness of the signal. If a custom function is used than ``flat`` can be
    used to reject epochs based on any criteria (including maxima and
    minima).
    If ``'existing'``, then the flat parameters set during epoch creation are
    used.
"""

_fmin_fmax = """\
fmin, fmax : float
    The lower- and upper-bound on frequencies of interest. Default is
    {}"""

docdict["fmin_fmax_psd"] = _fmin_fmax.format(
    "``fmin=0, fmax=np.inf`` (spans all frequencies present in the data)."
)

docdict["fmin_fmax_psd_topo"] = _fmin_fmax.format("``fmin=0, fmax=100``.")
docdict["fmin_fmax_tfr"] = _fmin_fmax.format(
    """``None``
    which is equivalent to ``fmin=0, fmax=np.inf`` (spans all frequencies
    present in the data)."""
)

docdict["fmin_fmid_fmax"] = """
fmin : float
    Minimum value in colormap (uses real fmin if None).
fmid : float
    Intermediate value in colormap (fmid between fmin and
    fmax if None).
fmax : float
    Maximum value in colormap (uses real max if None).
"""

docdict["fname_epochs"] = """
fname : path-like | file-like
    The epochs to load. If a filename, should end with ``-epo.fif`` or
    ``-epo.fif.gz``. If a file-like object, preloading must be used.
"""

docdict["fname_export_params"] = """
fname : str
    Name of the output file.
"""

docdict["fname_fwd"] = """
fname : path-like
    File name to save the forward solution to. It should end with
    ``-fwd.fif`` or ``-fwd.fif.gz`` to save to FIF, or ``-fwd.h5`` to save to
    HDF5.
"""

docdict["fnirs"] = """
fnirs : str | list | dict | bool | None
    Can be "channels", "pairs", "detectors", and/or "sources" to show the
    fNIRS channel locations, optode locations, or line between
    source-detector pairs, or a combination like ``('pairs', 'channels')``.
    True translates to ``('pairs',)``. A dict can also be used to specify
    alpha values (but only "channels" and "pairs" will be used), e.g.
    ``dict(channels=0.2, pairs=0.7)``.

    .. versionchanged:: 1.6
       Added support for specifying alpha values as a dict.
"""

docdict["focalpoint"] = """
focalpoint : tuple, shape (3,) | str | None
    The focal point of the camera rendering the view: (x, y, z) in
    plot units (either m or mm). When ``"auto"``, it is set to the center of
    mass of the visible bounds.
"""

docdict["font_color"] = """
font_color : color
    The color of tick labels in the colorbar. Defaults to white.
"""

docdict["forward_set_eeg_reference"] = """
forward : instance of Forward | None
    Forward solution to use. Only used with ``ref_channels='REST'``.

    .. versionadded:: 0.21
"""
_freqs_tfr_template = """
freqs : array-like |{auto} None
    The frequencies at which to compute the power estimates.
    {stockwell} be an array of shape (n_freqs,). ``None`` (the
    default) only works when using ``__setstate__`` and will raise an error otherwise.
"""
docdict["freqs_tfr"] = _freqs_tfr_template.format(auto="", stockwell="Must")
docdict["freqs_tfr_array"] = """
freqs : ndarray, shape (n_freqs,)
    The frequencies in Hz.
"""
docdict["freqs_tfr_attr"] = """
freqs : array
    Frequencies at which power has been computed."""
docdict["freqs_tfr_epochs"] = _freqs_tfr_template.format(
    auto=" 'auto' | ",
    stockwell="""If ``method='stockwell'`` this must be a length 2 iterable specifying lowest
    and highest frequencies, or ``'auto'`` (to use all available frequencies).
    For other methods, must""",  # noqa E501
)

docdict["fullscreen"] = """
fullscreen : bool
    Whether to start in fullscreen (``True``) or windowed mode
    (``False``).
"""

applyfun_fun_base = """
fun : callable
    A function to be applied to the channels. The first argument of
    fun has to be a timeseries (:class:`numpy.ndarray`). The function must
    operate on an array of shape ``(n_times,)`` {}.
    The function must return an :class:`~numpy.ndarray` shaped like its input.

    .. note::
        If ``channel_wise=True``, one can optionally access the index and/or the
        name of the currently processed channel within the applied function.
        This can enable tailored computations for different channels.
        To use this feature, add ``ch_idx`` and/or ``ch_name`` as
        additional argument(s) to your function definition.
"""
docdict["fun_applyfun"] = applyfun_fun_base.format(
    " if ``channel_wise=True`` and ``(len(picks), n_times)`` otherwise"
)
docdict["fun_applyfun_evoked"] = applyfun_fun_base.format(
    " because it will apply channel-wise"
)
docdict["fun_applyfun_stc"] = applyfun_fun_base.format(
    " because it will apply vertex-wise"
)

docdict["fwd"] = """
fwd : instance of Forward
    The forward solution. If present, the orientations of the dipoles
    present in the forward solution are displayed.
"""

docdict["fwhm_morlet_notes"] = r"""
Convolution of a signal with a Morlet wavelet will impose temporal smoothing
that is determined by the duration of the wavelet. In MNE-Python, the duration
of the wavelet is determined by the ``sigma`` parameter, which gives the
standard deviation of the wavelet's Gaussian envelope (our wavelets extend to
±5 standard deviations to ensure values very close to zero at the endpoints).
Some authors (e.g., :footcite:t:`Cohen2019`) recommend specifying and reporting
wavelet duration in terms of the full-width half-maximum (FWHM) of the
wavelet's Gaussian envelope. The FWHM is related to ``sigma`` by the following
identity: :math:`\mathrm{FWHM} = \sigma \times 2 \sqrt{2 \ln{2}}` (or the
equivalent in Python code: ``fwhm = sigma * 2 * np.sqrt(2 * np.log(2))``).
If ``sigma`` is not provided, it is computed from ``n_cycles`` as
:math:`\frac{\mathtt{n\_cycles}}{2 \pi f}` where :math:`f` is the frequency of
the wavelet oscillation (given by ``freqs``). Thus when ``sigma=None`` the FWHM
will be given by

.. math::

    \mathrm{FWHM} = \frac{\mathtt{n\_cycles} \times \sqrt{2 \ln{2}}}{\pi \times f}

(cf. eq. 4 in :footcite:`Cohen2019`). To create wavelets with a chosen FWHM,
one can compute::

    n_cycles = desired_fwhm * np.pi * np.array(freqs) / np.sqrt(2 * np.log(2))

to get an array of values for ``n_cycles`` that yield the desired FWHM at each
frequency in ``freqs``.  If you want different FWHM values at each frequency,
do the same computation with ``desired_fwhm`` as an array of the same shape as
``freqs``.
"""

# %%
# G

docdict["get_peak_parameters"] = """
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

_getitem_spectrum_base = """
data : ndarray
    The selected spectral data. Shape will be
    ``({n_epo}n_channels, n_freqs)`` for normal power spectra,
    ``({n_epo}n_channels, n_freqs, n_segments)`` for unaggregated
    Welch estimates, or ``({n_epo}n_channels, n_tapers, n_freqs)``
    for unaggregated multitaper estimates.
"""
_getitem_tfr_base = """
data : ndarray
    The selected time-frequency data. Shape will be
    ``({n_epo}n_channels, n_freqs, n_times)`` for Morlet, Stockwell, and aggregated
    (``output='power'``) multitaper methods, or
    ``({n_epo}n_channels, n_tapers, n_freqs, n_times)`` for unaggregated
    (``output='complex'``) multitaper method.
"""
n_epo = "n_epochs, "
docdict["getitem_epochspectrum_return"] = _getitem_spectrum_base.format(n_epo=n_epo)
docdict["getitem_epochstfr_return"] = _getitem_tfr_base.format(n_epo=n_epo)
docdict["getitem_spectrum_return"] = _getitem_spectrum_base.format(n_epo="")
docdict["getitem_tfr_return"] = _getitem_tfr_base.format(n_epo="")


docdict["group_by_browse"] = """
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

# %%
# H

docdict["h_freq"] = """
h_freq : float | None
    For FIR filters, the upper pass-band edge; for IIR filters, the upper
    cutoff frequency. If None the data are only high-passed.
"""

docdict["h_trans_bandwidth"] = """
h_trans_bandwidth : float | str
    Width of the transition band at the high cut-off frequency in Hz
    (low pass or cutoff 2 in bandpass). Can be "auto"
    (default in 0.14) to use a multiple of ``h_freq``::

        min(max(h_freq * 0.25, 2.), info['sfreq'] / 2. - h_freq)

    Only used for ``method='fir'``.
"""

docdict["head_pos"] = """
head_pos : None | path-like | dict | tuple | array
    Path to the position estimates file. Should be in the format of
    the files produced by MaxFilter. If dict, keys should
    be the time points and entries should be 4x4 ``dev_head_t``
    matrices. If None, the original head position (from
    ``info['dev_head_t']``) will be used. If tuple, should have the
    same format as data returned by ``head_pos_to_trans_rot_t``.
    If array, should be of the form returned by
    :func:`mne.chpi.read_head_pos`.
"""

docdict["head_pos_maxwell"] = """
head_pos : array | None
    If array, movement compensation will be performed.
    The array should be of shape (N, 10), holding the position
    parameters as returned by e.g. ``read_head_pos``.
"""

docdict["head_source"] = """
head_source : str | list of str
    Head source(s) to use. See the ``source`` option of
    :func:`mne.get_head_surf` for more information.
"""

docdict["hitachi_fname"] = """
fname : list | str
    Path(s) to the Hitachi CSV file(s). This should only be a list for
    multiple probes that were acquired simultaneously.

    .. versionchanged:: 1.2
        Added support for list-of-str.
"""

docdict["hitachi_notes"] = """
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

# %%
# I

docdict["idx_pctf"] = """
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

docdict["ignore_ref_maxwell"] = """
ignore_ref : bool
    If True, do not include reference channels in compensation. This
    option should be True for KIT files, since Maxwell filtering
    with reference channels is not currently supported.
"""

docdict["iir_params"] = """
iir_params : dict | None
    Dictionary of parameters to use for IIR filtering.
    If ``iir_params=None`` and ``method="iir"``, 4th order Butterworth will be used.
    For more information, see :func:`mne.filter.construct_iir_filter`.
"""

docdict["image_args"] = """
image_args : dict | None
    Keyword arguments to pass to :meth:`mne.time_frequency.AverageTFR.plot`. ``axes``
    and ``show`` are ignored. Defaults to ``None`` (i.e., and empty :class:`dict`).
"""

docdict["image_format_report"] = """
image_format : 'png' | 'svg' | 'gif' | None
    The image format to be used for the report, can be ``'png'``,
    ``'svg'``, or ``'gif'``.
    None (default) will use the default specified during `~mne.Report`
    instantiation.
"""

docdict["image_interp_topomap"] = """
image_interp : str
    The image interpolation to be used. Options are ``'cubic'`` (default)
    to use :class:`scipy.interpolate.CloughTocher2DInterpolator`,
    ``'nearest'`` to use :class:`scipy.spatial.Voronoi` or
    ``'linear'`` to use :class:`scipy.interpolate.LinearNDInterpolator`.
"""

docdict["include_tmax"] = """
include_tmax : bool
    If True (default), include tmax. If False, exclude tmax (similar to how
    Python indexing typically works).

    .. versionadded:: 0.19
"""

_index_df_base = """
index : {} | None
    Kind of index to use for the DataFrame. If ``None``, a sequential
    integer index (:class:`pandas.RangeIndex`) will be used. If ``'time'``, a
    ``pandas.Index``{} or :class:`pandas.TimedeltaIndex` will be used
    (depending on the value of ``time_format``). {}
"""

datetime = ", :class:`pandas.DatetimeIndex`,"
multiindex = (
    "If a list of two or more string values, a "
    ":class:`pandas.MultiIndex` will be created. "
)
raw = ("'time'", datetime, "")
epo = ("str | list of str", "", multiindex)
evk = ("'time'", "", "")

docdict["index_df_epo"] = _index_df_base.format(*epo)
docdict["index_df_evk"] = _index_df_base.format(*evk)
docdict["index_df_raw"] = _index_df_base.format(*raw)

_info_base = (
    "The :class:`mne.Info` object with information about the "
    "sensors and methods of measurement."
)

docdict["info"] = f"""
info : mne.Info | None
    {_info_base}
"""

docdict["info_not_none"] = f"""
info : mne.Info
    {_info_base}
"""

docdict["info_str"] = f"""
info : mne.Info | path-like
    {_info_base} If ``path-like``, it should be a :class:`str` or
    :class:`pathlib.Path` to a file with measurement information
    (e.g. :class:`mne.io.Raw`).
"""

docdict["inst_tfr"] = """
inst : instance of RawTFR, EpochsTFR, or AverageTFR
"""

docdict["int_order_maxwell"] = """
int_order : int
    Order of internal component of spherical expansion.
"""

docdict["interaction_scene"] = """
interaction : 'trackball' | 'terrain'
    How interactions with the scene via an input device (e.g., mouse or
    trackpad) modify the camera position. If ``'terrain'``, one axis is
    fixed, enabling "turntable-style" rotations. If ``'trackball'``,
    movement along all axes is possible, which provides more freedom of
    movement, but you may incidentally perform unintentional rotations along
    some axes.
"""

docdict["interaction_scene_none"] = """
interaction : 'trackball' | 'terrain' | None
    How interactions with the scene via an input device (e.g., mouse or
    trackpad) modify the camera position. If ``'terrain'``, one axis is
    fixed, enabling "turntable-style" rotations. If ``'trackball'``,
    movement along all axes is possible, which provides more freedom of
    movement, but you may incidentally perform unintentional rotations along
    some axes.
    If ``None``, the setting stored in the MNE-Python configuration file is
    used.
"""

docdict["interp"] = """
interp : str
    Either ``'hann'``, ``'cos2'`` (default), ``'linear'``, or ``'zero'``, the type of
    forward-solution interpolation to use between forward solutions
    at different head positions.
"""

docdict["interpolation_brain_time"] = """
interpolation : str | None
    Interpolation method (:class:`scipy.interpolate.interp1d` parameter).
    Must be one of ``'linear'``, ``'nearest'``, ``'zero'``, ``'slinear'``,
    ``'quadratic'`` or ``'cubic'``.
"""

docdict["inversion_bf"] = """
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

docdict["item"] = """
item : int | slice | array-like | str
"""

# %%
# J

docdict["joint_set_eeg_reference"] = """
joint : bool
    How to handle list-of-str ``ch_type``. If False (default), one projector
    is created per channel type. If True, one projector is created across
    all channel types. This is only used when ``projection=True``.

    .. versionadded:: 1.2
"""

# %%
# K

docdict["keep_his_anonymize_info"] = """
keep_his : bool
    If ``True``, ``his_id`` of ``subject_info`` will **not** be overwritten.
    Defaults to ``False``.

    .. warning:: This could mean that ``info`` is not fully
                 anonymized. Use with caution.
"""

docdict["kit_badcoils"] = """
bad_coils : array-like of int | None
    Indices of (up to two) bad marker coils to be removed.
    These marker coils must be present in the elp and mrk files.
"""

docdict["kit_elp"] = """
elp : path-like | array of shape (8, 3) | None
    Digitizer points representing the location of the fiducials and the
    marker coils with respect to the digitized head shape, or path to a
    file containing these points.
"""

docdict["kit_hsp"] = """
hsp : path-like | array of shape (n_points, 3) | None
    Digitizer head shape points, or path to head shape file. If more than
    10,000 points are in the head shape, they are automatically decimated.
"""

docdict["kit_mrk"] = """
mrk : path-like | array of shape (5, 3) | list | None
    Marker points representing the location of the marker coils with
    respect to the MEG sensors, or path to a marker file.
    If list, all of the markers will be averaged together.
"""

docdict["kit_slope"] = r"""
slope : ``'+'`` | ``'-'``
    How to interpret values on KIT trigger channels when synthesizing a
    Neuromag-style stim channel. With ``'+'``\, a positive slope (low-to-high)
    is interpreted as an event. With ``'-'``\, a negative slope (high-to-low)
    is interpreted as an event.
"""

docdict["kit_stim"] = r"""
stim : list of int | ``'<'`` | ``'>'`` | None
    Channel-value correspondence when converting KIT trigger channels to a
    Neuromag-style stim channel. For ``'<'``\, the largest values are
    assigned to the first channel (default). For ``'>'``\, the largest
    values are assigned to the last channel. Can also be specified as a
    list of trigger channel indexes. If None, no synthesized channel is
    generated.
"""

docdict["kit_stimcode"] = """
stim_code : ``'binary'`` | ``'channel'``
    How to decode trigger values from stim channels. ``'binary'`` read stim
    channel events as binary code, 'channel' encodes channel number.
"""

docdict["kit_stimthresh"] = """
stimthresh : float | None
    The threshold level for accepting voltage changes in KIT trigger
    channels as a trigger event. If None, stim must also be set to None.
"""

docdict["kwargs_fun"] = """
**kwargs : dict
    Additional keyword arguments to pass to ``fun``.
"""

# %%
# L

docdict["l_freq"] = """
l_freq : float | None
    For FIR filters, the lower pass-band edge; for IIR filters, the lower
    cutoff frequency. If None the data are only low-passed.
"""

docdict["l_freq_ecg_filter"] = """
l_freq : float
    Low pass frequency to apply to the ECG channel while finding events.
h_freq : float
    High pass frequency to apply to the ECG channel while finding events.
"""

docdict["l_trans_bandwidth"] = """
l_trans_bandwidth : float | str
    Width of the transition band at the low cut-off frequency in Hz
    (high pass or cutoff 1 in bandpass). Can be "auto"
    (default) to use a multiple of ``l_freq``::

        min(max(l_freq * 0.25, 2), l_freq)

    Only used for ``method='fir'``.
"""

docdict["label_tc_el_returns"] = """
label_tc : array | list (or generator) of array, shape (n_labels[, n_orient], n_times)
    Extracted time course for each label and source estimate.
"""

docdict["labels_eltc"] = """
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
docdict["layout_scale"] = """
layout_scale : float
    Scaling factor for adjusting the relative size of the layout on the canvas.
"""
docdict["layout_spectrum_plot_topo"] = """\
layout : instance of Layout | None
    Layout instance specifying sensor positions (does not need to be
    specified for Neuromag data). If ``None`` (default), the layout is
    inferred from the data (if possible).
"""

docdict["line_alpha_plot_psd"] = """\
line_alpha : float | None
    Alpha for the PSD line. Can be None (default) to use 1.0 when
    ``average=True`` and 0.1 when ``average=False``.
"""

_long_format_df_base = """
long_format : bool
    If True, the DataFrame is returned in long format where each row is one
    observation of the signal at a unique combination of {}.
    {}Defaults to ``False``.
"""

ch_type = (
    "For convenience, a ``ch_type`` column is added to facilitate "
    "subsetting the resulting DataFrame. "
)
raw = ("time point and channel", ch_type)
epo = ("time point, channel, epoch number, and condition", ch_type)
stc = ("time point and vertex", "")
spe = ("frequency and channel", ch_type)

docdict["long_format_df_epo"] = _long_format_df_base.format(*epo)
docdict["long_format_df_raw"] = _long_format_df_base.format(*raw)
docdict["long_format_df_spe"] = _long_format_df_base.format(*spe)
docdict["long_format_df_stc"] = _long_format_df_base.format(*stc)

docdict["loose"] = """
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

# %%
# M

docdict["mag_scale_maxwell"] = """
mag_scale : float | str
    The magenetometer scale-factor used to bring the magnetometers
    to approximately the same order of magnitude as the gradiometers
    (default 100.), as they have different units (T vs T/m).
    Can be ``'auto'`` to use the reciprocal of the physical distance
    between the gradiometer pickup loops (e.g., 0.0168 m yields
    59.5 for VectorView).
"""

docdict["mapping_rename_channels_duplicates"] = """
mapping : dict | callable
    A dictionary mapping the old channel to a new channel name
    e.g. ``{'EEG061' : 'EEG161'}``. Can also be a callable function
    that takes and returns a string.

    .. versionchanged:: 0.10.0
       Support for a callable function.
allow_duplicates : bool
    If True (default False), allow duplicates, which will automatically
    be renamed with ``-N`` at the end.

    .. versionadded:: 0.22.0
"""

_mask_base = """
mask : ndarray of bool, shape {shape} | None
    Array indicating channel{shape_appendix} to highlight with a distinct
    plotting style{example}. Array elements set to ``True`` will be plotted
    with the parameters given in ``mask_params``. Defaults to ``None``,
    equivalent to an array of all ``False`` elements.
"""
docdict["mask_alpha_tfr_plot"] = """
mask_alpha : float
    Relative opacity of the masked region versus the unmasked region, given as a
    :class:`float` between 0 and 1 (i.e., 0 means masked areas are not visible at all).
    Defaults to ``0.1``.
"""
docdict["mask_cmap_tfr_plot"] = """
mask_cmap : matplotlib colormap | str | None
    Colormap to use for masked areas of the plot. If a :class:`str`, must be a valid
    Matplotlib colormap name. If None, ``cmap`` is used for both masked and unmasked
    areas. Ignored if ``mask`` is ``None``. Default is ``'Greys'``.
"""
docdict["mask_evoked_topomap"] = _mask_base.format(
    shape="(n_channels, n_times)",
    shape_appendix="-time combinations",
    example=" (useful for, e.g. marking which channels at which times a "
    "statistical test of the data reaches significance)",
)
docdict["mask_params_topomap"] = """
mask_params : dict | None
    Additional plotting parameters for plotting significant sensors.
    Default (None) equals::

        dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                linewidth=0, markersize=4)
"""
docdict["mask_patterns_topomap"] = _mask_base.format(
    shape="(n_channels, n_patterns)", shape_appendix="-pattern combinations", example=""
)
docdict["mask_style_tfr_plot"] = """
mask_style : None | 'both' | 'contour' | 'mask'
    How to distinguish the masked/unmasked regions of the plot. If ``"contour"``, a
    line is drawn around the areas where the mask is ``True``. If ``"mask"``, areas
    where the mask is ``False`` will be (partially) transparent, as determined by
    ``mask_alpha``. If ``"both"``, both a contour and transparency are used. Default is
    ``None``, which is silently ignored if ``mask`` is ``None`` and is interpreted like
    ``"both"`` otherwise.
"""
docdict["mask_tfr_plot"] = """
mask : ndarray | None
    An :class:`array <numpy.ndarray>` of :class:`boolean <bool>` values, of the same
    shape as the data. Data that corresponds to ``False`` entries in the mask are
    plotted differently, as determined by ``mask_style``, ``mask_alpha``, and
    ``mask_cmap``. Useful for, e.g., highlighting areas of statistical significance.
"""
docdict["mask_topomap"] = _mask_base.format(
    shape="(n_channels,)", shape_appendix="(s)", example=""
)

docdict["match_alias"] = """
match_alias : bool | dict
    Whether to use a lookup table to match unrecognized channel location names
    to their known aliases. If True, uses the mapping in
    ``mne.io.constants.CHANNEL_LOC_ALIASES``. If a :class:`dict` is passed, it
    will be used instead, and should map from non-standard channel names to
    names in the specified ``montage``. Default is ``False``.

    .. versionadded:: 0.23
"""

docdict["match_case"] = """
match_case : bool
    If True (default), channel name matching will be case sensitive.

    .. versionadded:: 0.20
"""

docdict["max_dist_ieeg"] = """
max_dist : float
    The maximum distance to project a sensor to the pial surface in meters.
    Sensors that are greater than this distance from the pial surface will
    not be assigned locations. Projections can be done to the inflated or
    flat brain.
"""

docdict["max_iter_multitaper"] = """
max_iter : int
    Maximum number of iterations to reach convergence when combining the
    tapered spectra with adaptive weights (see argument ``adaptive``). This
    argument has not effect if ``adaptive`` is set to ``False``."""

docdict["max_step_clust"] = """
max_step : int
    Maximum distance between samples along the second axis of ``X`` to be
    considered adjacent (typically the second axis is the "time" dimension).
    Only used when ``adjacency`` has shape (n_vertices, n_vertices), that is,
    when adjacency is only specified for sensors (e.g., via
    :func:`mne.channels.find_ch_adjacency`), and not via sensors **and**
    further dimensions such as time points (e.g., via an additional call of
    :func:`mne.stats.combine_adjacency`).
"""

docdict["measure"] = """
measure : 'zscore' | 'correlation'
    Which method to use for finding outliers among the components:

    - ``'zscore'`` (default) is the iterative z-scoring method. This method
      computes the z-score of the component's scores and masks the components
      with a z-score above threshold. This process is repeated until no
      supra-threshold component remains.
    - ``'correlation'`` is an absolute raw correlation threshold ranging from 0
      to 1.

    .. versionadded:: 0.21"""

docdict["meg"] = """
meg : str | list | dict | bool | None
    Can be "helmet", "sensors" or "ref" to show the MEG helmet, sensors or
    reference sensors respectively, or a combination like
    ``('helmet', 'sensors')`` (same as None, default). True translates to
    ``('helmet', 'sensors', 'ref')``. Can also be a dict to specify alpha values,
    e.g. ``{"helmet": 0.1, "sensors": 0.8}``.

    .. versionchanged:: 1.6
       Added support for specifying alpha values as a dict.
"""

_metadata_attr_template = """
metadata : instance of pandas.DataFrame | None
    A :class:`pandas.DataFrame` specifying metadata about each epoch{or_none}.{extra}
"""
_metadata_template = _metadata_attr_template.format(
    or_none="",
    extra="""
    If not ``None``, ``len(metadata)`` must equal ``len(events)``. For
    save/load compatibility, the :class:`~pandas.DataFrame` may only contain
    :class:`str`, :class:`int`, :class:`float`, and :class:`bool` values.
    If not ``None``, then pandas-style queries may be used to select
    subsets of data, see :meth:`mne.Epochs.__getitem__`. When the {obj} object
    is subsetted, the metadata is subsetted accordingly, and the row indices
    will be modified to match ``{obj}.selection``.""",
)
docdict["metadata_attr"] = _metadata_attr_template.format(
    or_none=" (or ``None``)", extra=""
)
docdict["metadata_epochs"] = _metadata_template.format(obj="Epochs")
docdict["metadata_epochstfr"] = _metadata_template.format(obj="EpochsTFR")

docdict["method_fir"] = """
method : str
    ``'fir'`` will use overlap-add FIR filtering, ``'iir'`` will use IIR
    forward-backward filtering (via :func:`~scipy.signal.filtfilt`).
"""

_method_kw_tfr_template = """
**method_kw
    Additional keyword arguments passed to the spectrotemporal estimation function
    (e.g., ``n_cycles, use_fft, zero_mean`` for Morlet method{stockwell}
    or ``n_cycles, use_fft, zero_mean, time_bandwidth`` for multitaper method).
    See :func:`~mne.time_frequency.tfr_array_morlet`{stockwell_crossref}
    and :func:`~mne.time_frequency.tfr_array_multitaper` for additional details.
"""

docdict["method_kw_epochs_tfr"] = _method_kw_tfr_template.format(
    stockwell=", ``n_fft, width`` for Stockwell method,",
    stockwell_crossref=", :func:`~mne.time_frequency.tfr_array_stockwell`,",
)

docdict["method_kw_psd"] = """\
**method_kw
    Additional keyword arguments passed to the spectral estimation
    function (e.g., ``n_fft, n_overlap, n_per_seg, average, window``
    for Welch method, or ``bandwidth, adaptive, low_bias, normalization``
    for multitaper method). See :func:`~mne.time_frequency.psd_array_welch`
    and :func:`~mne.time_frequency.psd_array_multitaper` for details. Note
    that for Welch method if ``n_fft`` is unspecified its default will be
    the smaller of ``2048`` or the number of available time samples (taking into
    account ``tmin`` and ``tmax``), not ``256`` as in
    :func:`~mne.time_frequency.psd_array_welch`.
"""

docdict["method_kw_tfr"] = _method_kw_tfr_template.format(
    stockwell="", stockwell_crossref=""
)

_method_psd = """
method : ``'welch'`` | ``'multitaper'``{}
    Spectral estimation method. ``'welch'`` uses Welch's
    method :footcite:p:`Welch1967`, ``'multitaper'`` uses DPSS
    tapers :footcite:p:`Slepian1978`.{}
"""
docdict["method_plot_psd_auto"] = _method_psd.format(
    " | ``'auto'``",
    (
        " ``'auto'`` (default) uses Welch's method for continuous data and "
        "multitaper for :class:`~mne.Epochs` or :class:`~mne.Evoked` data."
    ),
)
docdict["method_psd"] = _method_psd.format("", "")
docdict["method_psd_auto"] = _method_psd.format(" | ``'auto'``", "")

docdict["method_resample"] = """
method : str
    Resampling method to use. Can be ``"fft"`` (default) or ``"polyphase"``
    to use FFT-based on polyphase FIR resampling, respectively. These wrap to
    :func:`scipy.signal.resample` and :func:`scipy.signal.resample_poly`, respectively.
"""

_method_tfr_template = """
method : ``'morlet'`` | ``'multitaper'``{literals} | None
    Spectrotemporal power estimation method. ``'morlet'`` uses Morlet wavelets,
    ``'multitaper'`` uses DPSS tapers :footcite:p:`Slepian1978`{cites}. ``None`` (the
    default) only works when using ``__setstate__`` and will raise an error otherwise.
"""
docdict["method_tfr"] = _method_tfr_template.format(literals="", cites="")
docdict["method_tfr_array"] = """
method : str | None
    Comment on the method used to compute the data, e.g., ``"hilbert"``.
    Default is ``None``.
"""
docdict["method_tfr_attr"] = """
method : str
    The method used to compute the spectra (e.g., ``"morlet"``, ``"multitaper"``
    or ``"stockwell"``).
"""
docdict["method_tfr_epochs"] = _method_tfr_template.format(
    literals=" | ``'stockwell'``",
    cites=", and ``'stockwell'`` uses the S-transform "
    ":footcite:p:`Stockwell2007,MoukademEtAl2014,WheatEtAl2010,JonesEtAl2006`",
)

docdict["mode_eltc"] = """
mode : str
    Extraction mode, see Notes.
"""

docdict["mode_pctf"] = """
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

docdict["mode_tfr_plot"] = """
mode : 'mean' | 'ratio' | 'logratio' | 'percent' | 'zscore' | 'zlogratio'
    Perform baseline correction by

    - subtracting the mean of baseline values ('mean') (default)
    - dividing by the mean of baseline values ('ratio')
    - dividing by the mean of baseline values and taking the log
      ('logratio')
    - subtracting the mean of baseline values followed by dividing by
      the mean of baseline values ('percent')
    - subtracting the mean of baseline values and dividing by the
      standard deviation of baseline values ('zscore')
    - dividing by the mean of baseline values, taking the log, and
      dividing by the standard deviation of log baseline values
      ('zlogratio')
"""

docdict["montage"] = """
montage : None | str | DigMontage
    A montage containing channel positions. If a string or
    :class:`~mne.channels.DigMontage` is
    specified, the existing channel information will be updated with the
    channel positions from the montage. Valid strings are the names of the
    built-in montages that ship with MNE-Python; you can list those via
    :func:`mne.channels.get_builtin_montages`.
    If ``None`` (default), the channel positions will be removed from the
    :class:`~mne.Info`.
"""

docdict["montage_types"] = """EEG/sEEG/ECoG/DBS/fNIRS"""

docdict["montage_units"] = """
montage_units : str
    Units that channel positions are represented in. Defaults to "mm"
    (millimeters), but can be any prefix + "m" combination (including just
    "m" for meters).

    .. versionadded:: 1.3
"""

docdict["morlet_reference"] = """
The Morlet wavelets follow the formulation in :footcite:t:`Tallon-BaudryEtAl1997`.
"""

docdict["moving"] = """
moving : instance of SpatialImage
    The image to morph ("from" volume).
"""

docdict["mri_resolution_eltc"] = """
mri_resolution : bool
    If True (default), the volume source space will be upsampled to the
    original MRI resolution via trilinear interpolation before the atlas values
    are extracted. This ensnures that each atlas label will contain source
    activations. When False, only the original source space points are used,
    and some atlas labels thus may not contain any source space vertices.

    .. versionadded:: 0.21.0
"""

# %%
# N

docdict["n_comp_pctf_n"] = """
n_comp : int
    Number of PSF/CTF components to return for mode='max' or mode='svd'.
    Default n_comp=1.
"""

docdict["n_cycles_tfr"] = """
n_cycles : int | array of int, shape (n_freqs,)
    Number of cycles in the wavelet, either a fixed number or one per
    frequency. The number of cycles ``n_cycles`` and the frequencies of
    interest ``freqs`` define the temporal window length. See notes for
    additional information about the relationship between those arguments
    and about time and frequency smoothing.
"""

docdict["n_jobs"] = """\
n_jobs : int | None
    The number of jobs to run in parallel. If ``-1``, it is set
    to the number of CPU cores. Requires the :mod:`joblib` package.
    ``None`` (default) is a marker for 'unset' that will be interpreted
    as ``n_jobs=1`` (sequential execution) unless the call is performed under
    a :class:`joblib:joblib.parallel_config` context manager that sets another
    value for ``n_jobs``.
"""

docdict["n_jobs_cuda"] = """
n_jobs : int | str
    Number of jobs to run in parallel. Can be ``'cuda'`` if ``cupy``
    is installed properly.
"""

docdict["n_jobs_fir"] = """
n_jobs : int | str
    Number of jobs to run in parallel. Can be ``'cuda'`` if ``cupy``
    is installed properly and ``method='fir'``.
"""

docdict["n_pca_components_apply"] = """
n_pca_components : int | float | None
    The number of PCA components to be kept, either absolute (int)
    or fraction of the explained variance (float). If None (default),
    the ``ica.n_pca_components`` from initialization will be used in 0.22;
    in 0.23 all components will be used.
"""

docdict["n_permutations_clust_all"] = """
n_permutations : int | 'all'
    The number of permutations to compute. Can be 'all' to perform
    an exact test.
"""

docdict["n_permutations_clust_int"] = """
n_permutations : int
    The number of permutations to compute.
"""

docdict["n_proj_vectors"] = """
n_grad : int | float between ``0`` and ``1``
    Number of vectors for gradiometers. Either an integer or a float between 0 and 1
    to select the number of vectors to explain the cumulative variance greater than
    ``n_grad``.
n_mag : int | float between ``0`` and ``1``
    Number of vectors for magnetometers. Either an integer or a float between 0 and
    1 to select the number of vectors to explain the cumulative variance greater
    than ``n_mag``.
n_eeg : int | float between ``0`` and ``1``
    Number of vectors for EEG channels. Either an integer or a float between 0 and 1
    to select the number of vectors to explain the cumulative variance greater than
    ``n_eeg``.
"""

docdict["names_topomap"] = """\
names : None | list
    Labels for the sensors. If a :class:`list`, labels should correspond
    to the order of channels in ``data``. If ``None`` (default), no channel
    names are plotted.
"""

docdict["nave_tfr_attr"] = """
nave : int
    The number of epochs that were averaged to yield the result. This may reflect
    epochs averaged *before* time-frequency analysis (as in
    ``epochs.average(...).compute_tfr(...)``) or *after* time-frequency analysis (as
    in ``epochs.compute_tfr(...).average(...)``).
"""
docdict["nirx_notes"] = """
This function has only been tested with NIRScout and NIRSport devices,
and with the NIRStar software version 15 and above and Aurora software
2021 and above.

The NIRSport device can detect if the amplifier is saturated.
Starting from NIRStar 14.2, those saturated values are replaced by NaNs
in the standard .wlX files.
The raw unmodified measured values are stored in another file
called .nosatflags_wlX. As NaN values can cause unexpected behaviour with
mathematical functions the default behaviour is to return the
saturated data.
"""

docdict["niter"] = """
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

docdict["norm_pctf"] = """
norm : None | 'max' | 'norm'
    Whether and how to normalise the PSFs and CTFs. This will be applied
    before computing summaries as specified in 'mode'.
    Can be:

    * None : Use un-normalized PSFs/CTFs (Default).
    * 'max' : Normalize to maximum absolute value across all PSFs/CTFs.
    * 'norm' : Normalize to maximum norm across all PSFs/CTFs.
"""

docdict["normalization"] = """normalization : 'full' | 'length'
    Normalization strategy. If "full", the PSD will be normalized by the
    sampling rate as well as the length of the signal (as in
    :ref:`Nitime <nitime:users-guide>`). Default is ``'length'``."""

docdict["normalize_psd_topo"] = """
normalize : bool
    If True, each band will be divided by the total power. Defaults to
    False.
"""

docdict["notes_2d_backend"] = """\
MNE-Python provides two different backends for browsing plots (i.e.,
:meth:`raw.plot()<mne.io.Raw.plot>`, :meth:`epochs.plot()<mne.Epochs.plot>`,
and :meth:`ica.plot_sources()<mne.preprocessing.ICA.plot_sources>`). One is
based on :mod:`matplotlib`, and the other is based on
:doc:`PyQtGraph<pyqtgraph:index>`. You can set the backend temporarily with the
context manager :func:`mne.viz.use_browser_backend`, you can set it for the
duration of a Python session using :func:`mne.viz.set_browser_backend`, and you
can set the default for your computer via
:func:`mne.set_config('MNE_BROWSER_BACKEND', 'matplotlib')<mne.set_config>`
(or ``'qt'``).

.. note:: For the PyQtGraph backend to run in IPython with ``block=False``
          you must run the magic command ``%gui qt5`` first.
.. note:: To report issues with the PyQtGraph backend, please use the
          `issues <https://github.com/mne-tools/mne-qt-browser/issues>`_
          of ``mne-qt-browser``.
"""

_notes_plot_psd = """\
This {} exists to support legacy code; for new code the preferred
idiom is ``inst.compute_psd().plot()`` (where ``inst`` is an instance
of :class:`~mne.io.Raw`, :class:`~mne.Epochs`, or :class:`~mne.Evoked`).
"""

docdict["notes_plot_*_psd_func"] = _notes_plot_psd.format("function")
docdict["notes_plot_psd_meth"] = _notes_plot_psd.format("method")

docdict["notes_spectrum_array"] = """
If the data passed in is real-valued, it is assumed to represent spectral *power* (not
amplitude, phase, etc), and downstream methods (such as
:meth:`~mne.time_frequency.SpectrumArray.plot`) assume power data. If you pass in
real-valued data that is not power, axis labels will be incorrect.

If the data passed in is complex-valued, it is assumed to represent Fourier
coefficients. Downstream plotting methods will treat the data as such, attempting to
convert this to power before visualisation. If you pass in complex-valued data that is
not Fourier coefficients, axis labels will be incorrect.
"""

docdict["notes_timefreqs_tfr_plot_joint"] = """
``timefreqs`` has three different modes: tuples, dicts, and auto. For (list of) tuple(s)
mode, each tuple defines a pair (time, frequency) in s and Hz on the TFR plot.
For example, to look at 10 Hz activity 1 second into the epoch and 3 Hz activity 300 ms
into the epoch, ::

    timefreqs=((1, 10), (.3, 3))

If provided as a dictionary, (time, frequency) tuples are keys and (time_window,
frequency_window) tuples are the values — indicating the width of the windows (centered
on the time and frequency indicated by the key) to be averaged over. For example, ::

    timefreqs={(1, 10): (0.1, 2)}

would translate into a window that spans 0.95 to 1.05 seconds and 9 to 11 Hz. If
``None``, a single topomap will be plotted at the absolute peak across the
time-frequency representation.
"""

docdict["notes_tmax_included_by_default"] = """
Unlike Python slices, MNE time intervals by default include **both**
their end points; ``crop(tmin, tmax)`` returns the interval
``tmin <= t <= tmax``. Pass ``include_tmax=False`` to specify the half-open
interval ``tmin <= t < tmax`` instead.
"""

docdict["npad"] = """
npad : int | str
    Amount to pad the start and end of the data. Can also be ``"auto"`` to use a padding
    that will result in a power-of-two size (can be much faster).
"""

docdict["npad_resample"] = (
    docdict["npad"]
    + """
    Only used when ``method="fft"``.
"""
)
docdict["nrows_ncols_ica_components"] = """
nrows, ncols : int | 'auto'
    The number of rows and columns of topographies to plot. If both ``nrows``
    and ``ncols`` are ``'auto'``, will plot up to 20 components in a 5×4 grid,
    and return multiple figures if more than 20 components are requested.
    If one is ``'auto'`` and the other a scalar, a single figure is generated.
    If scalars are provided for both arguments, will plot up to ``nrows*ncols``
    components in a grid and return multiple figures as needed. Default is
    ``nrows='auto', ncols='auto'``.
"""

docdict["nrows_ncols_topomap"] = """
nrows, ncols : int | 'auto'
    The number of rows and columns of topographies to plot. If either ``nrows``
    or ``ncols`` is ``'auto'``, the necessary number will be inferred. Defaults
    to ``nrows=1, ncols='auto'``.
"""

# %%
# O

docdict["offset_decim"] = """
offset : int
    Apply an offset to where the decimation starts relative to the
    sample corresponding to t=0. The offset is in samples at the
    current sampling rate.

    .. versionadded:: 0.12
"""

docdict["on_baseline_ica"] = """
on_baseline : str
    How to handle baseline-corrected epochs or evoked data.
    Can be ``'raise'`` to raise an error, ``'warn'`` (default) to emit a
    warning, ``'ignore'`` to ignore, or "reapply" to reapply the baseline
    after applying ICA.

    .. versionadded:: 1.2
"""
docdict["on_defects"] = """
on_defects : 'raise' | 'warn' | 'ignore'
    What to do if the surface is found to have topological defects.
    Can be ``'raise'`` (default) to raise an error, ``'warn'`` to emit a
    warning, or ``'ignore'`` to ignore when one or more defects are found.
    Note that a lot of computations in MNE-Python assume the surfaces to be
    topologically correct, topological defects may still make other
    computations (e.g., `mne.make_bem_model` and `mne.make_bem_solution`)
    fail irrespective of this parameter.
"""

docdict["on_header_missing"] = """
on_header_missing : str
    Can be ``'raise'`` (default) to raise an error, ``'warn'`` to emit a
    warning, or ``'ignore'`` to ignore when the FastSCAN header is missing.

    .. versionadded:: 0.22
"""

_on_missing_base = """\
Can be ``'raise'`` (default) to raise an error, ``'warn'`` to emit a
    warning, or ``'ignore'`` to ignore when"""


docdict["on_mismatch_info"] = f"""
on_mismatch : 'raise' | 'warn' | 'ignore'
    {_on_missing_base} the device-to-head transformation differs between
    instances.

    .. versionadded:: 0.24
"""

docdict["on_missing_ch_names"] = f"""
on_missing : 'raise' | 'warn' | 'ignore'
    {_on_missing_base} entries in ch_names are not present in the raw instance.

    .. versionadded:: 0.23.0
"""

docdict["on_missing_chpi"] = f"""
on_missing : 'raise' | 'warn' | 'ignore'
    {_on_missing_base} no cHPI information can be found. If ``'ignore'`` or
    ``'warn'``, all return values will be empty arrays or ``None``. If
    ``'raise'``, an exception will be raised.
"""

docdict["on_missing_epochs"] = """
on_missing : 'raise' | 'warn' | 'ignore'
    What to do if one or several event ids are not found in the recording.
    Valid keys are 'raise' | 'warn' | 'ignore'
    Default is ``'raise'``. If ``'warn'``, it will proceed but
    warn; if ``'ignore'``, it will proceed silently.

    .. note::
       If none of the event ids are found in the data, an error will be
       automatically generated irrespective of this parameter.
"""

docdict["on_missing_events"] = f"""
on_missing : 'raise' | 'warn' | 'ignore'
    {_on_missing_base} event numbers from ``event_id`` are missing from
    :term:`events`. When numbers from :term:`events` are missing from
    ``event_id`` they will be ignored and a warning emitted; consider
    using ``verbose='error'`` in this case.

    .. versionadded:: 0.21
"""

docdict["on_missing_fiducials"] = f"""
on_missing : 'raise' | 'warn' | 'ignore'
    {_on_missing_base} some necessary fiducial points are missing.
"""

docdict["on_missing_fwd"] = f"""
on_missing : 'raise' | 'warn' | 'ignore'
    {_on_missing_base} ``stc`` has vertices that are not in ``fwd``.
"""

docdict["on_missing_montage"] = f"""
on_missing : 'raise' | 'warn' | 'ignore'
    {_on_missing_base} channels have missing coordinates.

    .. versionadded:: 0.20.1
"""

docdict["on_rank_mismatch"] = """
on_rank_mismatch : str
    If an explicit MEG value is passed, what to do when it does not match
    an empirically computed rank (only used for covariances).
    Can be 'raise' to raise an error, 'warn' (default) to emit a warning, or
    'ignore' to ignore.

    .. versionadded:: 0.23
"""

docdict["on_split_missing"] = f"""
on_split_missing : str
    {_on_missing_base} split file is missing.

    .. versionadded:: 0.22
"""

docdict["ordered"] = """
ordered : bool
    If True (default), ensure that the order of the channels in
    the modified instance matches the order of ``ch_names``.

    .. versionadded:: 0.20.0
    .. versionchanged:: 1.7
        The default changed from False in 1.6 to True in 1.7.
"""

docdict["origin_maxwell"] = """
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

docdict["out_type_clust"] = """
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

docdict["outlines_topomap"] = """
outlines : 'head' | dict | None
    The outlines to be drawn. If 'head', the default head scheme will be
    drawn. If dict, each key refers to a tuple of x and y positions, the values
    in 'mask_pos' will serve as image mask.
    Alternatively, a matplotlib patch object can be passed for advanced
    masking options, either directly or as a function that returns patches
    (required for multi-axis plots). If None, nothing will be drawn.
    Defaults to 'head'.
"""

docdict["output_compute_tfr"] = """
output : str
    What kind of estimate to return. Allowed values are ``"complex"``, ``"phase"``,
    and ``"power"``. Default is ``"power"``.
"""

docdict["overview_mode"] = """
overview_mode : str | None
    Can be "channels", "empty", or "hidden" to set the overview bar mode
    for the ``'qt'`` backend. If None (default), the config option
    ``MNE_BROWSER_OVERVIEW_MODE`` will be used, defaulting to "channels"
    if it's not found.
"""

docdict["overwrite"] = """
overwrite : bool
    If True (default False), overwrite the destination file if it
    exists.
"""

# %%
# P

_pad_base = """
    all :func:`numpy.pad` ``mode`` options. Can also be ``"reflect_limited"``, which
    pads with a reflected version of each vector mirrored on the first and last values
    of the vector, followed by zeros.
"""

docdict["pad_fir"] = (
    """
pad : str
    The type of padding to use. Supports """
    + _pad_base
    + """\
    Only used for ``method='fir'``.
"""
)

docdict["pad_resample"] = (  # used when default is not "auto"
    """
pad : str
    The type of padding to use. When ``method="fft"``, supports """
    + _pad_base
    + """\
    When ``method="polyphase"``, supports all modes of :func:`scipy.signal.upfirdn`.
"""
)

docdict["pad_resample_auto"] = (  # used when default is "auto"
    docdict["pad_resample"]
    + """\
    The default ("auto") means ``'reflect_limited'`` for ``method='fft'`` and
    ``'reflect'`` for ``method='polyphase'``.
"""
)
docdict["pca_vars_pctf"] = """
pca_vars : array, shape (n_comp,) | list of array
    The explained variances of the first n_comp SVD components across the
    PSFs/CTFs for the specified vertices. Arrays for multiple labels are
    returned as list. Only returned if ``mode='svd'`` and ``return_pca_vars=True``.
"""

docdict["per_sample_metric"] = """
per_sample : bool
    If True the metric is computed for each sample
    separately. If False, the metric is spatio-temporal.
"""

docdict["phase"] = """
phase : str
    Phase of the filter.
    When ``method='fir'``, symmetric linear-phase FIR filters are constructed
    with the following behaviors when ``method="fir"``:

    ``"zero"`` (default)
        The delay of this filter is compensated for, making it non-causal.
    ``"minimum"``
        A minimum-phase filter will be constructed by decomposing the zero-phase filter
        into a minimum-phase and all-pass systems, and then retaining only the
        minimum-phase system (of the same length as the original zero-phase filter)
        via :func:`scipy.signal.minimum_phase`.
    ``"zero-double"``
        *This is a legacy option for compatibility with MNE <= 0.13.*
        The filter is applied twice, once forward, and once backward
        (also making it non-causal).
    ``"minimum-half"``
        *This is a legacy option for compatibility with MNE <= 1.6.*
        A minimum-phase filter will be reconstructed from the zero-phase filter with
        half the length of the original filter.

    When ``method='iir'``, ``phase='zero'`` (default) or equivalently ``'zero-double'``
    constructs and applies IIR filter twice, once forward, and once backward (making it
    non-causal) using :func:`~scipy.signal.filtfilt`; ``phase='forward'`` will apply
    the filter once in the forward (causal) direction using
    :func:`~scipy.signal.lfilter`.

    .. versionadded:: 0.13
    .. versionchanged:: 1.7

       The behavior for ``phase="minimum"`` was fixed to use a filter of the requested
       length and improved suppression.
"""

docdict["physical_range_export_params"] = """
physical_range : str | tuple
    The physical range of the data. If 'auto' (default), the physical range is inferred
    from the data, taking the minimum and maximum values per channel type. If
    'channelwise', the range will be defined per channel. If a tuple of minimum and
    maximum, this manual physical range will be used. Only used for exporting EDF files.
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

docdict["pick_ori"] = (
    """
pick_ori : None | "normal" | "vector"
"""
    + _pick_ori_novec
    + """
    - ``"vector"``
        No pooling of the orientations is done, and the vector result
        will be returned in the form of a :class:`mne.VectorSourceEstimate`
        object.
"""
)

docdict["pick_ori_bf"] = """
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

docdict["pick_ori_novec"] = (
    """
pick_ori : None | "normal"
"""
    + _pick_ori_novec
)

docdict["pick_types_params"] = """
meg : bool | str
    If True include MEG channels. If string it can be 'mag', 'grad',
    'planar1' or 'planar2' to select only magnetometers, all
    gradiometers, or a specific type of gradiometer.
eeg : bool
    If True include EEG channels.
stim : bool
    If True include stimulus channels.
eog : bool
    If True include EOG channels.
ecg : bool
    If True include ECG channels.
emg : bool
    If True include EMG channels.
ref_meg : bool | str
    If True include CTF / 4D reference channels. If 'auto', reference
    channels are included if compensations are present and ``meg`` is
    not False. Can also be the string options for the ``meg``
    parameter.
misc : bool
    If True include miscellaneous analog channels.
resp : bool
    If ``True`` include respiratory channels.
chpi : bool
    If True include continuous HPI coil channels.
exci : bool
    Flux excitation channel used to be a stimulus channel.
ias : bool
    Internal Active Shielding data (maybe on Triux only).
syst : bool
    System status channel information (on Triux systems only).
seeg : bool
    Stereotactic EEG channels.
dipole : bool
    Dipole time course channels.
gof : bool
    Dipole goodness of fit channels.
bio : bool
    Bio channels.
ecog : bool
    Electrocorticography channels.
fnirs : bool | str
    Functional near-infrared spectroscopy channels. If True include all
    fNIRS channels. If False (default) include none. If string it can
    be 'hbo' (to include channels measuring oxyhemoglobin) or 'hbr' (to
    include channels measuring deoxyhemoglobin).
csd : bool
    EEG-CSD channels.
dbs : bool
    Deep brain stimulation channels.
temperature : bool
    Temperature channels.
gsr : bool
    Galvanic skin response channels.
eyetrack : bool | str
    Eyetracking channels. If True include all eyetracking channels. If False
    (default) include none. If string it can be 'eyegaze' (to include
    eye position channels) or 'pupil' (to include pupil-size
    channels).
include : list of str
    List of additional channels to include. If empty do not include
    any.
exclude : list of str | str
    List of channels to exclude. If 'bads' (default), exclude channels
    in ``info['bads']``.
selection : list of str
    Restrict sensor channels (MEG, EEG, etc.) to this list of channel names.
"""

_picks_types = "str | array-like | slice | None"
_picks_header = f"picks : {_picks_types}"
_picks_desc = "Channels to include."
_picks_int = "Slices and lists of integers will be interpreted as channel indices."
_picks_str_types = """channel *type* strings (e.g., ``['meg', 'eeg']``) will
    pick channels of those types,"""
_picks_str_names = """channel *name* strings (e.g., ``['MEG0111', 'MEG2623']``
    will pick the given channels."""
_picks_str_values = """Can also be the string values ``'all'`` to pick
    all channels, or ``'data'`` to pick :term:`data channels`."""
_picks_str = f"""In lists, {_picks_str_types} {_picks_str_names}
    {_picks_str_values}
    None (default) will pick"""
_picks_str_notypes = f"""In lists, {_picks_str_names}
    None (default) will pick"""
_reminder = (
    "Note that channels in ``info['bads']`` *will be included* if "
    "their {}indices are explicitly provided."
)
reminder = _reminder.format("names or ")
reminder_nostr = _reminder.format("")
noref = f"(excluding reference MEG channels). {reminder}"
picks_base = f"""{_picks_header}
    {_picks_desc} {_picks_int} {_picks_str}"""
picks_base_notypes = f"""picks : list of int | list of str | slice | None
    {_picks_desc} {_picks_int} {_picks_str_notypes}"""
docdict["picks_all"] = _reflow_param_docstring(f"{picks_base} all channels. {reminder}")
docdict["picks_all_data"] = _reflow_param_docstring(
    f"{picks_base} all data channels. {reminder}"
)
docdict["picks_all_data_noref"] = _reflow_param_docstring(
    f"{picks_base} all data channels {noref}"
)
docdict["picks_all_notypes"] = _reflow_param_docstring(
    f"{picks_base_notypes} all channels. {reminder}"
)
docdict["picks_base"] = _reflow_param_docstring(picks_base)
docdict["picks_good_data"] = _reflow_param_docstring(
    f"{picks_base} good data channels. {reminder}"
)
docdict["picks_good_data_noref"] = _reflow_param_docstring(
    f"{picks_base} good data channels {noref}"
)
docdict["picks_header"] = _picks_header
docdict["picks_ica"] = """
picks : int | list of int | slice | None
    Indices of the independent components (ICs) to visualize.
    If an integer, represents the index of the IC to pick.
    Multiple ICs can be selected using a list of int or a slice.
    The indices are 0-indexed, so ``picks=1`` will pick the second
    IC: ``ICA001``. ``None`` will pick all independent components in the order fitted.
"""
docdict["picks_layout"] = """
picks : array-like of str or int | slice | ``'all'`` | None
    Channels to include in the layout. Slices and lists of integers will be interpreted
    as channel indices. Can also be the string value ``'all'`` to pick all channels.
    None (default) will pick all channels.
"""
docdict["picks_nostr"] = f"""picks : list | slice | None
    {_picks_desc} {_picks_int}
    None (default) will pick all channels. {reminder_nostr}"""

docdict["picks_plot_projs_joint_trace"] = f"""\
picks_trace : {_picks_types}
    Channels to show alongside the projected time courses. Typically
    these are the ground-truth channels for an artifact (e.g., ``'eog'`` or
    ``'ecg'``). {_picks_int} {_picks_str} no channels.
"""

docdict["pipeline"] = """
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

docdict["plot_psd_doc"] = """\
Plot power or amplitude spectra.

Separate plots are drawn for each channel type. When the data have been
processed with a bandpass, lowpass or highpass filter, dashed lines (╎)
indicate the boundaries of the filter. The line noise frequency is also
indicated with a dashed line (⋮). If ``average=False``, the plot will
be interactive, and click-dragging on the spectrum will generate a
scalp topography plot for the chosen frequency range in a new figure
"""
# lack of trailing . is intentional; it must be in actual docstring ↑↑↑ (D400)

_pos_topomap = """\
pos : array, shape (n_channels, 2){}
    Location information for the channels. If an array, should provide the x
    and y coordinates for plotting the channels in 2D.
"""
docdict["pos_topomap"] = _pos_topomap.format(" | instance of Info")
docdict["pos_topomap_psd"] = _pos_topomap.format("")

docdict["position"] = """
position : int
    The position for the progress bar.
"""

docdict["precompute"] = """
precompute : bool | str
    Whether to load all data (not just the visible portion) into RAM and
    apply preprocessing (e.g., projectors) to the full data array in a separate
    processor thread, instead of window-by-window during scrolling. The default
    None uses the ``MNE_BROWSER_PRECOMPUTE`` variable, which defaults to
    ``'auto'``. ``'auto'`` compares available RAM space to the expected size of
    the precomputed data, and precomputes only if enough RAM is available.
    This is only used with the Qt backend.

    .. versionadded:: 0.24
    .. versionchanged:: 1.0
       Support for the ``MNE_BROWSER_PRECOMPUTE`` config variable.
"""

docdict["preload"] = """
preload : bool or str (default False)
    Preload data into memory for data manipulation and faster indexing.
    If True, the data will be preloaded into memory (fast, requires
    large amount of memory). If preload is a string, preload is the
    file name of a memory-mapped file which is used to store the data
    on the hard drive (slower, requires less memory)."""

docdict["preload_concatenate"] = """
preload : bool, str, or None (default None)
    Preload data into memory for data manipulation and faster indexing.
    If True, the data will be preloaded into memory (fast, requires
    large amount of memory). If preload is a string, preload is the
    file name of a memory-mapped file which is used to store the data
    on the hard drive (slower, requires less memory). If preload is
    None, preload=True or False is inferred using the preload status
    of the instances passed in.
"""

docdict["proj_epochs"] = """
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

docdict["proj_plot"] = """
proj : bool | 'interactive' | 'reconstruct'
    If true SSP projections are applied before display. If ``'interactive'``,
    a check box for reversible selection of SSP projection vectors will
    be shown. If ``'reconstruct'``, projection vectors will be applied and then
    M/EEG data will be reconstructed via field mapping to reduce the signal
    bias caused by projection.

    .. versionchanged:: 0.21
       Support for 'reconstruct' was added.
"""

docdict["proj_psd"] = """\
proj : bool
    Whether to apply SSP projection vectors before spectral estimation.
    Default is ``False``.
"""

docdict["projection_set_eeg_reference"] = """
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

docdict["projs"] = """
projs : list of Projection
    List of computed projection vectors.
"""

docdict["projs_report"] = """
projs : bool | None
    Whether to add SSP projector plots if projectors are present in
    the data. If ``None``, use ``projs`` from `~mne.Report` creation.
"""

# %%
# R

docdict["random_state"] = """
random_state : None | int | instance of ~numpy.random.RandomState
    A seed for the NumPy random number generator (RNG). If ``None`` (default),
    the seed will be  obtained from the operating system
    (see  :class:`~numpy.random.RandomState` for details), meaning it will most
    likely produce different output every time this function or method is run.
    To achieve reproducible results, pass a value here to explicitly initialize
    the RNG with a defined state.
"""

_rank_base = """
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

docdict["rank"] = _rank_base
docdict["rank_info"] = _rank_base + "\n    The default is ``'info'``."
docdict["rank_none"] = _rank_base + "\n    The default is ``None``."

docdict["raw_epochs"] = """
raw : Raw object
    An instance of `~mne.io.Raw`.
"""

docdict["raw_sfreq"] = """
raw_sfreq : float
    The original Raw object sampling rate. If None, then it is set to
    ``info['sfreq']``.
"""

docdict["reduce_rank"] = """
reduce_rank : bool
    If True, the rank of the denominator of the beamformer formula (i.e.,
    during pseudo-inversion) will be reduced by one for each spatial location.
    Setting ``reduce_rank=True`` is typically necessary if you use a single
    sphere model with MEG data.

    .. versionchanged:: 0.20
        Support for reducing rank in all modes (previously only supported
        ``pick='max_power'`` with weight normalization).
"""

docdict["ref_channels"] = """
ref_channels : str | list of str
    Name of the electrode(s) which served as the reference in the
    recording. If a name is provided, a corresponding channel is added
    and its data is set to 0. This is useful for later re-referencing.
"""

docdict["ref_channels_set_eeg_reference"] = """
ref_channels : list of str | str | dict
    Can be:

    - The name(s) of the channel(s) used to construct the reference for
      every channel of ``ch_type``.
    - ``'average'`` to apply an average reference (default)
    - ``'REST'`` to use the Reference Electrode Standardization Technique
      infinity reference :footcite:`Yao2001`.
    - A dictionary mapping names of data channels to (lists of) names of
      reference channels. For example, {'A1': 'A3'} would replace the
      data in channel 'A1' with the difference between 'A1' and 'A3'. To take
      the average of multiple channels as reference, supply a list of channel
      names as the dictionary value, e.g. {'A1': ['A2', 'A3']} would replace
      channel A1 with ``A1 - mean(A2, A3)``.
    - An empty list, in which case MNE will not attempt any re-referencing of
      the data
"""

docdict["reg_affine"] = """
reg_affine : ndarray of float, shape (4, 4)
    The affine that registers one volume to another.
"""

docdict["regularize_maxwell_reg"] = """
regularize : str | None
    Basis regularization type, must be ``"in"`` or None.
    ``"in"`` is the same algorithm as the ``-regularize in`` option in
    MaxFilter™.
"""


_reject_by_annotation_base = """
reject_by_annotation : bool
    Whether to omit bad segments from the data before fitting. If ``True``
    (default), annotated segments whose description begins with ``'bad'`` are
    omitted. If ``False``, no rejection based on annotations is performed.
"""

docdict["reject_by_annotation_all"] = _reject_by_annotation_base

docdict["reject_by_annotation_epochs"] = """
reject_by_annotation : bool
    Whether to reject based on annotations. If ``True`` (default), epochs
    overlapping with segments whose description begins with ``'bad'`` are
    rejected. If ``False``, no rejection based on annotations is performed.
"""

docdict["reject_by_annotation_psd"] = """\
reject_by_annotation : bool
    Whether to omit bad spans of data before spectral estimation. If
    ``True``, spans with annotations whose description begins with
    ``bad`` will be omitted.
"""

docdict["reject_by_annotation_raw"] = (
    _reject_by_annotation_base
    + """
    Has no effect if ``inst`` is not a :class:`mne.io.Raw` object.
"""
)

docdict["reject_by_annotation_tfr"] = """
reject_by_annotation : bool
    Whether to omit bad spans of data before spectrotemporal power
    estimation. If ``True``, spans with annotations whose description
    begins with ``bad`` will be represented with ``np.nan`` in the
    time-frequency representation.
"""

_reject_common = """\
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

docdict["reject_drop_bad"] = """\
reject : dict | str | None
    Reject epochs based on **maximum** peak-to-peak signal amplitude (PTP)
    or custom functions. Peak-to-peak signal amplitude is defined as
    the absolute difference between the lowest and the highest signal
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

    Custom rejection criteria can be also be used by passing a callable,
    e.g., to check for 99th percentile of absolute values of any channel
    across time being bigger than :unit:`1 mV`. The callable must return a
    ``(good, reason)`` tuple: ``good`` must be :class:`bool` and ``reason``
    must be :class:`str`, :class:`list`, or :class:`tuple` where each entry
    is a :class:`str`::

        reject = dict(
            eeg=lambda x: (
                (np.percentile(np.abs(x), 99, axis=1) > 1e-3).any(),
                "signal > 1 mV somewhere",
            )
        )

    .. note:: If rejection is based on a signal **difference**
            calculated for each channel separately, applying baseline
            correction does not affect the rejection procedure, as the
            difference will be preserved.

    .. note:: If ``reject`` is a callable, than **any** criteria can be
            used to reject epochs (including maxima and minima).

    If ``reject`` is ``None``, no rejection is performed. If ``'existing'``
    (default), then the rejection parameters set at instantiation are used.
"""  # noqa: E501

docdict["reject_epochs"] = f"""
reject : dict | None
{_reject_common}
    .. note:: To constrain the time period used for estimation of signal
              quality, pass the ``reject_tmin`` and ``reject_tmax`` parameters.

    If ``reject`` is ``None`` (default), no rejection is performed.
"""

docdict["remove_dc"] = """
remove_dc : bool
    If ``True``, the mean is subtracted from each segment before computing
    its spectrum.
"""

docdict["replace_report"] = """
replace : bool
    If ``True``, content already present that has the same ``title`` and
    ``section`` will be replaced. Defaults to ``False``, which will cause
    duplicate entries in the table of contents if an entry for ``title``
    already exists.
"""

docdict["res_topomap"] = """
res : int
    The resolution of the topomap image (number of pixels along each side).
"""

docdict["return_pca_vars_pctf"] = """
return_pca_vars : bool
    Whether or not to return the explained variances across the specified
    vertices for individual SVD components. This is only valid if ``mode='svd'``.
    Default to False.
"""

docdict["roll"] = """
roll : float | None
    The roll of the camera rendering the view in degrees.
"""

# %%
# S

docdict["saturated"] = """saturated : str
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

docdict["scalings"] = """
scalings : 'auto' | dict | None
    Scaling factors for the traces. If a dictionary where any
    value is ``'auto'``, the scaling factor is set to match the 99.5th
    percentile of the respective data. If ``'auto'``, all scalings (for all
    channel types) are set to ``'auto'``. If any values are ``'auto'`` and the
    data is not preloaded, a subset up to 100 MB will be loaded. If ``None``,
    defaults to::

        dict(mag=1e-12, grad=4e-11, eeg=20e-6, eog=150e-6, ecg=5e-4,
             emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1,
             resp=1, chpi=1e-4, whitened=1e2)

    .. note::
        A particular scaling value ``s`` corresponds to half of the visualized
        signal range around zero (i.e. from ``0`` to ``+s`` or from ``0`` to
        ``-s``). For example, the default scaling of ``20e-6`` (20µV) for EEG
        signals means that the visualized range will be 40 µV (20 µV in the
        positive direction and 20 µV in the negative direction).
"""

docdict["scalings_df"] = """
scalings : dict | None
    Scaling factor applied to the channels picked. If ``None``, defaults to
    ``dict(eeg=1e6, mag=1e15, grad=1e13)`` — i.e., converts EEG to µV,
    magnetometers to fT, and gradiometers to fT/cm.
"""

docdict["scalings_topomap"] = """
scalings : dict | float | None
    The scalings of the channel types to be applied for plotting.
    If None, defaults to ``dict(eeg=1e6, grad=1e13, mag=1e15)``.
"""

docdict["scoring"] = """
scoring : callable | str | None
    Score function (or loss function) with signature
    ``score_func(y, y_pred, **kwargs)``.
    Note that the "predict" method is automatically identified if scoring is
    a string (e.g. ``scoring='roc_auc'`` calls ``predict_proba``), but is
    **not**  automatically set if ``scoring`` is a callable (e.g.
    ``scoring=sklearn.metrics.roc_auc_score``).
"""

docdict["sdr_morph"] = """
sdr_morph : instance of dipy.align.DiffeomorphicMap
    The class that applies the the symmetric diffeomorphic registration
    (SDR) morph.
"""

docdict["section_report"] = """
section : str | None
    The name of the section (or content block) to add the content to. This
    feature is useful for grouping multiple related content elements
    together under a single, collapsible section. Each content element will
    retain its own title and functionality, but not appear separately in the
    table of contents. Hence, using sections is a way to declutter the table
    of contents, and to easy navigation of the report.

    .. versionadded:: 1.1
"""

docdict["seed"] = """
seed : None | int | instance of ~numpy.random.RandomState
    A seed for the NumPy random number generator (RNG). If ``None`` (default),
    the seed will be  obtained from the operating system
    (see  :class:`~numpy.random.RandomState` for details), meaning it will most
    likely produce different output every time this function or method is run.
    To achieve reproducible results, pass a value here to explicitly initialize
    the RNG with a defined state.
"""

docdict["seeg"] = """
seeg : bool
    If True (default), show sEEG electrodes.
"""

docdict["selection"] = """
selection : iterable | None
    Iterable of indices of selected epochs. If ``None``, will be
    automatically generated, corresponding to all non-zero events.
"""
docdict["selection_attr"] = """
selection : ndarray
    Array of indices of *selected* epochs (i.e., epochs that were not rejected, dropped,
    or ignored)."""

docdict["sensor_colors"] = """
sensor_colors : array-like of color | dict | None
    Colors to use for the sensor glyphs. Can be None (default) to use default colors.
    A dict should provide the colors (values) for each channel type (keys), e.g.::

        dict(eeg=eeg_colors)

    Where the value (``eeg_colors`` above) can be broadcast to an array of colors with
    length that matches the number of channels of that type, i.e., is compatible with
    :func:`matplotlib.colors.to_rgba_array`. A few examples of this for the case above
    are the string ``"k"``, a list of ``n_eeg`` color strings, or an NumPy ndarray of
    shape ``(n_eeg, 3)`` or ``(n_eeg, 4)``.
"""

docdict["sensor_scales"] = """
sensor_scales : int | float | array-like | dict | None
    Scale to use for the sensor glyphs. Can be None (default) to use default scale.
    A dict should provide the Scale (values) for each channel type (keys), e.g.::

        dict(eeg=eeg_scales)

    Where the value (``eeg_scales`` above) can be broadcast to an array of values with
    length that matches the number of channels of that type. A few examples of this
    for the case above are the value ``10e-3``, a list of ``n_eeg`` values, or an NumPy
    ndarray of shape ``(n_eeg,)``.
"""

docdict["sensors_topomap"] = """
sensors : bool | str
    Whether to add markers for sensor locations. If :class:`str`, should be a
    valid matplotlib format string (e.g., ``'r+'`` for red plusses, see the
    Notes section of :meth:`~matplotlib.axes.Axes.plot`). If ``True`` (the
    default), black circles will be used.
"""

docdict["set_eeg_reference_see_also_notes"] = """
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

- Different references for different channels
    Set ``ref_channels`` to a dictionary mapping source channel names (str)
    to the reference channel names (str or list of str). Unlike the other
    approaches where the same reference is applied globally, you can set
    different references for different channels with this method. For example,
    to re-reference channel 'A1' to 'A2' and 'B1' to the average of 'B2' and
    'B3', set ``ref_channels={'A1': 'A2', 'B1': ['B2', 'B3']}``. Warnings are
    issued when a mapping involves bad channels or channels of different types.

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

docdict["sfreq_tfr_attr"] = """
sfreq : int | float
    The sampling frequency (read from ``info``)."""
docdict["shape_tfr_attr"] = """
shape : tuple of int
    The shape of the data."""

docdict["show"] = """\
show : bool
    Show the figure if ``True``.
"""

docdict["show_names_topomap"] = """
show_names : bool | callable
    If ``True``, show channel names next to each sensor marker. If callable,
    channel names will be formatted using the callable; e.g., to
    delete the prefix 'MEG ' from all channel names, pass the function
    ``lambda x: x.replace('MEG ', '')``. If ``mask`` is not ``None``, only
    non-masked sensor names will be shown.
"""

docdict["show_scalebars"] = """
show_scalebars : bool
    Whether to show scale bars when the plot is initialized. Can be toggled
    after initialization by pressing :kbd:`s` while the plot window is focused.
    Default is ``True``.
"""

docdict["show_scrollbars"] = """
show_scrollbars : bool
    Whether to show scrollbars when the plot is initialized. Can be toggled
    after initialization by pressing :kbd:`z` ("zen mode") while the plot
    window is focused. Default is ``True``.

    .. versionadded:: 0.19.0
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

docdict["size_topomap"] = """
size : float
    Side length of each subplot in inches.
"""

docdict["skip_by_annotation"] = """
skip_by_annotation : str | list of str
    If a string (or list of str), any annotation segment that begins
    with the given string will not be included in filtering, and
    segments on either side of the given excluded annotated segment
    will be filtered separately (i.e., as independent signals).
    The default (``('edge', 'bad_acq_skip')`` will separately filter
    any segments that were concatenated by :func:`mne.concatenate_raws`
    or :meth:`mne.io.Raw.append`, or separated during acquisition.
    To disable, provide an empty list. Only used if ``inst`` is raw.
"""

docdict["skip_by_annotation_maxwell"] = """
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

docdict["smooth"] = """
smooth : float in [0, 1)
    The smoothing factor to be applied. Default 0 is no smoothing.
"""

docdict["spatial_colors"] = """\
spatial_colors : bool | 'auto'
    If True, the lines are color coded by mapping physical sensor
    coordinates into color values. Spatially similar channels will have
    similar colors. Bad channels will be dotted. If False, the good
    channels are plotted black and bad channels red. If ``'auto'``, uses
    True if channel locations are present, and False if channel locations
    are missing or if the data contains only a single channel. Defaults to
    ``'auto'``.
"""

docdict["spatial_colors_psd"] = """\
spatial_colors : bool
    Whether to color spectrum lines by channel location. Ignored if
    ``average=True``.
"""

docdict["sphere_topomap_auto"] = f"""\
sphere : float | array-like | instance of ConductorModel | None  | 'auto' | 'eeglab'
    The sphere parameters to use for the head outline. Can be array-like of
    shape (4,) to give the X/Y/Z origin and radius in meters, or a single float
    to give just the radius (origin assumed 0, 0, 0). Can also be an instance
    of a spherical :class:`~mne.bem.ConductorModel` to use the origin and
    radius from that object. If ``'auto'`` the sphere is fit to digitization
    points. If ``'eeglab'`` the head circle is defined by EEG electrodes
    ``'Fpz'``, ``'Oz'``, ``'T7'``, and ``'T8'`` (if ``'Fpz'`` is not present,
    it will be approximated from the coordinates of ``'Oz'``). ``None`` (the
    default) is equivalent to ``'auto'`` when enough extra digitization points
    are available, and (0, 0, 0, {HEAD_SIZE_DEFAULT}) otherwise.

    .. versionadded:: 0.20
    .. versionchanged:: 1.1 Added ``'eeglab'`` option.
"""

docdict["splash"] = """
splash : bool
    If True (default), a splash screen is shown during the application startup. Only
    applicable to the ``qt`` backend.
"""

docdict["split_naming"] = """
split_naming : 'neuromag' | 'bids'
    When splitting files, append a filename partition with the appropriate
    naming schema. For ``'neuromag'``, a split file ``fname.fif`` will be named
    ``fname.fif``, ``fname-1.fif``, ``fname-2.fif``, and so on. For ``'bids'``,
    a filename is expected to consist of parts separated by underscores, like
    ``<part-1>_<part-N>_<suffix>.fif``, and the according split naming will
    return filenames like ``<part-1>_<part-N>_split-01_<suffix>.fif``,
    ``<part-1>_<part-N>_split-02_<suffix>.fif``, and so on.
"""

docdict["src_eltc"] = """
src : instance of SourceSpaces
    The source spaces for the source time courses.
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

docdict["st_fixed_maxwell_only"] = """
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

docdict["standardize_names"] = """
standardize_names : bool
    If True, standardize MEG and EEG channel names to be
    ``'MEG ###'`` and ``'EEG ###'``. If False (default), native
    channel names in the file will be used when possible.
"""

_stat_fun_clust_base = """
stat_fun : callable | None
    Function called to calculate the test statistic. Must accept 1D-array as
    input and return a 1D array. If ``None`` (the default), uses
    `mne.stats.{}`.
"""

docdict["stat_fun_clust_f"] = _stat_fun_clust_base.format("f_oneway")

docdict["stat_fun_clust_t"] = _stat_fun_clust_base.format("ttest_1samp_no_p")

docdict["static"] = """
static : instance of SpatialImage
    The image to align with ("to" volume).
"""

docdict["stc_est_metric"] = """
stc_est : instance of (Vol|Mixed)SourceEstimate
    The source estimates containing estimated values
    e.g. obtained with a source imaging method.
"""

docdict["stc_metric"] = """
metric : float | array, shape (n_times,)
    The metric. float if per_sample is False, else
    array with the values computed for each time point.
"""

docdict["stc_plot_kwargs_report"] = """
stc_plot_kwargs : dict
    Dictionary of keyword arguments to pass to
    :class:`mne.SourceEstimate.plot`. Only used when plotting in 3D
    mode.
"""

docdict["stc_true_metric"] = """
stc_true : instance of (Vol|Mixed)SourceEstimate
    The source estimates containing correct values.
"""

docdict["stcs_pctf"] = """
stcs : instance of SourceEstimate | list of instances of SourceEstimate
    The PSFs or CTFs as STC objects. All PSFs/CTFs will be returned as
    successive samples in STC objects, in the order they are specified
    in idx. STCs for different labels willbe returned as a list.
    If resmat was computed with n_orient_inv==3 for CTFs or
    n_orient_fwd==3 for PSFs then 3 functions per vertex will be returned
    as successive samples (i.e. one function per orientation).
    If vector=False (default) and resmat was computed with
    n_orient_inv==3 for PSFs or n_orient_fwd==3 for CTFs, then the three
    values per vertex will be combined into one intensity value per
    vertex in a SourceEstimate object. If vector=True, PSFs or CTFs
    with 3 values per vertex (one per orientation) will be returned in
    a VectorSourceEstimate object.
"""

docdict["std_err_by_event_type_returns"] = """
std_err : instance of Evoked | list of Evoked
    The standard error over epochs.
    When ``by_event_type=True`` was specified, a list is returned containing a
    separate :class:`~mne.Evoked` object for each event type. The list has the
    same order as the event types as specified in the ``event_id``
    dictionary.
"""

docdict["step_down_p_clust"] = """
step_down_p : float
    To perform a step-down-in-jumps test, pass a p-value for clusters to
    exclude from each successive iteration. Default is zero, perform no
    step-down test (since no clusters will be smaller than this value).
    Setting this to a reasonable value, e.g. 0.05, can increase sensitivity
    but costs computation time.
"""

docdict["subject"] = """
subject : str
    The FreeSurfer subject name.
"""

docdict["subject_label"] = """
subject : str | None
    Subject which this label belongs to. Should only be specified if it is not
    specified in the label.
"""

docdict["subject_none"] = """
subject : str | None
    The FreeSurfer subject name.
"""

docdict["subject_optional"] = """
subject : str
    The FreeSurfer subject name. While not necessary, it is safer to set the
    subject parameter to avoid analysis errors.
"""

docdict["subjects_dir"] = """
subjects_dir : path-like | None
    The path to the directory containing the FreeSurfer subjects
    reconstructions. If ``None``, defaults to the ``SUBJECTS_DIR`` environment
    variable.
"""

docdict["surface"] = """surface : str
    The surface along which to do the computations, defaults to ``'white'``
    (the gray-white matter boundary).
"""

# %%
# T

docdict["t_power_clust"] = """
t_power : float
    Power to raise the statistical values (usually t-values) by before
    summing (sign will be retained). Note that ``t_power=0`` will give a
    count of locations in each cluster, ``t_power=1`` will weight each location
    by its statistical score.
"""

docdict["t_window_chpi_t"] = """
t_window : float
    Time window to use to estimate the amplitudes, default is
    0.2 (200 ms).
"""

docdict["tags_report"] = """
tags : array-like of str | str
    Tags to add for later interactive filtering. Must not contain spaces.
"""

docdict["tail_clust"] = """
tail : int
    If tail is 1, the statistic is thresholded above threshold.
    If tail is -1, the statistic is thresholded below threshold.
    If tail is 0, the statistic is thresholded on both sides of
    the distribution.
"""

docdict["temporal_window_tfr_intro"] = """
In spectrotemporal analysis (as with traditional fourier methods),
the temporal and spectral resolution are interrelated: longer temporal windows
allow more precise frequency estimates; shorter temporal windows "smear"
frequency estimates while providing more precise timing information.

Time-frequency representations are computed using a sliding temporal window.
Either the temporal window has a fixed length independent of frequency, or the
temporal window decreases in length with increased frequency.

.. image:: https://www.fieldtriptoolbox.org/assets/img/tutorial/timefrequencyanalysis/figure1.png

*Figure: Time and frequency smoothing. (a) For a fixed length temporal window
the time and frequency smoothing remains fixed. (b) For temporal windows that
decrease with frequency, the temporal smoothing decreases and the frequency
smoothing increases with frequency.*
Source: `FieldTrip tutorial: Time-frequency analysis using Hanning window,
multitapers and wavelets <https://www.fieldtriptoolbox.org/tutorial/timefrequencyanalysis>`_.
"""  # noqa: E501

docdict["temporal_window_tfr_morlet_notes"] = r"""
In MNE-Python, the length of the Morlet wavelet is affected by the arguments
``freqs`` and ``n_cycles``, which define the frequencies of interest
and the number of cycles, respectively. For the time-frequency representation,
the length of the wavelet is defined such that both tails of
the wavelet extend five standard deviations from the midpoint of its Gaussian
envelope and that there is a sample at time zero.

The length of the wavelet is thus :math:`10\times\mathtt{sfreq}\cdot\sigma-1`,
which is equal to :math:`\frac{5}{\pi} \cdot \frac{\mathtt{n\_cycles} \cdot
\mathtt{sfreq}}{\mathtt{freqs}} - 1`, where
:math:`\sigma = \frac{\mathtt{n\_cycles}}{2\pi f}` corresponds to the standard
deviation of the wavelet's Gaussian envelope. Note that the length of the
wavelet must not exceed the length of your signal.

For more information on the Morlet wavelet, see :func:`mne.time_frequency.morlet`.
"""

docdict["temporal_window_tfr_multitaper_notes"] = r"""
In MNE-Python, the multitaper temporal window length is defined by the arguments
``freqs`` and ``n_cycles``, respectively defining the frequencies of interest
and the number of cycles: :math:`T = \frac{\mathtt{n\_cycles}}{\mathtt{freqs}}`

A fixed number of cycles for all frequencies will yield a temporal window which
decreases with frequency. For example, ``freqs=np.arange(1, 6, 2)`` and
``n_cycles=2`` yields ``T=array([2., 0.7, 0.4])``.

To use a temporal window with fixed length, the number of cycles has to be
defined based on the frequency. For example, ``freqs=np.arange(1, 6, 2)`` and
``n_cycles=freqs / 2`` yields ``T=array([0.5, 0.5, 0.5])``.
"""

_theme = """\
theme : str | path-like
    Can be "auto", "light", or "dark" or a path-like to a
    custom stylesheet. For Dark-Mode and automatic Dark-Mode-Detection,
    `qdarkstyle <https://github.com/ColinDuquesnoy/QDarkStyleSheet>`__ and
    `darkdetect <https://github.com/albertosottile/darkdetect>`__,
    respectively, are required.\
    If None (default), the config option {config_option} will be used,
    defaulting to "auto" if it's not found.\
"""

docdict["theme_3d"] = """
{theme}
""".format(theme=_theme.format(config_option="MNE_3D_OPTION_THEME"))

docdict["theme_pg"] = """
{theme}
    Only supported by the ``'qt'`` backend.
""".format(theme=_theme.format(config_option="MNE_BROWSER_THEME"))

docdict["thresh"] = """
thresh : None or float
    Not supported yet.
    If not None, values below thresh will not be visible.
"""

_threshold_clust_base = """
threshold : float | dict | None
    The so-called "cluster forming threshold" in the form of a test statistic
    (note: this is not an alpha level / "p-value").
    If numeric, vertices with data values more extreme than ``threshold`` will
    be used to form clusters. If ``None``, {} will be chosen
    automatically that corresponds to a p-value of 0.05 for the given number of
    observations (only valid when using {}). If ``threshold`` is a
    :class:`dict` (with keys ``'start'`` and ``'step'``) then threshold-free
    cluster enhancement (TFCE) will be used (see the
    :ref:`TFCE example <tfce_example>` and :footcite:`SmithNichols2009`).
    See Notes for an example on how to compute a threshold based on
    a particular p-value for one-tailed or two-tailed tests.
"""

f_test = ("an F-threshold", "an F-statistic")
docdict["threshold_clust_f"] = _threshold_clust_base.format(*f_test)

docdict["threshold_clust_f_notes"] = """
For computing a ``threshold`` based on a p-value, use the conversion
from :meth:`scipy.stats.rv_continuous.ppf`::

    pval = 0.001  # arbitrary
    dfn = n_conditions - 1  # degrees of freedom numerator
    dfd = n_observations - n_conditions  # degrees of freedom denominator
    thresh = scipy.stats.f.ppf(1 - pval, dfn=dfn, dfd=dfd)  # F distribution
"""

t_test = ("a t-threshold", "a t-statistic")
docdict["threshold_clust_t"] = _threshold_clust_base.format(*t_test)

docdict["threshold_clust_t_notes"] = """
For computing a ``threshold`` based on a p-value, use the conversion
from :meth:`scipy.stats.rv_continuous.ppf`::

    pval = 0.001  # arbitrary
    df = n_observations - 1  # degrees of freedom for the test
    thresh = scipy.stats.t.ppf(1 - pval / 2, df)  # two-tailed, t distribution

For a one-tailed test (``tail=1``), don't divide the p-value by 2.
For testing the lower tail (``tail=-1``), don't subtract ``pval`` from 1.
"""

docdict["time_bandwidth_tfr"] = """
time_bandwidth : float ``≥ 2.0``
    Product between the temporal window length (in seconds) and the *full*
    frequency bandwidth (in Hz). This product can be seen as the surface of the
    window on the time/frequency plane and controls the frequency bandwidth
    (thus the frequency resolution) and the number of good tapers. See notes
    for additional information."""

docdict["time_bandwidth_tfr_notes"] = r"""
In MNE-Python's multitaper functions, the frequency bandwidth is
additionally affected by the parameter ``time_bandwidth``.
The ``n_cycles`` parameter determines the temporal window length based on the
frequencies of interest: :math:`T = \frac{\mathtt{n\_cycles}}{\mathtt{freqs}}`.
The ``time_bandwidth`` parameter defines the "time-bandwidth product", which is
the product of the temporal window length (in seconds) and the frequency
bandwidth (in Hz). Thus once ``n_cycles`` has been set, frequency bandwidth is
determined by :math:`\frac{\mathrm{time~bandwidth}}{\mathrm{time~window}}`, and
thus passing a larger ``time_bandwidth`` value will increase the frequency
bandwidth (thereby decreasing the frequency *resolution*).

The increased frequency bandwidth is reached by averaging spectral estimates
obtained from multiple tapers. Thus, ``time_bandwidth`` also determines the
number of tapers used. MNE-Python uses only "good" tapers (tapers with minimal
leakage from far-away frequencies); the number of good tapers is
``floor(time_bandwidth - 1)``. This means there is another trade-off at play,
between frequency resolution and the variance reduction that multitaper
analysis provides. Striving for finer frequency resolution (by setting
``time_bandwidth`` low) means fewer tapers will be used, which undermines what
is unique about multitaper methods — namely their ability to improve accuracy /
reduce noise in the power estimates by using several (orthogonal) tapers.

.. warning::

    In `~mne.time_frequency.tfr_array_multitaper` and
    `~mne.time_frequency.tfr_multitaper`, ``time_bandwidth`` defines the
    product of the temporal window length with the *full* frequency bandwidth
    For example, a full bandwidth of 4 Hz at a frequency of interest of 10 Hz
    will "smear" the frequency estimate between 8 Hz and 12 Hz.

    This is not the case for `~mne.time_frequency.psd_array_multitaper` where
    the argument ``bandwidth`` defines the *half* frequency bandwidth. In the
    example above, the half-frequency bandwidth is 2 Hz.
"""

docdict["time_format"] = """
time_format : 'float' | 'clock'
    Style of time labels on the horizontal axis. If ``'float'``, labels will be
    number of seconds from the start of the recording. If ``'clock'``,
    labels will show "clock time" (hours/minutes/seconds) inferred from
    ``raw.info['meas_date']``. Default is ``'float'``.

    .. versionadded:: 0.24
"""

_time_format_df_base = """
time_format : str | None
    Desired time format. If ``None``, no conversion is applied, and time values
    remain as float values in seconds. If ``'ms'``, time values will be rounded
    to the nearest millisecond and converted to integers. If ``'timedelta'``,
    time values will be converted to :class:`pandas.Timedelta` values. {}
    Default is ``None``.
"""

docdict["time_format_df"] = _time_format_df_base.format("")

_raw_tf = (
    "If ``'datetime'``, time values will be converted to "
    ":class:`pandas.Timestamp` values, relative to "
    "``raw.info['meas_date']`` and offset by ``raw.first_samp``. "
)
docdict["time_format_df_raw"] = _time_format_df_base.format(_raw_tf)

docdict["time_label"] = """
time_label : str | callable | None
    Format of the time label (a format string, a function that maps
    floating point time values to strings, or None for no label). The
    default is ``'auto'``, which will use ``time=%0.2f ms`` if there
    is more than one time point.
"""

docdict["time_unit"] = """\
time_unit : str
    The units for the time axis, can be "s" (default) or "ms".
"""

docdict["time_viewer_brain_screenshot"] = """
time_viewer : bool
    If True, include time viewer traces. Only used if
    ``time_viewer=True`` and ``separate_canvas=False``.
"""

docdict["timefreqs"] = """
timefreqs : None | list of tuple | dict of tuple
    The time-frequency point(s) for which topomaps will be plotted. See Notes.
"""

docdict["times"] = """
times : ndarray, shape (n_times,)
    The time values in seconds.
"""

docdict["title_none"] = """
title : str | None
    The title of the generated figure. If ``None`` (default), no title is
    displayed.
"""
docdict["title_tfr_plot"] = """
title : str | 'auto' | None
    Title for the plot. If ``"auto"``, will use the channel name (if ``combine`` is
    ``None``) or state the number and method of combined channels used to generate the
    plot. If ``None``, no title is shown. Default is ``None``.
"""
docdict["tmax_raw"] = """
tmax : float | None
    End time of the raw data to use in seconds (cannot exceed data duration).
    If ``None`` (default), the current end of the data is used.
"""

docdict["tmin"] = """
tmin : scalar
    Time point of the first sample in data.
"""

docdict["tmin_epochs"] = """
tmin : float
    Start time before event. If nothing provided, defaults to 0.
"""

docdict["tmin_raw"] = """
tmin : float
    Start time of the raw data to use in seconds (must be >= 0).
"""

docdict["tmin_tmax_psd"] = """\
tmin, tmax : float | None
    First and last times to include, in seconds. ``None`` uses the first or
    last time present in the data. Default is ``tmin=None, tmax=None`` (all
    times).
"""

docdict["tol_kind_rank"] = """
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

docdict["tol_rank"] = """
tol : float | 'auto'
    Tolerance for singular values to consider non-zero in
    calculating the rank. The singular values are calculated
    in this method such that independent data are expected to
    have singular value around one. Can be 'auto' to use the
    same thresholding as :func:`scipy.linalg.orth`.
"""

_topomap_args_template = """
{param} : dict | None
    Keyword arguments to pass to {func}.{extra}
"""
docdict["topomap_args"] = _topomap_args_template.format(
    param="topomap_args",
    func=":func:`mne.viz.plot_topomap`",
    extra=" ``axes`` and ``show`` are ignored. If ``times`` is not in this dict, "
    "automatic peak detection is used. Beyond that, if ``None``, no customizable "
    "arguments will be passed. Defaults to ``None`` (i.e., an empty :class:`dict`).",
)
docdict["topomap_kwargs"] = _topomap_args_template.format(
    param="topomap_kwargs", func="the topomap-generating functions", extra=""
)

_trans_base = """\
If str, the path to the head<->MRI transform ``*-trans.fif`` file produced
    during coregistration. Can also be ``'fsaverage'`` to use the built-in
    fsaverage transformation."""

docdict["trans"] = f"""
trans : path-like | dict | instance of Transform | ``"fsaverage"`` | None
    {_trans_base}
    If trans is None, an identity matrix is assumed.
"""

docdict["trans_not_none"] = f"""
trans : str | dict | instance of Transform
    {_trans_base}
"""

docdict["transparent"] = """
transparent : bool | None
    If True: use a linear transparency between fmin and fmid
    and make values below fmin fully transparent (symmetrically for
    divergent colormaps). None will choose automatically based on colormap
    type.
"""

docdict["tstart_ecg"] = """
tstart : float
    Start ECG detection after ``tstart`` seconds. Useful when the beginning
    of the run is noisy.
"""

docdict["tstep"] = """
tstep : scalar
    Time step between successive samples in data.
"""

# %%
# U

docdict["ui_event_name_source"] = """
name : str
    The name of the event (same as its class name but in snake_case).
source : matplotlib.figure.Figure | Figure3D
    The figure that published the event.
"""

docdict["uint16_codec"] = """
uint16_codec : str | None
    If your set file contains non-ascii characters, sometimes reading
    it may fail and give rise to error message stating that "buffer is
    too small". ``uint16_codec`` allows to specify what codec (for example:
    'latin1' or 'utf-8') should be used when reading character arrays and
    can therefore help you solve this problem.
"""

docdict["units"] = """
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

docdict["units_edf_bdf_io"] = """
units : dict | str
    The units of the channels as stored in the file. This argument
    is useful only if the units are missing from the original file.
    If a dict, it must map a channel name to its unit, and if str
    it is assumed that all channels have the same units.
"""

_units = """
units : {}str | None
    The units to use for the colorbar label. Ignored if ``colorbar=False``.
    If ``None`` {}the label will be "AU" indicating arbitrary units.
    Default is ``None``.
"""
docdict["units_topomap"] = _units.format("", "")
docdict["units_topomap_evoked"] = _units.format(
    "dict | ", "and ``scalings=None`` the unit is automatically determined, otherwise "
)

docdict["use_cps"] = """
use_cps : bool
    Whether to use cortical patch statistics to define normal orientations for
    surfaces (default True).
"""

docdict["use_cps_restricted"] = """
use_cps : bool
    Whether to use cortical patch statistics to define normal orientations for
    surfaces (default True).

    Only used when the inverse is free orientation (``loose=1.``),
    not in surface orientation, and ``pick_ori='normal'``.
"""

docdict["use_opengl"] = """
use_opengl : bool | None
    Whether to use OpenGL when rendering the plot (requires ``pyopengl``).
    May increase performance, but effect is dependent on system CPU and
    graphics hardware. Only works if using the Qt backend. Default is
    None, which will use False unless the user configuration variable
    ``MNE_BROWSER_USE_OPENGL`` is set to ``'true'``,
    see :func:`mne.set_config`.

    .. versionadded:: 0.24
"""

# %%
# V

docdict["vector_pctf"] = """
vector : bool
    Whether to return PSF/CTF as vector source estimate (3 values per
    location) or source estimate object (1 intensity value per location).
    Only allowed to be True if corresponding dimension of resolution matrix
    is 3 * n_dipoles. Defaults to False.

    .. versionadded:: 1.2
"""

docdict["verbose"] = """
verbose : bool | str | int | None
    Control verbosity of the logging output. If ``None``, use the default
    verbosity level. See the :ref:`logging documentation <tut-logging>` and
    :func:`mne.verbose` for details. Should only be passed as a keyword
    argument.
"""

docdict["vertices_volume"] = """
vertices : list of array of int
    The indices of the dipoles in the source space. Should be a single
    array of shape (n_dipoles,) unless there are subvolumes.
"""

docdict["view"] = """
view : str | None
    The name of the view to show (e.g. "lateral"). Other arguments
    take precedence and modify the camera starting from the ``view``.
    See :meth:`Brain.show_view <mne.viz.Brain.show_view>` for valid
    string shortcut options.
"""

docdict["view_layout"] = """
view_layout : str
    Can be "vertical" (default) or "horizontal". When using "horizontal" mode,
    the PyVista backend must be used and hemi cannot be "split".
"""

docdict["views"] = """
views : str | list
    View to use. Using multiple views (list) is not supported for mpl
    backend. See :meth:`Brain.show_view <mne.viz.Brain.show_view>` for
    valid string options.
"""

_vlim = """\
vlim : tuple of length 2{joint_param}
    Lower and upper bounds of the colormap, typically a numeric value in the same
    units as the data. {callable}
    If both entries are ``None``, the bounds are set at {bounds}.
    Providing ``None`` for just one entry will set the corresponding boundary at the
    min/max of the data. {extra}Defaults to ``(None, None)``.
"""
_joint_param = ' | "joint"'
_callable_sentence = """Elements of the :class:`tuple` may also be callable functions
    which take in a :class:`NumPy array <numpy.ndarray>` and return a scalar.
"""
_bounds_symmetric = """± the maximum absolute value
    of the data (yielding a colormap with midpoint at 0)"""
_bounds_minmax = "``(min(data), max(data))``"
_bounds_norm = "``(0, max(abs(data)))``"
_bounds_contingent = f"""{_bounds_symmetric}, or {_bounds_norm}
    if the (possibly baselined) data are all-positive"""
_joint_sentence = """If ``vlim="joint"``, will compute the colormap limits
    jointly across all {what}s of the same channel type (instead of separately
    for each {what}), using the min/max of the data for that channel type.
    {joint_extra}"""

docdict["vlim_plot_topomap"] = _vlim.format(
    joint_param="", callable="", bounds=_bounds_minmax, extra=""
)
docdict["vlim_plot_topomap_proj"] = _vlim.format(
    joint_param=_joint_param,
    callable=_callable_sentence,
    bounds=_bounds_contingent,
    extra=_joint_sentence.format(
        what="projector",
        joint_extra='If vlim is ``"joint"``, ``info`` must not be ``None``. ',
    ),
)
docdict["vlim_plot_topomap_psd"] = _vlim.format(
    joint_param=_joint_param,
    callable=_callable_sentence,
    bounds=_bounds_contingent,
    extra=_joint_sentence.format(what="topomap", joint_extra=""),
)
docdict["vlim_tfr_plot"] = _vlim.format(
    joint_param="", callable="", bounds=_bounds_contingent, extra=""
)
docdict["vlim_tfr_plot_joint"] = _vlim.format(
    joint_param="",
    callable="",
    bounds=_bounds_contingent,
    extra="""To specify the colormap separately for the topomap annotations,
    see ``topomap_args``. """,
)

_vmin_vmax_template = """
vmin, vmax : float | {allowed}None
    Lower and upper bounds of the colormap, in the same units as the data.
    If ``vmin`` and ``vmax`` are both ``None``, the bounds are set at
    {bounds}. If only one of ``vmin``, ``vmax`` is ``None``, will use
    ``min(data)`` or ``max(data)``, respectively.{extra}
"""

# ↓↓↓ this one still used, needs helper func refactor before we can migrate to `vlim`
docdict["vmin_vmax_tfr_plot_topo"] = _vmin_vmax_template.format(
    allowed="", bounds=_bounds_symmetric, extra=""
)
# ↓↓↓ this one still used in Evoked.animate_topomap(), should migrate to `vlim`
docdict["vmin_vmax_topomap"] = _vmin_vmax_template.format(
    allowed="callable | ",
    bounds=_bounds_symmetric,
    extra=""" If callable, should accept
    a :class:`NumPy array <numpy.ndarray>` of data and return a :class:`float`.""",
)


# %%
# W

docdict["weight_norm"] = """
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
    - ``'unit-noise-gain-invariant'``
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

docdict["window_psd"] = """\
window : str | float | tuple
    Windowing function to use. See :func:`scipy.signal.get_window`.
"""

docdict["window_resample"] = """
window : str | tuple
    When ``method="fft"``, this is the *frequency-domain* window to use in resampling,
    and should be the same length as the signal; see :func:`scipy.signal.resample`
    for details. When ``method="polyphase"``, this is the *time-domain* linear-phase
    window to use after upsampling the signal; see :func:`scipy.signal.resample_poly`
    for details. The default ``"auto"`` will use ``"boxcar"`` for ``method="fft"`` and
    ``("kaiser", 5.0)`` for ``method="polyphase"``.
"""

# %%
# X

docdict["xscale_plot_psd"] = """\
xscale : 'linear' | 'log'
    Scale of the frequency axis. Default is ``'linear'``.
"""

# %%
# Y

docdict["yscale_tfr_plot"] = """
yscale : 'auto' | 'linear' | 'log'
    The scale of the y (frequency) axis. 'linear' gives linear y axis, 'log' gives
    log-spaced y axis and 'auto' detects if frequencies are log-spaced and if so sets
    the y axis to 'log'. Default is 'auto'.
"""

# %%
# Z

# this is needed in test_docstring_parameters, which reads the file as text
docdict["¿test—üñɪçøɖɘ_keys*"] = "¿test—üñɪçøɖɘ_values*"

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
        icount = _indentcount_lines(lines[1:])
    # Insert this indent to dictionary docstrings
    try:
        indented = docdict_indented[icount]
    except KeyError:
        indent = " " * icount
        docdict_indented[icount] = indented = {}
        for name, dstr in docdict.items():
            lines = dstr.splitlines()
            try:
                newlines = [lines[0]]
                for line in lines[1:]:
                    newlines.append(indent + line)
                indented[name] = "\n".join(newlines)
            except IndexError:
                indented[name] = dstr
    try:
        f.__doc__ = docstring % indented
    except (TypeError, ValueError, KeyError) as exp:
        funcname = f.__name__
        funcname = docstring.split("\n")[0] if funcname is None else funcname
        raise RuntimeError(f"Error documenting {funcname}:\n{exp}")
    return f


##############################################################################
# Utilities for docstring manipulation.


def copy_doc(source):
    """Copy the docstring from another function (decorator).

    The docstring of the source function is prepepended to the docstring of the
    function wrapped by this decorator.

    This is useful when inheriting from a class and overloading a method. This
    decorator can be used to copy the docstring of the original method.

    Docstrings are processed by :func:`python:inspect.cleandoc` before being used.

    Parameters
    ----------
    source : function
        Function to copy the docstring from.

    Returns
    -------
    wrapper : function
        The decorated function.

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
    Docstring for m1
    this gets appended
    """

    def wrapper(func):
        if source.__doc__ is None or len(source.__doc__) == 0:
            raise ValueError("Cannot copy docstring: docstring was empty.")
        doc = source.__doc__
        if func.__doc__ is not None:
            doc += f"\n{inspect.cleandoc(func.__doc__)}"
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

    Docstrings are parsed by :func:`python:inspect.cleandoc` before being used.
    If indentation and newlines are important, make the first line ``.``, and the dot
    will be removed and all following lines dedented jointly.

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
    ...         '''.
    ...
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
    """  # noqa: D410, D411, D214, D215

    def wrapper(func):
        # Work with cleandoc'ed sources (py3.13-compat)
        doc = inspect.cleandoc(source.__doc__).split("\n")
        if func.__doc__ is not None:
            func_doc = inspect.cleandoc(func.__doc__)
            if func_doc[:2] == ".\n":
                func_doc = func_doc[2:]
            func_doc = f"\n{func_doc}"
        else:
            func_doc = ""

        if len(doc) == 1:
            func.__doc__ = f"{doc[0]}{func_doc}"
            return func

        # Find parameter block
        for line, text in enumerate(doc[:-2]):
            if text.strip() == "Parameters" and doc[line + 1].strip() == "----------":
                parameter_block = line
                break
        else:
            # No parameter block found
            raise ValueError(
                "Cannot copy function docstring: no parameter "
                "block found. To simply copy the docstring, use "
                "the @copy_doc decorator instead."
            )

        # Find first parameter
        for line, text in enumerate(doc[parameter_block:], parameter_block):
            if ":" in text:
                first_parameter = line
                parameter_indentation = len(text) - len(text.lstrip(" "))
                break
        else:
            raise ValueError(
                "Cannot copy function docstring: no parameters "
                "found. To simply copy the docstring, use the "
                "@copy_doc decorator instead."
            )

        # Find end of first parameter
        for line, text in enumerate(doc[first_parameter + 1 :], first_parameter + 1):
            # Ignore empty lines
            if len(text.strip()) == 0:
                continue

            line_indentation = len(text) - len(text.lstrip(" "))
            if line_indentation <= parameter_indentation:
                # Reach end of first parameter
                first_parameter_end = line

                # Of only one parameter is defined, remove the Parameters
                # heading as well
                if ":" not in text:
                    first_parameter = parameter_block

                break
        else:
            # End of docstring reached
            first_parameter_end = line + 1
            first_parameter = parameter_block

        # Copy the docstring, but remove the first parameter
        doc = (
            "\n".join(doc[:first_parameter])
            + "\n"
            + "\n".join(doc[first_parameter_end:])
        )
        func.__doc__ = f"{doc}{func_doc}"
        return func

    return wrapper


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

    if domain != "py":
        return None

    modname = info["module"]
    fullname = info["fullname"]

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except Exception:
            return None
    # deal with our decorators properly
    while hasattr(obj, "__wrapped__"):
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
    fn = "/".join(op.normpath(fn).split(os.sep))  # in case on Windows

    try:
        source, lineno = inspect.getsourcelines(obj)
    except Exception:
        lineno = None

    if lineno:
        linespec = f"#L{lineno}-L{lineno + len(source) - 1}"
    else:
        linespec = ""

    if "dev" in mne.__version__:
        kind = "main"
    else:
        kind = "maint/" + ".".join(mne.__version__.split(".")[:2])
    return f"http://github.com/mne-tools/mne-python/blob/{kind}/mne/{fn}{linespec}"


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
        kind = get_config("MNE_DOCS_KIND", "api")
    help_dict = dict(
        api="python_reference.html",
        tutorials="tutorials.html",
        examples="auto_examples/index.html",
    )
    _check_option("kind", kind, sorted(help_dict.keys()))
    kind = help_dict[kind]
    if version is None:
        version = get_config("MNE_DOCS_VERSION", "stable")
    _check_option("version", version, ["stable", "dev"])
    webbrowser.open_new_tab(f"https://mne.tools/{version}/{kind}")


class _decorator:
    """Inject code or modify the docstring of a class, method, or function."""

    def __init__(self, extra):
        self.kind = self.__class__.__name__
        self.extra = extra
        self.msg = f"NOTE: {{}}() is a {self.kind} {{}}. {self.extra}."

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
        if inspect.isclass(obj):
            obj_type = "class"
        else:
            # NB: detecting (bound and unbound) methods seems to be impossible
            assert inspect.isfunction(obj), f"decorator used on {type(obj)}"
            obj_type = "function"
        msg = self.msg.format(obj.__name__, obj_type)
        if obj_type == "class":
            obj.__init__ = self._make_fun(obj.__init__, msg)
            return obj
        return self._make_fun(obj, msg)

    def _make_fun(self, func, body):
        evaldict = dict(_function_=func)
        fm = FunctionMaker(func, None, None, None, None, func.__module__)
        attrs = dict(
            __wrapped__=func,
            __qualname__=func.__qualname__,
            __globals__=func.__globals__,
        )
        dep = fm.make(body, evaldict, addsource=True, **attrs)
        dep.__doc__ = self._update_doc(dep.__doc__)
        dep._deprecated_original = func
        return dep

    def _update_doc(self, olddoc):
        newdoc = f".. warning:: {self.kind.upper()}"
        if self.extra:
            newdoc = f"{newdoc}: {self.extra}"
        newdoc += "."
        if olddoc:
            # Get the spacing right to avoid sphinx warnings
            n_space = 4
            for li, line in enumerate(olddoc.split("\n")):
                if li > 0 and len(line.strip()):
                    n_space = len(line) - len(line.lstrip())
                    break
            newdoc = f"{newdoc}\n\n{' ' * n_space}{olddoc}"
        return newdoc


# Following deprecated class copied from scikit-learn
class deprecated(_decorator):
    """Mark a function, class, or method as deprecated (decorator).

    Originally adapted from sklearn and
    http://wiki.python.org/moin/PythonDecoratorLibrary, then modified to make
    arguments populate properly following our verbose decorator methods based
    on decorator.

    Parameters
    ----------
    extra : str
        Extra information beyond just saying the class/function/method is
        deprecated. Should be a complete sentence (trailing period will be
        added automatically). Will be included in FutureWarning messages
        and in a sphinx warning box in the docstring.
    """

    def _make_fun(self, func, msg):
        body = f"""\
def %(name)s(%(signature)s):\n
    import warnings
    warnings.warn({repr(msg)}, category=FutureWarning)
    return _function_(%(shortsignature)s)"""
        return super()._make_fun(func=func, body=body)


def deprecated_alias(dep_name, func, removed_in=None):
    """Inject a deprecated alias into the namespace."""
    if removed_in is None:
        from .. import __version__

        removed_in = __version__.split(".")[:2]
        removed_in[1] = str(int(removed_in[1]) + 1)
        removed_in = ".".join(removed_in)
    # Inject a deprecated version into the namespace
    inspect.currentframe().f_back.f_globals[dep_name] = deprecated(
        f"{dep_name} has been deprecated in favor of {func.__name__} and will "
        f"be removed in {removed_in}."
    )(deepcopy(func))


###############################################################################
# "legacy" decorator for parts of our API retained only for backward compat


class legacy(_decorator):
    """Mark a function, class, or method as legacy (decorator).

    Parameters
    ----------
    alt : str
        Description of the alternate, preferred way to achieve a comparable
        result.
    extra : str
        Extra information beyond just saying the class/function/method is
        legacy. Should be a complete sentence (trailing period will be
        added automatically). Will be included in logger.info messages
        and in a sphinx warning box in the docstring.
    """

    def __init__(self, alt, extra=""):
        period = ". " if len(extra) else ""
        extra = f"New code should use {alt}{period}{extra}"
        super().__init__(extra=extra)

    def _make_fun(self, func, msg):
        body = f"""\
def %(name)s(%(signature)s):\n
    from mne.utils import logger
    logger.info({repr(msg)})
    return _function_(%(shortsignature)s)"""
        return super()._make_fun(func=func, body=body)


###############################################################################
# The following tools were adapted (mostly trimmed) from SciPy's doccer.py


def _docformat(docstring, docdict=None, funcname=None):
    """Fill a function docstring from variables in dictionary.

    Adapt the indent of the inserted docs

    Parameters
    ----------
    docstring : string
        docstring from function, possibly with dict formatting strings
    docdict : dict, optional
        dictionary with keys that match the dict formatting strings
        and values that are docstring fragments to be inserted.  The
        indentation of the inserted docstrings is set to match the
        minimum indentation of the ``docstring`` by adding this
        indentation to all lines of the inserted string, except the
        first

    Returns
    -------
    outstring : string
        string with requested ``docdict`` strings inserted
    """
    if not docstring:
        return docstring
    if docdict is None:
        docdict = {}
    if not docdict:
        return docstring
    lines = docstring.expandtabs().splitlines()
    # Find the minimum indent of the main docstring, after first line
    if len(lines) < 2:
        icount = 0
    else:
        icount = _indentcount_lines(lines[1:])
    indent = " " * icount
    # Insert this indent to dictionary docstrings
    indented = {}
    for name, dstr in docdict.items():
        lines = dstr.expandtabs().splitlines()
        try:
            newlines = [lines[0]]
            for line in lines[1:]:
                newlines.append(indent + line)
            indented[name] = "\n".join(newlines)
        except IndexError:
            indented[name] = dstr
    funcname = docstring.split("\n")[0] if funcname is None else funcname
    try:
        return docstring % indented
    except (TypeError, ValueError, KeyError) as exp:
        raise RuntimeError(f"Error documenting {funcname}:\n{exp}")


def _indentcount_lines(lines):
    """Compute minimum indent for all lines in line list."""
    indentno = sys.maxsize
    for line in lines:
        stripped = line.lstrip()
        if stripped:
            indentno = min(indentno, len(line) - len(stripped))
    if indentno == sys.maxsize:
        return 0
    return indentno
