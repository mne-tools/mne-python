# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from collections.abc import Callable, Sequence
from copy import deepcopy
from inspect import getfullargspec
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

import numpy as np

from ._fiff.constants import FIFF
from ._fiff.meas_info import (
    ContainsMixin,
    Info,
    SetChannelsMixin,
    _ensure_infos_match,
    _read_extended_ch_info,
    _rename_list,
    read_meas_info,
    write_meas_info,
)
from ._fiff.open import fiff_open
from ._fiff.pick import _FNIRS_CH_TYPES_SPLIT, _picks_to_idx, pick_types
from ._fiff.proj import ProjMixin
from ._fiff.tag import read_tag
from ._fiff.tree import dir_tree_find
from ._fiff.write import (
    end_block,
    start_and_end_file,
    start_block,
    write_complex_float_matrix,
    write_float,
    write_float_matrix,
    write_id,
    write_int,
    write_string,
)
from .baseline import _check_baseline, _log_rescale, rescale
from .channels.channels import InterpolationMixin, ReferenceMixin, UpdateChannelsMixin
from .channels.layout import Layout, _merge_ch_data, _pair_grad_sensors
from .defaults import _BORDER_DEFAULT, _EXTRAPOLATE_DEFAULT, _INTERPOLATION_DEFAULT
from .filter import FilterMixin, _check_fun, detrend
from .html_templates import _get_html_template
from .parallel import parallel_func
from .time_frequency.spectrum import Spectrum, SpectrumMixin, _validate_method
from .utils import (
    ExtendedTimeMixin,
    SizeMixin,
    _build_data_frame,
    _check_fname,
    _check_option,
    _check_pandas_index_arguments,
    _check_pandas_installed,
    _check_preload,
    _check_time_format,
    _convert_times,
    _scale_dataframe_data,
    _validate_type,
    _verbose_control,
    check_fname,
    copy_function_doc_to_method_doc_static,
    fill_doc_static,
    logger,
    repr_html,
    sizeof_fmt,
    verbose_static,
    warn,
)
from .utils._typing import Color, Self

if TYPE_CHECKING:
    # Heavy/optional deps kept out of the runtime import path (see
    # mne/tests/test_import_nesting.py); referenced only via string annotations.
    from matplotlib.animation import FuncAnimation
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap, Normalize
    from matplotlib.figure import Figure
    from pandas import DataFrame

    from .bem import ConductorModel
    from .cov import Covariance
    from .time_frequency.tfr import AverageTFR
    from .viz import Brain, EvokedField, Figure3D

_aspect_dict = {
    "average": FIFF.FIFFV_ASPECT_AVERAGE,
    "standard_error": FIFF.FIFFV_ASPECT_STD_ERR,
    "single_epoch": FIFF.FIFFV_ASPECT_SINGLE,
    "partial_average": FIFF.FIFFV_ASPECT_SUBAVERAGE,
    "alternating_subaverage": FIFF.FIFFV_ASPECT_ALTAVERAGE,
    "sample_cut_out_by_graph": FIFF.FIFFV_ASPECT_SAMPLE,
    "power_density_spectrum": FIFF.FIFFV_ASPECT_POWER_DENSITY,
    "dipole_amplitude_cuvre": FIFF.FIFFV_ASPECT_DIPOLE_WAVE,
    "squid_modulation_lower_bound": FIFF.FIFFV_ASPECT_IFII_LOW,
    "squid_modulation_upper_bound": FIFF.FIFFV_ASPECT_IFII_HIGH,
    "squid_gate_setting": FIFF.FIFFV_ASPECT_GATE,
}
_aspect_rev = {val: key for key, val in _aspect_dict.items()}


@fill_doc_static("verbose", "info_not_none")
class Evoked(
    ProjMixin,
    ContainsMixin,
    UpdateChannelsMixin,
    ReferenceMixin,
    SetChannelsMixin,
    InterpolationMixin,
    FilterMixin,
    ExtendedTimeMixin,
    SizeMixin,
    SpectrumMixin,
):
    """Evoked data.

    Parameters
    ----------
    fname : path-like | None
        Name of evoked/average FIF file to load.
        If None no data is loaded.
    condition : int | str | None
        Dataset ID number (int) or comment/name (str). Optional if there is
        only one data set in file.
    proj : bool
        Apply SSP projection vectors.
    kind : str
        Either ``'average'`` or ``'standard_error'``. The type of data to read.
        Only used if 'condition' is a str.
    allow_maxshield : bool | str
        If True, allow loading of data that has been recorded with internal
        active compensation (MaxShield). Data recorded with MaxShield should
        generally not be loaded directly, but should first be processed using
        SSS/tSSS to remove the compensation signals that may also affect brain
        activity. Can also be ``"yes"`` to load without eliciting a warning.
    verbose : bool | str | int | None
        Control verbosity of the logging output. If ``None``, use the default
        verbosity level. See the :ref:`logging documentation <tut-logging>` and
        :func:`mne.verbose` for details. Should only be passed as a keyword
        argument.

    Attributes
    ----------
    info : mne.Info
        The :class:`mne.Info` object with information about the
        sensors and methods of measurement.
    ch_names : list of str
        List of channels' names.
    nave : int
        Number of averaged epochs.
    kind : str
        Type of data, either average or standard_error.
    comment : str
        Comment on dataset. Can be the condition.
    data : array of shape (n_channels, n_times)
        Evoked response.
    first : int
        First time sample.
    last : int
        Last time sample.
    tmin : float
        The first time point in seconds.
    tmax : float
        The last time point in seconds.
    times :  array
        Time vector in seconds. Goes from ``tmin`` to ``tmax``. Time interval
        between consecutive time samples is equal to the inverse of the
        sampling frequency.
    baseline : None | tuple of length 2
         This attribute reflects whether the data has been baseline-corrected
         (it will be a ``tuple`` then) or not (it will be ``None``).

    Notes
    -----
    Evoked objects can only contain the average of a single set of conditions.
    """

    @_verbose_control
    def __init__(
        self,
        fname: Path | str | None,
        condition: int | str | None = None,
        proj: bool = True,
        kind: str = "average",
        allow_maxshield: bool | str = False,
        *,
        verbose: bool | str | int | None = None,
    ):
        _validate_type(proj, bool, "'proj'")
        # Read the requested data
        fname = _check_fname(fname=fname, must_exist=True, overwrite="read")
        (
            self.info,
            self.nave,
            self._aspect_kind,
            self.comment,
            times,
            self.data,
            self.baseline,
        ) = _read_evoked(fname, condition, kind, allow_maxshield)
        self._set_times(times)
        self._raw_times = self.times.copy()
        self._decim = 1

        self._update_first_last()
        self.preload = True
        # project and baseline correct
        if proj:
            self.apply_proj()
        self.filename = fname

    @property
    def filename(self) -> Path | None:
        """The filename of the evoked object, if it exists.

        :type: :class:`~pathlib.Path` | None
        """
        return self._filename

    @filename.setter
    def filename(self, value: Path | str | None) -> None:
        self._filename = Path(value) if value is not None else value

    @property
    def kind(self) -> str:
        """The data kind."""
        return _aspect_rev[self._aspect_kind]

    @kind.setter
    def kind(self, kind: str) -> None:
        _check_option("kind", kind, list(_aspect_dict.keys()))
        self._aspect_kind = _aspect_dict[kind]

    @property
    def data(self) -> np.ndarray:
        """The data matrix."""
        return self._data

    @data.setter
    def data(self, data: np.ndarray) -> None:
        """Set the data matrix."""
        self._data = data

    @fill_doc_static("picks_all", "units")
    def get_data(
        self,
        picks: str | np.ndarray | slice | None = None,
        units: str | dict | None = None,
        tmin: float | None = None,
        tmax: float | None = None,
        exclude: list[str] | Literal["bads"] | tuple = (),
    ) -> np.ndarray:
        """Get evoked data as 2D array.

        Parameters
        ----------
        picks : str | array-like | slice | None
            Channels to include. Slices and lists of integers will be interpreted as
            channel indices. In lists, channel *type* strings (e.g., ``['meg',
            'eeg']``) will pick channels of those types, channel *name* strings (e.g.,
            ``['MEG0111', 'MEG2623']`` will pick the given channels. Can also be the
            string values ``'all'`` to pick all channels, or ``'data'`` to pick
            :term:`data channels`. None (default) will pick all channels. Bad channels
            are included by default. Note that channels in ``info['bads']`` *will be
            included* if their names or indices are explicitly provided.
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
        tmin : float | None
            Start time of data to get in seconds.
        tmax : float | None
            End time of data to get in seconds.
        exclude : list[str] | Literal["bads"]
            Channels to exclude. If ``'bads'``, channels in ``info['bads']`` are
            excluded; pass an empty list or tuple (the default) to include all
            channels.

            .. versionadded:: 1.13

        Returns
        -------
        data : ndarray, shape (n_channels, n_times)
            A view on evoked data.

        Notes
        -----
        .. versionadded:: 0.24
        """
        # Avoid circular import
        from .io.base import _get_ch_factors

        picks = _picks_to_idx(self.info, picks, "all", exclude=exclude)

        start, stop = self._handle_tmin_tmax(tmin, tmax)

        data = self.data[picks, start:stop]

        if units is not None:
            ch_factors = _get_ch_factors(self, units, picks)
            data *= ch_factors[:, np.newaxis]

        return data

    @verbose_static(
        "applyfun_summary_evoked",
        "fun_applyfun_evoked",
        "picks_all_data_noref",
        "dtype_applyfun",
        "n_jobs",
        "channel_wise_applyfun",
        "kwargs_fun",
    )
    def apply_function(
        self,
        fun: Callable,
        picks: str | np.ndarray | slice | None = None,
        dtype: np.dtype | None = None,
        n_jobs: int | None = None,
        channel_wise: bool = True,
        *,
        verbose: bool | str | int | None = None,
        **kwargs,
    ) -> Self:
        """Apply a function to a subset of channels.

        The function ``fun`` is applied to the channels defined in ``picks``. The
        evoked object's data is modified in-place. If the function returns a
        different data type (e.g. :py:obj:`numpy.complex128`) it must be specified
        using the ``dtype`` parameter, which causes the data type of **all** the data
        to change (even if the function is only applied to channels in
        ``picks``).

        .. note:: If ``n_jobs`` > 1, more memory is required as
                  ``len(picks) * n_times`` additional time points need to
                  be temporarily stored in memory.
        .. note:: If the data type changes (``dtype != None``), more memory is
                  required since the original and the converted data needs
                  to be stored in memory.

        Parameters
        ----------
        fun : callable
            A function to be applied to the channels. The first argument of
            fun has to be a timeseries (:class:`numpy.ndarray`). The function must
            operate on an array of shape ``(n_times,)``  because it will apply
            channel-wise.
            The function must return an :class:`~numpy.ndarray` shaped like its input.

            .. note::
                If ``channel_wise=True``, one can optionally access the index and/or the
                name of the currently processed channel within the applied function.
                This can enable tailored computations for different channels.
                To use this feature, add ``ch_idx`` and/or ``ch_name`` as
                additional argument(s) to your function definition.
        picks : str | array-like | slice | None
            Channels to include. Slices and lists of integers will be interpreted as
            channel indices. In lists, channel *type* strings (e.g., ``['meg',
            'eeg']``) will pick channels of those types, channel *name* strings (e.g.,
            ``['MEG0111', 'MEG2623']`` will pick the given channels. Can also be the
            string values ``'all'`` to pick all channels, or ``'data'`` to pick
            :term:`data channels`. None (default) will pick all data channels
            (excluding reference MEG channels). Note that channels in ``info['bads']``
            *will be included* if their names or indices are explicitly provided.
        dtype : numpy.dtype
            Data type to use after applying the function. If None
            (default) the data type is not modified.
        n_jobs : int | None
            The number of jobs to run in parallel. If ``-1``, it is set
            to the number of CPU cores. Requires the :mod:`joblib` package.
            ``None`` (default) is a marker for 'unset' that will be interpreted
            as ``n_jobs=1`` (sequential execution) unless the call is performed under
            a :class:`joblib:joblib.parallel_config` context manager that sets another
            value for ``n_jobs``.
            Ignored if ``channel_wise=False`` as the workload
            is split across channels.
        channel_wise : bool
            Whether to apply the function to each channel individually. If
            ``False``, the function will be applied to all channels at once.
            Default ``True``.

            .. versionadded:: 1.6
        verbose : bool | str | int | None
            Control verbosity of the logging output. If ``None``, use the default
            verbosity level. See the :ref:`logging documentation <tut-logging>` and
            :func:`mne.verbose` for details. Should only be passed as a keyword
            argument.
        **kwargs : dict
            Additional keyword arguments to pass to ``fun``.

        Returns
        -------
        self : instance of Evoked
            The evoked object with transformed data.
        """
        _check_preload(self, "evoked.apply_function")
        picks = _picks_to_idx(self.info, picks, exclude=(), with_ref_meg=False)

        if not callable(fun):
            raise ValueError("fun needs to be a function")

        data_in = self._data
        if dtype is not None and dtype != self._data.dtype:
            self._data = self._data.astype(dtype)

        args = getfullargspec(fun).args + getfullargspec(fun).kwonlyargs
        if channel_wise is False:
            if ("ch_idx" in args) or ("ch_name" in args):
                raise ValueError(
                    "apply_function cannot access ch_idx or ch_name "
                    "when channel_wise=False"
                )
        if "ch_idx" in args:
            logger.info("apply_function requested to access ch_idx")
        if "ch_name" in args:
            logger.info("apply_function requested to access ch_name")

        # check the dimension of the incoming evoked data
        _check_option("evoked.ndim", self._data.ndim, [2])

        if channel_wise:
            parallel, p_fun, n_jobs = parallel_func(_check_fun, n_jobs)
            if n_jobs == 1:
                # modify data inplace to save memory
                for ch_idx in picks:
                    if "ch_idx" in args:
                        kwargs.update(ch_idx=ch_idx)
                    if "ch_name" in args:
                        kwargs.update(ch_name=self.info["ch_names"][ch_idx])
                    self._data[ch_idx, :] = _check_fun(
                        fun, data_in[ch_idx, :], **kwargs
                    )
            else:
                # use parallel function
                data_picks_new = parallel(
                    p_fun(
                        fun,
                        data_in[ch_idx, :],
                        **kwargs,
                        **{
                            k: v
                            for k, v in [
                                ("ch_name", self.info["ch_names"][ch_idx]),
                                ("ch_idx", ch_idx),
                            ]
                            if k in args
                        },
                    )
                    for ch_idx in picks
                )
                for run_idx, ch_idx in enumerate(picks):
                    self._data[ch_idx, :] = data_picks_new[run_idx]
        else:
            self._data[picks, :] = _check_fun(fun, data_in[picks, :], **kwargs)

        return self

    @verbose_static("baseline_evoked")
    def apply_baseline(
        self,
        baseline: tuple[float | None, float | None] | None = (None, 0),
        *,
        verbose: bool | str | int | None = None,
    ) -> Self:
        """Baseline correct evoked data.

        Parameters
        ----------
        baseline : None | tuple of length 2
            The time interval to consider as "baseline" when applying baseline
            correction. If ``None``, do not apply baseline correction.
            If a tuple ``(a, b)``, the interval is between ``a`` and ``b``
            (in seconds), including the endpoints.
            If ``a`` is ``None``, the **beginning** of the data is used; and if ``b``
            is ``None``, it is set to the **end** of the data.
            If ``(None, None)``, the entire time interval is used.

            .. note::
                The baseline ``(a, b)`` includes both endpoints, i.e. all timepoints
                ``t`` such that ``a <= t <= b``.

            Correction is applied **to each channel individually** in the following
            way:

            1. Calculate the mean signal of the baseline period.
            2. Subtract this mean from the **entire** ``Evoked``.

            Defaults to ``(None, 0)``, i.e. beginning of the data until
            time point zero.
        verbose : bool | str | int | None
            Control verbosity of the logging output. If ``None``, use the default
            verbosity level. See the :ref:`logging documentation <tut-logging>` and
            :func:`mne.verbose` for details. Should only be passed as a keyword
            argument.

        Returns
        -------
        evoked : instance of Evoked
            The baseline-corrected Evoked object.

        Notes
        -----
        Baseline correction can be done multiple times.

        .. versionadded:: 0.13.0
        """
        baseline = _check_baseline(baseline, times=self.times, sfreq=self.info["sfreq"])
        if self.baseline is not None and baseline is None:
            raise ValueError(
                "The data has already been baseline-corrected. "
                "Cannot remove existing baseline correction."
            )
        elif baseline is None:
            # Do not rescale
            logger.info(_log_rescale(None))
        else:
            # Actually baseline correct the data. Logging happens in rescale().
            self.data = rescale(self.data, self.times, baseline, copy=False)
            self.baseline = baseline

        return self

    @verbose_static("overwrite")
    def save(
        self,
        fname: Path | str,
        *,
        overwrite: bool = False,
        verbose: bool | str | int | None = None,
    ) -> None:
        """Save evoked data to a file.

        Parameters
        ----------
        fname : path-like
            The name of the file, which should end with ``-ave.fif(.gz)`` or
            ``_ave.fif(.gz)``.
        overwrite : bool
            If True (default False), overwrite the destination file if it
            exists.
        verbose : bool | str | int | None
            Control verbosity of the logging output. If ``None``, use the default
            verbosity level. See the :ref:`logging documentation <tut-logging>` and
            :func:`mne.verbose` for details. Should only be passed as a keyword
            argument.

        Notes
        -----
        To write multiple conditions into a single file, use
        `mne.write_evokeds`.

        .. versionchanged:: 0.23
            Information on baseline correction will be stored with the data,
            and will be restored when reading again via `mne.read_evokeds`.
        """
        write_evokeds(fname, self, overwrite=overwrite)

    @verbose_static(
        "export_fmt_support_evoked",
        "export_warning",
        "fname_export_params",
        "export_fmt_params_evoked",
        "overwrite",
        "export_warning_note_evoked",
    )
    def export(
        self,
        fname: str,
        fmt: Literal["auto", "mff"] = "auto",
        *,
        overwrite: bool = False,
        verbose: bool | str | int | None = None,
    ) -> None:
        """Export Evoked to external formats.

        Supported formats:

        - MFF (``.mff``, uses :func:`mne.export.export_evokeds_mff`)

        .. warning::
            Since we are exporting to external formats, there's no guarantee that all
            the info will be preserved in the external format. See Notes for details.

        Parameters
        ----------
        fname : str
            Name of the output file.
        fmt : 'auto' | 'mff'
            Format of the export. Defaults to ``'auto'``, which will infer the format
            from the filename extension. See supported formats above for more
            information.
        overwrite : bool
            If True (default False), overwrite the destination file if it
            exists.
        verbose : bool | str | int | None
            Control verbosity of the logging output. If ``None``, use the default
            verbosity level. See the :ref:`logging documentation <tut-logging>` and
            :func:`mne.verbose` for details. Should only be passed as a keyword
            argument.

        Notes
        -----
        .. versionadded:: 1.1

        Export to external format may not preserve all the information from the
        instance. To save in native MNE format (``.fif``) without information loss,
        use :meth:`mne.Evoked.save` instead.
        Export does not apply projector(s). Unapplied projector(s) will be lost.
        Consider applying projector(s) before exporting with
        :meth:`mne.Evoked.apply_proj`.
        """
        from .export import export_evokeds

        export_evokeds(fname, self, fmt, overwrite=overwrite, verbose=verbose)

    def __repr__(self):  # noqa: D105
        max_comment_length = 1000
        if len(self.comment) > max_comment_length:
            comment = self.comment[:max_comment_length]
            comment += "..."
        else:
            comment = self.comment
        s = f"'{comment}' ({self.kind}, N={self.nave})"
        s += f", {self.times[0]:0.5g} – {self.times[-1]:0.5g} s"
        s += ", baseline "
        if self.baseline is None:
            s += "off"
        else:
            s += f"{self.baseline[0]:g} – {self.baseline[1]:g} s"
            if self.baseline != _check_baseline(
                self.baseline,
                times=self.times,
                sfreq=self.info["sfreq"],
                on_baseline_outside_data="adjust",
            ):
                s += " (baseline period was cropped after baseline correction)"
        s += f", {self.data.shape[0]} ch"
        s += f", ~{sizeof_fmt(self._size)}"
        return f"<Evoked | {s}>"

    @repr_html
    def _repr_html_(self):
        t = _get_html_template("repr", "evoked.html.jinja")
        fname = self.filename
        t = t.render(
            inst=self,
            filenames=[Path(fname).name] if fname is not None else None,
        )
        return t

    @property
    def ch_names(self) -> list[str]:
        """Channel names."""
        return self.info["ch_names"]

    @copy_function_doc_to_method_doc_static("func:mne.viz.plot_evoked")
    def plot(
        self,
        picks: str | np.ndarray | slice | None = None,
        exclude: list[str] | Literal["bads"] = "bads",
        unit: bool = True,
        show: bool = True,
        ylim: dict | None = None,
        xlim: Literal["tight"] | tuple | None = "tight",
        proj: bool | Literal["interactive", "reconstruct"] = False,
        hline: list[float] | None = None,
        units: dict | None = None,
        scalings: dict | None = None,
        titles: dict | None = None,
        axes: "Axes | list | None" = None,
        gfp: bool | Literal["only"] = False,
        window_title: str | None = None,
        spatial_colors: bool | Literal["auto"] = "auto",
        zorder: str | Callable = "unsorted",
        selectable: bool = True,
        noise_cov: "Covariance | str | None" = None,
        time_unit: str = "s",
        sphere: "float | Annotated[Sequence[float], 4] | np.ndarray[tuple[Literal[4]], np.dtype[np.floating]] | ConductorModel | Literal['auto', 'cardinal', 'eeg', 'extra', 'hpi', 'eeglab'] | list[Literal['cardinal', 'eeg', 'extra', 'hpi']] | None" = None,  # noqa E501
        *,
        highlight: np.ndarray | None = None,
        verbose: bool | str | int | None = None,
    ) -> "Figure":
        """Plot evoked data using butterfly plots.

        Left click to a line shows the channel name. Selecting an area by clicking
        and holding left mouse button plots a topographic map of the painted area.

        .. note:: If bad channels are not excluded they are shown in red.

        Parameters
        ----------
        picks : str | array-like | slice | None
            Channels to include. Slices and lists of integers will be interpreted as
            channel indices. In lists, channel *type* strings (e.g., ``['meg',
            'eeg']``) will pick channels of those types, channel *name* strings (e.g.,
            ``['MEG0111', 'MEG2623']`` will pick the given channels. Can also be the
            string values ``'all'`` to pick all channels, or ``'data'`` to pick
            :term:`data channels`. None (default) will pick all channels. Bad channels
            are included by default. Note that channels in ``info['bads']`` *will be
            included* if their names or indices are explicitly provided.
        exclude : list of str | ``'bads'``
            Channels names to exclude from being shown. If ``'bads'``, the
            bad channels are excluded.
        unit : bool
            Scale plot with channel (SI) unit.
        show : bool
            Show figure if True.
        ylim : dict | None
            Y-axis limits for plots (after scaling has been applied). :class:`dict` keys
            should match channel types; valid keys are for instance ``eeg``, ``mag``,
            ``grad``, ``misc``, ``csd``, .. (example: ``ylim=dict(eeg=[-20, 20])``). If
            ``None``, the y-axis limits will be set automatically by matplotlib.
            Defaults to ``None``.
        xlim : ``'tight'`` | tuple | None
            Limits for the X-axis of the plots.
        proj : bool | 'interactive' | 'reconstruct'
            If true SSP projections are applied before display. If ``'interactive'``,
            a check box for reversible selection of SSP projection vectors will
            be shown. If ``'reconstruct'``, projection vectors will be applied and then
            M/EEG data will be reconstructed via field mapping to reduce the signal
            bias caused by projection.

            .. versionchanged:: 0.21
               Support for 'reconstruct' was added.
        hline : list of float | None
            The values at which to show an horizontal line.
        units : dict | None
            The units of the channel types used for axes labels. If None,
            defaults to ``dict(eeg='µV', grad='fT/cm', mag='fT')``.
        scalings : dict | None
            The scalings of the channel types to be applied for plotting. If None,
            defaults to ``dict(eeg=1e6, grad=1e13, mag=1e15)``.
        titles : dict | None
            The titles associated with the channels. If None, defaults to
            ``dict(eeg='EEG', grad='Gradiometers', mag='Magnetometers')``.
        axes : instance of Axes | list | None
            The axes to plot to. If list, the list must be a list of Axes of
            the same length as the number of channel types. If instance of
            Axes, there must be only one channel type plotted.
        gfp : bool | ``'only'``
            Plot the global field power (GFP) or the root mean square (RMS) of the
            data. For MEG data, this will plot the RMS. For EEG, it plots GFP,
            i.e. the standard deviation of the signal across channels. The GFP is
            equivalent to the RMS of an average-referenced signal.

            - ``True``
                Plot GFP or RMS (for EEG and MEG, respectively) and traces for all
                channels.
            - ``'only'``
                Plot GFP or RMS (for EEG and MEG, respectively), and omit the
                traces for individual channels.

            The color of the GFP/RMS trace will be green if
            ``spatial_colors=False``, and black otherwise.

            .. versionchanged:: 0.23
               Plot GFP for EEG instead of RMS. Label RMS traces correctly as such.
        window_title : str | None
            The title to put at the top of the figure.
        spatial_colors : bool | 'auto'
            If True, the lines are color coded by mapping physical sensor
            coordinates into color values. Spatially similar channels will have
            similar colors. Bad channels will be dotted. If False, the good
            channels are plotted black and bad channels red. If ``'auto'``, uses
            True if channel locations are present, and False if channel locations
            are missing or if the data contains only a single channel. Defaults to
            ``'auto'``.
        zorder : str | callable
            Which channels to put in the front or back. Only matters if
            ``spatial_colors`` is used.
            If str, must be ``std`` or ``unsorted`` (defaults to ``unsorted``). If
            ``std``, data with the lowest standard deviation (weakest effects) will
            be put in front so that they are not obscured by those with stronger
            effects. If ``unsorted``, channels are z-sorted as in the evoked
            instance.
            If callable, must take one argument: a numpy array of the same
            dimensionality as the evoked raw data; and return a list of
            unique integers corresponding to the number of channels.

            .. versionadded:: 0.13.0

        selectable : bool
            Whether to use interactive features. If True (default), it is possible
            to paint an area to draw topomaps. When False, the interactive features
            are disabled. Disabling interactive features reduces memory consumption
            and is useful when using ``axes`` parameter to draw multiaxes figures.

            .. versionadded:: 0.13.0

        noise_cov : instance of Covariance | str | None
            Noise covariance used to whiten the data while plotting.
            Whitened data channel names are shown in italic.
            Can be a string to load a covariance from disk.
            See also :meth:`mne.Evoked.plot_white` for additional inspection
            of noise covariance properties when whitening evoked data.
            For data processed with SSS, the effective dependence between
            magnetometers and gradiometers may introduce differences in scaling,
            consider using :meth:`mne.Evoked.plot_white`.

            .. versionadded:: 0.16.0
        time_unit : str
            The units for the time axis, can be "s" (default) or "ms".

            .. versionadded:: 0.16
        sphere : float | array-like of float | instance of ConductorModel | {"auto", "cardinal", "eeg", "extra", "hpi", "eeglab"} | list of str | None
            The sphere parameters to use for the head outline.
            Can be array-like of shape (4,) to give the X/Y/Z origin and radius in
            meters, or a single float to give just the radius (origin assumed 0, 0, 0).
            Can also be an instance of a spherical :class:`~mne.bem.ConductorModel` to
            use the origin and radius from that object.
            Can also be a ``str``, in which case:

            - ``'auto'``: the sphere is fit to external digitization points first, and
              to external + EEG digitization points if the former fails.

            - ``'eeglab'``: the head circle is defined by EEG electrodes ``'Fpz'``,
              ``'Oz'``, ``'T7'``, and ``'T8'`` (if ``'Fpz'`` is not present, it will be
              approximated from the coordinates of ``'Oz'``).

              - ``'extra'``: the sphere is fit to external digitization points.

              - ``'eeg'``: the sphere is fit to EEG digitization points.

              - ``'cardinal'``: the sphere is fit to cardinal digitization points.

              - ``'hpi'``: the sphere is fit to HPI coil digitization points.

            Can also be a list of ``str``, in which case the sphere is fit to the
            specified digitization points, which can be any combination of ``'extra'``,
            ``'eeg'``, ``'cardinal'``, and ``'hpi'``, as specified above.
            ``None`` (the default) will look for an existing head outline in the
            ``.info`` dictionary and use that. If no outline is present, it is
            equivalent to ``'auto'`` when enough extra digitization points are
            available, and ``(0, 0, 0, 0.095)`` otherwise.

            .. versionadded:: 0.20
            .. versionchanged:: 1.1 Added ``'eeglab'`` option.
            .. versionchanged:: 1.11 Added ``'extra'``, ``'eeg'``, ``'cardinal'``,
               ``'hpi'`` and list of ``str`` options.
        highlight : array-like of float, shape(2,) | array-like of float, shape (n, 2) | None
            Segments of the data to highlight by means of a light-yellow
            background color. Can be used to put visual emphasis on certain
            time periods. The time periods must be specified as ``array-like``
            objects in the form of ``(t_start, t_end)`` in the unit given by the
            ``time_unit`` parameter.
            Multiple time periods can be specified by passing an ``array-like``
            object of individual time periods (e.g., for 3 time periods, the shape
            of the passed object would be ``(3, 2)``. If ``None``, no highlighting
            is applied.

            .. versionadded:: 1.1
        verbose : bool | str | int | None
            Control verbosity of the logging output. If ``None``, use the default
            verbosity level. See the :ref:`logging documentation <tut-logging>` and
            :func:`mne.verbose` for details. Should only be passed as a keyword
            argument.

        Returns
        -------
        fig : instance of matplotlib.figure.Figure
            Figure containing the butterfly plots.

        See Also
        --------
        mne.viz.plot_evoked_white

        Notes
        -----
        The figure will publish and subscribe to the following UI events:

        * :class:`~mne.viz.ui_events.TimeChange`

        .. versionadded:: 1.13.0
        """  # noqa: E501
        from .viz import plot_evoked

        return plot_evoked(
            self,
            picks=picks,
            exclude=exclude,
            unit=unit,
            show=show,
            ylim=ylim,
            proj=proj,
            xlim=xlim,
            hline=hline,
            units=units,
            scalings=scalings,
            titles=titles,
            axes=axes,
            gfp=gfp,
            window_title=window_title,
            spatial_colors=spatial_colors,
            zorder=zorder,
            selectable=selectable,
            noise_cov=noise_cov,
            time_unit=time_unit,
            sphere=sphere,
            highlight=highlight,
            verbose=verbose,
        )

    @copy_function_doc_to_method_doc_static("func:mne.viz.plot_evoked_image")
    def plot_image(
        self,
        picks: str | np.ndarray | slice | None = None,
        exclude: list[str] | Literal["bads"] = "bads",
        unit: bool = True,
        show: bool = True,
        clim: dict | None = None,
        xlim: Literal["tight"] | tuple | None = "tight",
        proj: bool | Literal["interactive"] = False,
        units: dict | None = None,
        scalings: dict | None = None,
        titles: dict | None = None,
        axes: "Axes | list | dict | None" = None,
        cmap: "str | Colormap | tuple" = "RdBu_r",
        colorbar: bool = True,
        mask: np.ndarray | None = None,
        mask_style: Literal["both", "contour", "mask"] | None = None,
        mask_cmap: "str | Colormap | tuple" = "Greys",
        mask_alpha: float = 0.25,
        time_unit: str = "s",
        show_names: bool | Literal["auto", "all"] | None = None,
        group_by: dict | None = None,
        sphere: "float | Annotated[Sequence[float], 4] | np.ndarray[tuple[Literal[4]], np.dtype[np.floating]] | ConductorModel | Literal['auto', 'cardinal', 'eeg', 'extra', 'hpi', 'eeglab'] | list[Literal['cardinal', 'eeg', 'extra', 'hpi']] | None" = None,  # noqa E501
    ) -> "Figure":
        """Plot evoked data as images.

        Parameters
        ----------
        picks : str | array-like | slice | None
            Channels to include. Slices and lists of integers will be interpreted as
            channel indices. In lists, channel *type* strings (e.g., ``['meg',
            'eeg']``) will pick channels of those types, channel *name* strings (e.g.,
            ``['MEG0111', 'MEG2623']`` will pick the given channels. Can also be the
            string values ``'all'`` to pick all channels, or ``'data'`` to pick
            :term:`data channels`. None (default) will pick all channels. Bad channels
            are included by default. Note that channels in ``info['bads']`` *will be
            included* if their names or indices are explicitly provided.
            This parameter can also be used to set the order the channels
            are shown in, as the channel image is sorted by the order of picks.
        exclude : list of str | 'bads'
            Channels names to exclude from being shown. If 'bads', the
            bad channels are excluded.
        unit : bool
            Scale plot with channel (SI) unit.
        show : bool
            Show figure if True.
        clim : dict | None
            Color limits for plots (after scaling has been applied). e.g.
            ``clim = dict(eeg=[-20, 20])``.
            Valid keys are eeg, mag, grad, misc. If None, the clim parameter
            for each channel equals the pyplot default.
        xlim : 'tight' | tuple | None
            X limits for plots.
        proj : bool | 'interactive'
            If true SSP projections are applied before display. If 'interactive',
            a check box for reversible selection of SSP projection vectors will
            be shown.
        units : dict | None
            The units of the channel types used for axes labels. If None,
            defaults to ``dict(eeg='µV', grad='fT/cm', mag='fT')``.
        scalings : dict | None
            The scalings of the channel types to be applied for plotting. If None,`
            defaults to ``dict(eeg=1e6, grad=1e13, mag=1e15)``.
        titles : dict | None
            The titles associated with the channels. If None, defaults to
            ``dict(eeg='EEG', grad='Gradiometers', mag='Magnetometers')``.
        axes : instance of Axes | list | dict | None
            The axes to plot to. If list, the list must be a list of Axes of
            the same length as the number of channel types. If instance of
            Axes, there must be only one channel type plotted.
            If ``group_by`` is a dict, this cannot be a list, but it can be a dict
            of lists of axes, with the keys matching those of ``group_by``. In that
            case, the provided axes will be used for the corresponding groups.
            Defaults to ``None``.
        cmap : matplotlib colormap | (colormap, bool) | 'interactive'
            Colormap. If tuple, the first value indicates the colormap to use and
            the second value is a boolean defining interactivity. In interactive
            mode the colors are adjustable by clicking and dragging the colorbar
            with left and right mouse button. Left mouse button moves the scale up
            and down and right mouse button adjusts the range. Hitting space bar
            resets the scale. Up and down arrows can be used to change the
            colormap. If 'interactive', translates to ``('RdBu_r', True)``.
            Defaults to ``'RdBu_r'``.
        colorbar : bool
            If True, plot a colorbar. Defaults to True.

            .. versionadded:: 0.16
        mask : ndarray | None
            An array of booleans of the same shape as the data. Entries of the
            data that correspond to ``False`` in the mask are masked (see
            ``do_mask`` below). Useful for, e.g., masking for statistical
            significance.

            .. versionadded:: 0.16
        mask_style : None | 'both' | 'contour' | 'mask'
            If ``mask`` is not None: if 'contour', a contour line is drawn around
            the masked areas (``True`` in ``mask``). If 'mask', entries not
            ``True`` in ``mask`` are shown transparently. If 'both', both a contour
            and transparency are used.
            If ``None``, defaults to 'both' if ``mask`` is not None, and is ignored
            otherwise.

             .. versionadded:: 0.16
        mask_cmap : matplotlib colormap | (colormap, bool) | 'interactive'
            The colormap chosen for masked parts of the image (see below), if
            ``mask`` is not ``None``. If None, ``cmap`` is reused. Defaults to
            ``Greys``. Not interactive. Otherwise, as ``cmap``.
        mask_alpha : float
            A float between 0 and 1. If ``mask`` is not None, this sets the
            alpha level (degree of transparency) for the masked-out segments.
            I.e., if 0, masked-out segments are not visible at all.
            Defaults to .25.

            .. versionadded:: 0.16
        time_unit : str
            The units for the time axis, can be "ms" or "s" (default).

            .. versionadded:: 0.16
        show_names : bool | 'auto' | 'all'
            Determines if channel names should be plotted on the y axis. If False,
            no names are shown. If True, ticks are set automatically by matplotlib
            and the corresponding channel names are shown. If "all", all channel
            names are shown. If "auto", is set to False if ``picks`` is ``None``,
            to ``True`` if ``picks`` contains 25 or more entries, or to "all"
            if ``picks`` contains fewer than 25 entries.
        group_by : None | dict
            If a dict, the values must be picks, and ``axes`` must also be a dict
            with matching keys, or None. If ``axes`` is None, one figure and one
            axis will be created for each entry in ``group_by``.Then, for each
            entry, the picked channels will be plotted to the corresponding axis.
            If ``titles`` are None, keys will become plot titles. This is useful
            for e.g. ROIs. Each entry must contain only one channel type.
            For example::

                group_by=dict(Left_ROI=[1, 2, 3, 4], Right_ROI=[5, 6, 7, 8])

            If None, all picked channels are plotted to the same axis.
        sphere : float | array-like of float | instance of ConductorModel | {"auto", "cardinal", "eeg", "extra", "hpi", "eeglab"} | list of str | None
            The sphere parameters to use for the head outline.
            Can be array-like of shape (4,) to give the X/Y/Z origin and radius in
            meters, or a single float to give just the radius (origin assumed 0, 0, 0).
            Can also be an instance of a spherical :class:`~mne.bem.ConductorModel` to
            use the origin and radius from that object.
            Can also be a ``str``, in which case:

            - ``'auto'``: the sphere is fit to external digitization points first, and
              to external + EEG digitization points if the former fails.

            - ``'eeglab'``: the head circle is defined by EEG electrodes ``'Fpz'``,
              ``'Oz'``, ``'T7'``, and ``'T8'`` (if ``'Fpz'`` is not present, it will be
              approximated from the coordinates of ``'Oz'``).

              - ``'extra'``: the sphere is fit to external digitization points.

              - ``'eeg'``: the sphere is fit to EEG digitization points.

              - ``'cardinal'``: the sphere is fit to cardinal digitization points.

              - ``'hpi'``: the sphere is fit to HPI coil digitization points.

            Can also be a list of ``str``, in which case the sphere is fit to the
            specified digitization points, which can be any combination of ``'extra'``,
            ``'eeg'``, ``'cardinal'``, and ``'hpi'``, as specified above.
            ``None`` (the default) will look for an existing head outline in the
            ``.info`` dictionary and use that. If no outline is present, it is
            equivalent to ``'auto'`` when enough extra digitization points are
            available, and ``(0, 0, 0, 0.095)`` otherwise.

            .. versionadded:: 0.20
            .. versionchanged:: 1.1 Added ``'eeglab'`` option.
            .. versionchanged:: 1.11 Added ``'extra'``, ``'eeg'``, ``'cardinal'``,
               ``'hpi'`` and list of ``str`` options.

        Returns
        -------
        fig : instance of matplotlib.figure.Figure
            Figure containing the images.
        """  # noqa: E501
        from .viz import plot_evoked_image

        return plot_evoked_image(
            self,
            picks=picks,
            exclude=exclude,
            unit=unit,
            show=show,
            clim=clim,
            xlim=xlim,
            proj=proj,
            units=units,
            scalings=scalings,
            titles=titles,
            axes=axes,
            cmap=cmap,
            colorbar=colorbar,
            mask=mask,
            mask_style=mask_style,
            mask_cmap=mask_cmap,
            mask_alpha=mask_alpha,
            time_unit=time_unit,
            show_names=show_names,
            group_by=group_by,
            sphere=sphere,
        )

    @copy_function_doc_to_method_doc_static("func:mne.viz.plot_evoked_topo")
    def plot_topo(
        self,
        layout: Layout | None = None,
        layout_scale: float = 0.945,
        color: list[Color] | Color | None = None,
        border: str = "none",
        ylim: dict | None = None,
        scalings: dict | None = None,
        title: str | None = None,
        proj: bool | Literal["interactive"] = False,
        vline: list[float] | tuple[float, ...] | float | None = (0.0,),
        fig_background: np.ndarray | None = None,
        merge_grads: bool = False,
        legend: bool | int | str | tuple = True,
        axes: "Axes | None" = None,
        background_color: Color = "w",
        noise_cov: "Covariance | str | None" = None,
        exclude: list[str] | Literal["bads"] = "bads",
        select: bool = False,
        show: bool = True,
    ) -> "Figure":
        """Plot 2D topography of evoked responses.

        Clicking on the plot of an individual sensor opens a new figure showing
        the evoked response for the selected sensor.

        Parameters
        ----------
        layout : instance of Layout | None
            Layout instance specifying sensor positions (does not need to
            be specified for Neuromag data). If possible, the correct layout is
            inferred from the data.
        layout_scale : float
            Scaling factor for adjusting the relative size of the layout
            on the canvas.
        color : list of color | color | None
            Everything matplotlib accepts to specify colors. If not list-like,
            the color specified will be repeated. If None, colors are
            automatically drawn.
        border : str
            Matplotlib borders style to be used for each sensor plot.
        ylim : dict | None
            Y-axis limits for plots (after scaling has been applied). :class:`dict` keys
            should match channel types; valid keys are for instance ``eeg``, ``mag``,
            ``grad``, ``misc``, ``csd``, .. (example: ``ylim=dict(eeg=[-20, 20])``). If
            ``None``, the y-axis limits will be set automatically by matplotlib.
            Defaults to ``None``.
        scalings : dict | None
            The scalings of the channel types to be applied for plotting. If None,`
            defaults to ``dict(eeg=1e6, grad=1e13, mag=1e15)``.
        title : str
            Title of the figure.
        proj : bool | ``'interactive'``
            If true SSP projections are applied before display. If ``'interactive'``,
            a check box for reversible selection of SSP projection vectors will
            be shown.
        vline : list of float | float | None
            The values at which to show a vertical line.
        fig_background : None | ndarray
            A background image for the figure. This must work with a call to
            ``plt.imshow``. Defaults to None.
        merge_grads : bool
            Whether to use RMS value of gradiometer pairs. Only works for Neuromag
            data. Defaults to False.
        legend : bool | int | str | tuple
            If True, create a legend based on evoked.comment. If False, disable the
            legend. Otherwise, the legend is created and the parameter value is
            passed as the location parameter to the matplotlib legend call. It can
            be an integer (e.g. 0 corresponds to upper right corner of the plot),
            a string (e.g. ``'upper right'``), or a tuple (x, y coordinates of the
            lower left corner of the legend in the axes coordinate system).
            See matplotlib documentation for more details.
        axes : instance of matplotlib Axes | None
            Axes to plot into. If None, axes will be created.
        background_color : color
            Background color. Typically ``'k'`` (black) or ``'w'`` (white; default).

            .. versionadded:: 0.15.0
        noise_cov : instance of Covariance | str | None
            Noise covariance used to whiten the data while plotting.
            Whitened data channel names are shown in italic.
            Can be a string to load a covariance from disk.

            .. versionadded:: 0.16.0
        exclude : list of str | ``'bads'``
            Channels names to exclude from the plot. If ``'bads'``, the
            bad channels are excluded. By default, exclude is set to ``'bads'``.
        select : bool
            Whether to enable the lasso-selection tool to enable the user to select
            channels. The selected channels will be available in
            ``fig.lasso.selection``.

            .. versionadded:: 1.10.0
        exclude : list of str | ``'bads'``
            Channels names to exclude from the plot. If ``'bads'``, the
            bad channels are excluded. By default, exclude is set to ``'bads'``.
        show : bool
            Show figure if True.

        Returns
        -------
        fig : instance of matplotlib.figure.Figure
            Images of evoked responses at sensor locations.

        Notes
        -----
        The figure will publish and subscribe to the following UI events:

        * :class:`~mne.viz.ui_events.TimeChange`

        .. versionadded:: 1.13.0
        """
        from .viz import plot_evoked_topo

        return plot_evoked_topo(
            self,
            layout=layout,
            layout_scale=layout_scale,
            color=color,
            border=border,
            ylim=ylim,
            scalings=scalings,
            title=title,
            proj=proj,
            vline=vline,
            fig_background=fig_background,
            merge_grads=merge_grads,
            legend=legend,
            axes=axes,
            background_color=background_color,
            noise_cov=noise_cov,
            exclude=exclude,
            select=select,
            show=show,
        )

    @copy_function_doc_to_method_doc_static("func:mne.viz.plot_evoked_topomap")
    def plot_topomap(
        self,
        times: float | np.ndarray | Literal["auto", "peaks", "interactive"] = "auto",
        *,
        average: float | np.ndarray | None = None,
        ch_type: Literal["mag", "grad", "planar1", "planar2", "eeg"] | None = None,
        scalings: dict | float | None = None,
        proj: bool | Literal["interactive", "reconstruct"] = False,
        sensors: bool | str = True,
        show_names: bool | Callable = False,
        mask: np.ndarray | None = None,
        mask_params: dict | None = None,
        mask_label_params: dict | None = None,
        contours: int | np.ndarray = 6,
        outlines: Literal["head"] | dict | None = "head",
        sphere: "float | Annotated[Sequence[float], 4] | np.ndarray[tuple[Literal[4]], np.dtype[np.floating]] | ConductorModel | Literal['auto', 'cardinal', 'eeg', 'extra', 'hpi', 'eeglab'] | list[Literal['cardinal', 'eeg', 'extra', 'hpi']] | None" = None,  # noqa E501
        image_interp: str = _INTERPOLATION_DEFAULT,
        extrapolate: str = _EXTRAPOLATE_DEFAULT,
        border: float | Literal["mean"] = _BORDER_DEFAULT,
        res: int = 64,
        size: float = 1,
        cmap: "str | Colormap | tuple | Literal['interactive'] | None" = None,
        vlim: tuple | Literal["joint"] = (None, None),
        cnorm: "Normalize | None" = None,
        colorbar: bool = True,
        cbar_fmt: str = "%3.1f",
        units: dict | str | None = None,
        axes: "Axes | list[Axes] | None" = None,
        time_unit: str = "s",
        time_format: str | None = None,
        nrows: int | Literal["auto"] = 1,
        ncols: int | Literal["auto"] = "auto",
        show: bool = True,
    ) -> "Figure":
        """Plot topographic maps of specific time points of evoked data.

        Parameters
        ----------
        times : float | array of float | "auto" | "peaks" | "interactive"
            The time point(s) to plot. If "auto", the number of ``axes`` determines
            the amount of time point(s). If ``axes`` is also None, at most 10
            topographies will be shown with a regular time spacing between the
            first and last time instant. If "peaks", finds time points
            automatically by checking for local maxima in global field power. If
            "interactive", the time can be set interactively at run-time by using a
            slider.
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
        ch_type : 'mag' | 'grad' | 'planar1' | 'planar2' | 'eeg' | None
            The channel type to plot. For ``'grad'``, the gradiometers are
            collected in pairs and the RMS for each pair is plotted. If ``None``
            the first available channel type from order
            shown above is used. Defaults to ``None``.
        scalings : dict | float | None
            The scalings of the channel types to be applied for plotting.
            If None, defaults to ``dict(eeg=1e6, grad=1e13, mag=1e15)``.
        proj : bool | 'interactive' | 'reconstruct'
            If true SSP projections are applied before display. If ``'interactive'``,
            a check box for reversible selection of SSP projection vectors will
            be shown. If ``'reconstruct'``, projection vectors will be applied and then
            M/EEG data will be reconstructed via field mapping to reduce the signal
            bias caused by projection.

            .. versionchanged:: 0.21
               Support for 'reconstruct' was added.
        sensors : bool | str
            Whether to add markers for sensor locations. If :class:`str`, should be a
            valid matplotlib format string (e.g., ``'r+'`` for red plusses, see the
            Notes section of :meth:`~matplotlib.axes.Axes.plot`). If ``True`` (the
            default), black circles will be used.
        show_names : bool | callable
            If ``True``, show channel names next to each sensor marker. If callable,
            channel names will be formatted using the callable; e.g., to
            delete the prefix 'MEG ' from all channel names, pass the function
            ``lambda x: x.replace('MEG ', '')``. If ``mask`` is not ``None``, only
            non-masked sensor names will be shown.
        mask : ndarray of bool, shape (n_channels, n_times) | None
            Array indicating channel-time combinations to highlight with a distinct
            plotting style (useful for, e.g. marking which channels at which times a
            statistical test of the data reaches significance).
            Array elements set to ``True`` will be plotted
            with the parameters given in ``mask_params``. Defaults to ``None``,
            equivalent to an array of all ``False`` elements.
        mask_params : dict | None
            Additional plotting parameters for plotting significant sensors.
            Default (None) equals::

                dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                        linewidth=0, markersize=4)
        mask_label_params : dict | None
            Additional plotting parameters for significant sensor labels.
            Default (None) equals::

                dict(fontsize='medium', fontweight='bold')

            .. versionadded:: 1.13
        contours : int | array-like
            The number of contour lines to draw. If ``0``, no contours will be drawn.
            If a positive integer, that number of contour levels are chosen using the
            matplotlib tick locator (may sometimes be inaccurate, use array for
            accuracy). If array-like, the array values are used as the contour levels.
            The values should be in µV for EEG, fT for magnetometers and fT/m for
            gradiometers. Default is ``6``.
        outlines : 'head' | dict | None
            The outlines to be drawn. If 'head', the default head scheme will be
            drawn. If dict, each key refers to a tuple of x and y positions, the values
            in 'mask_pos' will serve as image mask.
            Alternatively, a matplotlib patch object can be passed for advanced
            masking options, either directly or as a function that returns patches
            (required for multi-axis plots). If None, nothing will be drawn.
            Defaults to 'head'.
        sphere : float | array-like of float | instance of ConductorModel | {"auto", "cardinal", "eeg", "extra", "hpi", "eeglab"} | list of str | None
            The sphere parameters to use for the head outline.
            Can be array-like of shape (4,) to give the X/Y/Z origin and radius in
            meters, or a single float to give just the radius (origin assumed 0, 0, 0).
            Can also be an instance of a spherical :class:`~mne.bem.ConductorModel` to
            use the origin and radius from that object.
            Can also be a ``str``, in which case:

            - ``'auto'``: the sphere is fit to external digitization points first, and
              to external + EEG digitization points if the former fails.

            - ``'eeglab'``: the head circle is defined by EEG electrodes ``'Fpz'``,
              ``'Oz'``, ``'T7'``, and ``'T8'`` (if ``'Fpz'`` is not present, it will be
              approximated from the coordinates of ``'Oz'``).

              - ``'extra'``: the sphere is fit to external digitization points.

              - ``'eeg'``: the sphere is fit to EEG digitization points.

              - ``'cardinal'``: the sphere is fit to cardinal digitization points.

              - ``'hpi'``: the sphere is fit to HPI coil digitization points.

            Can also be a list of ``str``, in which case the sphere is fit to the
            specified digitization points, which can be any combination of ``'extra'``,
            ``'eeg'``, ``'cardinal'``, and ``'hpi'``, as specified above.
            ``None`` (the default) will look for an existing head outline in the
            ``.info`` dictionary and use that. If no outline is present, it is
            equivalent to ``'auto'`` when enough extra digitization points are
            available, and ``(0, 0, 0, 0.095)`` otherwise.

            .. versionadded:: 0.20
            .. versionchanged:: 1.1 Added ``'eeglab'`` option.
            .. versionchanged:: 1.11 Added ``'extra'``, ``'eeg'``, ``'cardinal'``,
               ``'hpi'`` and list of ``str`` options.
        image_interp : str
            The image interpolation to be used. Options are ``'cubic'`` (default)
            to use :class:`scipy.interpolate.CloughTocher2DInterpolator`,
            ``'nearest'`` to use :class:`scipy.spatial.Voronoi` or
            ``'linear'`` to use :class:`scipy.interpolate.LinearNDInterpolator`.
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

            .. versionadded:: 0.18

            .. versionchanged:: 0.21

               - The default was changed to ``'local'`` for MEG sensors.
               - ``'local'`` was changed to use a convex hull mask
               - ``'head'`` was changed to extrapolate out to the clipping circle.
        border : float | 'mean'
            Value to extrapolate to on the topomap borders. If ``'mean'`` (default),
            then each extrapolated point has the average value of its neighbours.

            .. versionadded:: 0.20
        res : int
            The resolution of the topomap image (number of pixels along each side).
        size : float
            Side length of each subplot in inches.
        cmap : str | matplotlib.colors.Colormap | tuple | 'interactive' | None
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
        vlim : tuple of length 2 | "joint"
            Lower and upper bounds of the colormap, typically a numeric value in the
            same units as the data. Elements of the :class:`tuple` may also be
            callable functions which take in a :class:`NumPy array <numpy.ndarray>` and
            return a scalar.

            If both entries are ``None``, the bounds are set at
            ± the maximum absolute value
            of the data (yielding a colormap with midpoint at 0), or
            ``(0, max(abs(data)))`` if the (possibly baselined) data are all-positive.
            Providing ``None`` for just one entry will set the corresponding boundary
            at the min/max of the data. If ``vlim="joint"``, will compute the colormap
            limits jointly across all topomaps of the same channel type (instead of
            separately for each topomap), using the min/max of the data for that
            channel type. Defaults to ``(None, None)``.

            .. versionadded:: 1.2
        cnorm : matplotlib.colors.Normalize | None
            How to normalize the colormap. If ``None``, standard linear normalization
            is performed. If not ``None``, ``vmin`` and ``vmax`` will be ignored.
            See :ref:`Matplotlib docs <matplotlib:colormapnorms>`
            for more details on colormap normalization, and
            :ref:`the ERDs example<cnorm-example>` for an example of its use.

            .. versionadded:: 1.2
        colorbar : bool
            Plot a colorbar in the rightmost column of the figure.
        cbar_fmt : str
            Formatting string for colorbar tick labels. See :ref:`formatspec` for
            details.
        units : dict | str | None
            The units to use for the colorbar label. Ignored if ``colorbar=False``.
            If ``None`` and ``scalings=None`` the unit is automatically determined,
            otherwise the label will be "AU" indicating arbitrary units.
            Default is ``None``.
        axes : instance of Axes | list of Axes | None
            The axes to plot into. If ``None``, a new :class:`~matplotlib.figure.Figure`
            will be created with the correct number of axes. If
            :class:`~matplotlib.axes.Axes` are provided (either as a single instance or
            a :class:`list` of axes), the number of axes provided must
            match the number of ``times`` provided (unless ``times`` is
            ``None``). Default is ``None``.
        time_unit : str
            The units for the time axis, can be "ms" or "s" (default).

            .. versionadded:: 0.16
        time_format : str | None
            String format for topomap values. Defaults (None) to "%01d ms" if
            ``time_unit='ms'``, "%0.3f s" if ``time_unit='s'``, and
            "%g" otherwise. Can be an empty string to omit the time label.
        nrows, ncols : int | 'auto'
            The number of rows and columns of topographies to plot. If either ``nrows``
            or ``ncols`` is ``'auto'``, the necessary number will be inferred. Defaults
            to ``nrows=1, ncols='auto'``.
            Ignored when times == 'interactive'.

            .. versionadded:: 0.20
        show : bool
            Show the figure if ``True``. When shown, blocking follows
            :func:`matplotlib.pyplot.show`: the call blocks until the window is closed
            unless Matplotlib's interactive mode is on (enabled with
            :func:`matplotlib.pyplot.ion` or IPython's ``%%matplotlib`` magic command),
            in which case it returns immediately. Interactive mode is off by default, so
            a plain script or REPL blocks. Pass ``show=False`` to build several figures
            and display them together with a single :func:`matplotlib.pyplot.show` call.

        Returns
        -------
        fig : instance of matplotlib.figure.Figure
           The figure.

        Notes
        -----
        When existing ``axes`` are provided and ``colorbar=True``, note that the
        colorbar scale will only accurately reflect topomaps that are generated in
        the same call as the colorbar. Note also that the colorbar will not be
        resized automatically when ``axes`` are provided; use Matplotlib's
        :meth:`axes.set_position() <matplotlib.axes.Axes.set_position>` method or
        :ref:`gridspec <matplotlib:arranging_axes>` interface to adjust the colorbar
        size yourself.

        The defaults for ``contours`` and ``vlim`` are handled as follows:

        * When neither ``vlim`` nor a list of ``contours`` is passed, MNE sets
          ``vlim`` at ± the maximum absolute value of the data and then chooses
          contours within those bounds.

        * When ``vlim`` but not a list of ``contours`` is passed, MNE chooses
          contours to be within the ``vlim``.

        * When a list of ``contours`` but not ``vlim`` is passed, MNE chooses
          ``vlim`` to encompass the ``contours`` and the maximum absolute value of the
          data.

        * When both a list of ``contours`` and ``vlim`` are passed, MNE uses them
          as-is.

        When ``time=="interactive"``, the figure will publish and subscribe to the
        following UI events:

        * :class:`~mne.viz.ui_events.TimeChange` whenever a new time is selected.
        """  # noqa: E501
        from .viz import plot_evoked_topomap

        return plot_evoked_topomap(
            self,
            times=times,
            ch_type=ch_type,
            vlim=vlim,
            cmap=cmap,
            cnorm=cnorm,
            sensors=sensors,
            colorbar=colorbar,
            scalings=scalings,
            units=units,
            res=res,
            size=size,
            cbar_fmt=cbar_fmt,
            time_unit=time_unit,
            time_format=time_format,
            proj=proj,
            show=show,
            show_names=show_names,
            mask=mask,
            mask_params=mask_params,
            mask_label_params=mask_label_params,
            outlines=outlines,
            contours=contours,
            image_interp=image_interp,
            average=average,
            axes=axes,
            extrapolate=extrapolate,
            sphere=sphere,
            border=border,
            nrows=nrows,
            ncols=ncols,
        )

    @copy_function_doc_to_method_doc_static("func:mne.viz.plot_evoked_field")
    def plot_field(
        self,
        surf_maps: list,
        time: float | None = None,
        time_label: str | None = "t = %0.0f ms",
        n_jobs: int | None = None,
        fig: "Figure3D | Brain | None" = None,
        vmax: float | dict | None = None,
        n_contours: int = 21,
        *,
        show_density: bool = True,
        alpha: float | dict | None = None,
        interpolation: str | None = "nearest",
        interaction: Literal["trackball", "terrain"] = "terrain",
        time_viewer: bool | str = "auto",
        verbose: bool | str | int | None = None,
    ) -> "Figure3D | EvokedField":
        """Plot MEG/EEG fields on head surface and helmet in 3D.

        Parameters
        ----------
        surf_maps : list
            The surface mapping information obtained with make_field_map.
        time : float | None
            The time point at which the field map shall be displayed. If None,
            the average peak latency (across sensor types) is used.
        time_label : str | None
            How to print info about the time instant visualized.
        n_jobs : int | None
            The number of jobs to run in parallel. If ``-1``, it is set
            to the number of CPU cores. Requires the :mod:`joblib` package.
            ``None`` (default) is a marker for 'unset' that will be interpreted
            as ``n_jobs=1`` (sequential execution) unless the call is performed under
            a :class:`joblib:joblib.parallel_config` context manager that sets another
            value for ``n_jobs``.
        fig : Figure3D | mne.viz.Brain | None
            If None (default), a new figure will be created, otherwise it will
            plot into the given figure.

            .. versionadded:: 0.20
            .. versionadded:: 1.4
                ``fig`` can also be a ``Brain`` figure.
        vmax : float | dict | None
            Maximum intensity. Can be a dictionary with two entries ``"eeg"`` and ``"meg"``
            to specify separate values for EEG and MEG fields respectively. Can be
            ``None`` to use the maximum value of the data.

            .. versionadded:: 0.21
            .. versionadded:: 1.4
                ``vmax`` can be a dictionary to specify separate values for EEG and
                MEG fields.
        n_contours : int
            The number of contours.

            .. versionadded:: 0.21
        show_density : bool
            Whether to draw the field density as an overlay on top of the helmet/head
            surface. Defaults to ``True``.

            .. versionadded:: 1.6
        alpha : float | dict | None
            Opacity of the meshes (between 0 and 1). Can be a dictionary with two
            entries ``"eeg"`` and ``"meg"`` to specify separate values for EEG and
            MEG fields respectively. Can be ``None`` to use 1.0 when a single field
            map is shown, or ``dict(eeg=1.0, meg=0.5)`` when both field maps are shown.

            .. versionadded:: 1.4
        interpolation : str | None
            Interpolation method (:class:`scipy.interpolate.interp1d` parameter).
            Must be one of ``'linear'``, ``'nearest'``, ``'zero'``, ``'slinear'``,
            ``'quadratic'`` or ``'cubic'``.

            .. versionadded:: 1.6
        interaction : 'trackball' | 'terrain'
            How interactions with the scene via an input device (e.g., mouse or
            trackpad) modify the camera position. If ``'terrain'``, one axis is
            fixed, enabling "turntable-style" rotations. If ``'trackball'``,
            movement along all axes is possible, which provides more freedom of
            movement, but you may incidentally perform unintentional rotations along
            some axes.
            Defaults to ``'terrain'``.

            .. versionadded:: 1.1
        time_viewer : bool | str
            Display time viewer GUI. Can also be ``"auto"``, which will mean
            ``True`` if there is more than one time point and ``False`` otherwise.

            .. versionadded:: 1.6
        verbose : bool | str | int | None
            Control verbosity of the logging output. If ``None``, use the default
            verbosity level. See the :ref:`logging documentation <tut-logging>` and
            :func:`mne.verbose` for details. Should only be passed as a keyword
            argument.

        Returns
        -------
        fig : Figure3D | mne.viz.EvokedField
            Without the time viewer active, the figure is returned. With the time
            viewer active, an object is returned that can be used to control
            different aspects of the figure.
        """  # noqa: E501
        from .viz import plot_evoked_field

        return plot_evoked_field(
            self,
            surf_maps,
            time=time,
            time_label=time_label,
            n_jobs=n_jobs,
            fig=fig,
            vmax=vmax,
            n_contours=n_contours,
            show_density=show_density,
            alpha=alpha,
            interpolation=interpolation,
            interaction=interaction,
            time_viewer=time_viewer,
            verbose=verbose,
        )

    @copy_function_doc_to_method_doc_static("func:mne.viz.evoked.plot_evoked_white")
    def plot_white(
        self,
        noise_cov: "list | Covariance | Path | str",
        show: bool = True,
        rank: Literal["info", "full"] | dict | None = None,
        time_unit: str = "s",
        sphere: "float | Annotated[Sequence[float], 4] | np.ndarray[tuple[Literal[4]], np.dtype[np.floating]] | ConductorModel | Literal['auto', 'cardinal', 'eeg', 'extra', 'hpi', 'eeglab'] | list[Literal['cardinal', 'eeg', 'extra', 'hpi']] | None" = None,  # noqa E501
        axes: list | None = None,
        *,
        spatial_colors: bool | Literal["auto"] = "auto",
        verbose: bool | str | int | None = None,
    ) -> "Figure":
        """Plot whitened evoked response.

        Plots the whitened evoked response and the whitened GFP as described in
        :footcite:`EngemannGramfort2015`. This function is especially useful for
        investigating noise covariance properties to determine if data are
        properly whitened (e.g., achieving expected values in line with model
        assumptions, see Notes below).

        Parameters
        ----------
        noise_cov : list | instance of Covariance | path-like
            The noise covariance. Can be a string to load a covariance from disk.
        show : bool
            Show figure if True.
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

            The default is ``None``.
        time_unit : str
            The units for the time axis, can be "ms" or "s" (default).

            .. versionadded:: 0.16
        sphere : float | array-like of float | instance of ConductorModel | {"auto", "cardinal", "eeg", "extra", "hpi", "eeglab"} | list of str | None
            The sphere parameters to use for the head outline.
            Can be array-like of shape (4,) to give the X/Y/Z origin and radius in
            meters, or a single float to give just the radius (origin assumed 0, 0, 0).
            Can also be an instance of a spherical :class:`~mne.bem.ConductorModel` to
            use the origin and radius from that object.
            Can also be a ``str``, in which case:

            - ``'auto'``: the sphere is fit to external digitization points first, and
              to external + EEG digitization points if the former fails.

            - ``'eeglab'``: the head circle is defined by EEG electrodes ``'Fpz'``,
              ``'Oz'``, ``'T7'``, and ``'T8'`` (if ``'Fpz'`` is not present, it will be
              approximated from the coordinates of ``'Oz'``).

              - ``'extra'``: the sphere is fit to external digitization points.

              - ``'eeg'``: the sphere is fit to EEG digitization points.

              - ``'cardinal'``: the sphere is fit to cardinal digitization points.

              - ``'hpi'``: the sphere is fit to HPI coil digitization points.

            Can also be a list of ``str``, in which case the sphere is fit to the
            specified digitization points, which can be any combination of ``'extra'``,
            ``'eeg'``, ``'cardinal'``, and ``'hpi'``, as specified above.
            ``None`` (the default) will look for an existing head outline in the
            ``.info`` dictionary and use that. If no outline is present, it is
            equivalent to ``'auto'`` when enough extra digitization points are
            available, and ``(0, 0, 0, 0.095)`` otherwise.

            .. versionadded:: 0.20
            .. versionchanged:: 1.1 Added ``'eeglab'`` option.
            .. versionchanged:: 1.11 Added ``'extra'``, ``'eeg'``, ``'cardinal'``,
               ``'hpi'`` and list of ``str`` options.
        axes : list | None
            List of axes to plot into.

            .. versionadded:: 0.21.0
        spatial_colors : bool | 'auto'
            If True, the lines are color coded by mapping physical sensor
            coordinates into color values. Spatially similar channels will have
            similar colors. Bad channels will be dotted. If False, the good
            channels are plotted black and bad channels red. If ``'auto'``, uses
            True if channel locations are present, and False if channel locations
            are missing or if the data contains only a single channel. Defaults to
            ``'auto'``.

            .. versionadded:: 1.8.0
        verbose : bool | str | int | None
            Control verbosity of the logging output. If ``None``, use the default
            verbosity level. See the :ref:`logging documentation <tut-logging>` and
            :func:`mne.verbose` for details. Should only be passed as a keyword
            argument.

        Returns
        -------
        fig : instance of matplotlib.figure.Figure
            The figure object containing the plot.

        See Also
        --------
        mne.Evoked.plot

        Notes
        -----
        If baseline signals match the assumption of Gaussian white noise,
        values should be centered at 0, and be within 2 standard deviations
        (±1.96) for 95% of the time points. For the global field power (GFP),
        we expect it to fluctuate around a value of 1.

        If one single covariance object is passed, the GFP panel (bottom)
        will depict different sensor types. If multiple covariance objects are
        passed as a list, the left column will display the whitened evoked
        responses for each channel based on the whitener from the noise covariance
        that has the highest log-likelihood. The left column will depict the
        whitened GFPs based on each estimator separately for each sensor type.
        Instead of numbers of channels the GFP display shows the estimated rank.
        Note. The rank estimation will be printed by the logger
        (if ``verbose=True``) for each noise covariance estimator that is passed.

        References
        ----------
        .. [1] Engemann D. and Gramfort A. (2015) Automated model selection in
               covariance estimation and spatial whitening of MEG and EEG
               signals, vol. 108, 328-342, NeuroImage.
        """  # noqa: E501
        from .viz.evoked import plot_evoked_white

        return plot_evoked_white(
            self,
            noise_cov=noise_cov,
            rank=rank,
            show=show,
            time_unit=time_unit,
            sphere=sphere,
            axes=axes,
            spatial_colors=spatial_colors,
            verbose=verbose,
        )

    @copy_function_doc_to_method_doc_static("func:mne.viz.evoked.plot_evoked_joint")
    def plot_joint(
        self,
        times: float | np.ndarray | Literal["auto", "peaks"] = "peaks",
        title: str | None = "",
        picks: str | np.ndarray | slice | None = None,
        exclude: list[str] | Literal["bads"] = "bads",
        show: bool = True,
        ts_args: dict | None = None,
        topomap_args: dict | None = None,
    ) -> "Figure | list":
        """Plot evoked data as butterfly plot and add topomaps for time points.

        .. note:: Axes to plot in can be passed by the user through ``ts_args`` or
                  ``topomap_args``. In that case both ``ts_args`` and
                  ``topomap_args`` axes have to be used. Be aware that when the
                  axes are provided, their position may be slightly modified.

        Parameters
        ----------
        times : float | array of float | "auto" | "peaks"
            The time point(s) to plot. If ``"auto"``, 5 evenly spaced topographies
            between the first and last time instant will be shown. If ``"peaks"``,
            finds time points automatically by checking for 3 local maxima in
            Global Field Power. Defaults to ``"peaks"``.
        title : str | None
            The title. If ``None``, suppress printing channel type title. If an
            empty string, a default title is created. Defaults to ''. If custom
            axes are passed make sure to set ``title=None``, otherwise some of your
            axes may be removed during placement of the title axis.
        picks : str | array-like | slice | None
            Channels to include. Slices and lists of integers will be interpreted as
            channel indices. In lists, channel *type* strings (e.g., ``['meg',
            'eeg']``) will pick channels of those types, channel *name* strings (e.g.,
            ``['MEG0111', 'MEG2623']`` will pick the given channels. Can also be the
            string values ``'all'`` to pick all channels, or ``'data'`` to pick
            :term:`data channels`. None (default) will pick all channels. Bad channels
            are included by default. Note that channels in ``info['bads']`` *will be
            included* if their names or indices are explicitly provided.
        exclude : list of str | 'bads'
            Channels names to exclude from being shown. If ``'bads'``, the
            bad channels are excluded. Defaults to ``'bads'``.
        show : bool
            Show figure if ``True``. Defaults to ``True``.
        ts_args : None | dict
            A dict of ``kwargs`` that are forwarded to :meth:`mne.Evoked.plot` to
            style the butterfly plot. If they are not in this dict, the following
            defaults are passed: ``spatial_colors=True``, ``zorder='std'``.
            ``show`` and ``exclude`` are illegal.
            If ``None``, no customizable arguments will be passed.
            Defaults to ``None``.
        topomap_args : None | dict
            A dict of ``kwargs`` that are forwarded to
            :meth:`mne.Evoked.plot_topomap` to style the topomaps.
            If it is not in this dict, ``outlines='head'`` will be passed.
            ``show``, ``times``, ``colorbar`` are illegal.
            If ``None``, no customizable arguments will be passed.
            Defaults to ``None``.

        Returns
        -------
        fig : instance of matplotlib.figure.Figure | list
            The figure object containing the plot. If ``evoked`` has multiple
            channel types, a list of figures, one for each channel type, is
            returned.

        Notes
        -----
        .. versionadded:: 0.12.0
        """
        from .viz.evoked import plot_evoked_joint

        return plot_evoked_joint(
            self,
            times=times,
            title=title,
            picks=picks,
            exclude=exclude,
            show=show,
            ts_args=ts_args,
            topomap_args=topomap_args,
        )

    @verbose_static(
        "average_plot_evoked_topomap",
        "ch_type_topomap",
        "scalings_topomap",
        "proj_plot",
        "sensors_topomap",
        "show_names_topomap",
        "mask_evoked_topomap",
        "mask_params_topomap",
        "mask_label_params_topomap",
        "contours_topomap",
        "outlines_topomap",
        "sphere_topomap_auto",
        "image_interp_topomap",
        "extrapolate_topomap",
        "border_topomap",
        "res_topomap",
        "size_topomap",
        "cmap_topomap",
        "vlim_plot_topomap_psd",
        "cnorm",
        "colorbar_topomap",
        "cbar_fmt_topomap",
        "units_topomap_evoked",
    )
    def animate_topomap(
        self,
        *,
        times: np.ndarray | None = None,
        average: float | np.ndarray | None = None,
        ch_type: Literal["mag", "grad", "planar1", "planar2", "eeg"] | None = None,
        scalings: dict | float | None = None,
        proj: bool | Literal["interactive", "reconstruct"] = False,
        sensors: bool | str = True,
        show_names: bool | Callable = False,
        mask: np.ndarray | None = None,
        mask_params: dict | None = None,
        mask_label_params: dict | None = None,
        contours: int | np.ndarray = 6,
        outlines: Literal["head"] | dict | None = "head",
        sphere: "float | Annotated[Sequence[float], 4] | np.ndarray[tuple[Literal[4]], np.dtype[np.floating]] | ConductorModel | Literal['auto', 'cardinal', 'eeg', 'extra', 'hpi', 'eeglab'] | list[Literal['cardinal', 'eeg', 'extra', 'hpi']] | None" = None,  # noqa E501
        image_interp: str = _INTERPOLATION_DEFAULT,
        extrapolate: str = _EXTRAPOLATE_DEFAULT,
        border: float | Literal["mean"] = _BORDER_DEFAULT,
        res: int = 64,
        size: float = 1.0,
        cmap: "str | Colormap | tuple | Literal['interactive'] | None" = None,
        vlim: tuple | Literal["joint"] = (None, None),
        cnorm: "Normalize | None" = None,
        colorbar: bool = True,
        cbar_fmt: str = "%3.1f",
        units: dict | str | None = None,
        axes: "list[Axes] | None" = None,
        time_unit: str = "s",
        time_format: str | None = None,
        frame_rate: int | None = None,
        butterfly: bool = False,
        blit: bool = True,
        show: bool = True,
        verbose: bool | str | int | None = None,
    ) -> tuple["Figure", "FuncAnimation"]:
        """Make animation of evoked data as topomap timeseries.

        The animation can be paused/resumed with left mouse button.
        Left and right arrow keys can be used to move backward or forward
        in time.

        Parameters
        ----------
        times : array of float | None
            The time points to plot. If None (default), 10 evenly spaced samples are
            calculated over the evoked time series.
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
        ch_type : 'mag' | 'grad' | 'planar1' | 'planar2' | 'eeg' | None
            The channel type to plot. For ``'grad'``, the gradiometers are
            collected in pairs and the RMS for each pair is plotted. If ``None``
            the first available channel type from order
            shown above is used. Defaults to ``None``.
        scalings : dict | float | None
            The scalings of the channel types to be applied for plotting.
            If None, defaults to ``dict(eeg=1e6, grad=1e13, mag=1e15)``.
        proj : bool | 'interactive' | 'reconstruct'
            If true SSP projections are applied before display. If ``'interactive'``,
            a check box for reversible selection of SSP projection vectors will
            be shown. If ``'reconstruct'``, projection vectors will be applied and then
            M/EEG data will be reconstructed via field mapping to reduce the signal
            bias caused by projection.

            .. versionchanged:: 0.21
               Support for 'reconstruct' was added.
        sensors : bool | str
            Whether to add markers for sensor locations. If :class:`str`, should be a
            valid matplotlib format string (e.g., ``'r+'`` for red plusses, see the
            Notes section of :meth:`~matplotlib.axes.Axes.plot`). If ``True`` (the
            default), black circles will be used.
        show_names : bool | callable
            If ``True``, show channel names next to each sensor marker. If callable,
            channel names will be formatted using the callable; e.g., to
            delete the prefix 'MEG ' from all channel names, pass the function
            ``lambda x: x.replace('MEG ', '')``. If ``mask`` is not ``None``, only
            non-masked sensor names will be shown.
        mask : ndarray of bool, shape (n_channels, n_times) | None
            Array indicating channel-time combinations to highlight with a distinct
            plotting style (useful for, e.g. marking which channels at which times a
            statistical test of the data reaches significance).
            Array elements set to ``True`` will be plotted
            with the parameters given in ``mask_params``. Defaults to ``None``,
            equivalent to an array of all ``False`` elements.
        mask_params : dict | None
            Additional plotting parameters for plotting significant sensors.
            Default (None) equals::

                dict(marker='o', markerfacecolor='w', markeredgecolor='k',
                        linewidth=0, markersize=4)
        mask_label_params : dict | None
            Additional plotting parameters for significant sensor labels.
            Default (None) equals::

                dict(fontsize='medium', fontweight='bold')

            .. versionadded:: 1.13
        contours : int | array-like
            The number of contour lines to draw. If ``0``, no contours will be drawn.
            If a positive integer, that number of contour levels are chosen using the
            matplotlib tick locator (may sometimes be inaccurate, use array for
            accuracy). If array-like, the array values are used as the contour levels.
            The values should be in µV for EEG, fT for magnetometers and fT/m for
            gradiometers. Default is ``6``.
        outlines : 'head' | dict | None
            The outlines to be drawn. If 'head', the default head scheme will be
            drawn. If dict, each key refers to a tuple of x and y positions, the values
            in 'mask_pos' will serve as image mask.
            Alternatively, a matplotlib patch object can be passed for advanced
            masking options, either directly or as a function that returns patches
            (required for multi-axis plots). If None, nothing will be drawn.
            Defaults to 'head'.
        sphere : float | array-like of float | instance of ConductorModel | {"auto", "cardinal", "eeg", "extra", "hpi", "eeglab"} | list of str | None
            The sphere parameters to use for the head outline.
            Can be array-like of shape (4,) to give the X/Y/Z origin and radius in
            meters, or a single float to give just the radius (origin assumed 0, 0, 0).
            Can also be an instance of a spherical :class:`~mne.bem.ConductorModel` to
            use the origin and radius from that object.
            Can also be a ``str``, in which case:

            - ``'auto'``: the sphere is fit to external digitization points first, and
              to external + EEG digitization points if the former fails.

            - ``'eeglab'``: the head circle is defined by EEG electrodes ``'Fpz'``,
              ``'Oz'``, ``'T7'``, and ``'T8'`` (if ``'Fpz'`` is not present, it will be
              approximated from the coordinates of ``'Oz'``).

              - ``'extra'``: the sphere is fit to external digitization points.

              - ``'eeg'``: the sphere is fit to EEG digitization points.

              - ``'cardinal'``: the sphere is fit to cardinal digitization points.

              - ``'hpi'``: the sphere is fit to HPI coil digitization points.

            Can also be a list of ``str``, in which case the sphere is fit to the
            specified digitization points, which can be any combination of ``'extra'``,
            ``'eeg'``, ``'cardinal'``, and ``'hpi'``, as specified above.
            ``None`` (the default) will look for an existing head outline in the
            ``.info`` dictionary and use that. If no outline is present, it is
            equivalent to ``'auto'`` when enough extra digitization points are
            available, and ``(0, 0, 0, 0.095)`` otherwise.

            .. versionadded:: 0.20
            .. versionchanged:: 1.1 Added ``'eeglab'`` option.
            .. versionchanged:: 1.11 Added ``'extra'``, ``'eeg'``, ``'cardinal'``,
               ``'hpi'`` and list of ``str`` options.
        image_interp : str
            The image interpolation to be used. Options are ``'cubic'`` (default)
            to use :class:`scipy.interpolate.CloughTocher2DInterpolator`,
            ``'nearest'`` to use :class:`scipy.spatial.Voronoi` or
            ``'linear'`` to use :class:`scipy.interpolate.LinearNDInterpolator`.
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
        border : float | 'mean'
            Value to extrapolate to on the topomap borders. If ``'mean'`` (default),
            then each extrapolated point has the average value of its neighbours.
        res : int
            The resolution of the topomap image (number of pixels along each side).
        size : float
            Side length of each subplot in inches.
        cmap : str | matplotlib.colors.Colormap | tuple | 'interactive' | None
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
        vlim : tuple of length 2 | "joint"
            Lower and upper bounds of the colormap, typically a numeric value in the
            same units as the data. Elements of the :class:`tuple` may also be
            callable functions which take in a :class:`NumPy array <numpy.ndarray>` and
            return a scalar.

            If both entries are ``None``, the bounds are set at
            ± the maximum absolute value
            of the data (yielding a colormap with midpoint at 0), or
            ``(0, max(abs(data)))`` if the (possibly baselined) data are all-positive.
            Providing ``None`` for just one entry will set the corresponding boundary
            at the min/max of the data. If ``vlim="joint"``, will compute the colormap
            limits jointly across all topomaps of the same channel type (instead of
            separately for each topomap), using the min/max of the data for that
            channel type. Defaults to ``(None, None)``.
        cnorm : matplotlib.colors.Normalize | None
            How to normalize the colormap. If ``None``, standard linear normalization
            is performed. If not ``None``, ``vmin`` and ``vmax`` will be ignored.
            See :ref:`Matplotlib docs <matplotlib:colormapnorms>`
            for more details on colormap normalization, and
            :ref:`the ERDs example<cnorm-example>` for an example of its use.
        colorbar : bool
            Plot a colorbar in the rightmost column of the figure.
        cbar_fmt : str
            Formatting string for colorbar tick labels. See :ref:`formatspec` for
            details.
        units : dict | str | None
            The units to use for the colorbar label. Ignored if ``colorbar=False``.
            If ``None`` and ``scalings=None`` the unit is automatically determined,
            otherwise the label will be "AU" indicating arbitrary units.
            Default is ``None``.
        axes : list of matplotlib.axes.Axes | None
            The axes to use for plotting. Must have one axis for the topomap,
            then one for the colorbar (if ``colorbar=True``), then one for the
            butterfly axes (if ``butterfly=True``).
        time_unit : str
            The units for the time axis, can be "ms" or "s" (default).
        time_format : str | None
            String format for topomap values. Defaults (None) to "%01d ms" if
            ``time_unit='ms'``, "%0.3f s" if ``time_unit='s'``, and
            "%g" otherwise. Can be an empty string to omit the time label.
        frame_rate : int | None
            Frame rate for the animation in Hz. If None,
            frame rate = sfreq / 10. Defaults to None.
        butterfly : bool
            Whether to plot the data as butterfly plot under the topomap.
            Defaults to False.
        blit : bool
            Whether to use blit to optimize drawing. In general, it is
            recommended to use blit in combination with ``show=True``. If you
            intend to save the animation it is better to disable blit.
            Defaults to True.
        show : bool
            Whether to show the animation. Defaults to True.
        verbose : bool | str | int | None
            Control verbosity of the logging output. If ``None``, use the default
            verbosity level. See the :ref:`logging documentation <tut-logging>` and
            :func:`mne.verbose` for details. Should only be passed as a keyword
            argument.

        Returns
        -------
        fig : instance of matplotlib.figure.Figure
            The figure.
        anim : instance of matplotlib.animation.FuncAnimation
            Animation of the topomap.

        Notes
        -----
        .. versionchanged:: 1.13.0
           The ``vmin`` and ``vmax`` parameters were deprecated in favor of a single
           ``vlim`` parameter, and parameters were added and reordered to follow
           :meth:`~mne.Evoked.plot_topomap`.
        .. versionadded:: 0.12.0
        """  # noqa: E501
        from .viz.topomap import _topomap_animation

        return _topomap_animation(
            evoked=self,
            times=times,
            average=average,
            ch_type=ch_type,
            scalings=scalings,
            proj=proj,
            sensors=sensors,
            show_names=show_names,
            mask=mask,
            mask_params=mask_params,
            mask_label_params=mask_label_params,
            contours=contours,
            outlines=outlines,
            sphere=sphere,
            image_interp=image_interp,
            extrapolate=extrapolate,
            border=border,
            res=res,
            size=size,
            cmap=cmap,
            vlim=vlim,
            cnorm=cnorm,
            colorbar=colorbar,
            cbar_fmt=cbar_fmt,
            units=units,
            axes=axes,
            time_unit=time_unit,
            time_format=time_format,
            frame_rate=frame_rate,
            butterfly=butterfly,
            blit=blit,
            show=show,
        )

    def as_type(self, ch_type: str = "grad", mode: str = "fast") -> "Evoked":
        """Compute virtual evoked using interpolated fields.

        .. Warning:: Using virtual evoked to compute inverse can yield
            unexpected results. The virtual channels have ``'_v'`` appended
            at the end of the names to emphasize that the data contained in
            them are interpolated.

        Parameters
        ----------
        ch_type : str
            The destination channel type. It can be 'mag' or 'grad'.
        mode : str
            Either ``'accurate'`` or ``'fast'``, determines the quality of the
            Legendre polynomial expansion used. ``'fast'`` should be sufficient
            for most applications.

        Returns
        -------
        evoked : instance of mne.Evoked
            The transformed evoked object containing only virtual channels.

        Notes
        -----
        This method returns a copy and does not modify the data it
        operates on. It also returns an EvokedArray instance.

        .. versionadded:: 0.9.0
        """
        from .forward import _as_meg_type_inst

        return _as_meg_type_inst(self, ch_type=ch_type, mode=mode)

    @fill_doc_static("picks_good_data")
    def detrend(
        self, order: int = 1, picks: str | np.ndarray | slice | None = None
    ) -> Self:
        """Detrend data.

        This function operates in-place.

        Parameters
        ----------
        order : int
            Either 0 or 1, the order of the detrending. 0 is a constant
            (DC) detrend, 1 is a linear detrend.
        picks : str | array-like | slice | None
            Channels to include. Slices and lists of integers will be interpreted as
            channel indices. In lists, channel *type* strings (e.g., ``['meg',
            'eeg']``) will pick channels of those types, channel *name* strings (e.g.,
            ``['MEG0111', 'MEG2623']`` will pick the given channels. Can also be the
            string values ``'all'`` to pick all channels, or ``'data'`` to pick
            :term:`data channels`. None (default) will pick good data channels. Note
            that channels in ``info['bads']`` *will be included* if their names or
            indices are explicitly provided.

        Returns
        -------
        evoked : instance of Evoked
            The detrended evoked object.
        """
        picks = _picks_to_idx(self.info, picks)
        self.data[picks] = detrend(self.data[picks], order, axis=-1)
        return self

    def copy(self) -> Self:
        """Copy the instance of evoked.

        Returns
        -------
        evoked : instance of Evoked
            A copy of the object.
        """
        evoked = deepcopy(self)
        return evoked

    def __neg__(self):
        """Negate channel responses.

        Returns
        -------
        evoked_neg : instance of Evoked
            The Evoked instance with channel data negated and '-'
            prepended to the comment.
        """
        out = self.copy()
        out.data *= -1

        if out.comment is not None and " + " in out.comment:
            out.comment = f"({out.comment})"  # multiple conditions in evoked
        out.comment = f"- {out.comment or 'unknown'}"
        return out

    def get_peak(
        self,
        ch_type: str | None = None,
        tmin: float | None = None,
        tmax: float | None = None,
        mode: Literal["pos", "neg", "abs"] = "abs",
        time_as_index: bool = False,
        merge_grads: bool = False,
        return_amplitude: bool = False,
        *,
        strict: bool = True,
    ) -> tuple:
        """Get location and latency of peak amplitude.

        Parameters
        ----------
        ch_type : str | None
            The channel type to use. Defaults to None. If more than one channel
            type is present in the data, this value **must** be provided.
        tmin : float | None
            The minimum point in time to be considered for peak getting.
            If None (default), the beginning of the data is used.
        tmax : float | None
            The maximum point in time to be considered for peak getting.
            If None (default), the end of the data is used.
        mode : 'pos' | 'neg' | 'abs'
            How to deal with the sign of the data. If 'pos' only positive
            values will be considered. If 'neg' only negative values will
            be considered. If 'abs' absolute values will be considered.
            Defaults to 'abs'.
        time_as_index : bool
            Whether to return the time index instead of the latency in seconds.
        merge_grads : bool
            If True, compute peak from merged gradiometer data.
        return_amplitude : bool
            If True, return also the amplitude at the maximum response.

            .. versionadded:: 0.16
        strict : bool
            If True, raise an error if values are all positive when detecting
            a minimum (mode='neg'), or all negative when detecting a maximum
            (mode='pos'). Defaults to True.

            .. versionadded:: 1.7

        Returns
        -------
        ch_name : str
            The channel exhibiting the maximum response.
        latency : float | int
            The time point of the maximum response, either latency in seconds
            or index.
        amplitude : float
            The amplitude of the maximum response. Only returned if
            return_amplitude is True.

            .. versionadded:: 0.16
        """  # noqa: E501
        supported = (
            "mag",
            "grad",
            "eeg",
            "seeg",
            "dbs",
            "ecog",
            "misc",
            "None",
        ) + _FNIRS_CH_TYPES_SPLIT
        types_used = self.get_channel_types(unique=True, only_data_chs=True)

        _check_option("ch_type", str(ch_type), supported)

        if ch_type is not None and ch_type not in types_used:
            raise ValueError(
                f'Channel type "{ch_type}" not found in this evoked object.'
            )

        elif len(types_used) > 1 and ch_type is None:
            raise RuntimeError(
                'Multiple data channel types found. Please pass the "ch_type" '
                "parameter."
            )

        if merge_grads:
            if ch_type != "grad":
                raise ValueError('Channel type must be "grad" for merge_grads')
            elif mode == "neg":
                raise ValueError(
                    "Negative mode (mode=neg) does not make sense with merge_grads=True"
                )

        meg = eeg = misc = seeg = dbs = ecog = fnirs = False
        picks = None
        if ch_type in ("mag", "grad"):
            meg = ch_type
        elif ch_type == "eeg":
            eeg = True
        elif ch_type == "misc":
            misc = True
        elif ch_type == "seeg":
            seeg = True
        elif ch_type == "dbs":
            dbs = True
        elif ch_type == "ecog":
            ecog = True
        elif ch_type in _FNIRS_CH_TYPES_SPLIT:
            fnirs = ch_type

        if ch_type is not None:
            if merge_grads:
                picks = _pair_grad_sensors(self.info, topomap_coords=False)
            else:
                picks = pick_types(
                    self.info,
                    meg=meg,
                    eeg=eeg,
                    misc=misc,
                    seeg=seeg,
                    ecog=ecog,
                    ref_meg=False,
                    fnirs=fnirs,
                    dbs=dbs,
                )
        data = self.data
        ch_names = self.ch_names

        if picks is not None:
            data = data[picks]
            ch_names = [ch_names[k] for k in picks]

        if merge_grads:
            data, _ = _merge_ch_data(data, ch_type, [])
            ch_names = [ch_name[:-1] + "X" for ch_name in ch_names[::2]]

        ch_idx, time_idx, max_amp = _get_peak(
            data,
            self.times,
            tmin,
            tmax,
            mode,
            strict=strict,
        )

        out = (ch_names[ch_idx], time_idx if time_as_index else self.times[time_idx])

        if return_amplitude:
            out += (max_amp,)

        return out

    @verbose_static(
        "method_psd",
        "fmin_fmax_psd",
        "tmin_tmax_psd",
        "picks_good_data_noref",
        "proj_psd",
        "remove_dc",
        "exclude_psd",
        "n_jobs",
        "method_kw_psd",
    )
    def compute_psd(
        self,
        method: Literal["welch", "multitaper"] = "multitaper",
        fmin: float = 0,
        fmax: float = np.inf,
        tmin: float | None = None,
        tmax: float | None = None,
        picks: str | np.ndarray | slice | None = None,
        proj: bool = False,
        remove_dc: bool = True,
        exclude: list[str] | tuple[str, ...] | Literal["bads"] = (),
        *,
        n_jobs: int | None = 1,
        verbose: bool | str | int | None = None,
        **method_kw,
    ) -> Spectrum:
        """Perform spectral analysis on sensor data.

        Parameters
        ----------
        method : ``'welch'`` | ``'multitaper'``
            Spectral estimation method. ``'welch'`` uses Welch's
            method :footcite:p:`Welch1967`, ``'multitaper'`` uses DPSS
            tapers :footcite:p:`Slepian1978`.
            Default is ``'multitaper'``.
        fmin, fmax : float
            The lower- and upper-bound on frequencies of interest. Default is
            ``fmin=0, fmax=np.inf`` (spans all frequencies present in the data).
        tmin, tmax : float | None
            First and last times to include, in seconds. ``None`` uses the first or
            last time present in the data. Default is ``tmin=None, tmax=None`` (all
            times).
        picks : str | array-like | slice | None
            Channels to include. Slices and lists of integers will be interpreted as
            channel indices. In lists, channel *type* strings (e.g., ``['meg',
            'eeg']``) will pick channels of those types, channel *name* strings (e.g.,
            ``['MEG0111', 'MEG2623']`` will pick the given channels. Can also be the
            string values ``'all'`` to pick all channels, or ``'data'`` to pick
            :term:`data channels`. None (default) will pick good data channels
            (excluding reference MEG channels). Note that channels in ``info['bads']``
            *will be included* if their names or indices are explicitly provided.
        proj : bool
            Whether to apply SSP projection vectors before spectral estimation.
            Default is ``False``.
        remove_dc : bool
            If ``True``, the mean is subtracted from each segment before computing
            its spectrum.
        exclude : list of str | 'bads'
            Channel names to exclude. If ``'bads'``, channels
            in ``info['bads']`` are excluded; pass an empty list to
            include all channels (including "bad" channels, if any).
        n_jobs : int | None
            The number of jobs to run in parallel. If ``-1``, it is set
            to the number of CPU cores. Requires the :mod:`joblib` package.
            ``None`` (default) is a marker for 'unset' that will be interpreted
            as ``n_jobs=1`` (sequential execution) unless the call is performed under
            a :class:`joblib:joblib.parallel_config` context manager that sets another
            value for ``n_jobs``.
        verbose : bool | str | int | None
            Control verbosity of the logging output. If ``None``, use the default
            verbosity level. See the :ref:`logging documentation <tut-logging>` and
            :func:`mne.verbose` for details. Should only be passed as a keyword
            argument.
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

        Returns
        -------
        spectrum : instance of Spectrum
            The spectral representation of the data.

        Notes
        -----
        .. versionadded:: 1.2

        References
        ----------
        .. footbibliography::
        """
        method = _validate_method(method, type(self).__name__)
        self._set_legacy_nfft_default(tmin, tmax, method, method_kw)

        return Spectrum(
            self,
            method=method,
            fmin=fmin,
            fmax=fmax,
            tmin=tmin,
            tmax=tmax,
            picks=picks,
            exclude=exclude,
            proj=proj,
            remove_dc=remove_dc,
            reject_by_annotation=False,
            n_jobs=n_jobs,
            verbose=verbose,
            **method_kw,
        )

    @verbose_static(
        "method_tfr",
        "freqs_tfr",
        "tmin_tmax_psd",
        "picks_good_data_noref",
        "proj_psd",
        "output_compute_tfr",
        "decim_tfr",
        "n_jobs",
        "method_kw_tfr",
    )
    def compute_tfr(
        self,
        method: Literal["morlet", "multitaper"] | None,
        freqs: np.ndarray | None,
        *,
        tmin: float | None = None,
        tmax: float | None = None,
        picks: str | np.ndarray | slice | None = None,
        proj: bool = False,
        output: str = "power",
        decim: int | slice = 1,
        n_jobs: int | None = None,
        verbose: bool | str | int | None = None,
        **method_kw,
    ) -> "AverageTFR":
        """Compute a time-frequency representation of evoked data.

        Parameters
        ----------
        method : ``'morlet'`` | ``'multitaper'`` | None
            Spectrotemporal power estimation method. ``'morlet'`` uses Morlet wavelets,
            ``'multitaper'`` uses DPSS tapers :footcite:p:`Slepian1978`.
            ``None`` (the default) only works when using ``__setstate__`` and will
            raise an error otherwise.
        freqs : array-like | None
            The frequencies at which to compute the power estimates.
            Must be an array of shape (n_freqs,). ``None`` (the
            default) only works when using ``__setstate__`` and will raise an
            error otherwise.
        tmin, tmax : float | None
            First and last times to include, in seconds. ``None`` uses the first or
            last time present in the data. Default is ``tmin=None, tmax=None`` (all
            times).
        picks : str | array-like | slice | None
            Channels to include. Slices and lists of integers will be interpreted as
            channel indices. In lists, channel *type* strings (e.g., ``['meg',
            'eeg']``) will pick channels of those types, channel *name* strings (e.g.,
            ``['MEG0111', 'MEG2623']`` will pick the given channels. Can also be the
            string values ``'all'`` to pick all channels, or ``'data'`` to pick
            :term:`data channels`. None (default) will pick good data channels
            (excluding reference MEG channels). Note that channels in ``info['bads']``
            *will be included* if their names or indices are explicitly provided.
        proj : bool
            Whether to apply SSP projection vectors before spectral estimation.
            Default is ``False``.
        output : str
            What kind of estimate to return. Allowed values are ``"complex"``,
            ``"phase"``, and ``"power"``. Default is ``"power"``.
        decim : int | slice
            Decimation factor, applied *after* time-frequency decomposition.

            - if :class:`int`, returns ``tfr[..., ::decim]`` (keep only every Nth
              sample along the time axis).
            - if :class:`slice`, returns ``tfr[..., decim]`` (keep only the specified
              slice along the time axis).

            .. note::
                Decimation is done after convolutions and may create aliasing
                artifacts.
        n_jobs : int | None
            The number of jobs to run in parallel. If ``-1``, it is set
            to the number of CPU cores. Requires the :mod:`joblib` package.
            ``None`` (default) is a marker for 'unset' that will be interpreted
            as ``n_jobs=1`` (sequential execution) unless the call is performed under
            a :class:`joblib:joblib.parallel_config` context manager that sets another
            value for ``n_jobs``.
        verbose : bool | str | int | None
            Control verbosity of the logging output. If ``None``, use the default
            verbosity level. See the :ref:`logging documentation <tut-logging>` and
            :func:`mne.verbose` for details. Should only be passed as a keyword
            argument.
        **method_kw
            Additional keyword arguments passed to the spectrotemporal estimation
            function (e.g., ``n_cycles, use_fft, zero_mean`` for Morlet
            method
            or ``n_cycles, use_fft, zero_mean, time_bandwidth`` for multitaper method).
            See :func:`~mne.time_frequency.tfr_array_morlet`
            and :func:`~mne.time_frequency.tfr_array_multitaper` for additional details.

        Returns
        -------
        tfr : instance of AverageTFR
            The time-frequency-resolved power estimates of the data.

        Notes
        -----
        .. versionadded:: 1.7

        References
        ----------
        .. footbibliography::
        """
        from .time_frequency.tfr import AverageTFR

        _check_option("output", output, ("power", "phase", "complex"))
        method_kw["output"] = output
        return AverageTFR(
            inst=self,
            method=method,
            freqs=freqs,
            tmin=tmin,
            tmax=tmax,
            picks=picks,
            proj=proj,
            decim=decim,
            n_jobs=n_jobs,
            verbose=verbose,
            **method_kw,
        )

    @verbose_static(
        "fmin_fmax_psd",
        "tmin_tmax_psd",
        "picks_good_data_noref",
        "proj_psd",
        "method_plot_psd_auto",
        "average_plot_psd",
        "dB_plot_psd",
        "estimate_plot_psd",
        "xscale_plot_psd",
        "area_mode_plot_psd",
        "area_alpha_plot_psd",
        "color_plot_psd",
        "line_alpha_plot_psd",
        "spatial_colors_psd",
        "sphere_topomap_auto",
        "ax_plot_psd",
        "show",
        "n_jobs",
        "method_kw_psd",
        "notes_plot_psd_meth",
    )
    def plot_psd(
        self,
        fmin: float = 0,
        fmax: float = np.inf,
        tmin: float | None = None,
        tmax: float | None = None,
        picks: str | np.ndarray | slice | None = None,
        proj: bool = False,
        *,
        method: Literal["welch", "multitaper", "auto"] = "auto",
        average: bool = False,
        dB: bool = True,
        estimate: str = "power",
        xscale: Literal["linear", "log"] = "linear",
        area_mode: str | None = "std",
        area_alpha: float = 0.33,
        color: str | tuple = "black",
        line_alpha: float | None = None,
        spatial_colors: bool = True,
        sphere: "float | Annotated[Sequence[float], 4] | np.ndarray[tuple[Literal[4]], np.dtype[np.floating]] | ConductorModel | Literal['auto', 'cardinal', 'eeg', 'extra', 'hpi', 'eeglab'] | list[Literal['cardinal', 'eeg', 'extra', 'hpi']] | None" = None,  # noqa E501
        exclude: list[str] | Literal["bads"] = "bads",
        ax: "Axes | list[Axes] | None" = None,
        show: bool = True,
        n_jobs: int | None = 1,
        verbose: bool | str | int | None = None,
        **method_kw,
    ) -> "Figure":
        """Plot power or amplitude spectra.

        Separate plots are drawn for each channel type. When the data have been
        processed with a bandpass, lowpass or highpass filter, dashed lines (╎)
        indicate the boundaries of the filter. The line noise frequency is also
        indicated with a dashed line (⋮). If ``average=False``, the plot will
        be interactive, and click-dragging on the spectrum will generate a
        scalp topography plot for the chosen frequency range in a new figure.

        Parameters
        ----------
        fmin, fmax : float
            The lower- and upper-bound on frequencies of interest. Default is
            ``fmin=0, fmax=np.inf`` (spans all frequencies present in the data).
        tmin, tmax : float | None
            First and last times to include, in seconds. ``None`` uses the first or
            last time present in the data. Default is ``tmin=None, tmax=None`` (all
            times).
        picks : str | array-like | slice | None
            Channels to include. Slices and lists of integers will be interpreted as
            channel indices. In lists, channel *type* strings (e.g., ``['meg',
            'eeg']``) will pick channels of those types, channel *name* strings (e.g.,
            ``['MEG0111', 'MEG2623']`` will pick the given channels. Can also be the
            string values ``'all'`` to pick all channels, or ``'data'`` to pick
            :term:`data channels`. None (default) will pick good data channels
            (excluding reference MEG channels). Note that channels in ``info['bads']``
            *will be included* if their names or indices are explicitly provided.
        proj : bool
            Whether to apply SSP projection vectors before spectral estimation.
            Default is ``False``.
        method : ``'welch'`` | ``'multitaper'`` | ``'auto'``
            Spectral estimation method. ``'welch'`` uses Welch's
            method :footcite:p:`Welch1967`, ``'multitaper'`` uses DPSS
            tapers :footcite:p:`Slepian1978`. ``'auto'`` (default) uses Welch's
            method for continuous data and multitaper for
            :class:`~mne.Epochs` or :class:`~mne.Evoked` data.
        average : bool
            If False, the PSDs of all channels is displayed. No averaging
            is done and parameters area_mode and area_alpha are ignored. When
            False, it is possible to paint an area (hold left mouse button and
            drag) to plot a topomap.
        dB : bool
            Plot power spectral density (PSD) in units (dB/Hz) if ``dB=True`` and
            ``estimate='power'``. Plot PSD in units (amplitude**2/Hz) if ``dB=False``
            and ``estimate='power'``. Plot amplitude spectral density (ASD) in units
            (amplitude/sqrt(Hz)) if ``dB=False`` and ``estimate='amplitude'``. Plot ASD
            in units (dB/sqrt(Hz)) if ``dB=True`` and ``estimate='amplitude'``.
        estimate : str, {'power', 'amplitude'}
            Can be "power" for power spectral density (PSD; default), "amplitude" for
            amplitude spectrum density (ASD).
        xscale : 'linear' | 'log'
            Scale of the frequency axis. Default is ``'linear'``.
        area_mode : str | None
            Mode for plotting area. If 'std', the mean +/- 1 STD (across channels)
            will be plotted. If 'range', the min and max (across channels) will be
            plotted. Bad channels will be excluded from these calculations.
            If None, no area will be plotted. If average=False, no area is plotted.
        area_alpha : float
            Alpha for the area.
        color : str | tuple
            A matplotlib-compatible color to use. Has no effect when
            spatial_colors=True.
        line_alpha : float | None
            Alpha for the PSD line. Can be None (default) to use 1.0 when
            ``average=True`` and 0.1 when ``average=False``.
        spatial_colors : bool
            Whether to color spectrum lines by channel location. Ignored if
            ``average=True``.
        sphere : float | array-like of float | instance of ConductorModel | {"auto", "cardinal", "eeg", "extra", "hpi", "eeglab"} | list of str | None
            The sphere parameters to use for the head outline.
            Can be array-like of shape (4,) to give the X/Y/Z origin and radius in
            meters, or a single float to give just the radius (origin assumed 0, 0, 0).
            Can also be an instance of a spherical :class:`~mne.bem.ConductorModel` to
            use the origin and radius from that object.
            Can also be a ``str``, in which case:

            - ``'auto'``: the sphere is fit to external digitization points first, and
              to external + EEG digitization points if the former fails.

            - ``'eeglab'``: the head circle is defined by EEG electrodes ``'Fpz'``,
              ``'Oz'``, ``'T7'``, and ``'T8'`` (if ``'Fpz'`` is not present, it will be
              approximated from the coordinates of ``'Oz'``).

              - ``'extra'``: the sphere is fit to external digitization points.

              - ``'eeg'``: the sphere is fit to EEG digitization points.

              - ``'cardinal'``: the sphere is fit to cardinal digitization points.

              - ``'hpi'``: the sphere is fit to HPI coil digitization points.

            Can also be a list of ``str``, in which case the sphere is fit to the
            specified digitization points, which can be any combination of ``'extra'``,
            ``'eeg'``, ``'cardinal'``, and ``'hpi'``, as specified above.
            ``None`` (the default) will look for an existing head outline in the
            ``.info`` dictionary and use that. If no outline is present, it is
            equivalent to ``'auto'`` when enough extra digitization points are
            available, and ``(0, 0, 0, 0.095)`` otherwise.

            .. versionadded:: 0.20
            .. versionchanged:: 1.1 Added ``'eeglab'`` option.
            .. versionchanged:: 1.11 Added ``'extra'``, ``'eeg'``, ``'cardinal'``,
               ``'hpi'`` and list of ``str`` options.

            .. versionadded:: 0.22.0
        exclude : list of str | 'bads'
            Channels names to exclude from being shown. If 'bads', the bad
            channels are excluded. Pass an empty list to plot all channels
            (including channels marked "bad", if any).

            .. versionadded:: 0.24.0
        ax : instance of Axes | list of Axes | None
            The axes to plot into. If ``None``, a new :class:`~matplotlib.figure.Figure`
            will be created with the correct number of axes. If
            :class:`~matplotlib.axes.Axes` are provided (either as a single instance or
            a :class:`list` of axes), the number of axes provided must
            match the number of channel types present in
            the object. Default is ``None``.
        show : bool
            Show the figure if ``True``. When shown, blocking follows
            :func:`matplotlib.pyplot.show`: the call blocks until the window is closed
            unless Matplotlib's interactive mode is on (enabled with
            :func:`matplotlib.pyplot.ion` or IPython's ``%%matplotlib`` magic command),
            in which case it returns immediately. Interactive mode is off by default, so
            a plain script or REPL blocks. Pass ``show=False`` to build several figures
            and display them together with a single :func:`matplotlib.pyplot.show` call.
        n_jobs : int | None
            The number of jobs to run in parallel. If ``-1``, it is set
            to the number of CPU cores. Requires the :mod:`joblib` package.
            ``None`` (default) is a marker for 'unset' that will be interpreted
            as ``n_jobs=1`` (sequential execution) unless the call is performed under
            a :class:`joblib:joblib.parallel_config` context manager that sets another
            value for ``n_jobs``.
        verbose : bool | str | int | None
            Control verbosity of the logging output. If ``None``, use the default
            verbosity level. See the :ref:`logging documentation <tut-logging>` and
            :func:`mne.verbose` for details. Should only be passed as a keyword
            argument.
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

        Returns
        -------
        fig : instance of Figure
            Figure with frequency spectra of the data channels.

        Notes
        -----
        This method exists to support legacy code; for new code the preferred
        idiom is ``inst.compute_psd().plot()`` (where ``inst`` is an instance
        of :class:`~mne.io.Raw`, :class:`~mne.Epochs`, or :class:`~mne.Evoked`).
        """  # noqa: E501
        return super().plot_psd(
            fmin=fmin,
            fmax=fmax,
            tmin=tmin,
            tmax=tmax,
            picks=picks,
            proj=proj,
            reject_by_annotation=False,
            method=method,
            average=average,
            dB=dB,
            estimate=estimate,
            xscale=xscale,
            area_mode=area_mode,
            area_alpha=area_alpha,
            color=color,
            line_alpha=line_alpha,
            spatial_colors=spatial_colors,
            sphere=sphere,
            exclude=exclude,
            ax=ax,
            show=show,
            n_jobs=n_jobs,
            verbose=verbose,
            **method_kw,
        )

    @verbose_static(
        "picks_all",
        "index_df_evk",
        "scalings_df",
        "copy_df",
        "long_format_df_raw",
        "time_format_df",
        "df_return",
    )
    def to_data_frame(
        self,
        picks: str | np.ndarray | slice | None = None,
        index: Literal["time"] | None = None,
        scalings: dict | None = None,
        copy: bool = True,
        long_format: bool = False,
        time_format: str | None = None,
        *,
        verbose: bool | str | int | None = None,
    ) -> "DataFrame":
        """Export data in tabular structure as a pandas DataFrame.

        Channels are converted to columns in the DataFrame. By default,
        an additional column "time" is added, unless ``index='time'``
        (in which case time values form the DataFrame's index).

        Parameters
        ----------
        picks : str | array-like | slice | None
            Channels to include. Slices and lists of integers will be interpreted as
            channel indices. In lists, channel *type* strings (e.g., ``['meg',
            'eeg']``) will pick channels of those types, channel *name* strings (e.g.,
            ``['MEG0111', 'MEG2623']`` will pick the given channels. Can also be the
            string values ``'all'`` to pick all channels, or ``'data'`` to pick
            :term:`data channels`. None (default) will pick all channels. Bad channels
            are included by default. Note that channels in ``info['bads']`` *will be
            included* if their names or indices are explicitly provided.
        index : 'time' | None
            Kind of index to use for the DataFrame. If ``None``, a sequential
            integer index (:class:`pandas.RangeIndex`) will be used. If ``'time'``, a
            ``pandas.Index`` or
            :class:`pandas.TimedeltaIndex` will be used
            (depending on the value of ``time_format``).
            Defaults to ``None``.
        scalings : dict | None
            Scaling factor applied to the channels picked. If ``None``, defaults to
            ``dict(eeg=1e6, mag=1e15, grad=1e13)`` — i.e., converts EEG to µV,
            magnetometers to fT, and gradiometers to fT/cm. See :term:`data channels`
            and :term:`non-data channels` for full list of default scalings.
        copy : bool
            If ``True``, data will be copied. Otherwise data may be modified in place.
            Defaults to ``True``.
        long_format : bool
            If True, the DataFrame is returned in long format where each row is one
            observation of the signal at a unique combination of
            time point and channel.
            For convenience, a ``ch_type`` column is added to facilitate
            subsetting the resulting DataFrame. Defaults to ``False``.
        time_format : str | None
            Desired time format. If ``None``, no conversion is applied, and time values
            remain as float values in seconds. If ``'ms'``, time values will be rounded
            to the nearest millisecond and converted to integers. If ``'timedelta'``,
            time values will be converted to
            :class:`pandas.Timedelta` values.
            Default is ``None`` unless specified otherwise.

            .. versionadded:: 0.20
        verbose : bool | str | int | None
            Control verbosity of the logging output. If ``None``, use the default
            verbosity level. See the :ref:`logging documentation <tut-logging>` and
            :func:`mne.verbose` for details. Should only be passed as a keyword
            argument.

        Returns
        -------
        df : instance of pandas.DataFrame
            A dataframe suitable for usage with other statistical/plotting/analysis
            packages.
        """
        # check pandas once here, instead of in each private utils function
        pd = _check_pandas_installed()  # noqa
        # arg checking
        valid_index_args = ["time"]
        valid_time_formats = ["ms", "timedelta"]
        index = _check_pandas_index_arguments(index, valid_index_args)
        time_format = _check_time_format(time_format, valid_time_formats)
        # get data
        picks = _picks_to_idx(self.info, picks, "all", exclude=())
        data = self.data[picks, :]
        times = self.times
        data = data.T
        if copy:
            data = data.copy()
        data = _scale_dataframe_data(self, data, picks, scalings)
        # prepare extra columns / multiindex
        mindex = list()
        times = _convert_times(times, time_format, meas_date=self.info["meas_date"])
        mindex.append(("time", times))
        # build DataFrame
        df = _build_data_frame(
            self, data, picks, long_format, mindex, index, default_index=["time"]
        )
        return df


@fill_doc_static("info_not_none", "baseline_evoked", "verbose")
class EvokedArray(Evoked):
    """Evoked object from numpy array.

    Parameters
    ----------
    data : array of shape (n_channels, n_times)
        The channels' evoked response. See notes for proper units of measure.
    info : mne.Info
        The :class:`mne.Info` object with information about the
        sensors and methods of measurement.
        Consider using :func:`mne.create_info` to populate this
        structure.
    tmin : float
        Start time before event. Defaults to 0.
    comment : str
        Comment on dataset. Can be the condition. Defaults to ''.
    nave : int
        Number of averaged epochs. Defaults to 1.
    kind : str
        Type of data, either average or standard_error. Defaults to 'average'.
    baseline : None | tuple of length 2
        The time interval to consider as "baseline" when applying baseline
        correction. If ``None``, do not apply baseline correction.
        If a tuple ``(a, b)``, the interval is between ``a`` and ``b``
        (in seconds), including the endpoints.
        If ``a`` is ``None``, the **beginning** of the data is used; and if ``b``
        is ``None``, it is set to the **end** of the data.
        If ``(None, None)``, the entire time interval is used.

        .. note::
            The baseline ``(a, b)`` includes both endpoints, i.e. all timepoints
            ``t`` such that ``a <= t <= b``.

        Correction is applied **to each channel individually** in the following
        way:

        1. Calculate the mean signal of the baseline period.
        2. Subtract this mean from the **entire** ``Evoked``.

        Defaults to ``None``, i.e. no baseline correction.

        .. versionadded:: 0.23
    verbose : bool | str | int | None
        Control verbosity of the logging output. If ``None``, use the default
        verbosity level. See the :ref:`logging documentation <tut-logging>` and
        :func:`mne.verbose` for details. Should only be passed as a keyword
        argument.

    See Also
    --------
    EpochsArray, io.RawArray, create_info

    Notes
    -----
    Proper units of measure:

    * V: eeg, eog, seeg, dbs, emg, ecg, bio, ecog
    * T: mag
    * T/m: grad
    * M: hbo, hbr
    * Am: dipole
    * AU: misc
    """

    @_verbose_control
    def __init__(
        self,
        data: np.ndarray,
        info: Info,
        tmin: float = 0.0,
        comment: str | None = "",
        nave: int = 1,
        kind: str = "average",
        baseline: tuple[float | None, float | None] | None = None,
        *,
        verbose: bool | str | int | None = None,
    ):
        dtype = np.complex128 if np.iscomplexobj(data) else np.float64
        data = np.asanyarray(data, dtype=dtype)

        if data.ndim != 2:
            raise ValueError(
                "Data must be a 2D array of shape (n_channels, n_samples), got shape "
                f"{data.shape}"
            )

        if len(info["ch_names"]) != np.shape(data)[0]:
            raise ValueError(
                f"Info ({len(info['ch_names'])}) and data ({np.shape(data)[0]}) must "
                "have same number of channels."
            )

        self.data = data

        self.first = int(round(tmin * info["sfreq"]))
        self.last = self.first + np.shape(data)[-1] - 1
        self._set_times(
            np.arange(self.first, self.last + 1, dtype=np.float64) / info["sfreq"]
        )
        self._raw_times = self.times.copy()
        self._decim = 1
        self.info = info.copy()  # do not modify original info
        self.nave = nave
        self.kind = kind
        self.comment = comment
        self.picks = None
        self.preload = True
        self._projector = None
        _validate_type(self.kind, "str", "kind")
        if self.kind not in _aspect_dict:
            raise ValueError(
                f'unknown kind "{self.kind}", should be "average" or "standard_error"'
            )
        self._aspect_kind = _aspect_dict[self.kind]

        self.baseline = baseline
        if self.baseline is not None:  # omit log msg if not baselining
            self.apply_baseline(self.baseline)
        self._filename = None


def _get_entries(fid, evoked_node, allow_maxshield=False):
    """Get all evoked entries."""
    comments = list()
    aspect_kinds = list()
    for ev in evoked_node:
        for k in range(ev["nent"]):
            my_kind = ev["directory"][k].kind
            pos = ev["directory"][k].pos
            if my_kind == FIFF.FIFF_COMMENT:
                tag = read_tag(fid, pos)
                comments.append(tag.data)
        my_aspect = _get_aspect(ev, allow_maxshield)[0]
        for k in range(my_aspect["nent"]):
            my_kind = my_aspect["directory"][k].kind
            pos = my_aspect["directory"][k].pos
            if my_kind == FIFF.FIFF_ASPECT_KIND:
                tag = read_tag(fid, pos)
                aspect_kinds.append(int(tag.data.item()))
    comments = np.atleast_1d(comments)
    aspect_kinds = np.atleast_1d(aspect_kinds)
    if len(comments) != len(aspect_kinds) or len(comments) == 0:
        fid.close()
        raise ValueError("Dataset names in FIF file could not be found.")
    t = [_aspect_rev[a] for a in aspect_kinds]
    t = ['"' + c + '" (' + tt + ")" for tt, c in zip(t, comments)]
    t = "\n".join(t)
    return comments, aspect_kinds, t


def _get_aspect(evoked, allow_maxshield):
    """Get Evoked data aspect."""
    from .io.base import _check_maxshield

    is_maxshield = False
    aspect = dir_tree_find(evoked, FIFF.FIFFB_ASPECT)
    if len(aspect) == 0:
        _check_maxshield(allow_maxshield)
        aspect = dir_tree_find(evoked, FIFF.FIFFB_IAS_ASPECT)
        is_maxshield = True
    if len(aspect) > 1:
        logger.info("Multiple data aspects found. Taking first one.")
    return aspect[0], is_maxshield


def _get_evoked_node(fname):
    """Get info in evoked file."""
    f, tree, _ = fiff_open(fname)
    with f as fid:
        _, meas = read_meas_info(fid, tree, verbose=False)
        evoked_node = dir_tree_find(meas, FIFF.FIFFB_EVOKED)
    return evoked_node


def _check_evokeds_ch_names_times(all_evoked, inplace=False):
    evoked = all_evoked[0]
    ch_names = evoked.ch_names
    for ii, ev in enumerate(all_evoked[1:]):
        if ev.ch_names != ch_names:
            if set(ev.ch_names) != set(ch_names):
                raise ValueError(f"{evoked} and {ev} do not contain the same channels.")
            else:
                warn("Order of channels differs, reordering channels ...")
                if not inplace:
                    ev = ev.copy()
                ev.reorder_channels(ch_names)
                all_evoked[ii + 1] = ev
        if not np.max(np.abs(ev.times - evoked.times)) < 1e-7:
            raise ValueError(f"{evoked} and {ev} do not contain the same time instants")
    return all_evoked


def combine_evoked(
    all_evoked: list[Evoked], weights: list[float] | Literal["equal", "nave"]
) -> Evoked:
    """Merge evoked data by weighted addition or subtraction.

    Each `~mne.Evoked` in ``all_evoked`` should have the same channels and the
    same time instants. Subtraction can be performed by passing
    ``weights=[1, -1]``.

    .. Warning::
        Other than cases like simple subtraction mentioned above (where all
        weights are ``-1`` or ``1``), if you provide numeric weights instead of using
        ``'equal'`` or ``'nave'``, the resulting `~mne.Evoked` object's
        ``.nave`` attribute (which is used to scale noise covariance when
        applying the inverse operator) may not be suitable for inverse imaging.

    Parameters
    ----------
    all_evoked : list of Evoked
        The evoked datasets.
    weights : list of float | ``'equal'`` | ``'nave'``
        The weights to apply to the data of each evoked instance, or a string
        describing the weighting strategy to apply: ``'nave'`` computes
        sum-to-one weights proportional to each object's ``nave`` attribute;
        ``'equal'`` weights each `~mne.Evoked` by ``1 / len(all_evoked)``.

    Returns
    -------
    evoked : Evoked
        The new evoked data.

    Notes
    -----
    .. versionadded:: 0.9.0
    """
    naves = np.array([evk.nave for evk in all_evoked], float)
    if isinstance(weights, str):
        _check_option("weights", weights, ["nave", "equal"])
        if weights == "nave":
            weights_arr = naves / naves.sum()
        else:
            weights_arr = np.ones_like(naves) / len(naves)
    else:
        weights_arr = np.array(weights, float)

    if weights_arr.ndim != 1 or weights_arr.size != len(all_evoked):
        raise ValueError("weights must be the same size as all_evoked")

    # cf. https://en.wikipedia.org/wiki/Weighted_arithmetic_mean, section on
    # "weighted sample variance". The variance of a weighted sample mean is:
    #
    #    σ² = w₁² σ₁² + w₂² σ₂² + ... + wₙ² σₙ²
    #
    # We estimate the variance of each evoked instance as 1 / nave to get:
    #
    #    σ² = w₁² / nave₁ + w₂² / nave₂ + ... + wₙ² / naveₙ
    #
    # And our resulting nave is the reciprocal of this:
    new_nave = 1.0 / np.sum(weights_arr**2 / naves)
    # This general formula is equivalent to formulae in Matti's manual
    # (pp 128-129), where:
    # new_nave = sum(naves) when weights='nave' and
    # new_nave = 1. / sum(1. / naves) when weights are all 1.

    all_evoked = _check_evokeds_ch_names_times(all_evoked)
    evoked = all_evoked[0].copy()

    # use union of bad channels
    bads = list(set(b for e in all_evoked for b in e.info["bads"]))
    evoked.info["bads"] = bads
    evoked.data = sum(w * e.data for w, e in zip(weights_arr, all_evoked))
    evoked.nave = new_nave

    comment = ""
    for idx, (w, e) in enumerate(zip(weights_arr, all_evoked)):
        # pick sign
        sign = "" if w >= 0 else "-"
        # format weight
        weight = "" if np.isclose(abs(w), 1.0) else f"{abs(w):0.3f}"
        # format multiplier
        multiplier = " × " if weight else ""
        # format comment
        if e.comment is not None and " + " in e.comment:  # multiple conditions
            this_comment = f"({e.comment})"
        else:
            this_comment = f"{e.comment or 'unknown'}"
        # assemble everything
        if idx == 0:
            comment += f"{sign}{weight}{multiplier}{this_comment}"
        else:
            comment += f" {sign or '+'} {weight}{multiplier}{this_comment}"
    # special-case: combine_evoked([e1, -e2], [1, -1])
    evoked.comment = comment.replace(" - - ", " + ")
    return evoked


@verbose_static("baseline_evoked")
def read_evokeds(
    fname: Path | str,
    condition: int | str | list[int] | list[str] | None = None,
    baseline: tuple[float | None, float | None] | None = None,
    kind: str = "average",
    proj: bool = True,
    allow_maxshield: bool | str = False,
    verbose: bool | str | int | None = None,
) -> list[Evoked] | Evoked:
    """Read evoked dataset(s).

    Parameters
    ----------
    fname : path-like
        The filename, which should end with ``-ave.fif`` or ``-ave.fif.gz``.
    condition : int | str | list of int | list of str | None
        The index or list of indices of the evoked dataset to read. FIF files
        can contain multiple datasets. If None, all datasets are returned as a
        list.
    baseline : None | tuple of length 2
        The time interval to consider as "baseline" when applying baseline
        correction. If ``None``, do not apply baseline correction.
        If a tuple ``(a, b)``, the interval is between ``a`` and ``b``
        (in seconds), including the endpoints.
        If ``a`` is ``None``, the **beginning** of the data is used; and if ``b``
        is ``None``, it is set to the **end** of the data.
        If ``(None, None)``, the entire time interval is used.

        .. note::
            The baseline ``(a, b)`` includes both endpoints, i.e. all timepoints
            ``t`` such that ``a <= t <= b``.

        Correction is applied **to each channel individually** in the following
        way:

        1. Calculate the mean signal of the baseline period.
        2. Subtract this mean from the **entire** ``Evoked``.

        If ``None`` (default), do not apply baseline correction.

        .. note:: Note that if the read  `~mne.Evoked` objects have already
                  been baseline-corrected, the data retrieved from disk will
                  **always** be baseline-corrected (in fact, only the
                  baseline-corrected version of the data will be saved, so
                  there is no way to undo this procedure). Only **after** the
                  data has been loaded, a custom (additional) baseline
                  correction **may** be optionally applied by passing a tuple
                  here. Passing ``None`` will **not** remove an existing
                  baseline correction, but merely omit the optional, additional
                  baseline correction.
    kind : str
        Either ``'average'`` or ``'standard_error'``, the type of data to read.
    proj : bool
        If False, available projectors won't be applied to the data.
    allow_maxshield : bool | str
        If True, allow loading of data that has been recorded with internal
        active compensation (MaxShield). Data recorded with MaxShield should
        generally not be loaded directly, but should first be processed using
        SSS/tSSS to remove the compensation signals that may also affect brain
        activity. Can also be ``"yes"`` to load without eliciting a warning.
    verbose : bool | str | int | None
        Control verbosity of the logging output. If ``None``, use the default
        verbosity level. See the :ref:`logging documentation <tut-logging>` and
        :func:`mne.verbose` for details. Should only be passed as a keyword
        argument.

    Returns
    -------
    evoked : Evoked or list of Evoked
        The evoked dataset(s); one `~mne.Evoked` if ``condition`` is an
        integer or string; or a list of `~mne.Evoked` if ``condition`` is
        ``None`` or a list.

    See Also
    --------
    write_evokeds

    Notes
    -----
    .. versionchanged:: 0.23
        If the read `~mne.Evoked` objects had been baseline-corrected before
        saving, this will be reflected in their ``baseline`` attribute after
        reading.
    """
    fname = _check_fname(fname, overwrite="read", must_exist=True)
    check_fname(fname, "evoked", ("-ave.fif", "-ave.fif.gz", "_ave.fif", "_ave.fif.gz"))
    logger.info(f"Reading {fname} ...")
    return_list = True
    conditions: range | list
    if condition is None:
        evoked_node = _get_evoked_node(fname)
        conditions = range(len(evoked_node))
    elif not isinstance(condition, list):
        conditions = [condition]
        return_list = False
    else:
        conditions = condition

    out = []
    for c in conditions:
        evoked = Evoked(
            fname,
            c,
            kind=kind,
            proj=proj,
            allow_maxshield=allow_maxshield,
            verbose=verbose,
        )
        if baseline is None and evoked.baseline is None:
            logger.info(_log_rescale(None))
        elif baseline is None and evoked.baseline is not None:
            # Don't touch an existing baseline
            bmin, bmax = evoked.baseline
            logger.info(
                f"Loaded Evoked data is baseline-corrected "
                f"(baseline: [{bmin:g}, {bmax:g}] s)"
            )
        else:
            evoked.apply_baseline(baseline)
        out.append(evoked)

    return out if return_list else out[0]


def _read_evoked(fname, condition=None, kind="average", allow_maxshield=False):
    """Read evoked data from a FIF file."""
    if fname is None:
        raise ValueError("No evoked filename specified")

    f, tree, _ = fiff_open(fname)
    with f as fid:
        #   Read the measurement info
        info, meas = read_meas_info(fid, tree, clean_bads=True)

        #   Locate the data of interest
        processed = dir_tree_find(meas, FIFF.FIFFB_PROCESSED_DATA)
        if len(processed) == 0:
            raise ValueError("Could not find processed data")

        evoked_node = dir_tree_find(meas, FIFF.FIFFB_EVOKED)
        if len(evoked_node) == 0:
            raise ValueError("Could not find evoked data")

        # find string-based entry
        if isinstance(condition, str):
            if kind not in _aspect_dict.keys():
                raise ValueError('kind must be "average" or "standard_error"')

            comments, aspect_kinds, t = _get_entries(fid, evoked_node, allow_maxshield)
            goods = np.isin(comments, [condition]) & np.isin(
                aspect_kinds, [_aspect_dict[kind]]
            )
            found_cond = np.where(goods)[0]
            if len(found_cond) != 1:
                raise ValueError(
                    f'condition "{condition}" ({kind}) not found, out of found '
                    f"datasets:\n{t}"
                )
            condition = found_cond[0]
        elif condition is None:
            if len(evoked_node) > 1:
                _, _, conditions = _get_entries(fid, evoked_node, allow_maxshield)
                raise TypeError(
                    "Evoked file has more than one condition, the condition parameters "
                    f"must be specified from:\n{conditions}"
                )
            else:
                condition = 0

        if condition >= len(evoked_node) or condition < 0:
            raise ValueError("Data set selector out of range")

        my_evoked = evoked_node[condition]

        # Identify the aspects
        with info._unlock():
            my_aspect, info["maxshield"] = _get_aspect(my_evoked, allow_maxshield)

        # Now find the data in the evoked block
        nchan = 0
        sfreq = -1
        chs = []
        baseline = bmin = bmax = None
        comment = last = first = first_time = nsamp = None
        for k in range(my_evoked["nent"]):
            my_kind = my_evoked["directory"][k].kind
            pos = my_evoked["directory"][k].pos
            if my_kind == FIFF.FIFF_COMMENT:
                tag = read_tag(fid, pos)
                comment = tag.data
            elif my_kind == FIFF.FIFF_FIRST_SAMPLE:
                tag = read_tag(fid, pos)
                first = int(tag.data.item())
            elif my_kind == FIFF.FIFF_LAST_SAMPLE:
                tag = read_tag(fid, pos)
                last = int(tag.data.item())
            elif my_kind == FIFF.FIFF_NCHAN:
                tag = read_tag(fid, pos)
                nchan = int(tag.data.item())
            elif my_kind == FIFF.FIFF_SFREQ:
                tag = read_tag(fid, pos)
                sfreq = float(tag.data.item())
            elif my_kind == FIFF.FIFF_CH_INFO:
                tag = read_tag(fid, pos)
                chs.append(tag.data)
            elif my_kind == FIFF.FIFF_FIRST_TIME:
                tag = read_tag(fid, pos)
                first_time = float(tag.data.item())
            elif my_kind == FIFF.FIFF_NO_SAMPLES:
                tag = read_tag(fid, pos)
                nsamp = int(tag.data.item())
            elif my_kind == FIFF.FIFF_MNE_BASELINE_MIN:
                tag = read_tag(fid, pos)
                bmin = float(tag.data.item())
            elif my_kind == FIFF.FIFF_MNE_BASELINE_MAX:
                tag = read_tag(fid, pos)
                bmax = float(tag.data.item())

        if comment is None:
            comment = "No comment"

        if bmin is not None or bmax is not None:
            # None's should've been replaced with floats
            assert bmin is not None and bmax is not None
            baseline = (bmin, bmax)

        #   Local channel information?
        if nchan > 0:
            if chs is None:
                raise ValueError(
                    "Local channel information was not found when it was expected."
                )

            if len(chs) != nchan:
                raise ValueError(
                    "Number of channels and number of channel definitions are different"
                )

            ch_names_mapping = _read_extended_ch_info(chs, my_evoked, fid)
            info["chs"] = chs
            info["bads"][:] = _rename_list(info["bads"], ch_names_mapping)
            logger.info(
                f"    Found channel information in evoked data. nchan = {nchan}"
            )
            if sfreq > 0:
                info["sfreq"] = sfreq

        # Read the data in the aspect block
        nave = 1
        epoch = []
        for k in range(my_aspect["nent"]):
            kind = my_aspect["directory"][k].kind
            pos = my_aspect["directory"][k].pos
            if kind == FIFF.FIFF_COMMENT:
                tag = read_tag(fid, pos)
                comment = tag.data
            elif kind == FIFF.FIFF_ASPECT_KIND:
                tag = read_tag(fid, pos)
                aspect_kind = int(tag.data.item())
            elif kind == FIFF.FIFF_NAVE:
                tag = read_tag(fid, pos)
                nave = int(tag.data.item())
            elif kind == FIFF.FIFF_EPOCH:
                tag = read_tag(fid, pos)
                epoch.append(tag)

        nepoch = len(epoch)
        if nepoch != 1 and nepoch != info["nchan"]:
            raise ValueError(
                "Number of epoch tags is unreasonable "
                f"(nepoch = {nepoch} nchan = {info['nchan']})"
            )

        if nepoch == 1:
            # Only one epoch
            data = epoch[0].data
            # May need a transpose if the number of channels is one
            if data.shape[1] == 1 and info["nchan"] == 1:
                data = data.T
        else:
            # Put the old style epochs together
            data = np.concatenate([e.data[None, :] for e in epoch], axis=0)
        if np.isrealobj(data):
            data = data.astype(np.float64)
        else:
            data = data.astype(np.complex128)

        if first_time is not None and nsamp is not None:
            times = first_time + np.arange(nsamp) / info["sfreq"]
        elif first is not None:
            assert last is not None  # always read together with first
            nsamp = last - first + 1
            times = np.arange(first, last + 1) / info["sfreq"]
        else:
            raise RuntimeError("Could not read time parameters")
        del first, last
        if nsamp is not None and data.shape[1] != nsamp:
            raise ValueError(
                f"Incorrect number of samples ({data.shape[1]} instead of {nsamp})"
            )
        logger.info("    Found the data of interest:")
        logger.info(
            f"        t = {1000 * times[0]:10.2f} ... {1000 * times[-1]:10.2f} ms ("
            f"{comment})"
        )
        if info["comps"] is not None:
            logger.info(
                f"        {len(info['comps'])} CTF compensation matrices available"
            )
        logger.info(f"        nave = {nave} - aspect type = {aspect_kind}")

    # Calibrate
    cals = np.array(
        [
            info["chs"][k]["cal"] * info["chs"][k].get("scale", 1.0)
            for k in range(info["nchan"])
        ]
    )
    data *= cals[:, np.newaxis]

    return info, nave, aspect_kind, comment, times, data, baseline


@verbose_static("on_mismatch_info", "overwrite")
def write_evokeds(
    fname: Path | str,
    evoked: Evoked | list[Evoked],
    *,
    on_mismatch: Literal["raise", "warn", "ignore"] = "raise",
    overwrite: bool = False,
    verbose: bool | str | int | None = None,
) -> None:
    """Write an evoked dataset to a file.

    Parameters
    ----------
    fname : path-like
        The file name, which should end with ``-ave.fif`` or ``-ave.fif.gz``.
    evoked : Evoked instance, or list of Evoked instances
        The evoked dataset, or list of evoked datasets, to save in one file.
        Note that the measurement info from the first evoked instance is used,
        so be sure that information matches.
    on_mismatch : 'raise' | 'warn' | 'ignore'
        Can be ``'raise'`` (default) to raise an error, ``'warn'`` to emit a
        warning, or ``'ignore'`` to ignore
        when the device-to-head transformation differs between
        instances.

        .. versionadded:: 0.24
    overwrite : bool
        If True (default False), overwrite the destination file if it
        exists.

        .. versionadded:: 1.0
    verbose : bool | str | int | None
        Control verbosity of the logging output. If ``None``, use the default
        verbosity level. See the :ref:`logging documentation <tut-logging>` and
        :func:`mne.verbose` for details. Should only be passed as a keyword
        argument.

        .. versionadded:: 0.24

    See Also
    --------
    read_evokeds

    Notes
    -----
    .. versionchanged:: 0.23
        Information on baseline correction will be stored with each individual
        `~mne.Evoked` object, and will be restored when reading the data again
        via `mne.read_evokeds`.
    """
    _write_evokeds(fname, evoked, on_mismatch=on_mismatch, overwrite=overwrite)


def _write_evokeds(fname, evoked, check=True, *, on_mismatch="raise", overwrite=False):
    """Write evoked data."""
    from .dipole import DipoleFixed  # avoid circular import

    fname = _check_fname(fname=fname, overwrite=overwrite)
    if check:
        check_fname(
            fname, "evoked", ("-ave.fif", "-ave.fif.gz", "_ave.fif", "_ave.fif.gz")
        )

    if not isinstance(evoked, list | tuple):
        evoked = [evoked]
    if not len(evoked):
        raise ValueError("No evoked data to write")

    warned = False
    # Create the file and save the essentials
    with start_and_end_file(fname) as fid:
        start_block(fid, FIFF.FIFFB_MEAS)
        write_id(fid, FIFF.FIFF_BLOCK_ID)
        if evoked[0].info["meas_id"] is not None:
            write_id(fid, FIFF.FIFF_PARENT_BLOCK_ID, evoked[0].info["meas_id"])

        # Write measurement info
        write_meas_info(fid, evoked[0].info)

        # One or more evoked data sets
        start_block(fid, FIFF.FIFFB_PROCESSED_DATA)
        for ei, e in enumerate(evoked):
            if ei:
                _ensure_infos_match(
                    info1=evoked[0].info,
                    info2=e.info,
                    name=f"evoked[{ei}]",
                    on_mismatch=on_mismatch,
                )
            start_block(fid, FIFF.FIFFB_EVOKED)

            # Comment is optional
            if e.comment is not None and len(e.comment) > 0:
                write_string(fid, FIFF.FIFF_COMMENT, e.comment)

            # First time, num. samples, first and last sample
            write_float(fid, FIFF.FIFF_FIRST_TIME, e.times[0])
            write_int(fid, FIFF.FIFF_NO_SAMPLES, len(e.times))
            write_int(fid, FIFF.FIFF_FIRST_SAMPLE, e.first)
            write_int(fid, FIFF.FIFF_LAST_SAMPLE, e.last)

            # Baseline
            if not isinstance(e, DipoleFixed) and e.baseline is not None:
                bmin, bmax = e.baseline
                write_float(fid, FIFF.FIFF_MNE_BASELINE_MIN, bmin)
                write_float(fid, FIFF.FIFF_MNE_BASELINE_MAX, bmax)

            # The evoked data itself
            if e.info.get("maxshield"):
                aspect = FIFF.FIFFB_IAS_ASPECT
            else:
                aspect = FIFF.FIFFB_ASPECT
            start_block(fid, aspect)

            write_int(fid, FIFF.FIFF_ASPECT_KIND, e._aspect_kind)
            # convert nave to integer to comply with FIFF spec
            nave_int = int(round(e.nave))
            if nave_int != e.nave and not warned:
                warn(
                    'converting "nave" to integer before saving evoked; this '
                    "can have a minor effect on the scale of source "
                    'estimates that are computed using "nave".'
                )
                warned = True
            write_int(fid, FIFF.FIFF_NAVE, nave_int)
            del nave_int

            decal = np.zeros((e.info["nchan"], 1))
            for k in range(e.info["nchan"]):
                decal[k] = 1.0 / (
                    e.info["chs"][k]["cal"] * e.info["chs"][k].get("scale", 1.0)
                )

            if np.iscomplexobj(e.data):
                write_function = write_complex_float_matrix
            else:
                write_function = write_float_matrix

            write_function(fid, FIFF.FIFF_EPOCH, decal * e.data)
            end_block(fid, aspect)
            end_block(fid, FIFF.FIFFB_EVOKED)

        end_block(fid, FIFF.FIFFB_PROCESSED_DATA)
        end_block(fid, FIFF.FIFFB_MEAS)


def _get_peak(data, times, tmin=None, tmax=None, mode="abs", *, strict=True):
    """Get feature-index and time of maximum signal from 2D array.

    Note. This is a 'getter', not a 'finder'. For non-evoked type
    data and continuous signals, please use proper peak detection algorithms.

    Parameters
    ----------
    data : instance of numpy.ndarray (n_locations, n_times)
        The data, either evoked in sensor or source space.
    times : instance of numpy.ndarray (n_times)
        The times in seconds.
    tmin : float | None
        The minimum point in time to be considered for peak getting.
    tmax : float | None
        The maximum point in time to be considered for peak getting.
    mode : {'pos', 'neg', 'abs'}
        How to deal with the sign of the data. If 'pos' only positive
        values will be considered. If 'neg' only negative values will
        be considered. If 'abs' absolute values will be considered.
        Defaults to 'abs'.
    strict : bool
        If True, raise an error if values are all positive when detecting
        a minimum (mode='neg'), or all negative when detecting a maximum
        (mode='pos'). Defaults to True.

    Returns
    -------
    max_loc : int
        The index of the feature with the maximum value.
    max_time : int
        The time point of the maximum response, index.
    max_amp : float
        Amplitude of the maximum response.
    """
    _check_option("mode", mode, ["abs", "neg", "pos"])

    if tmin is None:
        tmin = times[0]
    if tmax is None:
        tmax = times[-1]

    if tmin < times.min() or tmax > times.max():
        if tmin < times.min():
            param_name = "tmin"
            param_val = tmin
        else:
            param_name = "tmax"
            param_val = tmax

        raise ValueError(
            f"{param_name} ({param_val}) is out of bounds. It must be "
            f"between {times.min()} and {times.max()}"
        )
    elif tmin > tmax:
        raise ValueError(f"tmin ({tmin}) must be <= tmax ({tmax})")

    time_win = (times >= tmin) & (times <= tmax)
    mask = np.ones_like(data).astype(bool)
    mask[:, time_win] = False

    maxfun = np.argmax
    if mode == "pos":
        if strict and not np.any(data[~mask] > 0):
            raise ValueError(
                "No positive values encountered. Cannot operate in pos mode."
            )
    elif mode == "neg":
        if strict and not np.any(data[~mask] < 0):
            raise ValueError(
                "No negative values encountered. Cannot operate in neg mode."
            )
        maxfun = np.argmin

    masked_index = np.ma.array(np.abs(data) if mode == "abs" else data, mask=mask)

    max_loc, max_time = np.unravel_index(maxfun(masked_index), data.shape)

    return max_loc, max_time, data[max_loc, max_time]
