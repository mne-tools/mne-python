# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from __future__ import annotations  # only needed for Python ≤ 3.9

from copy import deepcopy
from inspect import getfullargspec
from pathlib import Path

import numpy as np

from ._fiff.constants import FIFF
from ._fiff.meas_info import (
    ContainsMixin,
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
from .channels.layout import _merge_ch_data, _pair_grad_sensors
from .defaults import _BORDER_DEFAULT, _EXTRAPOLATE_DEFAULT, _INTERPOLATION_DEFAULT
from .filter import FilterMixin, _check_fun, detrend
from .html_templates import _get_html_template
from .parallel import parallel_func
from .time_frequency.spectrum import Spectrum, SpectrumMixin, _validate_method
from .time_frequency.tfr import AverageTFR
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
    check_fname,
    copy_function_doc_to_method_doc,
    fill_doc,
    logger,
    repr_html,
    sizeof_fmt,
    verbose,
    warn,
)
from .viz import (
    plot_evoked,
    plot_evoked_field,
    plot_evoked_image,
    plot_evoked_topo,
    plot_evoked_topomap,
)
from .viz.evoked import plot_evoked_joint, plot_evoked_white
from .viz.topomap import _topomap_animation

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


@fill_doc
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
    fname : path-like
        Name of evoked/average FIF file to load.
        If None no data is loaded.
    condition : int, or str
        Dataset ID number (int) or comment/name (str). Optional if there is
        only one data set in file.
    proj : bool, optional
        Apply SSP projection vectors.
    kind : str
        Either ``'average'`` or ``'standard_error'``. The type of data to read.
        Only used if 'condition' is a str.
    allow_maxshield : bool | str (default False)
        If True, allow loading of data that has been recorded with internal
        active compensation (MaxShield). Data recorded with MaxShield should
        generally not be loaded directly, but should first be processed using
        SSS/tSSS to remove the compensation signals that may also affect brain
        activity. Can also be ``"yes"`` to load without eliciting a warning.
    %(verbose)s

    Attributes
    ----------
    %(info_not_none)s
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

    @verbose
    def __init__(
        self,
        fname,
        condition=None,
        proj=True,
        kind="average",
        allow_maxshield=False,
        *,
        verbose=None,
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
    def filename(self, value):
        self._filename = Path(value) if value is not None else value

    @property
    def kind(self):
        """The data kind."""
        return _aspect_rev[self._aspect_kind]

    @kind.setter
    def kind(self, kind):
        _check_option("kind", kind, list(_aspect_dict.keys()))
        self._aspect_kind = _aspect_dict[kind]

    @property
    def data(self):
        """The data matrix."""
        return self._data

    @data.setter
    def data(self, data):
        """Set the data matrix."""
        self._data = data

    @fill_doc
    def get_data(self, picks=None, units=None, tmin=None, tmax=None):
        """Get evoked data as 2D array.

        Parameters
        ----------
        %(picks_all)s
        %(units)s
        tmin : float | None
            Start time of data to get in seconds.
        tmax : float | None
            End time of data to get in seconds.

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

        picks = _picks_to_idx(self.info, picks, "all", exclude=())

        start, stop = self._handle_tmin_tmax(tmin, tmax)

        data = self.data[picks, start:stop]

        if units is not None:
            ch_factors = _get_ch_factors(self, units, picks)
            data *= ch_factors[:, np.newaxis]

        return data

    @verbose
    def apply_function(
        self,
        fun,
        picks=None,
        dtype=None,
        n_jobs=None,
        channel_wise=True,
        *,
        verbose=None,
        **kwargs,
    ):
        """Apply a function to a subset of channels.

        %(applyfun_summary_evoked)s

        Parameters
        ----------
        %(fun_applyfun_evoked)s
        %(picks_all_data_noref)s
        %(dtype_applyfun)s
        %(n_jobs)s Ignored if ``channel_wise=False`` as the workload
            is split across channels.
        %(channel_wise_applyfun)s

            .. versionadded:: 1.6
        %(verbose)s
        %(kwargs_fun)s

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

    @verbose
    def apply_baseline(self, baseline=(None, 0), *, verbose=None):
        """Baseline correct evoked data.

        Parameters
        ----------
        %(baseline_evoked)s
            Defaults to ``(None, 0)``, i.e. beginning of the the data until
            time point zero.
        %(verbose)s

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

    @verbose
    def save(self, fname, *, overwrite=False, verbose=None):
        """Save evoked data to a file.

        Parameters
        ----------
        fname : path-like
            The name of the file, which should end with ``-ave.fif(.gz)`` or
            ``_ave.fif(.gz)``.
        %(overwrite)s
        %(verbose)s

        Notes
        -----
        To write multiple conditions into a single file, use
        `mne.write_evokeds`.

        .. versionchanged:: 0.23
            Information on baseline correction will be stored with the data,
            and will be restored when reading again via `mne.read_evokeds`.
        """
        write_evokeds(fname, self, overwrite=overwrite)

    @verbose
    def export(self, fname, fmt="auto", *, overwrite=False, verbose=None):
        """Export Evoked to external formats.

        %(export_fmt_support_evoked)s

        %(export_warning)s

        Parameters
        ----------
        %(fname_export_params)s
        %(export_fmt_params_evoked)s
        %(overwrite)s
        %(verbose)s

        Notes
        -----
        .. versionadded:: 1.1

        %(export_warning_note_evoked)s
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
        t = t.render(
            inst=self,
            filenames=(
                [Path(self.filename).name]
                if getattr(self, "filename", None) is not None
                else None
            ),
        )
        return t

    @property
    def ch_names(self):
        """Channel names."""
        return self.info["ch_names"]

    @copy_function_doc_to_method_doc(plot_evoked)
    def plot(
        self,
        picks=None,
        exclude="bads",
        unit=True,
        show=True,
        ylim=None,
        xlim="tight",
        proj=False,
        hline=None,
        units=None,
        scalings=None,
        titles=None,
        axes=None,
        gfp=False,
        window_title=None,
        spatial_colors="auto",
        zorder="unsorted",
        selectable=True,
        noise_cov=None,
        time_unit="s",
        sphere=None,
        *,
        highlight=None,
        verbose=None,
    ):
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

    @copy_function_doc_to_method_doc(plot_evoked_image)
    def plot_image(
        self,
        picks=None,
        exclude="bads",
        unit=True,
        show=True,
        clim=None,
        xlim="tight",
        proj=False,
        units=None,
        scalings=None,
        titles=None,
        axes=None,
        cmap="RdBu_r",
        colorbar=True,
        mask=None,
        mask_style=None,
        mask_cmap="Greys",
        mask_alpha=0.25,
        time_unit="s",
        show_names=None,
        group_by=None,
        sphere=None,
    ):
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

    @copy_function_doc_to_method_doc(plot_evoked_topo)
    def plot_topo(
        self,
        layout=None,
        layout_scale=0.945,
        color=None,
        border="none",
        ylim=None,
        scalings=None,
        title=None,
        proj=False,
        vline=(0.0,),
        fig_background=None,
        merge_grads=False,
        legend=True,
        axes=None,
        background_color="w",
        noise_cov=None,
        exclude="bads",
        select=False,
        show=True,
    ):
        """.

        Notes
        -----
        .. versionadded:: 0.10.0
        """
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

    @copy_function_doc_to_method_doc(plot_evoked_topomap)
    def plot_topomap(
        self,
        times="auto",
        *,
        average=None,
        ch_type=None,
        scalings=None,
        proj=False,
        sensors=True,
        show_names=False,
        mask=None,
        mask_params=None,
        contours=6,
        outlines="head",
        sphere=None,
        image_interp=_INTERPOLATION_DEFAULT,
        extrapolate=_EXTRAPOLATE_DEFAULT,
        border=_BORDER_DEFAULT,
        res=64,
        size=1,
        cmap=None,
        vlim=(None, None),
        cnorm=None,
        colorbar=True,
        cbar_fmt="%3.1f",
        units=None,
        axes=None,
        time_unit="s",
        time_format=None,
        nrows=1,
        ncols="auto",
        show=True,
    ):
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

    @copy_function_doc_to_method_doc(plot_evoked_field)
    def plot_field(
        self,
        surf_maps,
        time=None,
        time_label="t = %0.0f ms",
        n_jobs=None,
        fig=None,
        vmax=None,
        n_contours=21,
        *,
        show_density=True,
        alpha=None,
        interpolation="nearest",
        interaction="terrain",
        time_viewer="auto",
        verbose=None,
    ):
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

    @copy_function_doc_to_method_doc(plot_evoked_white)
    def plot_white(
        self,
        noise_cov,
        show=True,
        rank=None,
        time_unit="s",
        sphere=None,
        axes=None,
        *,
        spatial_colors="auto",
        verbose=None,
    ):
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

    @copy_function_doc_to_method_doc(plot_evoked_joint)
    def plot_joint(
        self,
        times="peaks",
        title="",
        picks=None,
        exclude="bads",
        show=True,
        ts_args=None,
        topomap_args=None,
    ):
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

    @fill_doc
    def animate_topomap(
        self,
        ch_type=None,
        times=None,
        frame_rate=None,
        butterfly=False,
        blit=True,
        show=True,
        time_unit="s",
        sphere=None,
        *,
        image_interp=_INTERPOLATION_DEFAULT,
        extrapolate=_EXTRAPOLATE_DEFAULT,
        vmin=None,
        vmax=None,
        verbose=None,
    ):
        """Make animation of evoked data as topomap timeseries.

        The animation can be paused/resumed with left mouse button.
        Left and right arrow keys can be used to move backward or forward
        in time.

        Parameters
        ----------
        ch_type : str | None
            Channel type to plot. Accepted data types: 'mag', 'grad', 'eeg',
            'hbo', 'hbr', 'fnirs_cw_amplitude',
            'fnirs_fd_ac_amplitude', 'fnirs_fd_phase', and 'fnirs_od'.
            If None, first available channel type from the above list is used.
            Defaults to None.
        times : array of float | None
            The time points to plot. If None, 10 evenly spaced samples are
            calculated over the evoked time series. Defaults to None.
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
        time_unit : str
            The units for the time axis, can be "ms" (default in 0.16)
            or "s" (will become the default in 0.17).

            .. versionadded:: 0.16
        %(sphere_topomap_auto)s
        %(image_interp_topomap)s
        %(extrapolate_topomap)s

            .. versionadded:: 0.22
        %(vmin_vmax_topomap)s

            .. versionadded:: 1.1.0
        %(verbose)s

        Returns
        -------
        fig : instance of matplotlib.figure.Figure
            The figure.
        anim : instance of matplotlib.animation.FuncAnimation
            Animation of the topomap.

        Notes
        -----
        .. versionadded:: 0.12.0
        """
        return _topomap_animation(
            self,
            ch_type=ch_type,
            times=times,
            frame_rate=frame_rate,
            butterfly=butterfly,
            blit=blit,
            show=show,
            time_unit=time_unit,
            sphere=sphere,
            image_interp=image_interp,
            extrapolate=extrapolate,
            vmin=vmin,
            vmax=vmax,
            verbose=verbose,
        )

    def as_type(self, ch_type="grad", mode="fast"):
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

    @fill_doc
    def detrend(self, order=1, picks=None):
        """Detrend data.

        This function operates in-place.

        Parameters
        ----------
        order : int
            Either 0 or 1, the order of the detrending. 0 is a constant
            (DC) detrend, 1 is a linear detrend.
        %(picks_good_data)s

        Returns
        -------
        evoked : instance of Evoked
            The detrended evoked object.
        """
        picks = _picks_to_idx(self.info, picks)
        self.data[picks] = detrend(self.data[picks], order, axis=-1)
        return self

    def copy(self):
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
        ch_type=None,
        tmin=None,
        tmax=None,
        mode="abs",
        time_as_index=False,
        merge_grads=False,
        return_amplitude=False,
        *,
        strict=True,
    ):
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

    @verbose
    def compute_psd(
        self,
        method="multitaper",
        fmin=0,
        fmax=np.inf,
        tmin=None,
        tmax=None,
        picks=None,
        proj=False,
        remove_dc=True,
        exclude=(),
        *,
        n_jobs=1,
        verbose=None,
        **method_kw,
    ):
        """Perform spectral analysis on sensor data.

        Parameters
        ----------
        %(method_psd)s
            Default is ``'multitaper'``.
        %(fmin_fmax_psd)s
        %(tmin_tmax_psd)s
        %(picks_good_data_noref)s
        %(proj_psd)s
        %(remove_dc)s
        %(exclude_psd)s
        %(n_jobs)s
        %(verbose)s
        %(method_kw_psd)s

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

    @verbose
    def compute_tfr(
        self,
        method,
        freqs,
        *,
        tmin=None,
        tmax=None,
        picks=None,
        proj=False,
        output="power",
        decim=1,
        n_jobs=None,
        verbose=None,
        **method_kw,
    ):
        """Compute a time-frequency representation of evoked data.

        Parameters
        ----------
        %(method_tfr)s
        %(freqs_tfr)s
        %(tmin_tmax_psd)s
        %(picks_good_data_noref)s
        %(proj_psd)s
        %(output_compute_tfr)s
        %(decim_tfr)s
        %(n_jobs)s
        %(verbose)s
        %(method_kw_tfr)s

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

    @verbose
    def plot_psd(
        self,
        fmin=0,
        fmax=np.inf,
        tmin=None,
        tmax=None,
        picks=None,
        proj=False,
        *,
        method="auto",
        average=False,
        dB=True,
        estimate="power",
        xscale="linear",
        area_mode="std",
        area_alpha=0.33,
        color="black",
        line_alpha=None,
        spatial_colors=True,
        sphere=None,
        exclude="bads",
        ax=None,
        show=True,
        n_jobs=1,
        verbose=None,
        **method_kw,
    ):
        """%(plot_psd_doc)s.

        Parameters
        ----------
        %(fmin_fmax_psd)s
        %(tmin_tmax_psd)s
        %(picks_good_data_noref)s
        %(proj_psd)s
        %(method_plot_psd_auto)s
        %(average_plot_psd)s
        %(dB_plot_psd)s
        %(estimate_plot_psd)s
        %(xscale_plot_psd)s
        %(area_mode_plot_psd)s
        %(area_alpha_plot_psd)s
        %(color_plot_psd)s
        %(line_alpha_plot_psd)s
        %(spatial_colors_psd)s
        %(sphere_topomap_auto)s

            .. versionadded:: 0.22.0
        exclude : list of str | 'bads'
            Channels names to exclude from being shown. If 'bads', the bad
            channels are excluded. Pass an empty list to plot all channels
            (including channels marked "bad", if any).

            .. versionadded:: 0.24.0
        %(ax_plot_psd)s
        %(show)s
        %(n_jobs)s
        %(verbose)s
        %(method_kw_psd)s

        Returns
        -------
        fig : instance of Figure
            Figure with frequency spectra of the data channels.

        Notes
        -----
        %(notes_plot_psd_meth)s
        """
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

    @verbose
    def to_data_frame(
        self,
        picks=None,
        index=None,
        scalings=None,
        copy=True,
        long_format=False,
        time_format=None,
        *,
        verbose=None,
    ):
        """Export data in tabular structure as a pandas DataFrame.

        Channels are converted to columns in the DataFrame. By default,
        an additional column "time" is added, unless ``index='time'``
        (in which case time values form the DataFrame's index).

        Parameters
        ----------
        %(picks_all)s
        %(index_df_evk)s
            Defaults to ``None``.
        %(scalings_df)s
        %(copy_df)s
        %(long_format_df_raw)s
        %(time_format_df)s

            .. versionadded:: 0.20
        %(verbose)s

        Returns
        -------
        %(df_return)s
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
        times = _convert_times(times, time_format, self.info["meas_date"])
        mindex.append(("time", times))
        # build DataFrame
        df = _build_data_frame(
            self, data, picks, long_format, mindex, index, default_index=["time"]
        )
        return df


@fill_doc
class EvokedArray(Evoked):
    """Evoked object from numpy array.

    Parameters
    ----------
    data : array of shape (n_channels, n_times)
        The channels' evoked response. See notes for proper units of measure.
    %(info_not_none)s Consider using :func:`mne.create_info` to populate this
        structure.
    tmin : float
        Start time before event. Defaults to 0.
    comment : str
        Comment on dataset. Can be the condition. Defaults to ''.
    nave : int
        Number of averaged epochs. Defaults to 1.
    kind : str
        Type of data, either average or standard_error. Defaults to 'average'.
    %(baseline_evoked)s
        Defaults to ``None``, i.e. no baseline correction.

        .. versionadded:: 0.23
    %(verbose)s

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

    @verbose
    def __init__(
        self,
        data,
        info,
        tmin=0.0,
        comment="",
        nave=1,
        kind="average",
        baseline=None,
        *,
        verbose=None,
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


def _check_evokeds_ch_names_times(all_evoked):
    evoked = all_evoked[0]
    ch_names = evoked.ch_names
    for ii, ev in enumerate(all_evoked[1:]):
        if ev.ch_names != ch_names:
            if set(ev.ch_names) != set(ch_names):
                raise ValueError(f"{evoked} and {ev} do not contain the same channels.")
            else:
                warn("Order of channels differs, reordering channels ...")
                ev = ev.copy()
                ev.reorder_channels(ch_names)
                all_evoked[ii + 1] = ev
        if not np.max(np.abs(ev.times - evoked.times)) < 1e-7:
            raise ValueError(f"{evoked} and {ev} do not contain the same time instants")
    return all_evoked


def combine_evoked(all_evoked, weights):
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
            weights = naves / naves.sum()
        else:
            weights = np.ones_like(naves) / len(naves)
    else:
        weights = np.array(weights, float)

    if weights.ndim != 1 or weights.size != len(all_evoked):
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
    new_nave = 1.0 / np.sum(weights**2 / naves)
    # This general formula is equivalent to formulae in Matti's manual
    # (pp 128-129), where:
    # new_nave = sum(naves) when weights='nave' and
    # new_nave = 1. / sum(1. / naves) when weights are all 1.

    all_evoked = _check_evokeds_ch_names_times(all_evoked)
    evoked = all_evoked[0].copy()

    # use union of bad channels
    bads = list(set(b for e in all_evoked for b in e.info["bads"]))
    evoked.info["bads"] = bads
    evoked.data = sum(w * e.data for w, e in zip(weights, all_evoked))
    evoked.nave = new_nave

    comment = ""
    for idx, (w, e) in enumerate(zip(weights, all_evoked)):
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


@verbose
def read_evokeds(
    fname,
    condition=None,
    baseline=None,
    kind="average",
    proj=True,
    allow_maxshield=False,
    verbose=None,
) -> list[Evoked] | Evoked:
    """Read evoked dataset(s).

    Parameters
    ----------
    fname : path-like
        The filename, which should end with ``-ave.fif`` or ``-ave.fif.gz``.
    condition : int or str | list of int or str | None
        The index or list of indices of the evoked dataset to read. FIF files
        can contain multiple datasets. If None, all datasets are returned as a
        list.
    %(baseline_evoked)s
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
    allow_maxshield : bool | str (default False)
        If True, allow loading of data that has been recorded with internal
        active compensation (MaxShield). Data recorded with MaxShield should
        generally not be loaded directly, but should first be processed using
        SSS/tSSS to remove the compensation signals that may also affect brain
        activity. Can also be ``"yes"`` to load without eliciting a warning.
    %(verbose)s

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
    if condition is None:
        evoked_node = _get_evoked_node(fname)
        condition = range(len(evoked_node))
    elif not isinstance(condition, list):
        condition = [condition]
        return_list = False

    out = []
    for c in condition:
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


@verbose
def write_evokeds(fname, evoked, *, on_mismatch="raise", overwrite=False, verbose=None):
    """Write an evoked dataset to a file.

    Parameters
    ----------
    fname : path-like
        The file name, which should end with ``-ave.fif`` or ``-ave.fif.gz``.
    evoked : Evoked instance, or list of Evoked instances
        The evoked dataset, or list of evoked datasets, to save in one file.
        Note that the measurement info from the first evoked instance is used,
        so be sure that information matches.
    %(on_mismatch_info)s
    %(overwrite)s

        .. versionadded:: 1.0
    %(verbose)s

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
