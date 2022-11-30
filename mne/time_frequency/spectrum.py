# -*- coding: utf-8 -*-
"""Container classes for spectral data."""

# Authors: Dan McCloy <dan@mccloy.info>
#
# License: BSD-3-Clause

from copy import deepcopy
from functools import partial
from inspect import signature

import numpy as np

from ..channels.channels import UpdateChannelsMixin, _get_ch_type
from ..channels.layout import _merge_ch_data
from ..defaults import (_BORDER_DEFAULT, _EXTRAPOLATE_DEFAULT,
                        _INTERPOLATION_DEFAULT, _handle_default)
from ..io.meas_info import ContainsMixin
from ..io.pick import _pick_data_channels, _picks_to_idx, pick_info
from ..utils import (GetEpochsMixin, _build_data_frame,
                     _check_pandas_index_arguments, _check_pandas_installed,
                     _check_sphere, _time_mask, _validate_type, fill_doc,
                     legacy, logger, object_diff, repr_html, verbose, warn)
from ..utils.check import (_check_fname, _check_option, _import_h5io_funcs,
                           _is_numeric, check_fname)
from ..utils.misc import _pl
from ..viz.topo import _plot_timeseries, _plot_timeseries_unified, _plot_topo
from ..viz.topomap import (_make_head_outlines, _prepare_topomap_plot,
                           plot_psds_topomap)
from ..viz.utils import (_format_units_psd, _plot_psd, _prepare_sensor_names,
                         plt_show)
from . import psd_array_multitaper, psd_array_welch
from .psd import _check_nfft


def _identity_function(x):
    return x


class SpectrumMixin():
    """Mixin providing spectral plotting methods to sensor-space containers."""

    @legacy(alt='.compute_psd().plot()')
    @verbose
    def plot_psd(self, fmin=0, fmax=np.inf, tmin=None, tmax=None, picks=None,
                 proj=False, reject_by_annotation=True, *, method='auto',
                 average=False, dB=True, estimate='auto', xscale='linear',
                 area_mode='std', area_alpha=0.33, color='black',
                 line_alpha=None, spatial_colors=True, sphere=None,
                 exclude='bads', ax=None, show=True, n_jobs=1, verbose=None,
                 **method_kw):
        """%(plot_psd_doc)s.

        Parameters
        ----------
        %(fmin_fmax_psd)s
        %(tmin_tmax_psd)s
        %(picks_good_data_noref)s
        %(proj_psd)s
        %(reject_by_annotation_psd)s
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
        from ..io import BaseRaw

        method = _validate_method(method, type(self).__name__)
        self._set_legacy_nfft_default(tmin, tmax, method, method_kw)
        # triage reject_by_annotation
        rba = dict()
        if isinstance(self, BaseRaw):
            rba = dict(reject_by_annotation=reject_by_annotation)

        spectrum = self.compute_psd(
            method=method, fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax,
            picks=picks, proj=proj, n_jobs=n_jobs, verbose=verbose, **rba,
            **method_kw)

        # translate kwargs
        amplitude = 'auto' if estimate == 'auto' else (estimate == 'amplitude')
        ci = 'sd' if area_mode == 'std' else area_mode
        # ↓ here picks="all" because we've already restricted the `info` to
        # ↓ have only `picks` channels
        fig = spectrum.plot(
            picks='all', average=average, dB=dB, amplitude=amplitude,
            xscale=xscale, ci=ci, ci_alpha=area_alpha, color=color,
            alpha=line_alpha, spatial_colors=spatial_colors, sphere=sphere,
            exclude=exclude, axes=ax, show=show)
        return fig

    @legacy(alt='.compute_psd().plot_topo()')
    @verbose
    def plot_psd_topo(self, tmin=None, tmax=None, fmin=0, fmax=100, proj=False,
                      *, method='auto', dB=True, layout=None, color='w',
                      fig_facecolor='k', axis_facecolor='k', axes=None,
                      block=False, show=True, n_jobs=None, verbose=None,
                      **method_kw):
        """Plot power spectral density, separately for each channel.

        Parameters
        ----------
        %(tmin_tmax_psd)s
        %(fmin_fmax_psd_topo)s
        %(proj_psd)s
        %(method_plot_psd_auto)s
        %(dB_spectrum_plot_topo)s
        %(layout_spectrum_plot_topo)s
        %(color_spectrum_plot_topo)s
        %(fig_facecolor)s
        %(axis_facecolor)s
        %(axes_spectrum_plot_topo)s
        %(block)s
        %(show)s
        %(n_jobs)s
        %(verbose)s
        %(method_kw_psd)s Defaults to ``dict(n_fft=2048)``.

        Returns
        -------
        fig : instance of matplotlib.figure.Figure
            Figure distributing one image per channel across sensor topography.
        """
        method = _validate_method(method, type(self).__name__)
        self._set_legacy_nfft_default(tmin, tmax, method, method_kw)

        spectrum = self.compute_psd(
            method=method, fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax,
            proj=proj, n_jobs=n_jobs, verbose=verbose, **method_kw)

        return spectrum.plot_topo(
            dB=dB, layout=layout, color=color, fig_facecolor=fig_facecolor,
            axis_facecolor=axis_facecolor, axes=axes, block=block, show=show)

    @legacy(alt='.compute_psd().plot_topomap()')
    @verbose
    def plot_psd_topomap(self, bands=None, tmin=None, tmax=None, ch_type=None,
                         *, proj=False, method='auto', normalize=False,
                         agg_fun=None, dB=False, sensors=True,
                         show_names=False, mask=None, mask_params=None,
                         contours=0, outlines='head', sphere=None,
                         image_interp=_INTERPOLATION_DEFAULT,
                         extrapolate=_EXTRAPOLATE_DEFAULT,
                         border=_BORDER_DEFAULT, res=64, size=1, cmap=None,
                         vlim=(None, None), cnorm=None, colorbar=True,
                         cbar_fmt='auto', units=None, axes=None, show=True,
                         n_jobs=None, verbose=None, **method_kw):
        """Plot scalp topography of PSD for chosen frequency bands.

        Parameters
        ----------
        %(bands_psd_topo)s
        %(tmin_tmax_psd)s
        %(ch_type_topomap_psd)s
        %(proj_psd)s
        %(method_plot_psd_auto)s
        %(normalize_psd_topo)s
        %(agg_fun_psd_topo)s
        %(dB_plot_topomap)s
        %(sensors_topomap)s
        %(show_names_topomap)s
        %(mask_evoked_topomap)s
        %(mask_params_topomap)s
        %(contours_topomap)s
        %(outlines_topomap)s
        %(sphere_topomap_auto)s
        %(image_interp_topomap)s
        %(extrapolate_topomap)s
        %(border_topomap)s
        %(res_topomap)s
        %(size_topomap)s
        %(cmap_topomap)s
        %(vlim_plot_topomap_psd)s
        %(cnorm)s

            .. versionadded:: 1.2
        %(colorbar_topomap)s
        %(cbar_fmt_topomap_psd)s
        %(units_topomap)s
        %(axes_spectrum_plot_topomap)s
        %(show)s
        %(n_jobs)s
        %(verbose)s
        %(method_kw_psd)s

        Returns
        -------
        fig : instance of Figure
            Figure showing one scalp topography per frequency band.
        """
        spectrum = self.compute_psd(
            method=method, tmin=tmin, tmax=tmax, proj=proj,
            n_jobs=n_jobs, verbose=verbose, **method_kw)

        fig = spectrum.plot_topomap(
            bands=bands, ch_type=ch_type, normalize=normalize, agg_fun=agg_fun,
            dB=dB, sensors=sensors, show_names=show_names, mask=mask,
            mask_params=mask_params, contours=contours, outlines=outlines,
            sphere=sphere, image_interp=image_interp, extrapolate=extrapolate,
            border=border, res=res, size=size, cmap=cmap, vlim=vlim,
            cnorm=cnorm, colorbar=colorbar, cbar_fmt=cbar_fmt, units=units,
            axes=axes, show=show)
        return fig

    def _set_legacy_nfft_default(self, tmin, tmax, method, method_kw):
        """Update method_kw with legacy n_fft default for plot_psd[_topo]().

        This method returns ``None`` and has a side effect of (maybe) updating
        the ``method_kw`` dict.
        """
        if method == 'welch' and method_kw.get('n_fft', None) is None:
            tm = _time_mask(self.times, tmin, tmax, sfreq=self.info['sfreq'])
            method_kw['n_fft'] = min(np.sum(tm), 2048)


class BaseSpectrum(ContainsMixin, UpdateChannelsMixin):
    """Base class for Spectrum and EpochsSpectrum."""

    def __init__(self, inst, method, fmin, fmax, tmin, tmax, picks,
                 proj, *, n_jobs, verbose=None, **method_kw):
        # arg checking
        self._sfreq = inst.info['sfreq']
        if np.isfinite(fmax) and (fmax > self.sfreq / 2):
            raise ValueError(
                f'Requested fmax ({fmax} Hz) must not exceed ½ the sampling '
                f'frequency of the data ({0.5 * inst.info["sfreq"]} Hz).')
        # method
        self._inst_type = type(inst)
        method = _validate_method(method, self._get_instance_type_string())

        # triage method and kwargs. partial() doesn't check validity of kwargs,
        # so we do it manually to save compute time if any are invalid.
        psd_funcs = dict(welch=psd_array_welch,
                         multitaper=psd_array_multitaper)
        invalid_ix = np.in1d(list(method_kw),
                             list(signature(psd_funcs[method]).parameters),
                             invert=True)
        if invalid_ix.any():
            invalid_kw = np.array(list(method_kw))[invalid_ix].tolist()
            s = _pl(invalid_kw)
            raise TypeError(
                f'Got unexpected keyword argument{s} {", ".join(invalid_kw)} '
                f'for PSD method "{method}".')
        self._psd_func = partial(psd_funcs[method], **method_kw)

        # apply proj if desired
        if proj:
            inst = inst.copy().apply_proj()
        self.inst = inst

        # prep times and picks
        self._time_mask = _time_mask(inst.times, tmin, tmax, sfreq=self.sfreq)
        self._picks = _picks_to_idx(inst.info, picks, 'data',
                                    with_ref_meg=False)

        # add the info object. bads and non-data channels were dropped by
        # _picks_to_idx() so we update the info accordingly:
        self.info = pick_info(inst.info, sel=self._picks, copy=True)

        # assign some attributes
        self.preload = True  # needed for __getitem__, doesn't mean anything
        self._method = method
        # self._dims may also get updated by child classes
        self._dims = ('channel', 'freq',)
        if method_kw.get('average', '') in (None, False):
            self._dims += ('segment',)
        if method_kw.get('output', '') == 'complex':
            self._dims = self._dims[:-1] + ('taper',) + self._dims[-1:]
        # record data type (for repr and html_repr)
        self._data_type = ('Fourier Coefficients' if 'taper' in self._dims
                           else 'Power Spectrum')

    def __eq__(self, other):
        """Test equivalence of two Spectrum instances."""
        return object_diff(vars(self), vars(other)) == ''

    def __getstate__(self):
        """Prepare object for serialization."""
        inst_type_str = self._get_instance_type_string()
        out = dict(method=self.method,
                   data=self._data,
                   sfreq=self.sfreq,
                   dims=self._dims,
                   freqs=self.freqs,
                   inst_type_str=inst_type_str,
                   data_type=self._data_type,
                   info=self.info)
        return out

    def __setstate__(self, state):
        """Unpack from serialized format."""
        from .. import Epochs, Evoked, Info
        from ..io import Raw

        self._method = state['method']
        self._data = state['data']
        self._freqs = state['freqs']
        self._dims = state['dims']
        self._sfreq = state['sfreq']
        self.info = Info(**state['info'])
        self._data_type = state['data_type']
        self.preload = True
        # instance type
        inst_types = dict(Raw=Raw, Epochs=Epochs, Evoked=Evoked)
        self._inst_type = inst_types[state['inst_type_str']]

    def __repr__(self):
        """Build string representation of the Spectrum object."""
        inst_type_str = self._get_instance_type_string()
        # shape & dimension names
        dims = ' × '.join(
            [f'{dim[0]} {dim[1]}s' for dim in zip(self.shape, self._dims)])
        freq_range = f'{self.freqs[0]:0.1f}-{self.freqs[-1]:0.1f} Hz'
        return (f'<{self._data_type} (from {inst_type_str}, '
                f'{self.method} method) | {dims}, {freq_range}>')

    @repr_html
    def _repr_html_(self, caption=None):
        """Build HTML representation of the Spectrum object."""
        from ..html_templates import repr_templates_env

        inst_type_str = self._get_instance_type_string()
        units = [f'{ch_type}: {unit}'
                 for ch_type, unit in self.units().items()]
        t = repr_templates_env.get_template('spectrum.html.jinja')
        t = t.render(spectrum=self, inst_type=inst_type_str, units=units)
        return t

    def _check_values(self):
        """Check PSD results for correct shape and bad values."""
        assert len(self._dims) == self._data.ndim
        assert self._data.shape == self._shape
        # negative values OK if the spectrum is really fourier coefficients
        if 'taper' in self._dims:
            return
        # TODO: should this be more fine-grained (report "chan X in epoch Y")?
        ch_dim = self._dims.index('channel')
        dims = np.arange(self._data.ndim).tolist()
        dims.pop(ch_dim)
        # take min() across all but the channel axis
        bad_value = self._data.min(axis=tuple(dims)) <= 0
        if bad_value.any():
            chs = np.array(self.ch_names)[bad_value].tolist()
            s = _pl(bad_value.sum())
            warn(f'Zero value in spectrum for channel{s} {", ".join(chs)}',
                 UserWarning)

    def _compute_spectra(self, data, fmin, fmax, n_jobs, method_kw, verbose):
        # make the spectra
        result = self._psd_func(
            data, self.sfreq, fmin=fmin, fmax=fmax, n_jobs=n_jobs,
            verbose=verbose)
        # assign ._data (handling unaggregated multitaper output)
        if method_kw.get('output', '') == 'complex':
            fourier_coefs, freqs, weights = result
            self._data = fourier_coefs
            self._mt_weights = weights
        else:
            psds, freqs = result
            self._data = psds
        # assign properties (._data already assigned above)
        self._freqs = freqs
        # this is *expected* shape, it gets asserted later in _check_values()
        # (and then deleted afterwards)
        self._shape = (len(self.ch_names), len(self.freqs))
        # append n_welch_segments
        if method_kw.get('average', '') in (None, False):
            n_welch_segments = _compute_n_welch_segments(data.shape[-1],
                                                         method_kw)
            self._shape += (n_welch_segments,)
        # insert n_tapers
        if method_kw.get('output', '') == 'complex':
            self._shape = (
                self._shape[:-1] + (self._mt_weights.size,) + self._shape[-1:])
        # we don't need these anymore, and they make save/load harder
        del self._picks
        del self._psd_func
        del self._time_mask

    def _get_instance_type_string(self):
        """Get string representation of the originating instance type."""
        from .. import BaseEpochs, Evoked, EvokedArray
        from ..io import BaseRaw

        parent_classes = self._inst_type.__bases__
        if BaseRaw in parent_classes:
            inst_type_str = 'Raw'
        elif BaseEpochs in parent_classes:
            inst_type_str = 'Epochs'
        elif self._inst_type in (Evoked, EvokedArray):
            inst_type_str = 'Evoked'
        else:
            raise RuntimeError(
                f'Unknown instance type {self._inst_type} in Spectrum')
        return inst_type_str

    @property
    def _detrend_picks(self):
        """Provide compatibility with __iter__."""
        return list()

    @property
    def ch_names(self):
        return self.info['ch_names']

    @property
    def freqs(self):
        return self._freqs

    @property
    def method(self):
        return self._method

    @property
    def sfreq(self):
        return self._sfreq

    @property
    def shape(self):
        return self._data.shape

    def copy(self):
        """Return copy of the Spectrum instance.

        Returns
        -------
        spectrum : instance of Spectrum
            A copy of the object.
        """
        return deepcopy(self)

    @fill_doc
    def get_data(self, picks=None, exclude='bads', fmin=0, fmax=np.inf,
                 return_freqs=False):
        """Get spectrum data in NumPy array format.

        Parameters
        ----------
        %(picks_good_data_noref)s
        %(exclude_spectrum_get_data)s
        %(fmin_fmax_psd)s
        return_freqs : bool
            Whether to return the frequency bin values for the requested
            frequency range. Default is ``False``.

        Returns
        -------
        data : array
            The requested data in a NumPy array.
        freqs : array
            The frequency values for the requested range. Only returned if
            ``return_freqs`` is ``True``.
        """
        picks = _picks_to_idx(self.info, picks, 'data_or_ica', exclude=exclude,
                              with_ref_meg=False)
        fmin_idx = np.searchsorted(self.freqs, fmin)
        fmax_idx = np.searchsorted(self.freqs, fmax, side='right')
        freq_picks = np.arange(fmin_idx, fmax_idx)
        freq_axis = self._dims.index('freq')
        chan_axis = self._dims.index('channel')
        # normally there's a risk of np.take reducing array dimension if there
        # were only one channel or frequency selected, but `_picks_to_idx`
        # always returns an array of picks, and np.arange always returns an
        # array of freq bin indices, so we're safe; the result will always be
        # 2D.
        data = self._data.take(picks, chan_axis).take(freq_picks, freq_axis)
        if return_freqs:
            freqs = self._freqs[fmin_idx:fmax_idx]
            return (data, freqs)
        return data

    @fill_doc
    def plot(self, *, picks=None, average=False, dB=True, amplitude='auto',
             xscale='linear', ci='sd', ci_alpha=0.3, color='black', alpha=None,
             spatial_colors=True, sphere=None, exclude='bads', axes=None,
             show=True):
        """%(plot_psd_doc)s.

        Parameters
        ----------
        %(picks_good_data_noref)s
        average : bool
            Whether to average across channels before plotting. If ``True``,
            interactive plotting of scalp topography is disabled, and
            parameters ``ci`` and ``ci_alpha`` control the style of the
            confidence band around the mean. Default is ``False``.
        %(dB_spectrum_plot)s
        amplitude : bool | 'auto'
            Whether to plot an amplitude spectrum (``True``) or power spectrum
            (``False``). If ``'auto'``, will plot a power spectrum when
            ``dB=True`` and an amplitude spectrum otherwise. Default is
            ``'auto'``.
        %(xscale_plot_psd)s
        ci : float | 'sd' | 'range' | None
            Type of confidence band drawn around the mean when
            ``average=True``. If ``'sd'`` the band spans ±1 standard deviation
            across channels. If ``'range'`` the band spans the range across
            channels at each frequency. If a :class:`float`, it indicates the
            (bootstrapped) confidence interval to display, and must satisfy
            ``0 < ci <= 100``. If ``None``, no band is drawn. Default is
            ``sd``.
        ci_alpha : float
            Opacity of the confidence band. Must satisfy
            ``0 <= ci_alpha <= 1``. Default is 0.3.
        %(color_plot_psd)s
        alpha : float | None
            Opacity of the spectrum line(s). If :class:`float`, must satisfy
            ``0 <= alpha <= 1``. If ``None``, opacity will be ``1`` when
            ``average=True`` and ``0.1`` when ``average=False``. Default is
            ``None``.
        %(spatial_colors_psd)s
        %(sphere_topomap_auto)s
        %(exclude_spectrum_plot)s
        %(axes_spectrum_plot_topomap)s
        %(show)s

        Returns
        -------
        fig : instance of matplotlib.figure.Figure
            Figure with spectra plotted in separate subplots for each channel
            type.
        """
        from ..viz._mpl_figure import _line_figure, _split_picks_by_type
        from .multitaper import _psd_from_mt

        # arg checking
        ci = _check_ci(ci)
        _check_option('xscale', xscale, ('log', 'linear'))
        sphere = _check_sphere(sphere, self.info)
        # defaults
        scalings = _handle_default('scalings', None)
        titles = _handle_default('titles', None)
        units = _handle_default('units', None)
        if amplitude == 'auto':
            estimate = 'power' if dB else 'amplitude'
        else:  # amplitude is boolean
            estimate = 'amplitude' if amplitude else 'power'
        # split picks by channel type
        picks = _picks_to_idx(self.info, picks, 'data', with_ref_meg=False)
        (picks_list, units_list, scalings_list, titles_list
         ) = _split_picks_by_type(self, picks, units, scalings, titles)
        # handle unaggregated multitaper
        if hasattr(self, '_mt_weights'):
            logger.info('Aggregating multitaper estimates before plotting...')
            _f = partial(_psd_from_mt, weights=self._mt_weights)
        # handle unaggregated Welch
        elif 'segment' in self._dims:
            logger.info(
                'Aggregating Welch estimates (median) before plotting...')
            seg_axis = self._dims.index('segment')
            _f = partial(np.nanmedian, axis=seg_axis)
        else:  # "normal" cases
            _f = _identity_function
        ch_axis = self._dims.index('channel')
        psd_list = [_f(self._data.take(_p, axis=ch_axis)) for _p in picks_list]
        # handle epochs
        if 'epoch' in self._dims:
            # XXX TODO FIXME decide how to properly aggregate across repeated
            # measures (epochs) and non-repeated but correlated measures
            # (channels) when calculating stddev or a CI. For across-channel
            # aggregation, doi:10.1007/s10162-012-0321-8 used hotellings T**2
            # with a correction factor that estimated data rank using monte
            # carlo simulations; seems like we could use our own data rank
            # estimation methods to similar effect. Their exact approach used
            # complex spectra though, here we've already converted to power;
            # not sure if that makes an important difference? Anyway that
            # aggregation would need to happen in the _plot_psd function
            # though, not here... for now we just average like we always did.

            # only log message if averaging will actually have an effect
            if self._data.shape[0] > 1:
                logger.info('Averaging across epochs...')
            # epoch axis should always be the first axis
            psd_list = [_p.mean(axis=0) for _p in psd_list]
        # initialize figure
        fig, axes = _line_figure(self, axes, picks=picks)
        # don't add ylabels & titles if figure has unexpected number of axes
        make_label = len(axes) == len(fig.axes)
        # Plot Frequency [Hz] xlabel only on the last axis
        xlabels_list = [False] * (len(axes) - 1) + [True]
        # plot
        _plot_psd(self, fig, self.freqs, psd_list, picks_list, titles_list,
                  units_list, scalings_list, axes, make_label, color,
                  area_mode=ci, area_alpha=ci_alpha, dB=dB, estimate=estimate,
                  average=average, spatial_colors=spatial_colors,
                  xscale=xscale, line_alpha=alpha,
                  sphere=sphere, xlabels_list=xlabels_list)
        fig.subplots_adjust(hspace=0.3)
        plt_show(show, fig)
        return fig

    @fill_doc
    def plot_topo(self, *, dB=True, layout=None, color='w',
                  fig_facecolor='k', axis_facecolor='k', axes=None,
                  block=False, show=True):
        """Plot power spectral density, separately for each channel.

        Parameters
        ----------
        %(dB_spectrum_plot_topo)s
        %(layout_spectrum_plot_topo)s
        %(color_spectrum_plot_topo)s
        %(fig_facecolor)s
        %(axis_facecolor)s
        %(axes_spectrum_plot_topo)s
        %(block)s
        %(show)s

        Returns
        -------
        fig : instance of matplotlib.figure.Figure
            Figure distributing one image per channel across sensor topography.
        """
        if layout is None:
            from ..channels.layout import find_layout
            layout = find_layout(self.info)

        psds, freqs = self.get_data(return_freqs=True)
        if dB:
            psds = 10 * np.log10(psds)
            y_label = 'dB'
        else:
            y_label = 'Power'
        show_func = partial(
            _plot_timeseries_unified, data=[psds], color=color, times=[freqs])
        click_func = partial(
            _plot_timeseries, data=[psds], color=color, times=[freqs])
        picks = _pick_data_channels(self.info)
        info = pick_info(self.info, picks)
        fig = _plot_topo(
            info, times=freqs, show_func=show_func, click_func=click_func,
            layout=layout, axis_facecolor=axis_facecolor,
            fig_facecolor=fig_facecolor, x_label='Frequency (Hz)',
            unified=True, y_label=y_label, axes=axes)
        plt_show(show, block=block)
        return fig

    @fill_doc
    def plot_topomap(self, bands=None, ch_type=None, *, normalize=False,
                     agg_fun=None, dB=False, sensors=True, show_names=False,
                     mask=None, mask_params=None, contours=6, outlines='head',
                     sphere=None, image_interp=_INTERPOLATION_DEFAULT,
                     extrapolate=_EXTRAPOLATE_DEFAULT, border=_BORDER_DEFAULT,
                     res=64, size=1, cmap=None, vlim=(None, None), cnorm=None,
                     colorbar=True, cbar_fmt='auto', units=None, axes=None,
                     show=True):
        """Plot scalp topography of PSD for chosen frequency bands.

        Parameters
        ----------
        %(bands_psd_topo)s
        %(ch_type_topomap_psd)s
        %(normalize_psd_topo)s
        %(agg_fun_psd_topo)s
        %(dB_plot_topomap)s
        %(sensors_topomap)s
        %(show_names_topomap)s
        %(mask_evoked_topomap)s
        %(mask_params_topomap)s
        %(contours_topomap)s
        %(outlines_topomap)s
        %(sphere_topomap_auto)s
        %(image_interp_topomap)s
        %(extrapolate_topomap)s
        %(border_topomap)s
        %(res_topomap)s
        %(size_topomap)s
        %(cmap_topomap)s
        %(vlim_plot_topomap_psd)s
        %(cnorm)s
        %(colorbar_topomap)s
        %(cbar_fmt_topomap_psd)s
        %(units_topomap)s
        %(axes_spectrum_plot_topomap)s
        %(show)s

        Returns
        -------
        fig : instance of Figure
            Figure showing one scalp topography per frequency band.
        """
        ch_type = _get_ch_type(self, ch_type)
        if units is None:
            units = _handle_default('units', None)
        unit = units[ch_type] if hasattr(units, 'keys') else units
        scalings = _handle_default('scalings', None)
        scaling = scalings[ch_type]

        picks, pos, merge_channels, names, ch_type, sphere, clip_origin = \
            _prepare_topomap_plot(self, ch_type, sphere=sphere)
        outlines = _make_head_outlines(sphere, pos, outlines, clip_origin)

        psds, freqs = self.get_data(picks=picks, return_freqs=True)
        if 'epoch' in self._dims:
            psds = np.mean(psds, axis=self._dims.index('epoch'))
        psds *= scaling**2

        if merge_channels:
            psds, names = _merge_ch_data(psds, ch_type, names, method='mean')

        names = _prepare_sensor_names(names, show_names)
        return plot_psds_topomap(
            psds=psds, freqs=freqs, pos=pos, bands=bands, ch_type=ch_type,
            normalize=normalize, agg_fun=agg_fun, dB=dB, sensors=sensors,
            names=names, mask=mask, mask_params=mask_params,
            contours=contours, outlines=outlines, sphere=sphere,
            image_interp=image_interp, extrapolate=extrapolate, border=border,
            res=res, size=size, cmap=cmap, vlim=vlim, cnorm=cnorm,
            colorbar=colorbar, cbar_fmt=cbar_fmt, unit=unit, axes=axes,
            show=show)

    @verbose
    def save(self, fname, *, overwrite=False, verbose=None):
        """Save spectrum data to disk (in HDF5 format).

        Parameters
        ----------
        fname : path-like
            Path of file to save to.
        %(overwrite)s
        %(verbose)s

        See Also
        --------
        mne.time_frequency.read_spectrum
        """
        _, write_hdf5 = _import_h5io_funcs()
        check_fname(fname, 'spectrum', ('.h5', '.hdf5'))
        fname = _check_fname(fname, overwrite=overwrite, verbose=verbose)
        out = self.__getstate__()
        write_hdf5(fname, out, overwrite=overwrite, title='mnepython')

    @verbose
    def to_data_frame(self, picks=None, index=None, copy=True,
                      long_format=False, *, verbose=None):
        """Export data in tabular structure as a pandas DataFrame.

        Channels are converted to columns in the DataFrame. By default,
        an additional column "frequency" is added, unless ``index='freq'``
        (in which case frequency values form the DataFrame's index).

        Parameters
        ----------
        %(picks_all)s
        index : str | list of str | None
            Kind of index to use for the DataFrame. If ``None``, a sequential
            integer index (:class:`pandas.RangeIndex`) will be used. If a
            :class:`str`, a :class:`pandas.Index`, :class:`pandas.Int64Index`,
            or :class:`pandas.Float64Index` will be used (see Notes). If a list
            of two or more string values, a :class:`pandas.MultiIndex` will be
            used. Defaults to ``None``.
        %(copy_df)s
        %(long_format_df_spe)s
        %(verbose)s

        Returns
        -------
        %(df_return)s

        Notes
        -----
        Valid values for ``index`` depend on whether the Spectrum was created
        from continuous data (:class:`~mne.io.Raw`, :class:`~mne.Evoked`) or
        discontinuous data (:class:`~mne.Epochs`). For continuous data, only
        ``None`` or ``'freq'`` is supported. For discontinuous data, additional
        valid values are ``'epoch'`` and ``'condition'``, or a :class:`list`
        comprising some of the valid string values (e.g.,
        ``['freq', 'epoch']``).
        """
        # check pandas once here, instead of in each private utils function
        pd = _check_pandas_installed()  # noqa
        # triage for Epoch-derived or unaggregated spectra
        from_epo = self._get_instance_type_string() == 'Epochs'
        unagg_welch = 'segment' in self._dims
        unagg_mt = 'taper' in self._dims
        # arg checking
        valid_index_args = ['freq']
        if from_epo:
            valid_index_args += ['epoch', 'condition']
        index = _check_pandas_index_arguments(index, valid_index_args)
        # get data
        picks = _picks_to_idx(self.info, picks, 'all', exclude=())
        data = self.get_data(picks)
        if copy:
            data = data.copy()
        # reshape
        if unagg_mt:
            data = np.moveaxis(data, self._dims.index('freq'), -2)
        if from_epo:
            n_epochs, n_picks, n_freqs = data.shape[:3]
        else:
            n_epochs, n_picks, n_freqs = (1,) + data.shape[:2]
        n_segs = data.shape[-1] if unagg_mt or unagg_welch else 1
        data = np.moveaxis(data, self._dims.index('channel'), -1)
        # at this point, should be ([epoch], freq, [segment/taper], channel)
        data = data.reshape(n_epochs * n_freqs * n_segs, n_picks)
        # prepare extra columns / multiindex
        mindex = list()
        default_index = list()
        if from_epo:
            rev_event_id = {v: k for k, v in self.event_id.items()}
            _conds = [rev_event_id[k] for k in self.events[:, 2]]
            conditions = np.repeat(_conds, n_freqs * n_segs)
            epoch_nums = np.repeat(self.selection, n_freqs * n_segs)
            mindex.extend([('condition', conditions), ('epoch', epoch_nums)])
            default_index.extend(['condition', 'epoch'])
        freqs = np.tile(np.repeat(self.freqs, n_segs), n_epochs)
        mindex.append(('freq', freqs))
        default_index.append('freq')
        if unagg_mt or unagg_welch:
            name = 'taper' if unagg_mt else 'segment'
            seg_nums = np.tile(np.arange(n_segs), n_epochs * n_freqs)
            mindex.append((name, seg_nums))
            default_index.append(name)
        # build DataFrame
        df = _build_data_frame(self, data, picks, long_format, mindex, index,
                               default_index=default_index)
        return df

    def units(self, latex=False):
        """Get the spectrum units for each channel type.

        Parameters
        ----------
        latex : bool
            Whether to format the unit strings as LaTeX. Default is ``False``.

        Returns
        -------
        units : dict
            Mapping from channel type to a string representation of the units
            for that channel type.
        """
        units = _handle_default('si_units', None)
        power = not hasattr(self, '_mt_weights')
        return {ch_type: _format_units_psd(units[ch_type], power=power,
                                           latex=latex)
                for ch_type in sorted(self.get_channel_types(unique=True))}


@fill_doc
class Spectrum(BaseSpectrum):
    """Data object for spectral representations of continuous data.

    .. warning:: The preferred means of creating Spectrum objects from
                 continuous or averaged data is via the instance methods
                 :meth:`mne.io.Raw.compute_psd` or
                 :meth:`mne.Evoked.compute_psd`. Direct class instantiation
                 is not supported.

    Parameters
    ----------
    inst : instance of Raw or Evoked
        The data from which to compute the frequency spectrum.
    %(method_psd_auto)s
        ``'auto'`` (default) uses Welch's method for continuous data
        and multitaper for :class:`~mne.Evoked` data.
    %(fmin_fmax_psd)s
    %(tmin_tmax_psd)s
    %(picks_good_data_noref)s
    %(proj_psd)s
    %(reject_by_annotation_psd)s
    %(n_jobs)s
    %(verbose)s
    %(method_kw_psd)s

    Attributes
    ----------
    ch_names : list
        The channel names.
    freqs : array
        Frequencies at which the amplitude, power, or fourier coefficients
        have been computed.
    %(info_not_none)s
    method : str
        The method used to compute the spectrum (``'welch'`` or
        ``'multitaper'``).

    See Also
    --------
    EpochsSpectrum
    mne.io.Raw.compute_psd
    mne.Epochs.compute_psd
    mne.Evoked.compute_psd

    References
    ----------
    .. footbibliography::
    """

    def __init__(self, inst, method, fmin, fmax, tmin, tmax, picks,
                 proj, reject_by_annotation, *, n_jobs, verbose=None,
                 **method_kw):
        from ..io import BaseRaw

        # triage reading from file
        if isinstance(inst, dict):
            self.__setstate__(inst)
            return
        # do the basic setup
        super().__init__(inst, method, fmin, fmax, tmin, tmax, picks, proj,
                         n_jobs=n_jobs, verbose=verbose, **method_kw)
        # get just the data we want
        if isinstance(self.inst, BaseRaw):
            start, stop = np.where(self._time_mask)[0][[0, -1]]
            rba = 'NaN' if reject_by_annotation else None
            data = self.inst.get_data(self._picks, start, stop + 1,
                                      reject_by_annotation=rba)
        else:  # Evoked
            data = self.inst.data[self._picks][:, self._time_mask]
        # compute the spectra
        self._compute_spectra(data, fmin, fmax, n_jobs, method_kw, verbose)
        # check for correct shape and bad values
        self._check_values()
        del self._shape
        # save memory
        del self.inst

    def __getitem__(self, item):
        """Get Spectrum data.

        Parameters
        ----------
        item : int | slice | array-like
            Indexing is similar to a :class:`NumPy array<numpy.ndarray>`; see
            Notes.

        Returns
        -------
        %(getitem_spectrum_return)s

        Notes
        -----
        Integer-, list-, and slice-based indexing is possible:

        - ``spectrum[0]`` gives all frequency bins in the first channel
        - ``spectrum[:3]`` gives all frequency bins in the first 3 channels
        - ``spectrum[[0, 2], 5]`` gives the value in the sixth frequency bin of
          the first and third channels
        - ``spectrum[(4, 7)]`` is the same as ``spectrum[4, 7]``.

        .. note::

           Unlike :class:`~mne.io.Raw` objects (which returns a tuple of the
           requested data values and the corresponding times), accessing
           :class:`~mne.time_frequency.Spectrum` values via subscript does
           **not** return the corresponding frequency bin values. If you need
           them, use ``spectrum.freqs[freq_indices]``.
        """
        from ..io import BaseRaw
        self._parse_get_set_params = partial(
            BaseRaw._parse_get_set_params, self)
        return BaseRaw._getitem(self, item, return_times=False)


@fill_doc
class EpochsSpectrum(BaseSpectrum, GetEpochsMixin):
    """Data object for spectral representations of epoched data.

    .. warning:: The preferred means of creating Spectrum objects from Epochs
                 is via the instance method :meth:`mne.Epochs.compute_psd`.
                 Direct class instantiation is not supported.

    Parameters
    ----------
    inst : instance of Epochs
        The data from which to compute the frequency spectrum.
    %(method_psd)s
    %(fmin_fmax_psd)s
    %(tmin_tmax_psd)s
    %(picks_good_data_noref)s
    %(proj_psd)s
    %(n_jobs)s
    %(verbose)s
    %(method_kw_psd)s

    Attributes
    ----------
    ch_names : list
        The channel names.
    freqs : array
        Frequencies at which the amplitude, power, or fourier coefficients
        have been computed.
    %(info_not_none)s
    method : str
        The method used to compute the spectrum ('welch' or 'multitaper').

    See Also
    --------
    Spectrum
    mne.io.Raw.compute_psd
    mne.Epochs.compute_psd
    mne.Evoked.compute_psd

    References
    ----------
    .. footbibliography::
    """

    def __init__(self, inst, method, fmin, fmax, tmin, tmax, picks, proj, *,
                 n_jobs, verbose=None, **method_kw):
        # triage reading from file
        if isinstance(inst, dict):
            self.__setstate__(inst)
            return
        # do the basic setup
        super().__init__(inst, method, fmin, fmax, tmin, tmax, picks, proj,
                         n_jobs=n_jobs, verbose=verbose, **method_kw)
        # get just the data we want
        data = self.inst.get_data(picks=self._picks)[:, :, self._time_mask]
        # compute the spectra
        self._compute_spectra(data, fmin, fmax, n_jobs, method_kw, verbose)
        self._dims = ('epoch',) + self._dims
        self._shape = (len(self.inst),) + self._shape
        # check for correct shape and bad values
        self._check_values()
        del self._shape
        # we need these for to_data_frame()
        self.event_id = self.inst.event_id.copy()
        self.events = self.inst.events.copy()
        self.selection = self.inst.selection.copy()
        # we need these for __getitem__()
        self.drop_log = deepcopy(self.inst.drop_log)
        self._metadata = self.inst.metadata
        # save memory
        del self.inst

    def __getitem__(self, item):
        """Subselect epochs from an EpochsSpectrum.

        Parameters
        ----------
        item : int | slice | array-like | str
            Access options are the same as for :class:`~mne.Epochs` objects,
            see the docstring of :meth:`mne.Epochs.__getitem__` for
            explanation.

        Returns
        -------
        %(getitem_epochspectrum_return)s
        """
        return super().__getitem__(item)

    def __getstate__(self):
        """Prepare object for serialization."""
        out = super().__getstate__()
        out.update(metadata=self._metadata,
                   drop_log=self.drop_log,
                   event_id=self.event_id,
                   events=self.events,
                   selection=self.selection)
        return out

    def __setstate__(self, state):
        """Unpack from serialized format."""
        super().__setstate__(state)
        self._metadata = state['metadata']
        self.drop_log = state['drop_log']
        self.event_id = state['event_id']
        self.events = state['events']
        self.selection = state['selection']

    def average(self, method='mean'):
        """Average the spectra across epochs.

        Parameters
        ----------
        method : 'mean' | 'median' | callable
            How to aggregate spectra across epochs. If callable, must take a
            :class:`NumPy array<numpy.ndarray>` of shape
            ``(n_epochs, n_channels, n_freqs)`` and return an array of shape
            ``(n_channels, n_freqs)``. Default is ``'mean'``.

        Returns
        -------
        spectrum : instance of Spectrum
            The aggregated spectrum object.
        """
        if isinstance(method, str):
            method = getattr(np, method)  # mean, median, std, etc
            method = partial(method, axis=0)
        if not callable(method):
            raise ValueError('"method" must be a valid string or callable, '
                             f'got a {type(method).__name__} ({method}).')
        # averaging unaggregated spectral estimates are not supported
        if hasattr(self, '_mt_weights'):
            raise NotImplementedError(
                'Averaging complex spectra is not supported. Consider '
                'averaging the signals before computing the complex spectrum.')
        elif 'segment' in self._dims:
            raise NotImplementedError(
                'Averaging individual Welch segments across epochs is not '
                'supported. Consider averaging the signals before computing '
                'the Welch spectrum estimates.')
        # serialize the object and update data, dims, and data type
        state = super().__getstate__()
        state['data'] = method(state['data'])
        state['dims'] = state['dims'][1:]
        state['data_type'] = f'Averaged {state["data_type"]}'
        defaults = dict(
            method=None, fmin=None, fmax=None, tmin=None, tmax=None,
            picks=None, proj=None, reject_by_annotation=None, n_jobs=None,
            verbose=None)
        return Spectrum(state, **defaults)


def read_spectrum(fname):
    """Load a :class:`mne.time_frequency.Spectrum` object from disk.

    Parameters
    ----------
    fname : path-like
        Path to a spectrum file in HDF5 format.

    Returns
    -------
    spectrum : instance of Spectrum
        The loaded Spectrum object.

    See Also
    --------
    mne.time_frequency.Spectrum.save
    """
    read_hdf5, _ = _import_h5io_funcs()
    _validate_type(fname, 'path-like', 'fname')
    fname = _check_fname(fname=fname, overwrite='read', must_exist=False)
    # read it in
    hdf5_dict = read_hdf5(fname, title='mnepython')
    defaults = dict(method=None, fmin=None, fmax=None, tmin=None, tmax=None,
                    picks=None, proj=None, reject_by_annotation=None,
                    n_jobs=None, verbose=None)
    Klass = (EpochsSpectrum if hdf5_dict['inst_type_str'] == 'Epochs'
             else Spectrum)
    return Klass(hdf5_dict, **defaults)


def _check_ci(ci):
    ci = 'sd' if ci == 'std' else ci  # be forgiving
    if _is_numeric(ci):
        if not (0 < ci <= 100):
            raise ValueError(f'ci must satisfy 0 < ci <= 100, got {ci}')
        ci /= 100.
    else:
        _check_option('ci', ci, [None, 'sd', 'range'])
    return ci


def _compute_n_welch_segments(n_times, method_kw):
    # get default values from psd_array_welch
    _defaults = dict()
    for param in ('n_fft', 'n_per_seg', 'n_overlap'):
        _defaults[param] = signature(psd_array_welch).parameters[param].default
    # override defaults with user-specified values
    for key, val in _defaults.items():
        _defaults.update({key: method_kw.get(key, val)})
    # sanity check values / replace `None`s with real numbers
    n_fft, n_per_seg, n_overlap = _check_nfft(n_times, **_defaults)
    # compute expected number of segments
    step = n_per_seg - n_overlap
    return (n_times - n_overlap) // step


def _validate_method(method, instance_type):
    """Convert 'auto' to a real method name, and validate."""
    if method == 'auto':
        method = 'welch' if instance_type.startswith('Raw') else 'multitaper'
    _check_option('method', method, ('welch', 'multitaper'))
    return method
