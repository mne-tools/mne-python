# -*- coding: utf-8 -*-
"""Container classes for spectral data."""

# Authors: Dan McCloy <dan@mccloy.info>
#
# License: BSD-3-Clause

from functools import partial
from inspect import signature

import numpy as np

from ..channels.channels import UpdateChannelsMixin
from ..defaults import _handle_default
from ..io.meas_info import ContainsMixin
from ..io.pick import _picks_to_idx, pick_info
from ..utils import _check_sphere, _time_mask, fill_doc, logger, verbose, warn
from ..utils.check import _check_option, _is_numeric
from ..utils.misc import _pl
from ..viz.utils import _plot_psd, plt_show
from . import psd_array_multitaper, psd_array_welch


class ToSpectrumMixin():
    """Mixin class providing spectral methods to sensor-space containers."""

    @verbose
    def compute_psd(self, method='auto', fmin=0, fmax=np.inf, tmin=None,
                    tmax=None, picks=None, proj=False,
                    reject_by_annotation=True, *, n_jobs=1, verbose=None,
                    **method_kw):
        """Perform spectral analysis on sensor data.

        Parameters
        ----------
        %(method_psd)s
        %(fmin_fmax_psd)s
        %(tmin_tmax_psd)s
        %(picks_good_data_noref)s
        %(proj_psd)s
        %(reject_by_annotation_psd)s
        %(n_jobs)s
        %(verbose)s
        %(method_kw_psd)s

        Returns
        -------
        spectrum : instance of Spectrum
            The spectral representation of the data.

        References
        ----------
        .. footbibliography::
        """
        from ..io import BaseRaw
        if method == 'auto':
            method = 'welch' if isinstance(self, BaseRaw) else 'multitaper'
        return Spectrum(
            self, method=method, fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax,
            picks=picks, proj=proj, reject_by_annotation=reject_by_annotation,
            n_jobs=n_jobs, verbose=verbose, **method_kw)

    @verbose
    def plot_psd(self, fmin=0, fmax=np.inf, tmin=None, tmax=None, picks=None,
                 proj=False, reject_by_annotation=True, *, method='auto',
                 ax=None, color='black', xscale='linear', area_mode='std',
                 area_alpha=0.33, dB=True, estimate='auto', show=True,
                 line_alpha=None, spatial_colors=True, sphere=None,
                 exclude='bads', n_jobs=1, average=False, verbose=None,
                 **method_kw):
        """%(plot_psd_doc)s.

        Parameters
        ----------
        %(fmin_fmax_psd)s
        %(tmin_tmax_psd)s
        %(picks_good_data_noref)s
        %(proj_psd)s
        %(reject_by_annotation_psd)s
        %(method_psd)s
        %(ax_psd)s
        %(color_plot_psd)s
        %(xscale_plot_psd)s
        %(area_mode_plot_psd)s
        %(area_alpha_plot_psd)s
        %(dB_plot_psd)s
        %(estimate_plot_psd)s
        %(show)s
        %(line_alpha_plot_psd)s
        %(spatial_colors_psd)s
        %(sphere_topomap_auto)s

            .. versionadded:: 0.22.0
        exclude : list of str | 'bads'
            Channels names to exclude from being shown. If 'bads', the bad
            channels are excluded. Pass an empty list to plot all channels
            (including channels marked "bad", if any).

            .. versionadded:: 0.24.0
        %(n_jobs)s
        %(average_plot_psd)s
        %(verbose)s
        %(method_kw_psd)s

        Returns
        -------
        fig : instance of Figure
            Figure with frequency spectra of the data channels.

        Notes
        -----
        This method exists to support legacy code; for new code the preferred
        idiom is ``inst.compute_psd().plot()`` (where ``inst`` is an instance
        of :class:`~mne.io.Raw`, :class:`~mne.Epochs`, or
        :class:`~mne.Evoked`).
        """
        # legacy n_fft default for plot_psd()
        if method_kw.get('n_fft', None) is None:
            tm = _time_mask(self.times, tmin, tmax, sfreq=self.info['sfreq'])
            method_kw['n_fft'] = min(np.sum(tm), 2048)

        spectrum = self.compute_psd(
            method=method, fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax,
            picks=picks, proj=proj, reject_by_annotation=reject_by_annotation,
            n_jobs=n_jobs, verbose=verbose, **method_kw)

        # translate kwargs
        amplitude = 'auto' if estimate == 'auto' else (estimate == 'amplitude')
        ci = 'sd' if area_mode == 'std' else area_mode

        fig = spectrum.plot(
            average=average, dB=dB, amplitude=amplitude, xscale=xscale,
            ci=ci, ci_alpha=area_alpha, color=color, alpha=line_alpha,
            spatial_colors=spatial_colors, sphere=sphere, exclude=exclude,
            ax=ax, show=show)
        return fig


@fill_doc
class Spectrum(ContainsMixin, UpdateChannelsMixin):
    """Data object for spectral representations of continuous data.

    .. warning:: The preferred means of creating Spectrum objects is via the
                 instance methods :meth:`mne.io.Raw.compute_psd`,
                 :meth:`mne.Epochs.compute_psd`, or
                 :meth:`mne.Evoked.compute_psd`. Direct instantiation is not
                 supported.

    Parameters
    ----------
    inst : instance of Raw, Epochs, or Evoked
        The data from which to compute the frequency spectrum.
    %(method_psd)s
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
        The method used to compute the spectrum ('welch' or 'multitaper').

    See Also
    --------
    mne.io.Raw.compute_psd
    mne.Epochs.compute_psd
    mne.Evoked.compute_psd
    """

    def __init__(self, inst, method, fmin, fmax, tmin, tmax, picks,
                 proj, reject_by_annotation, *, n_jobs, verbose, **method_kw):
        from .. import BaseEpochs
        from ..io import BaseRaw

        # arg checking
        sfreq = inst.info['sfreq']
        if np.isfinite(fmax) and (fmax > sfreq / 2):
            raise ValueError(
                f'Requested fmax ({fmax} Hz) must not exceed ½ the sampling '
                f'frequency of the data ({0.5 * inst.info["sfreq"]} Hz).')
        _check_option('method', method, ('welch', 'multitaper'))
        # get just the data we want to include (formerly _check_psd_data())
        time_mask = _time_mask(inst.times, tmin, tmax, sfreq=sfreq)
        picks = _picks_to_idx(inst.info, picks, 'data', with_ref_meg=False)
        if proj:
            inst = inst.copy().apply_proj()
        if isinstance(inst, BaseRaw):
            start, stop = np.where(time_mask)[0][[0, -1]]
            rba = 'NaN' if reject_by_annotation else None
            data = inst.get_data(picks, start, stop + 1,
                                 reject_by_annotation=rba)
        elif isinstance(inst, BaseEpochs):
            data = inst.get_data(picks=picks)[:, :, time_mask]
        else:  # Evoked
            data = inst.data[picks][:, time_mask]
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
        psd_func = partial(psd_funcs[method], **method_kw)
        # make the spectra
        result = psd_func(
            data, sfreq, fmin=fmin, fmax=fmax, n_jobs=n_jobs, verbose=verbose)
        # assign ._data (handling unaggregated multitaper output)
        if method_kw.get('output', '') == 'complex':
            fourier_coefs, freqs, weights = result
            self._data = fourier_coefs
            self._mt_weights = weights
        else:
            psds, freqs = result
            self._data = psds
        # add the info. bads were effectively dropped by _check_psd_data() so
        # we update the info accordingly
        self.info = pick_info(inst.info, sel=picks, copy=True)
        if proj:  # projs were already applied
            with self.info._unlock():
                for proj in self.info['projs']:
                    proj['active'] = True
        # assign properties (._data already assigned above)
        self._freqs = freqs
        self._inst_type = type(inst)
        self._method = method
        self.verbose = verbose
        # document dims
        self._dims = ('channel', 'freq',)
        expected_shape = (len(self.ch_names), len(self.freqs))
        if BaseEpochs in self._inst_type.__bases__:
            self._dims = ('epoch',) + self._dims
            expected_shape = (len(inst),) + expected_shape
        if method_kw.get('average', '') in (None, False):
            self._dims += ('segment',)
            # hard to know in advance how many welch segments, so fudge it here
            expected_shape += (psds.shape[-1],)
        if method_kw.get('output', '') == 'complex':
            self._dims = self._dims[:-1] + ('taper',) + self._dims[-1:]
            expected_shape = (
                expected_shape[:-1] + (weights.size,) + expected_shape[-1:])
        # record data type (for repr and html_repr)
        self._data_type = ('Fourier Coefficients' if 'taper' in self._dims
                           else 'Power Spectrum')
        # check for bad values
        self._check_values()
        assert len(self._dims) == self._data.ndim
        assert self._data.shape == expected_shape

    def _get_instance_type_string(self):
        """Get string representation of the originating instance type."""
        from .. import BaseEpochs, Evoked
        from ..io import BaseRaw

        if BaseRaw in self._inst_type.__bases__:
            inst_type = 'Raw'
        elif isinstance(self._inst_type, BaseEpochs):
            inst_type = 'Epochs'
        elif isinstance(self._inst_type, Evoked):
            inst_type = 'Evoked'
        else:
            raise RuntimeError(
                f'Unknown instance type {self._inst_type} in Spectrum')
        return inst_type

    def __repr__(self):
        """Build string representation of the Spectrum object."""
        inst_type = self._get_instance_type_string()
        # shape & dimension names
        dims = ' × '.join(
            [f'{dim[0]} {dim[1]}s'
             for dim in zip(self._data.shape, self._dims)])
        freq_range = f'{self.freqs[0]:0.1f}-{self.freqs[-1]:0.1f} Hz'
        return f'<{self._data_type} (from {inst_type}) | {dims}, {freq_range}>'

    def _repr_html_(self, caption=None):
        """Build HTML representation of the Spectrum object."""
        from ..html_templates import repr_templates_env

        inst_type = self._get_instance_type_string()
        t = repr_templates_env.get_template('spectrum.html.jinja')
        t = t.render(spectrum=self, inst_type=inst_type,
                     data_type=self._data_type)
        return t

    @property
    def ch_names(self):
        return self.info['ch_names']

    @fill_doc
    def get_data(self, picks=None, fmin=0, fmax=np.inf, return_freqs=False):
        """Get spectrum data in NumPy array format.

        Parameters
        ----------
        %(picks_good_data_noref)s
        %(fmin_fmax_psd)s
        return_freqs : bool
            Whether to return the frequency bin values for the requested
            frequency range. Default is ``True``.

        Returns
        -------
        data : array
            The requested data in a NumPy array.
        freqs : array
            The frequency values for the requested range. Only returned if
            ``return_freqs`` is ``True``.
        """
        picks = _picks_to_idx(self.info, picks, 'data', with_ref_meg=False)
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

    @property
    def freqs(self):
        return self._freqs

    @property
    def method(self):
        return self._method

    @fill_doc
    def plot(self, picks=None, average=False, dB=True, amplitude='auto',
             xscale='linear', ci='sd', ci_alpha=0.3, color='black', alpha=None,
             spatial_colors=True, sphere=None, exclude='bads', ax=None,
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
        %(dB_psd)s
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
        %(exclude_psd)s
        %(ax_psd)s
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
            amplitude = 'power' if dB else 'amplitude'
        # split picks by channel type
        picks = _picks_to_idx(self.info, picks, 'data', with_ref_meg=False)
        (picks_list, units_list, scalings_list, titles_list
         ) = _split_picks_by_type(self, picks, units, scalings, titles)
        # handle unaggregated multitaper
        ch_axis = self._dims.index('channel')
        if hasattr(self, '_mt_weights'):
            logger.info('Aggregating multitaper estimates before plotting...')
            psd_list = [_psd_from_mt(self._data.take(_p, axis=ch_axis),
                                     self._mt_weights) for _p in picks_list]
        # handle unaggregated Welch
        elif 'segment' in self._dims:
            logger.info(
                'Aggregating Welch estimates (median) before plotting...')
            seg_axis = self._dims.index('segment')
            psd_list = [
                np.nanmedian(self._data.take(_p, axis=ch_axis), axis=seg_axis)
                for _p in picks_list]
        else:  # "normal" cases
            psd_list = [self._data.take(_p, axis=ch_axis) for _p in picks_list]
        # handle epochs
        if 'epoch' in self._dims:
            # XXX TODO FIXME look up how to properly aggregate across repeated
            # measures (epochs) and non-repeated measures (channels) when
            # calculating stddev or a CI.
            # epoch axis should always be the first axis
            logger.info('Averaging across epochs...')
            psd_list = [_p.mean(axis=0) for _p in psd_list]
        # initialize figure
        fig, axes = _line_figure(self, ax, picks=picks)
        # don't add ylabels & titles if figure has unexpected number of axes
        make_label = len(axes) == len(fig.axes)
        # Plot Frequency [Hz] xlabel only on the last axis
        xlabels_list = [False] * (len(axes) - 1) + [True]
        # plot
        _plot_psd(self, fig, self.freqs, psd_list, picks_list, titles_list,
                  units_list, scalings_list, axes, make_label, color,
                  area_mode=ci, area_alpha=ci_alpha, dB=dB, estimate=amplitude,
                  average=average, spatial_colors=spatial_colors,
                  xscale=xscale, line_alpha=alpha,
                  sphere=sphere, xlabels_list=xlabels_list)
        fig.subplots_adjust(hspace=0.3)
        plt_show(show, fig)
        return fig

    def pick_epoch(self, epochs=None):
        """Extract one or more epochs into a new Spectrum object.

        Parameters
        ----------
        epochs : int | list of int
            Index (or list of indices) of which epochs to pick.

        Notes
        -----
        Like the channel-picking methods, this modifies the object in-place.
        If you don't want that, use ``.copy()`` first and assign the result to
        a new variable.
        """
        # XXX FIXME TODO figure out how to let people pick a specific epoch,
        # probably via __getitem__?  May be better with a separate
        # EpochSpectrum class since we don't need it for Raw / Evoked?
        # Would be cool if something like Spectrum['aud/left'] worked...
        pass

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
        # TODO: this should "know" whether the contents are power, amplitude,
        # or fourier coefficients.
        units = _handle_default('si_units', None)
        return {ch_type: self._format_units(units[ch_type], latex=latex)
                for ch_type in sorted(self.get_channel_types(unique=True))}

    def _check_values(self):
        """Check PSD results for bad values."""
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

    def _format_units(self, unit, latex, power=True):
        """Format the measurement units nicely."""
        if power:
            denom = 'Hz'
            exp = r'^{2}' if latex else '²'
            unit = f'({unit})' if '/' in unit else unit
        else:
            denom = r'\sqrt{Hz}' if latex else '√(Hz)'
            exp = ''
        pre, post = (r'$\mathrm{', r'}$') if latex else ('', '')
        return f'{pre}{unit}{exp}/{denom}{post}'


def _check_ci(ci):
    ci = 'sd' if ci == 'std' else ci  # be forgiving
    if _is_numeric(ci):
        if not (0 < ci <= 100):
            raise ValueError(f'ci must satisfy 0 < ci <= 100, got {ci}')
        ci /= 100.
    else:
        _check_option('ci', ci, [None, 'sd', 'range'])
    return ci
