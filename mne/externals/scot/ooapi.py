# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

"""
Summary
-------
Object oriented API to SCoT.

Extended Summary
----------------
The object oriented API provides a the `Workspace` class, which provides high-level functionality and serves as an
example usage of the low-level API.
"""

import numpy as np

from . import config
from .varica import mvarica, cspvarica
from .plainica import plainica
from .datatools import dot_special
from .connectivity import Connectivity
from .connectivity_statistics import surrogate_connectivity, bootstrap_connectivity, test_bootstrap_difference
from .connectivity_statistics import significance_fdr


class Workspace:
    """SCoT Workspace

    This class provides high-level functionality for source identification, connectivity estimation, and visualization.

    Parameters
    ----------
    var : {:class:`~scot.var.VARBase`-like object, dict}
        Vector autoregressive model (VAR) object that is used for model fitting.
        This can also be a dictionary that is passed as `**kwargs` to backend['var']() in order to
        construct a new VAR model object.
    locations : array_like, optional
        3D Electrode locations. Each row holds the x, y, and z coordinates of an electrode.
    reducedim : {int, float, 'no_pca'}, optional
        A number of less than 1 in interpreted as the fraction of variance that should remain in the data. All
        components that describe in total less than `1-reducedim` of the variance are removed by the PCA step.
        An integer numer of 1 or greater is interpreted as the number of components to keep after applying the PCA.
        If set to 'no_pca' the PCA step is skipped.
    nfft : int, optional
        Number of frequency bins for connectivity estimation.
    backend : dict-like, optional
        Specify backend to use. When set to None the backend configured in config.backend is used.

    Attributes
    ----------
    `unmixing_` : array
        Estimated unmixing matrix.
    `mixing_` : array
        Estimated mixing matrix.
    `plot_diagonal` : str
        Configures what is plotted in the diagonal subplots.
        **'topo'** (default) plots topoplots on the diagonal,
        **'S'** plots the spectral density of each component, and
        **'fill'** plots connectivity on the diagonal.
    `plot_outside_topo` : bool
        Whether to place topoplots in the left column and top row.
    `plot_f_range` : (int, int)
        Lower and upper frequency limits for plotting. Defaults to [0, fs/2].
    """
    def __init__(self, var, locations=None, reducedim=0.99, nfft=512, fs=2, backend=None):
        self.data_ = None
        self.cl_ = None
        self.fs_ = fs
        self.time_offset_ = 0
        self.unmixing_ = None
        self.mixing_ = None
        self.premixing_ = None
        self.activations_ = None
        self.connectivity_ = None
        self.locations_ = locations
        self.reducedim_ = reducedim
        self.nfft_ = nfft
        self.backend_ = backend

        self.trial_mask_ = []

        self.topo_ = None
        self.mixmaps_ = []
        self.unmixmaps_ = []

        self.var_multiclass_ = None
        self.var_model_ = None
        self.var_cov_ = None

        self.plot_diagonal = 'topo'
        self.plot_outside_topo = False
        self.plot_f_range = [0, fs/2]

        self._plotting = None

        if self.backend_ is None:
            self.backend_ = config.backend

        try:
            self.var_ = self.backend_['var'](**var)
        except TypeError:
            self.var_ = var

    def __str__(self):
        if self.data_ is not None:
            datastr = '%d samples, %d channels, %d trials' % self.data_.shape
        else:
            datastr = 'None'

        if self.cl_ is not None:
            clstr = str(np.unique(self.cl_))
        else:
            clstr = 'None'

        if self.unmixing_ is not None:
            sourcestr = str(self.unmixing_.shape[1])
        else:
            sourcestr = 'None'

        if self.var_ is None:
            varstr = 'None'
        else:
            varstr = str(self.var_)

        s = 'Workspace:\n'
        s += '  Data      : ' + datastr + '\n'
        s += '  Classes   : ' + clstr + '\n'
        s += '  Sources   : ' + sourcestr + '\n'
        s += '  VAR models: ' + varstr + '\n'

        return s

    def set_locations(self, locations):
        """ Set sensor locations.

        Parameters
        ----------
        locations : array_like
            3D Electrode locations. Each row holds the x, y, and z coordinates of an electrode.
        """
        self.locations_ = locations

    def set_premixing(self, premixing):
        """ Set premixing matrix.

        The premixing matrix maps data to physical channels. If the data is actual channel data,
        the premixing matrix can be set to identity. Use this functionality if the data was pre-
        transformed with e.g. PCA.

        Parameters
        ----------
        premixing : array_like, shape = [n_signals, n_channels]
            Matrix that maps data signals to physical channels.
        """
        self.premixing_ = premixing

    def set_data(self, data, cl=None, time_offset=0):
        """ Assign data to the workspace.

        This function assigns a new data set to the workspace. Doing so invalidates currently fitted VAR models,
        connectivity estimates, and activations.

        Parameters
        ----------
        data : array-like, shape = [n_samples, n_channels, n_trials] or [n_samples, n_channels]
            EEG data set
        cl : list of valid dict keys
            Class labels associated with each trial.
        time_offset : float, optional
            Trial starting time; used for labelling the x-axis of time/frequency plots.
        """
        self.data_ = np.atleast_3d(data)
        self.cl_ = np.asarray(cl if cl is not None else [None]*self.data_.shape[2])
        self.time_offset_ = time_offset
        self.var_model_ = None
        self.var_cov_ = None
        self.connectivity_ = None

        self.trial_mask_ = np.ones(self.cl_.size, dtype=bool)

        if self.unmixing_ is not None:
            self.activations_ = dot_special(self.data_, self.unmixing_)

    def set_used_labels(self, labels):
        """ Specify which trials to use in subsequent analysis steps.

        This function masks trials based on their class labels.

        Parameters
        ----------
        labels : list of class labels
            Marks all trials that have a label that is in the `labels` list for further processing.
        """
        mask = np.zeros(self.cl_.size, dtype=bool)
        for l in labels:
            mask = np.logical_or(mask, self.cl_ == l)
        self.trial_mask_ = mask

    def do_mvarica(self, varfit='ensemble'):
        """ Perform MVARICA

        Perform MVARICA source decomposition and VAR model fitting.

        Parameters
        ----------
        varfit : string
            Determines how to calculate the residuals for source decomposition.
            'ensemble' (default) fits one model to the whole data set,
            'class' fits a different model for each class, and
            'trial' fits a different model for each individual trial.

        Returns
        -------
        result : class
            see :func:`mvarica` for a description of the return value.

        Raises
        ------
        RuntimeError
            If the :class:`Workspace` instance does not contain data.
            
        See Also
        --------
        :func:`mvarica` : MVARICA implementation
        """
        if self.data_ is None:
            raise RuntimeError("MVARICA requires data to be set")
        result = mvarica(x=self.data_[:, :, self.trial_mask_], cl=self.cl_[self.trial_mask_], var=self.var_,
                         reducedim=self.reducedim_, backend=self.backend_, varfit=varfit)
        self.mixing_ = result.mixing
        self.unmixing_ = result.unmixing
        self.var_ = result.b
        self.connectivity_ = Connectivity(result.b.coef, result.b.rescov, self.nfft_)
        self.activations_ = dot_special(self.data_, self.unmixing_)
        self.mixmaps_ = []
        self.unmixmaps_ = []
        return result
    
    def do_cspvarica(self, varfit='ensemble'):
        """ Perform CSPVARICA

        Perform CSPVARICA source decomposition and VAR model fitting.

        Parameters
        ----------
        varfit : string
            Determines how to calculate the residuals for source decomposition.
            'ensemble' (default) fits one model to the whole data set,
            'class' fits a different model for each class, and
            'trial' fits a different model for each individual trial.

        Returns
        -------
        result : class
            see :func:`cspvarica` for a description of the return value.

        Raises
        ------
        RuntimeError
            If the :class:`Workspace` instance does not contain data.

        See Also
        --------
        :func:`cspvarica` : CSPVARICA implementation
        """
        if self.data_ is None:
            raise RuntimeError("CSPVARICA requires data to be set")
        try:
            sorted(self.cl_)
            for c in self.cl_:
                assert(c is not None)
        except (TypeError, AssertionError):
            raise RuntimeError("CSPVARICA requires orderable and hashable class labels that are not None")
        result = cspvarica(x=self.data_, var=self.var_, cl=self.cl_,
                           reducedim=self.reducedim_, backend=self.backend_, varfit=varfit)
        self.mixing_ = result.mixing
        self.unmixing_ = result.unmixing
        self.var_ = result.b
        self.connectivity_ = Connectivity(self.var_.coef, self.var_.rescov, self.nfft_)
        self.activations_ = dot_special(self.data_, self.unmixing_)
        self.mixmaps_ = []
        self.unmixmaps_ = []
        return result

    def do_ica(self):
        """ Perform ICA

        Perform plain ICA source decomposition.

        Returns
        -------
        result : class
            see :func:`plainica` for a description of the return value.

        Raises
        ------
        RuntimeError
            If the :class:`Workspace` instance does not contain data.
        """
        if self.data_ is None:
            raise RuntimeError("ICA requires data to be set")
        result = plainica(x=self.data_[:, :, self.trial_mask_], reducedim=self.reducedim_, backend=self.backend_)
        self.mixing_ = result.mixing
        self.unmixing_ = result.unmixing
        self.activations_ = dot_special(self.data_, self.unmixing_)
        self.var_model_ = None
        self.var_cov_ = None
        self.connectivity_ = None
        self.mixmaps_ = []
        self.unmixmaps_ = []
        return result

    def remove_sources(self, sources):
        """ Remove sources from the decomposition.

        This function removes sources from the decomposition. Doing so invalidates currently fitted VAR models and
        connectivity estimates.

        Parameters
        ----------
        sources : {slice, int, array of ints}
            Indices of components to remove.

        Raises
        ------
        RuntimeError
            If the :class:`Workspace` instance does not contain a source decomposition.
        """
        if self.unmixing_ is None or self.mixing_ is None:
            raise RuntimeError("No sources available (run do_mvarica first)")
        self.mixing_ = np.delete(self.mixing_, sources, 0)
        self.unmixing_ = np.delete(self.unmixing_, sources, 1)
        if self.activations_ is not None:
            self.activations_ = np.delete(self.activations_, sources, 1)
        self.var_model_ = None
        self.var_cov_ = None
        self.connectivity_ = None
        self.mixmaps_ = []
        self.unmixmaps_ = []

    def fit_var(self):
        """ Fit a var model to the source activations.

        Raises
        ------
        RuntimeError
            If the :class:`Workspace` instance does not contain source activations.
        """
        if self.activations_ is None:
            raise RuntimeError("VAR fitting requires source activations (run do_mvarica first)")
        self.var_.fit(data=self.activations_[:, :, self.trial_mask_])
        self.connectivity_ = Connectivity(self.var_.coef, self.var_.rescov, self.nfft_)

    def optimize_var(self):
        """ Optimize the var model's hyperparameters (such as regularization).

        Raises
        ------
        RuntimeError
            If the :class:`Workspace` instance does not contain source activations.
        """
        if self.activations_ is None:
            raise RuntimeError("VAR fitting requires source activations (run do_mvarica first)")

        self.var_.optimize(self.activations_[:, :, self.trial_mask_])

    def get_connectivity(self, measure_name, plot=False):
        """ Calculate spectral connectivity measure.

        Parameters
        ----------
        measure_name : str
            Name of the connectivity measure to calculate. See :class:`Connectivity` for supported measures.
        plot : {False, None, Figure object}, optional
            Whether and where to plot the connectivity. If set to **False**, nothing is plotted. Otherwise set to the
            Figure object. If set to **None**, a new figure is created.

        Returns
        -------
        measure : array, shape = [n_channels, n_channels, nfft]
            Values of the connectivity measure.
        fig : Figure object
            Instance of the figure in which was plotted. This is only returned if `plot` is not **False**.

        Raises
        ------
        RuntimeError
            If the :class:`Workspace` instance does not contain a fitted VAR model.
        """
        if self.connectivity_ is None:
            raise RuntimeError("Connectivity requires a VAR model (run do_mvarica or fit_var first)")

        cm = getattr(self.connectivity_, measure_name)()

        cm = np.abs(cm) if np.any(np.iscomplex(cm)) else cm

        if plot is None or plot:
            fig = plot
            if self.plot_diagonal == 'fill':
                diagonal = 0
            elif self.plot_diagonal == 'S':
                diagonal = -1
                sm = np.abs(self.connectivity_.S())
                sm /= np.max(sm)     # scale to 1 since components are scaled arbitrarily anyway
                fig = self.plotting.plot_connectivity_spectrum(sm, fs=self.fs_, freq_range=self.plot_f_range,
                                                          diagonal=1, border=self.plot_outside_topo, fig=fig)
            else:
                diagonal = -1

            fig = self.plotting.plot_connectivity_spectrum(cm, fs=self.fs_, freq_range=self.plot_f_range,
                                                      diagonal=diagonal, border=self.plot_outside_topo, fig=fig)

            return cm, fig

        return cm

    def get_surrogate_connectivity(self, measure_name, repeats=100, plot=False):
        """ Calculate spectral connectivity measure under the assumption of no actual connectivity.

        Repeatedly samples connectivity from phase-randomized data. This provides estimates of the connectivity
        distribution if there was no causal structure in the data.

        Parameters
        ----------
        measure_name : str
            Name of the connectivity measure to calculate. See :class:`Connectivity` for supported measures.
        repeats : int, optional
            How many surrogate samples to take.

        Returns
        -------
        measure : array, shape = [`repeats`, n_channels, n_channels, nfft]
            Values of the connectivity measure for each surrogate.

        See Also
        --------
        :func:`scot.connectivity_statistics.surrogate_connectivity` : Calculates surrogate connectivity
        """
        cs = surrogate_connectivity(measure_name, self.activations_[:, :, self.trial_mask_],
                                    self.var_, self.nfft_, repeats)

        if plot is None or plot:
            fig = plot
            if self.plot_diagonal == 'fill':
                diagonal = 0
            elif self.plot_diagonal == 'S':
                diagonal = -1
                sb = self.get_surrogate_connectivity('absS', repeats)
                sb /= np.max(sb)     # scale to 1 since components are scaled arbitrarily anyway
                su = np.percentile(sb, 95, axis=0)
                fig = self.plotting.plot_connectivity_spectrum([su], fs=self.fs_, freq_range=self.plot_f_range,
                                                          diagonal=1, border=self.plot_outside_topo, fig=fig)
            else:
                diagonal = -1
            cu = np.percentile(cs, 95, axis=0)
            fig = self.plotting.plot_connectivity_spectrum([cu], fs=self.fs_, freq_range=self.plot_f_range,
                                                      diagonal=diagonal, border=self.plot_outside_topo, fig=fig)
            return cs, fig

        return cs

    def get_bootstrap_connectivity(self, measure_names, repeats=100, num_samples=None, plot=False):
        """ Calculate bootstrap estimates of spectral connectivity measures.

        Bootstrapping is performed on trial level.

        Parameters
        ----------
        measure_names : {str, list of str}
            Name(s) of the connectivity measure(s) to calculate. See :class:`Connectivity` for supported measures.
        repeats : int, optional
            How many bootstrap estimates to take.
        num_samples : int, optional
            How many samples to take for each bootstrap estimates. Defaults to the same number of trials as present in
            the data.

        Returns
        -------
        measure : array, shape = [`repeats`, n_channels, n_channels, nfft]
            Values of the connectivity measure for each bootstrap estimate. If `measure_names` is a list of strings a
            dictionary is returned, where each key is the name of the measure, and the corresponding values are
            ndarrays of shape [`repeats`, n_channels, n_channels, nfft].

        See Also
        --------
        :func:`scot.connectivity_statistics.bootstrap_connectivity` : Calculates bootstrap connectivity
        """
        if num_samples is None:
            num_samples = np.sum(self.trial_mask_)

        cb = bootstrap_connectivity(measure_names, self.activations_[:, :, self.trial_mask_],
                                    self.var_, self.nfft_, repeats, num_samples)

        if plot is None or plot:
            fig = plot
            if self.plot_diagonal == 'fill':
                diagonal = 0
            elif self.plot_diagonal == 'S':
                diagonal = -1
                sb = self.get_bootstrap_connectivity('absS', repeats, num_samples)
                sb /= np.max(sb)     # scale to 1 since components are scaled arbitrarily anyway
                sm = np.median(sb, axis=0)
                sl = np.percentile(sb, 2.5, axis=0)
                su = np.percentile(sb, 97.5, axis=0)
                fig = self.plotting.plot_connectivity_spectrum([sm, sl, su], fs=self.fs_, freq_range=self.plot_f_range,
                                                          diagonal=1, border=self.plot_outside_topo, fig=fig)
            else:
                diagonal = -1
            cm = np.median(cb, axis=0)
            cl = np.percentile(cb, 2.5, axis=0)
            cu = np.percentile(cb, 97.5, axis=0)
            fig = self.plotting.plot_connectivity_spectrum([cm, cl, cu], fs=self.fs_, freq_range=self.plot_f_range,
                                                      diagonal=diagonal, border=self.plot_outside_topo, fig=fig)
            return cb, fig

        return cb

    def get_tf_connectivity(self, measure_name, winlen, winstep, plot=False, crange='default'):
        """ Calculate estimate of time-varying connectivity.

        Connectivity is estimated in a sliding window approach on the current data set. The window is stepped
        `n_steps` = (`n_samples` - `winlen`) // `winstep` times.

        Parameters
        ----------
        measure_name : str
            Name of the connectivity measure to calculate. See :class:`Connectivity` for supported measures.
        winlen : int
            Length of the sliding window (in samples).
        winstep : int
            Step size for sliding window (in samples).
        plot : {False, None, Figure object}, optional
            Whether and where to plot the connectivity. If set to **False**, nothing is plotted. Otherwise set to the
            Figure object. If set to **None**, a new figure is created.

        Returns
        -------
        result : array, shape = [n_channels, n_channels, nfft, n_steps]
            Values of the connectivity measure.
        fig : Figure object, optional
            Instance of the figure in which was plotted. This is only returned if `plot` is not **False**.

        Raises
        ------
        RuntimeError
            If the :class:`Workspace` instance does not contain a fitted VAR model.
        """
        if self.activations_ is None:
            raise RuntimeError("Time/Frequency Connectivity requires activations (call set_data after do_mvarica)")
        [n, m, _] = self.activations_.shape

        nstep = (n - winlen) // winstep

        result = np.zeros((m, m, self.nfft_, nstep), np.complex64)
        i = 0
        for j in range(0, n - winlen, winstep):
            win = np.arange(winlen) + j
            data = self.activations_[win, :, :]
            data = data[:, :, self.trial_mask_]
            self.var_.fit(data)
            con = Connectivity(self.var_.coef, self.var_.rescov, self.nfft_)
            result[:, :, :, i] = getattr(con, measure_name)()
            i += 1

        if plot is None or plot:
            fig = plot
            t0 = 0.5 * winlen / self.fs_ + self.time_offset_
            t1 = self.data_.shape[0] / self.fs_ - 0.5 * winlen / self.fs_ + self.time_offset_
            if self.plot_diagonal == 'fill':
                diagonal = 0
            elif self.plot_diagonal == 'S':
                diagonal = -1
                s = np.abs(self.get_tf_connectivity('S', winlen, winstep))
                if crange == 'default':
                    crange = [np.min(s), np.max(s)]
                fig = self.plotting.plot_connectivity_timespectrum(s, fs=self.fs_, crange=[np.min(s), np.max(s)],
                                                              freq_range=self.plot_f_range, time_range=[t0, t1],
                                                              diagonal=1, border=self.plot_outside_topo, fig=fig)
            else:
                diagonal = -1

            tfc = self._clean_measure(measure_name, result)
            if crange == 'default':
                if diagonal == -1:
                    for m in range(tfc.shape[0]):
                        tfc[m, m, :, :] = 0
                crange = [np.min(tfc), np.max(tfc)]
            fig = self.plotting.plot_connectivity_timespectrum(tfc, fs=self.fs_, crange=crange,
                                                          freq_range=self.plot_f_range, time_range=[t0, t1],
                                                          diagonal=diagonal, border=self.plot_outside_topo, fig=fig)

            return result, fig

        return result

    def compare_conditions(self, labels1, labels2, measure_name, alpha=0.01, repeats=100, num_samples=None, plot=False):
        """ Test for significant difference in connectivity of two sets of class labels.

        Connectivity estimates are obtained by bootstrapping. Correction for multiple testing is performed by
        controlling the false discovery rate (FDR).

        Parameters
        ----------
        labels1, labels2 : list of class labels
            The two sets of class labels to compare. Each set may contain more than one label.
        measure_name : str
            Name of the connectivity measure to calculate. See :class:`Connectivity` for supported measures.
        alpha : float, optional
            Maximum allowed FDR. The ratio of falsely detected significant differences is guaranteed to be less than
            `alpha`.
        repeats : int, optional
            How many bootstrap estimates to take.
        num_samples : int, optional
            How many samples to take for each bootstrap estimates. Defaults to the same number of trials as present in
            the data.
        plot : {False, None, Figure object}, optional
            Whether and where to plot the connectivity. If set to **False**, nothing is plotted. Otherwise set to the
            Figure object. If set to **None**, a new figure is created.

        Returns
        -------
        p : array, shape = [n_channels, n_channels, nfft]
            Uncorrected p-values.
        s : array, dtype=bool, shape = [n_channels, n_channels, nfft]
            FDR corrected significance. True means the difference is significant in this location.
        fig : Figure object, optional
            Instance of the figure in which was plotted. This is only returned if `plot` is not **False**.
        """
        self.set_used_labels(labels1)
        ca = self.get_bootstrap_connectivity(measure_name, repeats, num_samples)
        self.set_used_labels(labels2)
        cb = self.get_bootstrap_connectivity(measure_name, repeats, num_samples)

        p = test_bootstrap_difference(ca, cb)
        s = significance_fdr(p, alpha)

        if plot is None or plot:
            fig = plot
            if self.plot_diagonal == 'topo':
                diagonal = -1
            elif self.plot_diagonal == 'fill':
                diagonal = 0
            elif self.plot_diagonal is 'S':
                diagonal = -1
                self.set_used_labels(labels1)
                sa = self.get_bootstrap_connectivity('absS', repeats, num_samples)
                sm = np.median(sa, axis=0)
                sl = np.percentile(sa, 2.5, axis=0)
                su = np.percentile(sa, 97.5, axis=0)
                fig = self.plotting.plot_connectivity_spectrum([sm, sl, su], fs=self.fs_, freq_range=self.plot_f_range,
                                                          diagonal=1, border=self.plot_outside_topo, fig=fig)

                self.set_used_labels(labels2)
                sb = self.get_bootstrap_connectivity('absS', repeats, num_samples)
                sm = np.median(sb, axis=0)
                sl = np.percentile(sb, 2.5, axis=0)
                su = np.percentile(sb, 97.5, axis=0)
                fig = self.plotting.plot_connectivity_spectrum([sm, sl, su], fs=self.fs_, freq_range=self.plot_f_range,
                                                          diagonal=1, border=self.plot_outside_topo, fig=fig)

                p_s = test_bootstrap_difference(ca, cb)
                s_s = significance_fdr(p_s, alpha)

                self.plotting.plot_connectivity_significance(s_s, fs=self.fs_, freq_range=self.plot_f_range,
                                                        diagonal=1, border=self.plot_outside_topo, fig=fig)
            else:
                diagonal = -1

            cm = np.median(ca, axis=0)
            cl = np.percentile(ca, 2.5, axis=0)
            cu = np.percentile(ca, 97.5, axis=0)

            fig = self.plotting.plot_connectivity_spectrum([cm, cl, cu], fs=self.fs_, freq_range=self.plot_f_range,
                                                      diagonal=diagonal, border=self.plot_outside_topo, fig=fig)

            cm = np.median(cb, axis=0)
            cl = np.percentile(cb, 2.5, axis=0)
            cu = np.percentile(cb, 97.5, axis=0)

            fig = self.plotting.plot_connectivity_spectrum([cm, cl, cu], fs=self.fs_, freq_range=self.plot_f_range,
                                                      diagonal=diagonal, border=self.plot_outside_topo, fig=fig)

            self.plotting.plot_connectivity_significance(s, fs=self.fs_, freq_range=self.plot_f_range,
                                                    diagonal=diagonal, border=self.plot_outside_topo, fig=fig)

            return p, s, fig

        return p, s

    @staticmethod
    def show_plots():
        """Show current plots.

        This is only a convenience wrapper around :func:`matplotlib.pyplot.show_plots`.

        """
        self.plotting.show_plots()

    def plot_source_topos(self, common_scale=None):
        """ Plot topography of the Source decomposition.

        Parameters
        ----------
        common_scale : float, optional
            If set to None, each topoplot's color axis is scaled individually. Otherwise specifies the percentile
            (1-99) of values in all plot. This value is taken as the maximum color scale.
        """
        if self.unmixing_ is None and self.mixing_ is None:
            raise RuntimeError("No sources available (run do_mvarica first)")

        self._prepare_plots(True, True)

        self.plotting.plot_sources(self.topo_, self.mixmaps_, self.unmixmaps_, common_scale)

    def plot_connectivity_topos(self, fig=None):
        """ Plot scalp projections of the sources.

        This function only plots the topos. Use in combination with connectivity plotting.

        Parameters
        ----------
        fig : {None, Figure object}, optional
            Where to plot the topos. f set to **None**, a new figure is created. Otherwise plot into the provided
            figure object.

        Returns
        -------
        fig : Figure object
            Instance of the figure in which was plotted.
        """
        self._prepare_plots(True, False)
        if self.plot_outside_topo:
            fig = self.plotting.plot_connectivity_topos('outside', self.topo_, self.mixmaps_, fig)
        elif self.plot_diagonal == 'topo':
            fig = self.plotting.plot_connectivity_topos('diagonal', self.topo_, self.mixmaps_, fig)
        return fig

    def plot_connectivity_surrogate(self, measure_name, repeats=100, fig=None):
        """ Plot spectral connectivity measure under the assumption of no actual connectivity.

        Repeatedly samples connectivity from phase-randomized data. This provides estimates of the connectivity
        distribution if there was no causal structure in the data.

        Parameters
        ----------
        measure_name : str
            Name of the connectivity measure to calculate. See :class:`Connectivity` for supported measures.
        repeats : int, optional
            How many surrogate samples to take.
        fig : {None, Figure object}, optional
            Where to plot the topos. f set to **None**, a new figure is created. Otherwise plot into the provided
            figure object.

        Returns
        -------
        fig : Figure object
            Instance of the figure in which was plotted.
        """
        cb = self.get_surrogate_connectivity(measure_name, repeats)

        self._prepare_plots(True, False)

        cu = np.percentile(cb, 95, axis=0)

        fig = self.plotting.plot_connectivity_spectrum([cu], self.fs_, freq_range=self.plot_f_range, fig=fig)

        return fig

    @property
    def plotting(self):
        if not self._plotting:
            from . import plotting
            self._plotting = plotting
        return self._plotting

    def _prepare_plots(self, mixing=False, unmixing=False):
        if self.locations_ is None:
            raise RuntimeError("Need sensor locations for plotting")

        if self.topo_ is None:
            from scot.eegtopo.topoplot import Topoplot
            self.topo_ = Topoplot()
            self.topo_.set_locations(self.locations_)

        if mixing and not self.mixmaps_:
            premix = self.premixing_ if self.premixing_ is not None else np.eye(self.mixing_.shape[1])
            self.mixmaps_ = self.plotting.prepare_topoplots(self.topo_, np.dot(self.mixing_, premix))
            #self.mixmaps_ = self.plotting.prepare_topoplots(self.topo_, self.mixing_)

        if unmixing and not self.unmixmaps_:
            preinv = np.linalg.pinv(self.premixing_) if self.premixing_ is not None else np.eye(self.unmixing_.shape[0])
            self.unmixmaps_ = self.plotting.prepare_topoplots(self.topo_, np.dot(preinv, self.unmixing_).T)
            #self.unmixmaps_ = self.plotting.prepare_topoplots(self.topo_, self.unmixing_.transpose())

    @staticmethod
    def _clean_measure(measure, a):
        if measure in ['a', 'H', 'COH', 'pCOH']:
            return np.abs(a)
        elif measure in ['S', 'g']:
            return np.log(np.abs(a))
        else:
            return np.real(a)
