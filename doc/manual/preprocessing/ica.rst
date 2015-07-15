Independent Component Analysis (ICA)
####################################

.. contents:: Contents
   :local:
   :depth: 2


MNE-Python supports identifying artifacts and latent components
using temporal ICA.

Background
==========

Concepts
--------

The assumption behind ICA is that the measured data are the result of a
linear combination of
statistically independent latent time series, commonly named sources.

ICA estimates a mixing matrix and source time series that are maximally
non-Gaussian (kurtosis and skewness). Once these time series are uncovered,
particular ones, e.g., artifacts, can be dropped before reverting
the mixing process. As a consequence, only the selected time series will remain
in the signals.

The ICA procedure is based on an implementation of the
FastICA algorithm~\cite{Hyvarinen:2000vk} that is included with
the scikit-learn package~\cite{scikit-learn}. To reduce
the computation time and to improve the unmixing performance, dimensionality
reduction can be achieved using the randomized PCA
algorithm~\cite{martinsson-etal:10}.

To integrate data from different channel types that can have
signal amplitudes which are orders of magnitude different, a noise
covariance matrix can be included.
The ICA in MNE can be computed on either raw or epoched data.
The set of functions included allows one
 to interactively select noise-free sources or to perform a fully automated
artifact removal. ICA sources can be visualized using MNE functions for generating
trellis plots~\cite{becker1996tour} (cf. Fig \ref{fig:img_ica}) and sensitivity maps projected on
topographic channel layouts.} % DE maybe add figure

The ICA sources can be exported as a raw data object and saved into a FIF file,
hence allowing any sensor space analysis to be performed on the ICA time series:
time--frequency analysis, raster plots, connectivity, or statistics.
For example, one can create epochs of ICA sources around artifact onsets
and identify noisy ICA components by averaging.
