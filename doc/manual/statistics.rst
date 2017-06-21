==========
Statistics
==========

MNE-Python provides different parametric and
and non-parametric statistics in :mod:`mne.stats` which are specially designed
for analyzing mass-univariate hypotheses on neuroimaging data.


Parametric statistics
---------------------

Models
^^^^^^

- :func:`mne.stats.linear_regression` allows to compute ordinary least square
  regressions on multiple targets, e.g., sensors, time points across trials
  (samples). For each regressor it returns the beta values, t-staistics, and
  uncorrected significance values. While it can be used as a test it is
  particularly useful to compute weighted averages.

- :func:`mne.stats.f_mway_rm` computes a generalized M-way repeated
  measures ANOVA for balancd designs. It returns mass-univariate F-statistics
  and p-valus. The associated helper function
  :func:`mne.stats.f_threshold_mway_rm` allows to determine the F-threshold
  at a given significance level and set of degrees of freedom. Note that
  this set of functions was previously called `mne.stats.f_twoway_rm` and
  `mne.stats.f_threshold_twoway_rm`, respectively, only supporting 2-way
  factorial designs.

- :func:`mne.stats.ttest_1samp_no_p` is an optimized version of the one sample
  t-test provided by scipy. It is used by default for contrast enhancement in
  :func:`mne.stats.permutation_cluster_1samp_test` and
  :func:`mne.stats.spatio_temporal_cluster_1samp_test`.

- :func:`mne.stats.f_oneway` is an optimized version of the F-test
  for independent samples provided by scipy.
  It can be used in the context of non-paramteric permutation tests to
  compute various F-contrasts. It is used by default for contrast enhancement in
  :func:`mne.stats.spatio_temporal_cluster_test` and
  :func:`mne.stats.permutation_cluster_test`.


Multiple comparisons
^^^^^^^^^^^^^^^^^^^^

In MEG and EEG analyses typically involve multiple measurements
(sensors, time points) for each sample. In a mass-univariate analysis fitting
statistical models for each of these observations a multiple comparison problem
occurs (MCPP). MNE-Python provides the following functions to control for
multiple comparison:

- :func:`mne.stats.bonferroni_correction` returns a boolean mask of rejection
  decisions and the corrected p-values. The Bonferroni correction reflects the
  most conservative choice and corrects for the MCPP by multiplying the
  p-values by the number of observations

- :func:`mne.stats.fdr_correction` implements False discovery rate (FDR) and
  also returns a boolean mask of rejection decisions and the corrected p-values.

More flexible handling of the MCPP can be achieved by non-parametric statistics.


Non-paramteric statistics
-------------------------

Permutation clustering
^^^^^^^^^^^^^^^^^^^^^^

As MEG and EEG data are subject to considerable spatiotemporal correlation
the assumption of independence between observations is hard to justify.
As a consequence the MCPP is overestimated when employing paramteric
mass-univariate statistics. A flexble alternative is given by non-parametric
permutation clustering statistics which implement a spatiotemporal priors
and typically allow for clusterwise inference.
These tests can be applied over a wide range of situations inclduing single subject and group analyses
in time, space and frequency. The only requirement is that the scientific hypothesis can be mapped
onto an exchangeability null hypothesis in which two or more conditions can be compared and exchanged
across permutations to generate an empirical distribution.

The clustering permutation API in MNE-Python is grouped according to different contrasts of interest
and clustering connectivity prior, i.e., assumptions about the grouping and neighborhood of the observations.

- :func:`mne.stats.permutation_cluster_1samp_test` supports paired contrasts with spatial prior.

- :func:`mne.stats.permutation_cluster_test` supports F-contrasts with spatial prior.

- :func:`mne.stats.spatio_temporal_cluster_1samp_test` supports paired contrasts without spatial prior.

- :func:`mne.stats.spatio_temporal_cluster_test` supports F-contrasts without spatial prior.

Using the TFCE option observation- instead of cluster-wise hypothesis testing can be realized.


.. note:: Note that the permutation clustering functions do not constitute thresholding to paramterical tests.
    Although using F-tests and t-tests internally for contrast enhancement, the actual test statistic is
    the cluster size.

.. note:: Unless TFCE is used, the hypotheses tested are cluster-wise. This means that no inference is provided
    for individual time points, sensors, dipole locations or frequencies in such a cluster.
