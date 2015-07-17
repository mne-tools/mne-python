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
regressions on multiple targets, e.g., sensors, time points across trials (samples).
For each regressor it returns the beta values, t-staistics, and uncorrected
significance values. While it can be used as a test it is particularly useful
to compute weighted averages.

- :func:`mne.stats.f_mway_rm` computes a generalized M-way repeated
measures ANOVA for balancd designs. It returns mass-univariate F-statistics
and p-valus. The associated helper function
:func:`mne.stats.f_threshold_mway_rm` allows to determine the F-threshold
at a given significance level and set of degrees of freedom. Note that
this set of functions was previously called `mne.stats.f_twoway_rm` and
`mne.stats.f_threshold_twoway_rm`, respectively, only supporting 2-way factorial designs.

- :func:`mne.stats.ttest_1samp_no_p` is an optimized version of the one sample
t-test provided by scipy. It can be used in the context of non-paramteric
permutation tests to enhance paired-contrasts.

- :func:`mne.stats.parametric.f_oneway` is an optimized version of the F-test
for independent samples provided by scipy.
It can be used in the context of non-paramteric permutation tests to
compute various F-contrasts.


Multiple comparisons
^^^^^^^^^^^^^^^^^^^^

In MEG and EEG analyses typically invole multiple measurements
(sensors, time points) for each sample. In a mass-univariate analysis fitting
 statistical models for each of these observations a multiple comparison problem
occurs (MCPP). MNE-Python provides the following functions to control for
multiple comparison:

- :func:`mne.stats.bonferroni_correction` returns a boolean mask of rejection
decisions and the corrected p-values. The Bonferroni correction reflects the most conservative choice
and corrects for the MCPP by multiplying the p-values by the number of observations

- :func:`mne.stats.fdr_correction` implements False discovery rate (FDR) and also
returns a boolean mask of rejection decisions and the corrected p-values.

More flexible handling of the MCPP can be achieved by non-parametric statistics.


Non-paramteric statistics
-------------------------
