
.. _api_reference_statistics:

Statistics
==========

:py:mod:`mne.stats`:

.. automodule:: mne.stats
   :no-members:
   :no-inherited-members:

.. currentmodule:: mne.stats

Parametric statistics (see :mod:`scipy.stats` and :mod:`statsmodels` for more
options):

.. autosummary::
   :toctree: ../generated/

   ttest_1samp_no_p
   ttest_ind_no_p
   f_oneway
   f_mway_rm
   f_threshold_mway_rm
   linear_regression
   linear_regression_raw

Mass-univariate multiple comparison correction:

.. autosummary::
   :toctree: ../generated/

   bonferroni_correction
   fdr_correction

Non-parametric (clustering) resampling methods:

.. autosummary::
   :toctree: ../generated/

   combine_adjacency
   permutation_cluster_test
   permutation_cluster_1samp_test
   permutation_t_test
   spatio_temporal_cluster_test
   spatio_temporal_cluster_1samp_test
   summarize_clusters_stc
   bootstrap_confidence_interval

ERP-related statistics:

.. autosummary::
   :toctree: ../generated/

   erp.compute_sme

Compute ``adjacency`` matrices for cluster-level statistics:

.. currentmodule:: mne

.. autosummary::
   :toctree: ../generated/

   channels.find_ch_adjacency
   channels.read_ch_adjacency
   spatial_dist_adjacency
   spatial_src_adjacency
   spatial_tris_adjacency
   spatial_inter_hemi_adjacency
   spatio_temporal_src_adjacency
   spatio_temporal_tris_adjacency
   spatio_temporal_dist_adjacency
