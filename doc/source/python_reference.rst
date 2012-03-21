=========
Reference
=========

This is the classes and functions reference of mne-python.

.. currentmodule:: mne.fiff

.. automodule:: mne
   :no-members:
   :no-inherited-members:

Classes reference
=================

.. autosummary::
   :toctree: generated/
   :template: class.rst

   mne.fiff.Raw
   mne.fiff.Evoked
   mne.Epochs
   mne.Covariance
   mne.SourceEstimate

Functions reference
===================

.. autosummary::
   :toctree: generated/
   :template: function.rst

   read_cov
   write_cov
   label_time_courses
   read_label
   label_sign_flip
   write_label
   stc_to_label
   read_bem_surfaces
   read_surface
   write_bem_surface
   read_source_spaces
   save_stc_as_volume
   morph_data
   read_stc
   write_stc
   read_w
   write_w

.. currentmodule:: mne.fiff

.. autosummary::
   :toctree: generated/
   :template: function.rst

   pick_types
   pick_channels
   pick_types_evoked
   pick_channels_regexp
   pick_channels_forward
   pick_types_forward

.. currentmodule:: mne.minimum_norm

.. autosummary::
   :toctree: generated/
   :template: function.rst

   read_inverse_operator
   apply_inverse
   apply_inverse_raw
   apply_inverse_epochs
   make_inverse_operator
   write_inverse_operator
   source_band_induced_power
   source_induced_power

.. currentmodule:: mne.stats

.. autosummary::
   :toctree: generated/
   :template: function.rst

   permutation_t_test
   permutation_cluster_test
   permutation_cluster_1samp_test
   fdr_correction
   bonferroni_correction
