=========
Reference
=========

This is the classes and functions reference of mne-python.

.. _mne_ref:

.. automodule:: mne
   :no-members:
   :no-inherited-members:

Classes reference
=================

.. currentmodule:: mne

.. autosummary::
   :toctree: generated/
   :template: class.rst

   fiff.Raw
   fiff.Evoked
   Epochs
   Covariance
   SourceEstimate

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
   make_inverse_operator
   apply_inverse_epochs
   write_inverse_operator
   source_band_induced_power
   source_induced_power
