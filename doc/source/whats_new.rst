What's new
==========

.. _changes_0_2:

Current
-----------

Changelog
~~~~~~~~~

   - Sign flip computation for robust label average of signed values by `Alex Gramfort`_.

   - Reading and writing of .w files by `Martin Luessi`_.

   - Support for modifying Raw object and allow raw data preloading with memory mapping by `Martin Luessi`_ and `Alex Gramfort`_.

   - Support of arithmetic of Evoked data (useful to concatenate between runs and compute contrasts) by `Alex Gramfort`_.

   - Support for computing sensor space data from a source estimate using an MNE forward solution by `Martin Luessi`_.

   - Support of arithmetic of Covariance by `Alex Gramfort`_.

   - Write BEM surfaces in Python  by `Alex Gramfort`_.

   - Filtering operations and apply_function interface for Raw object by `Martin Luessi`_.

   - Support for complex valued raw fiff files and computation of analytic signal for Raw object by `Martin Luessi`_.

   - Write inverse operators (surface and volume) by `Alex Gramfort`_.

Version 0.2
-----------

Changelog
~~~~~~~~~

   - New stats functions for FDR correction and Bonferroni by `Alex Gramfort`_.

   - Faster time-frequency using downsampling trick by `Alex Gramfort`_.

   - Support for volume source spaces by `Alex Gramfort`_ (requires next MNE release or nightly).

   - Improved Epochs handling by `Martin Luessi`_ (slicing, drop_bad_epochs).

   - Bug fix in Epochs + ECG detection by Manfred Kitzbichler.

   - New pick_types_evoked function by `Alex Gramfort`_.

   - SourceEstimate now supports algebra by `Alex Gramfort`_.

API changes summary
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here are the code migration instructions when upgrading from mne-python
version 0.1:

  - New return values for the function find_ecg_events

Authors
~~~~~~~~~

The committer list for this release is the following (preceded by number
of commits):

    * 33  Alexandre Gramfort
    * 12  Martin Luessi
    *  2  Yaroslav Halchenko
    *  1  Manfred Kitzbichler

.. _Alex Gramfort: http://www-sop.inria.fr/members/Alexandre.Gramfort/

.. _Martin Luessi: http://ivpl.eecs.northwestern.edu/people/mluessi

.. _Yaroslav Halchenko: http://www.onerussian.com/
