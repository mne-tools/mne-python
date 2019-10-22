.. _changes_0_2:

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
~~~~~~~

The committer list for this release is the following (preceded by number
of commits):

* 33  Alexandre Gramfort
* 12  Martin Luessi
*  2  Yaroslav Halchenko
*  1  Manfred Kitzbichler
