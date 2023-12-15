.. NOTE: we use standard sphinx cross-references to highlight new functions and classes.
   References must be to where they are documented in the built API documentation, e.g.,
   :class:`mne.Epochs` not :class:`mne.epochs.Epochs` (even though the class is defined
   in mne/epochs.py).

   There are 5 separate sections for changes, based on type.
   Each should have a filename in this directory of the form NNNNN.<type>.rst,
   where NNNNN is the PR number (e.g., 12345.bugfix.rst). The types are:

   notable
       For overarching changes, e.g., adding type hints package-wide. These are rare.
   dependency
       For changes to dependencies, e.g., adding a new dependency or changing
       the minimum version of an existing dependency.
   bugfix
       For bug fixes. Can change code behavior with no deprecation period.
   deprecation
       Code behavior changes that require a deprecation period.
   enhancement
       For new features.
   other
       For changes that don't fit into any of the above categories, e.g.,
       internal refactorings.

   First-time contributors should use :newcontrib:`Firstname Lastname` instead of
   `Firstname Lastname`_ in their entries. Also add a corresponding entry for
   yourself in doc/changes/names.inc

.. _current:

.. towncrier-draft-entries:: Version |release| (development)
