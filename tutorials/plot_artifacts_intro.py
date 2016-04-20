"""
.. _tut_artifacts_intro:

Introduction: dealing with artifacts
====================================

Since MNE supports the data of many different acquisition systems, the
particular artifacts in your data might behave very differently from the
artifacts you can observe in our tutorials and examples.

Therefore you should be aware of the different approaches and of
the variability of artifact rejection (automatic/manual) procedures described
onwards. At the end consider always to visually inspect your data
after artifact rejection or correction.

Background: what is an artifact?
--------------------------------

Artifacts are signal interference that can be
endogenous (biological) and exogenous (environmental).
Typical biological artifacts are head movements, eye blinks
or eye movements, heart beats. The most common environmental
artifact is due to the power line, the so-called *line noise*.

How to handle artifacts?
------------------------

MNE deals with artifacts by first identifying them, and subsequently removing
them. Detection of artifacts can be done visually, or using automatic routines
(or a combination of both). After you know what the artifacts are, you need
remove them. This can be done by:

    - *ignoring* the piece of corrupted data
    - *fixing* the corrupted data

For the artifact detection the functions MNE provides depend on whether
your data is continuous (Raw) or epoch-based (Epochs) and depending on
whether your data is stored on disk or already in memory.

Detecting the artifacts without reading the complete data into memory allows
you to work with datasets that are too large to fit in memory all at once.
Detecting the artifacts in continuous data allows you to apply filters
(e.g. a band-pass filter to zoom in on the muscle artifacts on the temporal
channels) without having to worry about edge effects due to the filter
(i.e. filter ringing). Having the data in memory after segmenting/epoching is
however a very efficient way of browsing through the data which helps
in visualizing. So to conclude, there is not a single most optimal manner
to detect the artifacts: it just depends on the data properties and your
own preferences.


For how to detect artifacts visually or automatically
see :ref:`tut_artifacts_detect`.

For how to correct artifacts by rejection see :ref:`tut_artifacts_reject`.
To discover how to correct certain artifacts by filtering see
:ref:`tut_artifacts_filter` and to learn how to correct artifacts
with subspace methods like SSP and ICA see :ref:`tut_artifacts_correct_ssp`
and :ref:`tut_artifacts_correct_ica`.
"""
