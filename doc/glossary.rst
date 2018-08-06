:orphan:

.. include:: links.inc

.. _glossary:

==========================
Glossary
==========================

.. contents:: Contents
   :local:


MNE-Python core terminology
===========================

epochs
    Epoched data; dimensionality: epochs x sensors x times. Typically,
    corresponds to the trials of an experimental design.
    See :class:`Epochs <mne.Epochs>` for the API of the corresponding
    object class, and :ref:`sphx_glr_auto_tutorials_plot_object_raw.py` for a
    narrative overview.

evoked
    Averages over multiple epochs; dimensionality: sensors x times. E.g., an
    average over the trials of one subject for one condition, or over multiple
    subjects.
    See :class:`EvokedArray <mne.EvokedArray>` for the API of the corresponding
    object class, and :ref:`sphx_glr_auto_tutorials_plot_object_evoked.py`
    for a narrative overview.

events
    A collection of Specific events occurring at specific time points; e.g.,
    triggers, experimental condition events, ... MNE only knows about integer
    events. These can be stored as (3, n_events) numpy arrays, but are also
    stored with the raw in the form of a trigger channel.

info
    Collection of metadata regarding a Raw, Epochs or Evoked object; e.g.,
    channel locations, preprocessing history such as filters ...
    See :ref:`sphx_glr_auto_tutorials_plot_info.py` for a narrative
    overview.
    
montage
    EEG channel names and the relative positions of the sensor w.r.t the scalp.
    See :class:`mne.channels.Montage` for the API of the corresponding object
    class.

pick
    One specific sensor; in MNE, in integer format and referring to the
    position of the sensor in the list of sensors.

raw
    Continuous data; dimensionality: sensors x times. May or may not be
    preprocessed.
    See :class:`RawArray <mne.io.RawArray>` for the API of the corresponding
    object class, and :ref:`sphx_glr_auto_tutorials_plot_object_raw.py` for a
    narrative overview.

selection
    A set of picks. E.g., all sensors included in a Region of Interest.
