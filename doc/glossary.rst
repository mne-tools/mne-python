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

Basic Data Types
------------------------------------------------

raw
    Continuous data; dimensionality: sensors x times. May or may not be
    preprocessed.

epochs
    Epoched data; dimensionality: epochs x sensors x times. Typically,
    corresponds to the trials of an experimental design.

evoked
    Averages over multiple epochs; dimensionality: sensors x times. E.g., an
    average over the trials of one subject for one condition, or over multiple
    subjects.

Metadata Types
-----------------------------------------------

info
    blah

events
    blah

Sensors, channels, montages, ...
-----------------------------------------------

montage
    blah

picks
    blah

selection
    A set of picks. E.g., all sensors included in a Region of Interest.


