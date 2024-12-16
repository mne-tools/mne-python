:orphan:

Supported channel types
=======================

.. NOTE: part of this file is included in doc/overview/implementation.rst.
   Changes here are reflected there. If you want to link to this content, link
   to :ref:`channel-types` to link to that section of the implementation.rst
   page. The next line is a target for :start-after: so we can omit the title
   from the include:
   channel-types-begin-content

.. NOTE: In the future, this table should be automatically synchronized with
   the sensor types listed in the glossary. Perhaps a table showing data type
   channels as well as non-data type channels should be added to the glossary
   and displayed here too.

Channel types are represented in MNE-Python with shortened or abbreviated
names. This page lists all supported channel types, their abbreviated names,
and the measurement unit used to represent data of that type. Where channel
types occur in two or more sub-types, the sub-type abbreviations are given in
parentheses. More information about measurement units is given in the
:ref:`units` section.

.. NOTE: To include only the table, here's a different target for :start-after:
   channel-types-begin-table

.. cssclass:: table-bordered
.. rst-class:: midvalign

================= ========================================= =================
Channel type      Description                               Measurement unit
================= ========================================= =================
eeg               scalp electroencephalography (EEG)        Volts

meg (mag)         Magnetoencephalography (magnetometers)    Teslas

meg (grad)        Magnetoencephalography (gradiometers)     Teslas/meter

ecg               Electrocardiography (ECG)                 Volts

seeg              Stereotactic EEG channels                 Volts

dbs               Deep brain stimulation (DBS)              Volts

ecog              Electrocorticography (ECoG)               Volts

fnirs (hbo)       Functional near-infrared spectroscopy     Moles/liter
                  (oxyhemoglobin)

fnirs (hbr)       Functional near-infrared spectroscopy     Moles/liter
                  (deoxyhemoglobin)

emg               Electromyography (EMG)                    Volts

eog               Electrooculography  (EOG)                 Volts

bio               Miscellaneous biological channels (e.g.,  Arbitrary units
                  skin conductance)

stim              stimulus (a.k.a. trigger) channels        Arbitrary units

resp              respiration monitoring channel            Volts

chpi              continuous head position indicator        Teslas
                  (HPI) coil channels

exci              Flux excitation channel

ias               Internal Active Shielding data
                  (Triux systems only?)

syst              System status channel information
                  (Triux systems only)

temperature       Temperature                               Degrees Celsius

gsr               Galvanic skin response                    Siemens

ref_meg           Reference Magnetometers                   Teslas

dipole            Dipole amplitude                          Amperes

gof               Goodness of fit (GOF)                     Goodness-of-fit

cw-nirs (amp)     Continuous-wave functional near-infrared  Volts
                  spectroscopy (CW-fNIRS) (CW amplitude)

fd-nirs (ac amp)  Frequency-domain near-infrared            Volts
                  spectroscopy (FD-NIRS AC amplitude)

fd-nirs (phase)   Frequency-domain near-infrared            Radians
                  spectroscopy (FD-NIRS phase)

fnirs (od)        Functional near-infrared spectroscopy     Volts
                  (optical density)

csd               Current source density                    Volts per square
                                                            meter

eyegaze           Eye-tracking (gaze position)              Arbitrary units

pupil             Eye-tracking (pupil size)                 Arbitrary units
================= ========================================= =================