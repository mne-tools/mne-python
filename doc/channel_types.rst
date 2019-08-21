.. _channel-types:

Supported channel types
=======================

Channel types are represented in MNE-Python with shortened or abbreviated
names. This page lists all supported channel types, their abbreviated names,
and the measurement unit used to represent data of that type. Where channel
types occur in two or more sub-types, the sub-type abbreviations are given in
parentheses. More information about measurement units is given on the
:doc:`units` page.

.. NOTE: If this table gets included in another page, target cross-refs to
    :doc:`doc/channel_types` rather than to
    :ref:`doc/<other_page>/<section_name>`. The next line is a target for
    :start-after: so we can omit title and intro paragraph from the include:
    channel-types-begin-table

.. cssclass:: table-bordered
.. rst-class:: midvalign

=============  ========================================= =================
Channel type    Description                              Measurement unit
=============  ========================================= =================
eeg            scalp electroencephalography (EEG)        Volts

meg (mag)      Magnetoencephalography (magnetometers)    Teslas

meg (grad)     Magnetoencephalography (gradiometers)     Teslas/meter

ecg            Electrocardiography (ECG)                 Volts

seeg           Stereotactic EEG channels                 Volts

ecog           Electrocorticography (ECoG)               Volts

fnirs (hbo)    Functional near-infrared spectroscopy     Moles/liter
               (oxyhemoglobin)

fnirs (hbr)    Functional near-infrared spectroscopy     Moles/liter
               (deoxyhemoglobin)

emg            Electromyography (EMG)                    Volts

bio            Miscellaneous biological channels (e.g.,  Arbitrary units
               skin conductance)

stim           stimulus (a.k.a. trigger) channels        Arbitrary units

resp           response-trigger channel                  Arbitrary units

chpi           continuous head position indicator        Teslas
               (HPI) coil channels

exci           Flux excitation channel

ias            Internal Active Shielding data
               (Triux systems only?)

syst           System status channel information
               (Triux systems only)
=============  ========================================= =================
