OPM
===
:func:`mne.datasets.opm.data_path`

OPM data acquired using an Elekta DACQ, simply piping the data into Elekta
magnetometer channels. The FIF files thus appear to come from a TRIUX system
that is only acquiring a small number of magnetometer channels instead of the
whole array.

The OPM ``coil_type`` is custom, requiring a custom ``coil_def.dat``.
The new ``coil_type`` is 9999.

OPM co-registration differs a bit from the typical SQUID-MEG workflow.
No ``-trans.fif`` file is needed for the OPMs, the FIF files include proper
sensor locations in MRI coordinates and no digitization of RPA/LPA/Nasion.
Thus the MEG<->Head coordinate transform is taken to be an identity matrix
(i.e., everything is in MRI coordinates), even though this mis-identifies
the head coordinate frame (which is defined by the relationship of the
LPA, RPA, and Nasion).

Triggers include:

* Median nerve stimulation: trigger value 257.
* Magnetic trigger (in OPM measurement only): trigger value 260.
  1 second before the median nerve stimulation, a magnetic trigger is piped into the MSR.
  This was to be able to check the synchronization between OPMs retrospectively, as each
  sensor runs on an indepent clock. Synchronization turned out to be satisfactory

.. topic:: Examples

    * :ref:`sphx_glr_auto_examples_datasets_plot_opm_data.py`
    * :ref:`sphx_glr_auto_examples_datasets_plot_opm_rest_data.py`
