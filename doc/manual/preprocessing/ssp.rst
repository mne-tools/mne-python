.. _ssp-api:

The Signal-Space Projection API
###############################

This page describes the MNE-Python API for signal-space projection (SSP). For
background information on SSP, see :ref:`CACCHABI`. For a tutorial on applying
SSP for artifact correction, see :ref:`tut-artifact-ssp`.

.. contents:: Page contents
   :local:
   :depth: 2

.. note::
   Once a projector is applied to the data, it is said to be *active*.

The proj attribute
------------------

All the basic data containers (:class:`~mne.io.Raw`, :class:`~mne.Epochs`, and
:class:`~mne.Evoked`) have a ``proj`` attribute. It is ``True`` if at least one
projector is present and all of them are *active*.

Computing projectors
--------------------

In MNE-Python SSP vectors can be computed using general
purpose functions :func:`mne.compute_proj_epochs`,
:func:`mne.compute_proj_evoked`, and :func:`mne.compute_proj_raw`.
The general assumption these functions make is that the data passed contains
raw, epochs or averages of the artifact. Typically this involves continuous raw
data of empty room recordings or averaged ECG or EOG artifacts.

A second set of high-level convenience functions is provided to compute
projection vectors for typical use cases. This includes
:func:`mne.preprocessing.compute_proj_ecg` and
:func:`mne.preprocessing.compute_proj_eog` for computing the ECG and EOG
related artifact components, respectively. For computing the EEG reference
signal, the function :func:`mne.set_eeg_reference` can be used.

.. warning:: It is best to compute projectors only on channels that will be
             used (e.g., excluding bad channels). This ensures that
             projection vectors will remain ortho-normalized and that they
             properly capture the activity of interest.

.. _remove_projector:

Adding/removing projectors
--------------------------

To explicitly add a ``proj``, use ``add_proj``. For example::

    >>> projs = mne.read_proj('proj_a.fif')  # doctest: +SKIP
    >>> evoked.add_proj(projs)  # doctest: +SKIP

If projectors are already present in the raw :file:`fif` file, the new
projector will be added to the ``info`` dictionary automatically. To remove
existing projectors first, you can do::

	>>> evoked.add_proj([], remove_existing=True)  # doctest: +SKIP

Applying projectors
-------------------

Projectors can be applied at any stage of the pipeline. When the ``raw`` data
is read in, the projectors are not applied by default but this flag can be
turned on. However, at the ``epochs`` stage, the projectors are applied by
default.

To apply explicitly projs at any stage of the pipeline, use ``apply_proj``. For
example::

	>>> evoked.apply_proj()  # doctest: +SKIP

The projectors might not be applied if data are not :ref:`preloaded <memory>`.
In this case, it's the ``_projector`` attribute that indicates if a projector
will be applied when the data is loaded in memory. If the data is already in
memory, then the projectors applied to it are the ones marked as `active`. As
soon as you've applied the projectors, it will stay active in the remaining
pipeline.

.. Warning:: Once a projection operator is applied, it cannot be reversed.

.. Warning::
   Projections present in the info are applied during inverse computation
   whether or not they are *active*. Therefore, if a certain projection should
   not be applied, remove it from the info as described in Section
   :ref:`remove_projector`

Delayed projectors
------------------

The suggested pipeline is ``proj=True`` in epochs (it's computationally cheaper
to apply projectors to epochs than to raw). When you use delayed SSP in
``Epochs``, projectors are applied when you call :func:`mne.Epochs.get_data`
method. They are not applied to the ``evoked`` data unless you call
``apply_proj()``. The reason is that you want to reject epochs with projectors
although it's not stored in the projector mode.

.. topic:: Examples:

    * :ref:`tut-artifact-ssp`: SSP sensitivities in sensor space
    * :ref:`ex-sensitivity-maps`: SSP sensitivities in source space
