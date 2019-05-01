:orphan:

.. include:: links.inc

.. _glossary:

==========================
Glossary
==========================

.. .. contents:: Contents
..    :local:


.. currentmodule:: mne

MNE-Python core terminology and general concepts
================================================

.. glossary::


    annotations
        An annotation is defined by an onset, a duration, and a string
        description. It can contain information about the experiments, but
        also details on signals marked by a human: bad data segments,
        sleep scores, sleep events (spindles, K-complex) etc.
        An :class:`Annotations` object is a container of multiple annotations.
        See :class:`Annotations` page for the API of the corresponding
        object class and :ref:`sphx_glr_auto_tutorials_plot_object_annotations.py`
        for a tutorial on how to manipulate such objects.

    channels
        Channels refer to MEG sensors, EEG electrodes or any extra electrode
        or sensor such as EOG, ECG or sEEG, ECoG etc. Channels have typically
        a type, such as gradiometer, and a unit, such as Tesla/Meter that
        is used in the code base, e.g. for plotting.

    BEM
        BEM is the acronym for boundary element method or boundary element
        model. Both are related to the forward model computation and more
        specifically the definion of the conductor model. The
        boundary element model consists of surfaces such as the inner skull,
        outer skull and outer skiln (a.k.a. scalp) that define compartments
        of tissues of the head. You can compute the BEM surfaces with
        :func:`mne.bem.make_watershed_bem` or :func:`mne.bem.make_flash_bem`.
        See :ref:`sphx_glr_auto_tutorials_plot_forward.py` for usage demo.

    epochs
        Epochs are chunks of data extracted from raw continuous data. Typically,
        they correspond to the trials of an experimental design.
        See :class:`Epochs` for the API of the corresponding
        object class, and :ref:`sphx_glr_auto_tutorials_plot_object_epochs.py` for a
        narrative overview.

    evoked
        Evoked data are obtained by averaging epochs. Typically, an evoked object
        is constructed for each subject and each condition, but it can also be
        obtained by averaging a list of evoked over different subjects.
        See :class:`EvokedArray` for the API of the corresponding
        object class, and :ref:`sphx_glr_auto_tutorials_plot_object_evoked.py`
        for a narrative overview.

    events
        Events correspond to specific time points in raw data; e.g.,
        triggers, experimental condition events, etc. MNE represents events with
        integers that are stored in numpy arrays of shape (n_events, 3). Such arrays
        are classically obtained from a trigger channel, also referred to as
        stim channel.

    first_samp
        The attribute of raw objects called ``first_samp`` is an integer that
        refers to the number of time samples passed between the onset of the
        acquisition system and the time when data started to be written
        on disk. This is a specificity of the Vectorview MEG systems (fif files)
        but for consistency it is available for all file formats in MNE.
        One benefit of this system is that croppping data only boils
        down to a change of the ``first_samp`` attribute to know when cropped data
        was acquired.

    info
        Also called ``measurement info``, it is a collection of metadata regarding
        a Raw, Epochs or Evoked object; e.g.,
        channel locations and types, sampling frequency,
        preprocessing history such as filters ...
        See :ref:`sphx_glr_auto_tutorials_plot_info.py` for a narrative
        overview.

    label
        A :class:`Label` refers to a region in the cortex, also often called
        a region of interest (ROI) in the literature.

    montage
        EEG channel names and the relative positions of the sensor w.r.t. the scalp.
        See :class:`~channels.Montage` for the API of the corresponding object
        class.

    morphing
        Morphing refers to the operation of transferring source estimates from
        one anatomy to another. It is commonly referred as realignment in fMRI
        literature. This operation is necessary for group studies.
        See :ref:`ch_morph` for more details.

    pick
        An integer that is the index of a channel in the measurement info.
        It allows to obtain the information on a channel in the list of channels
        available in ``info['chs']``.

    projector, (abbr. ``proj``)
        A projector, also referred to a Signal Suspace Projection (SSP), defines
        a linear operation applied spatially to EEG or MEG data. You can see
        this as a matrix multiplication that reduces the rank of the data by
        projecting it to a lower dimensional subspace. Such a projection
        operator is applied to both the data and the forward operator for
        source localization. Note that EEG average referencing can be done
        using such a projection operator. It is stored in the measurement
        info in ``info['projs']``.

    raw
        It corresponds to continuous data (preprocessed or not). One typically
        manipulates raw data when reading recordings in a file on disk.
        See :class:`~io.RawArray` for the API of the corresponding
        object class, and :ref:`sphx_glr_auto_tutorials_plot_object_raw.py` for a
        narrative overview.

    source space (abbr. ``src``)
        A source space specifies where in the brain one wants to estimate the
        source amplitudes. It corresponds to locations of a set of
        candidate equivalent current dipoles (ECD). MNE mostly works
        with source spaces defined on the cortical surfaces estimated
        by FreeSurfer from a T1-weighted MRI image. See
        :ref:`sphx_glr_auto_tutorials_plot_forward.py` to read on
        how to compute a forward operator on a source space.
        See :class:`SourceSpaces` for the API of the corresponding
        object class.

    source estimates (abbr. ``stc``)
        Source estimates, commonly referred to as STC (Source Time Courses),
        are obtained from source localization methods,
        such as dSPM, sLORETA, LCMV or MxNE.
        It contains the amplitudes of the sources over time.
        An STC object only stores the amplitudes of activations but
        not the locations of the sources. To get access to the locations
        you need to have the source space used to compute the forward
        operator.
        See :class:`SourceEstimate`, :class:`VolSourceEstimate`
        :class:`VectorSourceEstimate`, :class:`MixedSourceEstimate`,
        for the API of the corresponding object classes.

    selection (abbr. sel)
        A set of picks. E.g., all sensors included in a Region of Interest.

    stim channel
        A stim channel, a.k.a. trigger channel, is a channel that encodes events
        during the recording. It is typically a channel that is always zero and that
        takes positive values when something happens such as the onset of a stimulus.
        Classical names for stim channels is ``STI 014`` or ``STI 101``.
        So-called events arrays are obtained from stim channels.

    trans
        A coordinate frame affine transformation, usually between the Neuromag head
        coordinate frame and the MRI Surface RAS coordinate frame used by Freesurfer.
