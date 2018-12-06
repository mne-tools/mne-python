
.. XXX: Things I've removed from inverse.rst

==================================
legacy
==================================

.. contents:: Contents
   :local:
   :depth: 2

.. _movies_and_snapshots:

Producing movies and snapshots
##############################
.. XXX: maybe movies & snapshots should go in using mne or as a side note in cookbook or somewhere in ploting

:ref:`mne_make_movie` is a program
for producing movies and snapshot graphics frames without any graphics
output to the screen. In addition, :ref:`mne_make_movie` can
produce stc or w files which contain the numerical current estimate
data in a simple binary format for postprocessing. These files can
be displayed in :ref:`mne_analyze`,
see :ref:`ch_interactive_analysis`, utilized in the cross-subject averaging
process, see :ref:`sphx_glr_auto_tutorials_plot_morph_stc.py`,
and read into Matlab using the MNE Matlab toolbox, see :ref:`ch_matlab`.

.. _computing_inverse:

Computing inverse from raw and evoked data
##########################################

The purpose of the utility :ref:`mne_compute_raw_inverse` is
to compute inverse solutions from either evoked-response or raw
data at specified ROIs (labels) and to save the results in a fif
file which can be viewed with :ref:`mne_browse_raw`,
read to Matlab directly using the MNE Matlab Toolbox, see :ref:`ch_matlab`,
or converted to Matlab format using either :ref:`mne_convert_mne_data`,
:ref:`mne_raw2mat`, or :ref:`mne_epochs2mat`. See
:ref:`mne_compute_raw_inverse` for command-line options.

.. _implementation_details:

Implementation details
======================

The fif files output from mne_compute_raw_inverse have
various fields of the channel information set to facilitate interpretation
by postprocessing software as follows:

**channel name**

    Will be set to J[xyz] <*number*> ,
    where the source component is indicated by the coordinat axis name
    and number is the vertex number, starting from zero, in the complete
    triangulation of the hemisphere in question.

**logical channel number**

    Will be set to is the vertex number, starting from zero, in the
    complete triangulation of the hemisphere in question.

**sensor location**

    The location of the vertex in head coordinates or in MRI coordinates,
    determined by the ``--mricoord`` flag.

**sensor orientation**

    The *x*-direction unit vector will point to the
    direction of the current. Other unit vectors are set to zero. Again,
    the coordinate system in which the orientation is expressed depends
    on the ``--mricoord`` flag.

The ``--align_z`` flag tries to align the signs
of the signals at different vertices of the label. For this purpose,
the surface normals within the label are collected into a :math:`n_{vert} \times 3` matrix.
The preferred orientation will be taken as the first right singular
vector of this matrix, corresponding to its largest singular value.
If the dot product of the surface normal of a vertex is negative,
the sign of the estimates at this vertex are inverted. The inversion
is reflected in the current direction vector listed in the channel
information, see above.

.. note:: The raw data files output by :ref:`mne_compute_raw_inverse` can be converted to mat files with :ref:`mne_raw2mat`. Alternatively, the files can be read directly from Matlab using the routines in the MNE Matlab toolbox, see :ref:`ch_matlab`. The evoked data output can be easily read directly from Matlab using the fiff_load_evoked routine in the MNE Matlab toolbox. Both raw data and evoked output files can be loaded into :ref:`mne_browse_raw`, see :ref:`ch_browse`.
