

.. _CHDDEFAB:

============
Introduction
============

This document describes a set of programs for preprocessing
and averaging of MEG and EEG data and for constructing cortically-constrained minimum-norm
estimates. This software package will in the sequel be referred to
as *MNE software*. The software is based on anatomical
MRI processing, forward modeling, and source estimation methods published in
Dale, Fischl, Hämäläinen, and others.
The software depends on anatomical MRI processing tools provided
by the FreeSurfer software.

:ref:`CHDBAFGJ` gives an overview of the software
modules included with MNE software. :ref:`ch_cookbook` is a concise cookbook
describing a typical workflow for a novice user employing the convenience
scripts as far as possible. :ref:`ch_browse` to :ref:`ch_misc` give more detailed
information about the software modules. :ref:`ch_sample_data` discusses
processing of the sample data set included with the MNE software. :ref:`ch_reading` lists
some useful background material for the methods employed in the
MNE software.

:ref:`create_bem_model` is an overview of the BEM model mesh
generation methods, :ref:`setup_martinos` contains information specific
to the setup at Martinos Center of Biomedical Imaging, :ref:`install_config` is
a software installation and configuration guide, :ref:`release_notes` summarizes
the software history, and :ref:`licence` contains the End-User
License Agreement.

.. note:: The most recent version of this manual is available    at ``$MNE_ROOT/share/doc/MNE-manual-`` <*version*> ``.pdf`` . For    the present manual, <*version*> = ``2.7`` .    For definition of the ``MNE_ROOT`` environment variable,    see :ref:`user_environment`.

We want to thank all MNE Software users at the Martinos Center and
in other institutions for their collaboration during the creation
of this software as well as for useful comments on the software
and its documentation.

The development of this software has been supported by the
NCRR *Center for Functional Neuroimaging Technologies* P41RR14075-06, the
NIH grants 1R01EB009048-01, R01 EB006385-A101, 1R01 HD40712-A1, 1R01
NS44319-01, and 2R01 NS37462-05, ell as by Department of Energy
under Award Number DE-FG02-99ER62764 to The MIND Institute. 
