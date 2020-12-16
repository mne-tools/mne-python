# -*- coding: utf-8 -*-
r"""
.. _tut-freesurfer-reconstruction:

=============================
FreeSurfer MRI reconstruction
=============================

FreeSurfer is an open source analysis toolbox for MRI data. It contains several
command line tools and graphical user interfaces. FreeSurfer can be obtained
from https://surfer.nmr.mgh.harvard.edu/

In MNE, FreeSurfer is used to provide structural information of various
kinds, for :ref:`source estimation <tut-inverse-methods>`. Thereby a
subject specific structural MRI will be used to obtain various structural
representations like spherical or inflated brain surfaces. Furthermore features
like curvature as well as various labels for areas of interest (such as V1) are
computed.

Thus FreeSurfer provides an easy way to shift anatomically related
data between different representations and spaces. See e.g.
:ref:`ch_morph` for information about how to
use FreeSurfer surface representations to allow functional data to morph
between different subjects.

.. contents::
    :local:

First steps
===========

After downloading and installing, the environment needs to be set up correctly.
This can be done by setting the FreeSurfer's root directory correctly and
sourcing the setup file::

    $ export FREESURFER_HOME=/path/to/FreeSurfer
    $ source $FREESURFER_HOME/SetUpFreeSurfer.sh

.. note::
    The FreeSurfer home directory might vary depending on your operating
    system. See the `FreeSurfer installation guide
    <https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall>`_ for more.

Another important step is to define the subject directory correctly.
``SUBJECTS_DIR`` must be defined such, that it contains the individual
subject's reconstructions in separate sub-folders. Those sub-folders will be
created upon the reconstruction of the anatomical data. Nevertheless the parent
directory has to be set beforehand::

    $ export SUBJECTS_DIR=~/subjects

Again see the `FreeSurfer installation guide
<https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall>`_ for more.

Once setup correctly, FreeSurfer will create a new subject folder in
``$SUBJECTS_DIR``.

Anatomical reconstruction
=========================

MNE-Python works together with FreeSurfer in order to compute the forward model
and setting up the corresponding :class:`source space <mne.SourceSpaces>`. See
:ref:`setting_up_source_space` for more information. Usually a full FreeSurfer
reconstruction is obtained by prompting the following command to a bash
console (e.g. Linux or MacOS Terminal)::

    $ my_subject=sample
    $ my_NIfTI=/path/to/NIfTI.nii.gz
    $ recon-all -i $my_NIfTI -s $my_subject -all

where :code:`i` stands for "input" and :code:`s` for "subject". Executing
this, will create the folder "~/subjects/sample", where all
results are stored.

.. note::
    This compution takes often several hours. Please be patient.

Within a single subject all the files MNE-Python uses (and some more) are
grouped into meaningful sub-folders (such that "surf" contains surface
representations, "mri" volumetric files, etc.).

FreeSurfer performs a hemispheric separation and most results are present
in a left and right hemisphere version. This is often indicated by the
prefix ``lh`` or ``rh`` to refer to the aforementioned. For that reason
data representations such as :class:`mne.SourceEstimate` carry two sets of
spatial locations (vertices) for both hemispheres separately. See also
:ref:`tut-source-estimate-class`.
"""

import mne
subjects_dir = mne.datasets.sample.data_path() + '/subjects'
Brain = mne.viz.get_brain_class()
brain = Brain('sample', hemi='lh', surf='pial',
              subjects_dir=subjects_dir, size=(800, 600))
brain.add_annotation('aparc.a2009s', borders=False)

###############################################################################
# 'fsaverage'
# ===========
#
# During installation, FreeSurfer copies a "default" subject, called
# ``'fsaverage'`` to ``$FREESURFER_HOME/subjects/fsaverage``. It contains all
# data types that a subject reconstruction would yield and is required by
# MNE-Python.
#
# See https://surfer.nmr.mgh.harvard.edu/fswiki/FsAverage for an overview, and
# https://surfer.nmr.mgh.harvard.edu/fswiki/Buckner40Notes for details about
# the included subjects. A copy of 'fsaverage' can be found in the
# :ref:`sample-dataset` dataset and is also distributed as a :ref:`standalone
# dataset <fsaverage>`.
#
# When using ``'fsaverage'`` as value for the definition
# of a subject when calling a function, the corresponding data will be read
# (e.g., ``subject='fsaverage'``) from '~/subjects/fsaverage'. This becomes
# especially handy, when attempting statistical analyses on group level, based
# on individual's brain space data. In that case ``'fsaverage'`` will by
# default act as reference space for
# :ref:`source estimate transformations <ch_morph>`.
#
# For example, to reproduce a typical header image used by FreeSurfer, we can
# plot the ``aparc`` parcellation:
#
# Use with MNE-Python
# ===================
#
# For source localization analyses to work properly, it is important that the
# FreeSurfer reconstruction has completed beforehand. Furthermore, when using
# related functions, such as :func:`mne.setup_source_space`, ``SUBJECTS_DIR``
# has to be defined either globally by setting :func:`mne.set_config` or for
# each function separately, by passing the respective keyword argument
# ``subjects_dir='~/subjects'``.
#
# See also :ref:`setting_up_source_space` to get an idea of how this works for
# one particular function, and :ref:`tut-freesurfer-mne` for how the two are
# integrated.
