"""
.. _tut-report:

Getting started with ``mne.Report``
===================================

This tutorial covers making interactive HTML summaries with
:class:`mne.Report`.

.. contents:: Page contents
   :local:
   :depth: 2

As usual we'll start by importing the modules we need and loading some
:ref:`example data <sample-dataset>`:
"""

import os
import mne

###############################################################################
# Before getting started with :class:`mne.Report`, make sure the files you want
# to render follow the filename conventions defined by MNE:
#
# .. cssclass:: table-bordered
# .. rst-class:: midvalign
#
# ============ ==============================================================
# Data object  Filename convention (ends with)
# ============ ==============================================================
# raw          -raw.fif(.gz), -raw_sss.fif(.gz), -raw_tsss.fif(.gz), _meg.fif
# events       -eve.fif(.gz)
# epochs       -epo.fif(.gz)
# evoked       -ave.fif(.gz)
# covariance   -cov.fif(.gz)
# trans        -trans.fif(.gz)
# forward      -fwd.fif(.gz)
# inverse      -inv.fif(.gz)
# ============ ==============================================================
#
#
# Basic reports
# ^^^^^^^^^^^^^
#
# The basic process for creating an HTML report is to instantiate the
# :class:`~mne.Report` class, then use the :meth:`~mne.Report.parse_folder`
# method to select particular files to include in the report. Which files are
# included depends on both the ``pattern`` parameter passed to
# :meth:`~mne.Report.parse_folder` and also the ``subject`` and
# ``subjects_dir`` parameters provided to the :class:`~mne.Report` constructor.
#
# .. sidebar: Viewing the report
#
#    On successful creation of the report, the :meth:`~mne.Report.save` method
#    will open the HTML in a new tab in the browser. To disable this, use the
#    ``open_browser=False`` parameter of :meth:`~mne.Report.save`.
#
# For our first example, we'll generate a barebones report for all the
# :file:`.fif` files containing raw data in the sample dataset, by passing the
# pattern ``*raw.fif`` to :meth:`~mne.Report.parse_folder`. We'll omit the
# ``subject`` and ``subjects_dir`` parameters from the :class:`~mne.Report`
# constructor, but we'll also pass ``render_bem=False`` to the
# :meth:`~mne.Report.parse_folder` method — otherwise we would get a warning
# about not being able to render MRI and ``trans`` files without knowing the
# subject.

path = mne.datasets.sample.data_path(verbose=False)
report = mne.Report(verbose=True)
report.parse_folder(path, pattern='*raw.fif', render_bem=False)
report.save('report_basic.html')

###############################################################################
# This report yields a textual summary of the :class:`~mne.io.Raw` files
# selected by the pattern. For a slightly more useful report, we'll ask for the
# power spectral density of the :class:`~mne.io.Raw` files, by passing
# ``raw_psd=True`` to the :class:`~mne.Report` constructor. Let's also refine
# our pattern to select only the filtered raw recording (omitting the
# unfiltered data and the empty-room noise recordings):

pattern = 'sample_audvis_filt-0-40_raw.fif'
report = mne.Report(verbose=True, raw_psd=True)
report.parse_folder(path, pattern=pattern, render_bem=False)
report.save('report_raw_psd.html')

###############################################################################
# This time we'll pass a specific ``subject`` and ``subjects_dir`` (even though
# there's only one subject in the sample dataset) and remove our
# ``render_bem=False`` parameter so we can see the MRI slices, with BEM
# contours overlaid on top if available. Since this is computationally
# expensive, we'll also pass the ``mri_decim`` parameter for the benefit of our
# documentation servers, and skip processing the :file:`.fif` files:

subjects_dir = os.path.join(path, 'subjects')
report = mne.Report(subject='sample', subjects_dir=subjects_dir, verbose=True)
report.parse_folder(path, pattern='', mri_decim=25)
report.save('report_mri_bem.html')

###############################################################################
# Now let's look at how :class:`~mne.Report` handles :class:`~mne.Evoked` data
# (we'll skip the MRIs to save computation time):

pattern = 'sample_audvis-no-filter-ave.fif'
report = mne.Report(verbose=True)
report.parse_folder(path, pattern=pattern, render_bem=False)
report.save('report_evoked.html')

###############################################################################
# To render whitened :class:`~mne.Evoked` files with baseline correction, add
# the noise covariance file. This will display ERP/F plots for both the
# original and whitened :class:`~mne.Evoked` objects, but scalp topomaps only
# for the original.

cov_fname = os.path.join(path, 'MEG', 'sample', 'sample_audvis-cov.fif')
report = mne.Report(cov_fname=cov_fname, verbose=True)
report.parse_folder(path, pattern=pattern, render_bem=False)
report.save('report_evoked_whitened.html')

###############################################################################
# If you want to actually *view* the noise covariance in the report, make sure
# it is captured by the pattern passed to :meth:`~mne.Report.parse_folder`, and
# also include a source for an :class:`~mne.Info` object (any of the
# :class:`~mne.io.Raw`, :class:`~mne.Epochs` or :class:`~mne.Evoked`
# :file:`.fif` files that contain subject data also contain the measurement
# information and should work):

pattern = 'sample_audvis-cov.fif'
info_fname = os.path.join(path, 'MEG', 'sample', 'sample_audvis-ave.fif')
report = mne.Report(info_fname=info_fname, verbose=True)
report.parse_folder(path, pattern=pattern, render_bem=False)
report.save('report_cov.html')

###############################################################################
# Adding custom plots to a report
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The python interface has greater flexibility compared to the :ref:`command
# line interface <gen_mne_report>`. For example, custom plots can be added via
# the :meth:`~mne.Report.add_figs_to_section` method:

# generate a custom plot:
fname_evoked = os.path.join(path, 'MEG', 'sample', 'sample_audvis-ave.fif')
evoked = mne.read_evokeds(fname_evoked,
                          condition='Left Auditory',
                          baseline=(None, 0),
                          verbose=True)
fig = evoked.plot(show=False)

# add the custom plot to the report:
report.add_figs_to_section(fig, captions='Left Auditory', section='evoked')
report.save('report_custom.html')

###############################################################################
# Managing report sections
# ^^^^^^^^^^^^^^^^^^^^^^^^
#
# The MNE report command internally manages the sections so that plots
# belonging to the same section are rendered consecutively. Within a section,
# the plots are ordered in the same order that they were added using the
# :meth:`~mne.Report.add_figs_to_section` command. Each section is identified
# by a toggle button in the top navigation bar of the report which can be used
# to show or hide the contents of the section. To toggle the show/hide state of
# all sections in the HTML report, press :kbd:`t`.
#
# .. note::
#
#    Although we've been generating separate reports in each example, you could
#    easily create a single report for all :file:`.fif` files (raw, evoked,
#    covariance, etc) by passing ``pattern='*.fif'``.
#
#
# Editing a saved report
# ^^^^^^^^^^^^^^^^^^^^^^
#
# Saving to HTML is a write-only operation, meaning that we cannot read an
# ``.html`` file back as a :class:`~mne.Report` object. In order to be able
# to edit a report once it's no longer in-memory in an active Python session,
# save it as an HDF5 file instead of HTML:

report.save('report.h5', overwrite=True)
report_from_disk = mne.open_report('report.h5')
print(report_from_disk)

###############################################################################
# This allows the possibility of multiple scripts adding figures to the same
# report. To make this even easier, :class:`mne.Report` can be used as a
# context manager:

with mne.open_report('report.h5') as report:
    report.add_figs_to_section(fig,
                               captions='Left Auditory',
                               section='evoked',
                               replace=True)
    report.save('report_final.html')

###############################################################################
# With the context manager, the updated report is also automatically saved
# back to :file:`report.h5` upon leaving the block.
