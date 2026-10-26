"""
.. _tut-xfit:

=====================================================================
Source localization by guided equivalent current dipole (ECD) fitting
=====================================================================

This combination of manual specification and automated fitting is one of the oldest MEG
source estimation techniques :footcite:`Sarvas1987`. We will manually identify where and
when dipole source are active, upon which the fitting algorithm will find the best
location for the source. The result is a sparse source estimate of several equivalent
current dipoles (ECDs) that together explain (most of) the MEG evoked response. ECDs are
especially suited for capturing individual components of an evoked response (e.g. N100m,
N400m, etc.). Once the set of ECDs has been established, their timecourses can be
computed for multiple :class:`~mne.Evoked` objects, for example different experimental
conditions.

This tutorial will demonstrate how to fit ECDs using the interactive GUI and also how to
drive that same GUI from Python code. Every screenshot below is of the *same* GUI
window, so you can follow along how its state evolves as we go.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# sphinx_gallery_preserve_gui = True

# %%
# Guided ECD fitting using the GUI
# --------------------------------
#
# Starting the GUI
# ~~~~~~~~~~~~~~~~
# We can start the GUI from the command line by passing the ``mne dipolefit`` program
# an evoked file (filename typically ends in ``*-ave.fif``):
#
# .. code-block:: console
#
#     $ mne dipolefit sample_audvis-ave.fif
#
# The GUI can also be started from an interactive python console. The minimal setup that
# can be used to fit dipoles is an :class:`~mne.Evoked` object and nothing else:
#
# .. code-block:: python
#
#     mne.gui.dipolefit(evoked)
#
# When only specifying an evoked object, the GUI shows the sensors, a spherical head
# model and the electro-magnetic field recorded by the sensors, using an ad-hoc noise
# covariance matrix. If we provide more information, we can create a more accurate head
# model that provides better ECD fits and gives us more guidance for determining
# sources. On the command line there are various options you can use to specify files
# containing the covariance matrix, BEM model and MRI<->head transformation, see the
# output of ``mne dipolefit --help``. In an interactive python console, we can provide
# the appropriate MNE-Python objects when starting the GUI:

import mne

path = mne.datasets.sample.data_path()
meg_dir = path / "MEG" / "sample"
subjects_dir = path / "subjects"

evoked = mne.read_evokeds(meg_dir / "sample_audvis-ave.fif", condition="Left Auditory")
evoked.apply_baseline()

cov = mne.read_cov(meg_dir / "sample_audvis-cov.fif")
bem = mne.read_bem_solution(
    subjects_dir / "sample" / "bem" / "sample-5120-5120-5120-bem-sol.fif"
)
trans = mne.read_trans(meg_dir / "sample_audvis_raw-trans.fif")

# A distributed source estimate is a helpful guide for our dipole fits.
inv = mne.minimum_norm.read_inverse_operator(
    meg_dir / "sample_audvis-meg-oct-6-meg-inv.fif"
)
stc = mne.minimum_norm.apply_inverse(evoked, inv)

# Open the GUI with a better head model.
fitting_gui = mne.gui.dipolefit(
    evoked,
    cov=cov,
    bem=bem,
    trans=trans,
    stc=stc,
    ch_type="meg",  # only use MEG sensors for this tutorial
    subject="sample",
    subjects_dir=subjects_dir,
)

# %%
# Fitting a dipole
# ~~~~~~~~~~~~~~~~
# During guided ECD fitting, we look for patterns in the electro-magnetic field to
# identify when and where sources may be active. We can use the time slider to examine
# how the field changes over time. The sample data is an evoked response to an auditory
# tone being played to the left of the participant and we can see the initial auditory
# response peaking at around 85 ms on the right hemisphere. The field shows a typical
# di-polar pattern with a pair of red/blue focii on either side of the source that
# should be located in auditory cortex (the distributed source estimate shows where it
# is).
#
# By pressing the "Fit dipole" button we instruct the algorithm to fit a dipole at the
# current time. After a few seconds of computation, the resulting dipole will be
# displayed as an arrow in the brain, indicating its source, as well as an arrow on the
# MEG helmet indicating the fit between the dipole and the field pattern. The timecourse
# of the dipole is shown below. On the right are controls to name, remove, temporarily
# (de-)activate, and save the dipole to a file. You also find a toggle switch to make
# the dipole's orientation dynamic or keep it fixed at the orientation it had at the
# time when it was fitted.
#
# Dragging the time slider and clicking the "Fit dipole" button have programmatic
# equivalents, which is what we will use here to build up the figures in this tutorial:

fitting_gui.set_time(0.085)
fitting_gui.fit_dipole()

# %%
# Adjusting the 3D view
# ~~~~~~~~~~~~~~~~~~~~~
# Each dipole is drawn deep inside the head, so the "Meshes" panel on the left offers a
# visibility checkbox and an opacity slider for every surface (the cortex, the head, the
# MEG helmet, the sensors, and the colorbar of the distributed source estimate) to
# uncover whatever the fit needs. Underneath are buttons for the five standard views of
# the head. Both have programmatic equivalents:

fitting_gui.toggle_mesh("helmet", show=False)
fitting_gui.set_mesh_opacity("head", 0.1)

# %%
# Selecting channels to guide the ECD modeling
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# At nearly the same time, auditory responses are occurring in both left and right
# auditory cortex. Hence, the single dipole that we fitted without any guidance will
# have some bias as the algorithm attempted to fit the ECD to the entire bi-lateral
# field pattern.
#
# To isolate portions of the field pattern that contain a single pair of red/blue focii,
# the ideal fitting target for the algorithm, we can restrict the analysis to a subset
# of sensors. To do so, first press the "Sensor data" button, which will open a new
# window showing the evoked response across all sensors. By clicking and dragging the
# mouse we can make a lasso selection around the sensors we wish to include in the
# analysis. Hold ``CTRL`` to add to the current selection and ``CTRL + SHIFT`` to remove
# from the current selection. The currently selected sensors are highlighted in green in
# the main window, showing the portion of the field pattern they cover. When you are
# happy with the selection, you can use the "Fit dipole" button as before to fit a
# dipole using the selected sensors at the current timepoint.
#
# Remove or de-activate the dipole we previously fitted to the entire field pattern and
# fit two dipoles using the left-side and right-side sensors respectively. It is helpful
# to name them. This part of the workflow is inherently interactive, so we cannot
# reproduce it here in code.

# %%
# Multi/single dipole modes
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# By default, the fitting algorithm is in "Multi dipole (MNE)" mode, meaning portions of
# the signal attributed to one dipole can not be attributed to a second dipole at the
# same time. You will notice that if you have two dipoles with similar orientations
# close to each other, their timecourses become a strange mixture as each dipole will
# claim a part of the same signal. To prevent this, we can switch the algorithm over to
# "Single dipole" using the "Dipole model" dropdown. In this mode, the timecourse of
# each dipole will be computed whilst ignoring all other dipoles, which is useful when
# evaluating multiple candidate dipoles for the same source.

# %%
# Saving and loading sets of dipoles
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We can save the fitted dipoles using the "Save dipoles" button. There are two possible
# file formats for this: plain text (``.dip``) and a binary format (``.bdip``). These
# formats are compatible with MEGIN's software, allowing interoperability between
# MNE-Python and Xfit. Saved dipoles can be read back with :func:`mne.read_dipole` and
# added to an existing dipole fitting GUI, optionally under a name of our choosing.
# Here, we add the dipole that a lasso selection of the left-side sensors would have
# produced:

dips_to_add = mne.read_dipole(meg_dir / "sample_audvis_set1.dip")
dips_to_add = dips_to_add[[33]]  # add only one of the 34 dipoles in the file
fitting_gui.add_dipole(dips_to_add, name="lh")

# %%
# Working with the dipoles from Python
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Although each dipole is fitted at a single time point, its timecourse is estimated
# across the entire epoch, so we can move the time cursor to any latency to inspect the
# dipole model there. The dipoles themselves are available as a list of
# :class:`mne.Dipole` objects and can be saved without touching the GUI at all:

fitting_gui.set_time(0.115)
fitted_dipoles = fitting_gui.dipoles  # the dipoles we fitted
print([dip.name for dip in fitted_dipoles])
# save with: fitting_gui.save("my_file.dip")

# %%
# References
# ----------
# .. footbibliography::
