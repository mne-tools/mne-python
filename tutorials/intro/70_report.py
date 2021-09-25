"""
.. _tut-report:

Getting started with ``mne.Report``
===================================

`mne.Report` is a way to create interactive HTML summaries of your data. These
reports can show many different visualizations of one or multiple subject's
data. A common use case is creating diagnostic summaries to check data quality
at different stages in the processing pipeline. The report can show things like
plots of data before and after each preprocessing step, epoch rejection
statistics, MRI slices with overlaid BEM shells, all the way up to plots of
estimated cortical activity.

Compared to a Jupyter notebook, `mne.Report` is easier to deploy (the HTML
pages it generates are self-contained and do not require a running Python
environment) but less flexible (you can't change code and re-run something
directly within the browser). This tutorial covers the basics of building a
`~mne.Report`. As usual we'll start by importing the modules and data we need:
"""

# %%

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import mne

data_path = Path(mne.datasets.sample.data_path(verbose=False))
sample_dir = data_path / 'MEG' / 'sample'
subjects_dir = data_path / 'subjects'

# %%
# Before getting started with :class:`mne.Report`, make sure the files you want
# to render follow the filename conventions defined by MNE:
#
# .. cssclass:: table-bordered
# .. rst-class:: midvalign
#
# =================================== =========================================
# Data object                         Filename convention (ends with)
# =================================== =========================================
# `~mne.io.Raw`                       ``-raw.fif(.gz)``, ``-raw_sss.fif(.gz)``,
#                                     ``-raw_tsss.fif(.gz)``,
#                                     ``_meg.fif(.gz)``, ``_eeg.fif(.gz)``,
#                                     ``_ieeg.fif(.gz)``
# events                              ``-eve.fif(.gz)``
# `~mne.Epochs`                       ``-epo.fif(.gz)``
# `~mne.Evoked`                       ``-ave.fif(.gz)``
# `~mne.Covariance`                   ``-cov.fif(.gz)``
# `~mne.Projection`                   ``-proj.fif(.gz)``
# `~mne.transforms.Transform`         ``-trans.fif(.gz)``
# `~mne.Forward`                      ``-fwd.fif(.gz)``
# `~mne.minimum_norm.InverseOperator` ``-inv.fif(.gz)``
# =================================== =========================================
#
# Alternatively, the dash ``-`` in the filename may be replaced with an
# underscore ``_``.
#
# The basic process for creating an HTML report is to instantiate the
# :class:`~mne.Report` class and then use one or more of its many methods to
# add content, one element at a time.
# 
# You may also use the :meth:`~mne.Report.parse_folder` method to select
# particular files to include in the report. But more on that later.
#
# .. sidebar: Viewing the report
#
#    On successful creation of the report, the :meth:`~mne.Report.save` method
#    will open the HTML in a new tab in the browser. To disable this, use the
#    ``open_browser=False`` parameter of :meth:`~mne.Report.save`.
#

# %%
# Adding `~mne.io.Raw` data
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Raw data can be added via the :meth:`mne.Report.add_raw` method. It can
# operate with a path to a raw file and `~mne.io.Raw` objects:

raw_path = sample_dir / 'sample_audvis_filt-0-40_raw.fif'
raw = mne.io.read_raw(raw_path)

report = mne.Report()
report.add_raw(raw=raw_path, title='Raw from Path')
report.add_raw(raw=raw, title='Raw from "raw"', psd=False)  # omit PSD plot
report.save('report_raw.html', overwrite=True)

# %%
# Adding events
# ^^^^^^^^^^^^^
#
# Events can be added via :meth:`mne.Report.add_events`. You also need to
# supply the sampling frequency used during the recording.

events_path = sample_dir / 'sample_audvis_filt-0-40_raw-eve.fif'
events = mne.read_events(events_path)
sfreq = raw.info['sfreq']

report = mne.Report()
report.add_events(events=events_path, title='Events from Path', sfreq=sfreq)
report.add_events(events=events, title='Events from "events"', sfreq=sfreq)
report.save('report_events.html', overwrite=True)

# %%
# Adding `~mne.Epochs`
# ^^^^^^^^^^^^^^^^^^^^
#
# Epochs can be added via :meth:`mne.Report.add_epochs`. Note that although
# this methods accepts a path to an epochs file too, in the following example
# we only add epochs that we create on the fly from raw data.

epochs = mne.Epochs(raw=raw, events=events)

report = mne.Report()
report.add_epochs(epochs=epochs, title='Epochs from "epochs"')
report.save('report_epochs.html', overwrite=True)

# %%
# Adding `~mne.Evoked`
# ^^^^^^^^^^^^^^^^^^^^
#
# Evoked data can be added via :meth:`mne.Report.add_evokeds`. By default, the
# ``Evoked.comment`` attribute of each evoked will be used as a title. We can
# specify custom titles via the ``titles`` parameter. Again, this method
# also accepts the path to an evoked file stored on disk; in the following
# example, however, we load the evokeds manually first, since we only want to
# add a subset of them to the report. The evokeds are not baseline-corrected,
# so we apply baseline correction, too. Lastly, by providing an (optional)
# noise covariance, we can add plots evokeds that were "whitened" using this
# covariance matrix.

evoked_path = sample_dir / 'sample_audvis-ave.fif'
cov_path = sample_dir / 'sample_audvis-cov.fif'

evokeds = mne.read_evokeds(evoked_path)
evokeds_subset = evokeds[:2]  # The first two
evokeds_subset_bl_corr = [e.apply_baseline((None, 0))  # Baseline correction
                          for e in evokeds_subset]

report = mne.Report()
report.add_evokeds(
    evokeds=evokeds_subset_bl_corr,
    titles=['evoked 1',  # Manually specify titles
            'evoked 2'],
    noise_cov=cov_path
)

report.save('report_evoked.html', overwrite=True)

# %%
# Adding `~mne.Covariance`
# ^^^^^^^^^^^^^^^^^^^^^^^^
#
# (Noise) covariance objects can be added via
# :meth:`mne.Report.add_covariance`. The method accepts `~mne.Covariance`
# objects and the path to a file on disk. It also expects us to pass an
# `~mne.Info` object or the path to a file to read the measurement info from,
# as well as a title.

cov_path = sample_dir / 'sample_audvis-cov.fif'

report = mne.Report()
report.add_covariance(cov=cov_path, info=raw_path, title='Covariance')

report.save('report_cov.html', overwrite=True)

# %%
# Adding SSP `~mne.Projection` vectors
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# `~mne.Projection` vectors can be added via
# :meth:`mne.Report.add_spp_projs`. The method requires an `~mne.Info` object
# (or the path to one) and a title. Projectors found in the `~mne.Info` will
# be visualized. You may also supply a list of `~mne.Projection` objects or
# a path to projectors stored on disk. In this case, the channel information
# is read from the `~mne.Info`, but projectors potentially included will be
# ignored; instead, only the explicitly passed projectors will be plotted.

ecg_proj_path = sample_dir / 'sample_audvis_ecg-proj.fif'
eog_proj_path = sample_dir / 'sample_audvis_eog-proj.fif'

report = mne.Report()
report.add_ssp_projs(info=raw_path, title='Projs from info')
report.add_ssp_projs(info=raw_path, projs=ecg_proj_path,
                     title='ECG projs from path')
report.add_ssp_projs(info=raw_path, projs=eog_proj_path,
                     title='EOG projs from path')

report.save('report_ssp_projs.html', overwrite=True)


# %%
# Adding MRI with BEM
# ^^^^^^^^^^^^^^^^^^^
#
# MRI slices with superimposed traces of the boundary element model (BEM)
# surfaces can be added via :meth:`mne.Report.add_bem`. All you need to pass is
# the FreeSurfer subject name and subjects directory, and a title.

report = mne.Report()
report.add_bem(subject='sample', subjects_dir=subjects_dir, title='MRI & BEM')
report.save('report_mri_and_bem.html', overwrite=True)

# %%
# Adding coregistration
# ^^^^^^^^^^^^^^^^^^^^^
#
# The `head -> mri` transformation ("coregistration") can be visualized via
# :meth:`mne.Report.add_trans`. The method expects the transformation either as
# a `~mne.Transform` object or as a path to a `trans.fif` file, the FreeSurfer
# subject name and subjects directory, and a title.

trans_path = sample_dir / 'sample_audvis_raw-trans.fif'

report = mne.Report()
report.add_trans(
    trans=trans_path, info=raw_path, subject='sample',
    subjects_dir=subjects_dir, title='Coregistration'
)

report.save('report_coregistration.html', overwrite=True)

# %%
# Adding a `~mne.Forward` solution
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Forward solutions ("leadfields") can be added by passing a `~mne.Forward`
# object or the path to a forward solution stored on disk to 
# meth:`mne.Report.add_forward`.

fwd_path = sample_dir / 'sample_audvis-meg-oct-6-fwd.fif'

report = mne.Report()
report.add_forward(forward=fwd_path, title='Forward solution')

report.save('report_forward_sol.html', overwrite=True)

# %%
# Adding an `~mne.minimum_norm.InverseOperator`
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# An inverse operator can be added via :meth:`mne.Report.add_inverse`. The
# method expects an `~mne.minimum_norm.InverseOperator` object or a path to one
# stored on disk, and a title.
# 
# Optionally, you may pass the corresponding FreeSurfer subject name, subjects
# directory, and a `head -> mri` transformation, either as a `~mne.Transform`
# object or as a path to a `trans.fif` file to add a visualization of the
# source space the provided inverse operator is based on.

inverse_op_path = sample_dir / 'sample_audvis-meg-oct-6-meg-inv.fif'

report = mne.Report()
report.add_inverse_op(inverse_op=inverse_op_path, title='Inverse operator',
                      subject='sample', subjects_dir=subjects_dir,
                      trans=trans_path)

report.save('report_inverse_op.html', overwrite=True)


# %%
# Adding a `~mne.SourceEstimate`
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# An inverse solution (also called source estimate or source time course, STC)
# can be added vua :meth:`mne.Report.add_stc`. The
# method expects an `~mne.SourceEstimate, the corresponding FreeSurfer subject
# name and subjects directory, and a title

stc_path = sample_dir / 'sample_audvis-meg'

report = mne.Report()
report.add_stc(stc=stc_path, subject='sample', subjects_dir=subjects_dir,
               title='Source estimate')

report.save('report_inverse_sol.html', overwrite=True)


# %%
# Adding system information
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# In order to improve reproducibility of results, it is useful to document some
# key information on the system that was used to create the report. The output
# of the helpful `mne.sys_info` command can be automatically added to a report
# via :meth:`mne.Report.add_sys_info`.

report = mne.Report()
report.add_sys_info(title='System info')
report.save('report_sys_info.html', overwrite=True)

# %%
# Adding code (e.g., a Python script)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# It is possible to add code or scripts (e.g., the scripts you used for
# analysis) to the report via :meth:`mne.Report.add_code`. The code blocks will
# be automatically syntax-highlighted. You may pass a string with the
# respective code snippet, or the path to a file. Optionally, you can specify
# which programming language to assume for syntax highlighting by passing the
# ``language`` parameter. By default, we'll assume the provided code is Python.

report = mne.Report()
report.add_code(
    code=mne.__file__,  # This will point to __init__.py in the MNE-Python root
    title='mne.__init__.py'
)
report.save('report_code.html', overwrite=True)

# %%
# Adding custom plots
# ^^^^^^^^^^^^^^^^^^^
#
# Custom Matplotlib figures can be added via :meth:`~mne.Report.add_figure`.
# You may even add captions to appear below the figure!

x = np.linspace(start=0, stop=10, num=100)
y = x**2

fig, ax = plt.subplots()
ax.plot(x, y, ls='--', lw=2, color='blue', label='my function')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.legend()

report = mne.Report(verbose=True)
report.add_figure(fig=fig, title='A custom figure',
                  caption='A blue dashed line reaches up into the sky …')
report.save('report_custom_figure.html', overwrite=True)

# %%
# Adding image files
# ^^^^^^^^^^^^^^^^^^
#
# Existing images (e.g., photos, screenshots, sketches etc.) can be added
# to the report via :meth:`mne.Report.add_image`. Supported image formats
# include JPEG, PNG, GIF, and SVG (and possibly others). Like with Matplotlib
# figures, you can specify a caption to appear below the image.
#
# In the following example, we'll add the MNE logo.

mne_logo_path = Path(mne.__file__).parent / 'icons' / 'mne_icon-cropped.png'

report = mne.Report(verbose=True)
report.add_image(image=mne_logo_path, title='MNE',
                 caption='Powered by 🧠 🧠 🧠 around the world!')
report.save('report_custom_image.html', overwrite=True)

# %%
# Adding a slider
# ^^^^^^^^^^^^^^^
#
# Sliders provide an intuitive way for users to interactively browse a
# predefined set of figures. You can add sliders via
# :meth:`~mne.Report.add_slider`. You need to provide a collection of figures,
# a title, and optionally a collection of captions, the index of the figure to
# display first (i.e., slider starting position), and the desired image format
# in which the figures will be saved before embedding them into the report.
#
# In the following example, we will read the MNE logo as a Matplotlib figure
# and rotate it with different angles. Each rotated figure and its respective
# caption will be added to a list, which is then used to create the slider.

fig_array = plt.imread(mne_logo_path)
rotation_angles = np.linspace(start=0, stop=360, num=17)

figs = []
captions = []
for angle in rotation_angles:
    # Rotate and remove some rounding errors to avoid Matplotlib warnings
    fig_array_rotated = scipy.ndimage.rotate(input=fig_array, angle=angle)
    fig_array_rotated = fig_array_rotated.clip(min=0, max=1)

    # Create the figure
    fig, ax = plt.subplots()
    ax.imshow(fig_array_rotated)
    ax.set_axis_off()

    # Add figure and caption to the lists we'll use to create the slider
    figs.append(fig)
    captions.append(f'Rotation angle: {round(angle, 1)}°')

report = mne.Report()
report.add_slider(figs=figs, title='Fun with sliders! 🥳', captions=captions)
report.save('report_slider.html', overwrite=True)

# %%
# Adding an entire folder of files
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# For our first example, we'll generate a barebones report for all the
# :file:`.fif` files containing raw data in the sample dataset, by passing the
# pattern ``*raw.fif`` to :meth:`~mne.Report.parse_folder`. We'll omit the
# ``subject`` and ``subjects_dir`` parameters from the :class:`~mne.Report`
# constructor, but we'll also pass ``render_bem=False`` to the
# :meth:`~mne.Report.parse_folder` method — otherwise we would get a warning
# about not being able to render MRI and ``trans`` files without knowing the
# subject.

report = mne.Report(verbose=True)
report.parse_folder(data_path, pattern='*raw.fif', render_bem=False)
report.save('report_basic.html', overwrite=True)

# %%
# This report yields a textual summary of the :class:`~mne.io.Raw` files
# selected by the pattern. For a slightly more useful report, we'll ask for the
# power spectral density of the :class:`~mne.io.Raw` files, by passing
# ``raw_psd=True`` to the :class:`~mne.Report` constructor. We'll also
# visualize the SSP projectors stored in the raw data's `~mne.Info` dictionary
# by setting ``projs=True``. Lastly, let's also refine our pattern to select
# only the filtered raw recording (omitting the unfiltered data and the
# empty-room noise recordings):

pattern = 'sample_audvis_filt-0-40_raw.fif'
report = mne.Report(raw_psd=True, projs=True, verbose=True)
report.parse_folder(data_path, pattern=pattern, render_bem=False)
report.save('report_raw_psd.html', overwrite=True)

# %%
# This time we'll pass a specific ``subject`` and ``subjects_dir`` (even though
# there's only one subject in the sample dataset) and remove our
# ``render_bem=False`` parameter so we can see the MRI slices, with BEM
# contours overlaid on top if available. Since this is computationally
# expensive, we'll also pass the ``mri_decim`` parameter for the benefit of our
# documentation servers, and skip processing the :file:`.fif` files:

report = mne.Report(subject='sample', subjects_dir=subjects_dir, verbose=True)
report.parse_folder(data_path, pattern='', mri_decim=25)
report.save('report_mri_bem.html', overwrite=True)

# %%
# Now let's look at how :class:`~mne.Report` handles :class:`~mne.Evoked` data
# (we will skip the MRIs to save computation time). The following code will
# produce butterfly plots, topomaps, and comparisons of the global field
# power (GFP) for different experimental conditions.

pattern = 'sample_audvis-no-filter-ave.fif'
report = mne.Report(verbose=True)
report.parse_folder(data_path, pattern=pattern, render_bem=False)
report.save('report_evoked.html', overwrite=True)

# %%
# You have probably noticed that the EEG recordings look particularly odd. This
# is because by default, `~mne.Report` does not apply baseline correction
# before rendering evoked data. So if the dataset you wish to add to the report
# has not been baseline-corrected already, you can request baseline correction
# here. The MNE sample dataset we're using in this example has **not** been
# baseline-corrected; so let's do this now for the report!
#
# To request baseline correction, pass a ``baseline`` argument to
# `~mne.Report`, which should be a tuple with the starting and ending time of
# the baseline period. For more details, see the documentation on
# `~mne.Evoked.apply_baseline`. Here, we will apply baseline correction for a
# baseline period from the beginning of the time interval to time point zero.

baseline = (None, 0)
pattern = 'sample_audvis-no-filter-ave.fif'
report = mne.Report(baseline=baseline, verbose=True)
report.parse_folder(data_path, pattern=pattern, render_bem=False)
report.save('report_evoked_baseline.html', overwrite=True)

# %%
# To render whitened :class:`~mne.Evoked` files with baseline correction, pass
# the ``baseline`` argument we just used, and add the noise covariance file.
# This will display ERP/ERF plots for both the original and whitened
# :class:`~mne.Evoked` objects, but scalp topomaps only for the original.

cov_fname = op.join(sample_dir, 'sample_audvis-cov.fif')
baseline = (None, 0)
report = mne.Report(cov_fname=cov_fname, baseline=baseline, verbose=True)
report.parse_folder(data_path, pattern=pattern, render_bem=False)
report.save('report_evoked_whitened.html', overwrite=True)

# %%
# If you want to actually *view* the noise covariance in the report, make sure
# it is captured by the pattern passed to :meth:`~mne.Report.parse_folder`, and
# also include a source for an :class:`~mne.Info` object (any of the
# :class:`~mne.io.Raw`, :class:`~mne.Epochs` or :class:`~mne.Evoked`
# :file:`.fif` files that contain subject data also contain the measurement
# information and should work):

pattern = 'sample_audvis-cov.fif'
info_fname = op.join(sample_dir, 'sample_audvis-ave.fif')
report = mne.Report(info_fname=info_fname, verbose=True)
report.parse_folder(data_path, pattern=pattern, render_bem=False)
report.save('report_cov.html', overwrite=True)




# %%
# Working with tags
# ^^^^^^^^^^^^^^^^^^
#
# by a toggle button in the top navigation bar of the report which can be used
# to show or hide the contents of the section. To toggle the show/hide state of
# all sections in the HTML report, press :kbd:`t`, or press the toggle-all
# button in the upper right.
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

# %%
# This allows the possibility of multiple scripts adding figures to the same
# report. To make this even easier, :class:`mne.Report` can be used as a
# context manager:

with mne.open_report('report.h5') as report:
    report.add_figs_to_section(fig_evoked,
                               captions='Left Auditory',
                               section='evoked',
                               replace=True)
    report.save('report_final.html', overwrite=True)

# %%
# With the context manager, the updated report is also automatically saved
# back to :file:`report.h5` upon leaving the block.
