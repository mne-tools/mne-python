# -*- coding: utf-8 -*-
"""
.. _tut-report:

===============================
Getting started with mne.Report
===============================

:class:`mne.Report` is a way to create interactive HTML summaries of your data.
These reports can show many different visualizations for one or multiple
participants. A common use case is creating diagnostic summaries to check data
quality at different stages in the processing pipeline. The report can show
things like plots of data before and after each preprocessing step, epoch
rejection statistics, MRI slices with overlaid BEM shells, all the way up to
plots of estimated cortical activity.

Compared to a Jupyter notebook, :class:`mne.Report` is easier to deploy (the
HTML pages it generates are self-contained and do not require a running Python
environment) but less flexible (you can't change code and re-run something
directly within the browser). This tutorial covers the basics of building a
:class:`~mne.Report`. As usual, we'll start by importing the modules and data
we need:
"""

# %%

from pathlib import Path
import tempfile
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
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
#    will open the HTML in a new tab in your browser. To disable this, use the
#    ``open_browser=False`` parameter of :meth:`~mne.Report.save`.
#
# Adding `~mne.io.Raw` data
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Raw data can be added via the :meth:`mne.Report.add_raw` method. It can
# operate with a path to a raw file and `~mne.io.Raw` objects, and will
# produce – among other output – a slider that allows you to scrub through 10
# equally-spaced 1-second segments of the data:
#
# .. warning::
#    In the following example, we crop the raw data to 60 seconds merely to
#    speed up processing; this is not usually recommended!

raw_path = sample_dir / 'sample_audvis_filt-0-40_raw.fif'
raw = mne.io.read_raw(raw_path)
raw.pick_types(eeg=True, eog=True, stim=True).crop(tmax=60).load_data()

report = mne.Report(title='Raw example')
# This method also accepts a path, e.g., raw=raw_path
report.add_raw(raw=raw, title='Raw', psd=False)  # omit PSD plot
report.save('report_raw.html', overwrite=True)

# %%
# Adding events
# ^^^^^^^^^^^^^
#
# Events can be added via :meth:`mne.Report.add_events`. You also need to
# supply the sampling frequency used during the recording; this information
# is used to generate a meaningful time axis.

events_path = sample_dir / 'sample_audvis_filt-0-40_raw-eve.fif'
events = mne.find_events(raw=raw)
sfreq = raw.info['sfreq']

report = mne.Report(title='Events example')
report.add_events(events=events_path, title='Events from Path', sfreq=sfreq)
report.add_events(events=events, title='Events from "events"', sfreq=sfreq)
report.save('report_events.html', overwrite=True)

# %%
# Adding :class:`~mne.Epochs`
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Epochs can be added via :meth:`mne.Report.add_epochs`. Note that although
# this method accepts a path to an epochs file too, in the following example
# we only add epochs that we create on the fly from raw data. To demonstrate
# the representation of epochs metadata, we'll add some of that too.

event_id = {
    'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
    'visual/right': 4, 'face': 5, 'buttonpress': 32
}

metadata, _, _ = mne.epochs.make_metadata(
    events=events,
    event_id=event_id,
    tmin=-0.2,
    tmax=0.5,
    sfreq=raw.info['sfreq']
)
epochs = mne.Epochs(
    raw=raw, events=events, event_id=event_id, metadata=metadata
)

report = mne.Report(title='Epochs example')
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
#
# By default, this method will produce snapshots at 21 equally-spaced time
# points (or fewer, if the data contains fewer time points). We can adjust this
# via the ``n_time_points`` parameter.

evoked_path = sample_dir / 'sample_audvis-ave.fif'
cov_path = sample_dir / 'sample_audvis-cov.fif'

evokeds = mne.read_evokeds(evoked_path, baseline=(None, 0))
evokeds_subset = evokeds[:2]  # The first two
for evoked in evokeds_subset:
    evoked.pick('eeg')  # just for speed of plotting

report = mne.Report(title='Evoked example')
report.add_evokeds(
    evokeds=evokeds_subset,
    titles=['evoked 1',  # Manually specify titles
            'evoked 2'],
    noise_cov=cov_path,
    n_time_points=5
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

report = mne.Report(title='Covariance example')
report.add_covariance(cov=cov_path, info=raw_path, title='Covariance')
report.save('report_cov.html', overwrite=True)

# %%
# Adding `~mne.Projection` vectors
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# `~mne.Projection` vectors can be added via
# :meth:`mne.Report.add_projs`. The method requires an `~mne.Info` object
# (or the path to one) and a title. Projectors found in the `~mne.Info` will
# be visualized. You may also supply a list of `~mne.Projection` objects or
# a path to projectors stored on disk. In this case, the channel information
# is read from the `~mne.Info`, but projectors potentially included will be
# ignored; instead, only the explicitly passed projectors will be plotted.

ecg_proj_path = sample_dir / 'sample_audvis_ecg-proj.fif'
eog_proj_path = sample_dir / 'sample_audvis_eog-proj.fif'

report = mne.Report(title='Projectors example')
report.add_projs(info=raw_path, title='Projs from info')
report.add_projs(info=raw_path, projs=ecg_proj_path,
                 title='ECG projs from path')
report.add_projs(info=raw_path, projs=eog_proj_path,
                 title='EOG projs from path')
report.save('report_projs.html', overwrite=True)

# %%
# Adding `~mne.preprocessing.ICA`
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# `~mne.preprocessing.ICA` objects can be added via
# :meth:`mne.Report.add_ica`. Aside from the parameters ``ica`` (that accepts
# an `~mne.preprocessing.ICA` instance or a path to an ICA object stored on
# disk) and the ``title``, there is a third required parameter, ``inst``.
# ``inst`` is used to specify a `~mne.io.Raw` or `~mne.Epochs` object for
# producing ICA property plots and overlay plots demonstrating
# the effects of ICA cleaning. If, instead, you only want to generate ICA
# component topography plots, explicitly pass ``inst=None``.
#
# .. note:: :meth:`mne.Report.add_ica` only works with fitted ICAs.
#
# You can optionally specify for which components to produce topography and
# properties plots by passing ``picks``. By default, all components will be
# shown. It is also possible to pass evoked signals based on ECG and EOG events
# via ``ecg_evoked`` and ``eog_evoked``. This allows you directly see the
# effects of ICA component removal on these artifactual signals.
# Artifact detection scores produced by
# :meth:`~mne.preprocessing.ICA.find_bads_ecg`
# and :meth:`~mne.preprocessing.ICA.find_bads_eog` can be passed via the
# ``ecg_scores`` and ``eog_scores`` parameters, respectively, producing
# visualizations of the scores for each ICA component.
#
# Lastly, by passing ``n_jobs``, you may largely speed up the generation of
# the properties plots by enabling parallel execution.
#
# .. warning::
#    In the following example, we request a small number of ICA components
#    to estimate, set the threshold for assuming ICA convergence to a very
#    liberal value, and only visualize 2 of the components. All of this is
#    done to largely reduce the processing time of this tutorial, and is
#    usually **not** recommended for an actual data analysis.

ica = mne.preprocessing.ICA(
    n_components=5,  # fit 5 ICA components
    fit_params=dict(tol=0.01)  # assume very early on that ICA has converged
)

ica.fit(inst=raw)

# create epochs based on EOG events, find EOG artifacts in the data via pattern
# matching, and exclude the EOG-related ICA components
eog_epochs = mne.preprocessing.create_eog_epochs(raw=raw)
eog_components, eog_scores = ica.find_bads_eog(
    inst=eog_epochs,
    ch_name='EEG 001',  # a channel close to the eye
    threshold=1  # lower than the default threshold
)
ica.exclude = eog_components

report = mne.Report(title='ICA example')
report.add_ica(
    ica=ica,
    title='ICA cleaning',
    picks=[0, 1],  # only plot the first two components
    inst=raw,
    eog_evoked=eog_epochs.average(),
    eog_scores=eog_scores,
    n_jobs=None  # could be increased!
)
report.save('report_ica.html', overwrite=True)

# %%
# Adding MRI with BEM
# ^^^^^^^^^^^^^^^^^^^
#
# MRI slices with superimposed traces of the boundary element model (BEM)
# surfaces can be added via :meth:`mne.Report.add_bem`. All you need to pass is
# the FreeSurfer subject name and subjects directory, and a title. To reduce
# the resulting file size, you may pass the ``decim`` parameter to only include
# every n-th volume slice, and ``width`` to specify the width of the resulting
# figures in pixels.

report = mne.Report(title='BEM example')
report.add_bem(
    subject='sample', subjects_dir=subjects_dir, title='MRI & BEM',
    decim=20,
    width=256
)
report.save('report_mri_and_bem.html', overwrite=True)

# %%
# Adding coregistration
# ^^^^^^^^^^^^^^^^^^^^^
#
# The sensor alignment (``head -> mri`` transformation obtained by
# "coregistration") can be visualized via :meth:`mne.Report.add_trans`. The
# method expects the transformation either as a `~mne.transforms.Transform`
# object or as a path to a ``trans.fif`` file, the FreeSurfer subject name and
# subjects directory, and a title. The ``alpha`` parameter can be used to
# control the transparency of the head, where a value of 1 means fully opaque.

trans_path = sample_dir / 'sample_audvis_raw-trans.fif'

report = mne.Report(title='Coregistration example')
report.add_trans(
    trans=trans_path, info=raw_path, subject='sample',
    subjects_dir=subjects_dir, alpha=1.0, title='Coregistration'
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

report = mne.Report(title='Forward solution example')
report.add_forward(forward=fwd_path, title='Forward solution')
report.save('report_forward_sol.html', overwrite=True)

# %%
# Adding an `~mne.minimum_norm.InverseOperator`
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# An inverse operator can be added via :meth:`mne.Report.add_inverse_operator`.
# The method expects an `~mne.minimum_norm.InverseOperator` object or a path to
# one stored on disk, and a title.

inverse_op_path = sample_dir / 'sample_audvis-meg-oct-6-meg-inv.fif'

report = mne.Report(title='Inverse operator example')
report.add_inverse_operator(
    inverse_operator=inverse_op_path, title='Inverse operator'
)
report.save('report_inverse_op.html', overwrite=True)

# %%
# Adding a `~mne.SourceEstimate`
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# An inverse solution (also called source estimate or source time course, STC)
# can be added via :meth:`mne.Report.add_stc`. The
# method expects an `~mne.SourceEstimate`, the corresponding FreeSurfer subject
# name and subjects directory, and a title. By default, it will produce
# snapshots at 51 equally-spaced time points (or fewer, if the data contains
# fewer time points). We can adjust this via the ``n_time_points`` parameter.

stc_path = sample_dir / 'sample_audvis-meg'

report = mne.Report(title='Source estimate example')
report.add_stc(
    stc=stc_path, subject='sample', subjects_dir=subjects_dir,
    title='Source estimate', n_time_points=2  # few for speed
)
report.save('report_inverse_sol.html', overwrite=True)

# %%
# Adding source code (e.g., a Python script)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# It is possible to add code or scripts (e.g., the scripts you used for
# analysis) to the report via :meth:`mne.Report.add_code`. The code blocks will
# be automatically syntax-highlighted. You may pass a string with the
# respective code snippet, or the path to a file. If you pass a path, it
# **must** be a `pathlib.Path` object (and not a string), otherwise it will be
# treated as a code literal.
#
# Optionally, you can specify which programming language to assume for syntax
# highlighting by passing the ``language`` parameter. By default, we'll assume
# the provided code is Python.

mne_init_py_path = Path(mne.__file__)  # __init__.py in the MNE-Python root
mne_init_py_content = mne_init_py_path.read_text(encoding='utf-8')

report = mne.Report(title='Code example')
report.add_code(
    code=mne_init_py_path,
    title="Code from Path"
)
report.add_code(
    code=mne_init_py_content,
    title="Code from string"
)

report.save('report_code.html', overwrite=True)

# %%
# Adding custom figures
# ^^^^^^^^^^^^^^^^^^^^^
#
# Custom Matplotlib figures can be added via :meth:`~mne.Report.add_figure`.
# Required parameters are the figure and a title. Optionally, may add a caption
# to appear below the figure. You can also specify the image format of the
# image file that will be generated from the figure, so it can be embedded in
# the HTML report.

x = np.linspace(start=0, stop=10, num=100)
y = x**2

fig, ax = plt.subplots()
ax.plot(x, y, ls='--', lw=2, color='blue', label='my function')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.legend()

report = mne.Report(title='Figure example')
report.add_figure(
    fig=fig, title='A custom figure',
    caption='A blue dashed line reaches up into the sky …',
    image_format='PNG'
)
report.save('report_custom_figure.html', overwrite=True)
plt.close(fig)

# %%
# Multiple figures can be grouped into a single section via the ``section``
# parameter.

fig_1, ax_1 = plt.subplots()
ax_1.plot([1, 2, 3])

fig_2, ax_2 = plt.subplots()
ax_2.plot([3, 2, 1])

section = 'Section example'

report = mne.Report(title='Figure section example')
report.add_figure(
    fig=fig_1,
    title='Figure 1',
    section=section,
    tags='fig-1'
)
report.add_figure(
    fig=fig_2,
    title='Figure 2',
    section=section,
    tags='fig-2'
)
report.save('report_custom_figure_sections.html', overwrite=True)
plt.close(fig_1)
plt.close(fig_2)

# %%
# The :meth:`mne.Report.add_figure` method can also add multiple figures at
# once. In this case, a slider will appear, allowing users to intuitively
# browse the figures. To make this work, you need to provide a collection o
# figures, a title, and optionally a collection of captions.
#
# In the following example, we will read the MNE logo as a Matplotlib figure
# and rotate it with different angles. Each rotated figure and its respective
# caption will be added to a list, which is then used to create the slider.

mne_logo_path = Path(mne.__file__).parent / 'icons' / 'mne_icon-cropped.png'
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

    # Store figure and caption
    figs.append(fig)
    captions.append(f'Rotation angle: {round(angle, 1)}°')

# can also be a MNEQtBrowser instance
figs.append(raw.plot())
captions.append('... plus a raw data plot')

report = mne.Report(title='Multiple figures example')
report.add_figure(fig=figs, title='Fun with figures! 🥳', caption=captions)
report.save('report_custom_figures.html', overwrite=True)
for fig in figs[:-1]:
    plt.close(fig)
figs[-1].close()
del figs

# %%
# Adding image files
# ^^^^^^^^^^^^^^^^^^
#
# Existing images (e.g., photos, screenshots, sketches etc.) can be added
# to the report via :meth:`mne.Report.add_image`. Supported image formats
# include JPEG, PNG, GIF, and SVG (and possibly others). Like with Matplotlib
# figures, you can specify a caption to appear below the image.

report = mne.Report(title='Image example')
report.add_image(
    image=mne_logo_path, title='MNE',
    caption='Powered by 🧠 🧠 🧠 around the world!'
)
report.save('report_custom_image.html', overwrite=True)

# %%
# Working with tags
# ^^^^^^^^^^^^^^^^^
#
# Each ``add_*`` method accepts a keyword parameter ``tags``, which can be
# used to pass one or more tags to associate with the respective content
# elements. By default, each ``add_*`` method adds a tag describing the data
# type, e.g., ``evoked`` or ``source-estimate``. When viewing the HTML report,
# the ``Filter by tags`` dropdown menu can be used to interactively show or
# hide content with specific tags. This allows you e.g. to only view
# ``evoked`` or ``participant-001`` data, should you have added those tags.
# Visible tags will appear with blue, and hidden tags with gray background
# color.
#
# To toggle the visibility of **all** tags, use the respective checkbox in the
# ``Filter by tags`` dropdown menu, or press :kbd:`T`.

report = mne.Report(title='Tags example')
report.add_image(
    image=mne_logo_path,
    title='MNE Logo',
    tags=('image', 'mne', 'logo', 'open-source')
)
report.save('report_tags.html', overwrite=True)

# %%
# Editing a saved report
# ^^^^^^^^^^^^^^^^^^^^^^
#
# Saving to HTML is a write-only operation, meaning that we cannot read an
# ``.html`` file back as a :class:`~mne.Report` object. In order to be able
# to edit a report once it's no longer in-memory in an active Python session,
# save it as an HDF5 file instead of HTML:

report = mne.Report(title='Saved report example', verbose=True)
report.add_image(image=mne_logo_path, title='MNE 1')
report.save('report_partial.hdf5', overwrite=True)

# %%
# The saved report can be read back and modified or amended. This allows the
# possibility to e.g. run multiple scripts in a processing pipeline, where each
# script adds new content to an existing report.

report_from_disk = mne.open_report('report_partial.hdf5')
report_from_disk.add_image(image=mne_logo_path, title='MNE 2')
report_from_disk.save('report_partial.hdf5', overwrite=True)

# %%
# To make this even easier, :class:`mne.Report` can be used as a
# context manager (note the ``with`` statement)`):

with mne.open_report('report_partial.hdf5') as report:
    report.add_image(image=mne_logo_path, title='MNE 3')
    report.save('report_final.html', overwrite=True)

# %%
# With the context manager, the updated report is also automatically saved
# back to :file:`report.h5` upon leaving the block.
#
# Adding an entire folder of files
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We also provide a way to add an entire **folder** of files to the report at
# once, without having to invoke the individual ``add_*`` methods outlined
# above for each file. This approach, while convenient, provides less
# flexibility with respect to content ordering, tags, titles, etc.
#
# For our first example, we'll generate a barebones report for all the
# :file:`.fif` files containing raw data in the sample dataset, by passing the
# pattern ``*raw.fif`` to :meth:`~mne.Report.parse_folder`. We'll omit the
# ``subject`` and ``subjects_dir`` parameters from the :class:`~mne.Report`
# constructor, but we'll also pass ``render_bem=False`` to the
# :meth:`~mne.Report.parse_folder` method — otherwise we would get a warning
# about not being able to render MRI and ``trans`` files without knowing the
# subject. To save some processing time in this tutorial, we're also going to
# disable rendering of the butterfly plots for the `~mne.io.Raw` data by
# passing ``raw_butterfly=False``.
#
# Which files are included depends on both the ``pattern`` parameter passed to
# :meth:`~mne.Report.parse_folder` and also the ``subject`` and
# ``subjects_dir`` parameters provided to the :class:`~mne.Report` constructor.

report = mne.Report(title='parse_folder example')
report.parse_folder(
    data_path=data_path, pattern='*raw.fif', render_bem=False,
    raw_butterfly=False
)
report.save('report_parse_folder_basic.html', overwrite=True)

# %%
# By default, the power spectral density and SSP projectors of the
# :class:`~mne.io.Raw` files are not shown to speed up report generation. You
# can add them by passing ``raw_psd=True`` and ``projs=True`` to the
# :class:`~mne.Report` constructor. Like in the previous example, we're going
# to omit the butterfly plots by passing ``raw_butterfly=False``. Lastly, let's
# also refine our pattern to select only the filtered raw recording (omitting
# the unfiltered data and the empty-room noise recordings).

pattern = 'sample_audvis_filt-0-40_raw.fif'
report = mne.Report(title='parse_folder example 2', raw_psd=True, projs=True)
report.parse_folder(
    data_path=data_path, pattern=pattern, render_bem=False, raw_butterfly=False
)
report.save('report_parse_folder_raw_psd_projs.html', overwrite=True)

# %%
# This time we'll pass a specific ``subject`` and ``subjects_dir`` (even though
# there's only one subject in the sample dataset) and remove our
# ``render_bem=False`` parameter so we can see the MRI slices, with BEM
# contours overlaid on top if available. Since this is computationally
# expensive, we'll also pass the ``mri_decim`` parameter for the benefit of our
# documentation servers, and skip processing the :file:`.fif` files.

report = mne.Report(
    title='parse_folder example 3', subject='sample', subjects_dir=subjects_dir
)
report.parse_folder(data_path=data_path, pattern='', mri_decim=25)
report.save('report_parse_folder_mri_bem.html', overwrite=True)

# %%
# Now let's look at how :class:`~mne.Report` handles :class:`~mne.Evoked`
# data (we will skip the MRIs to save computation time).
#
# The MNE sample dataset we're using in this example has **not** been
# baseline-corrected; so let's apply baseline correction this now for the
# report!
#
# To request baseline correction, pass a ``baseline`` argument to
# `~mne.Report`, which should be a tuple with the starting and ending time of
# the baseline period. For more details, see the documentation on
# `~mne.Evoked.apply_baseline`. Here, we will apply baseline correction for a
# baseline period from the beginning of the time interval to time point zero.
#
# Lastly, we want to render the "whitened" evoked data, too. Whitening
# requires us to specify the path to a covariance matrix file via the
# ``cov_fname`` parameter of `~mne.Report`.
#
# Now, let's put all of this together! Here we use a temporary directory
# for speed so we can render a single Evoked instance, using just EEG
# channels.

baseline = (None, 0)
cov_fname = sample_dir / 'sample_audvis-cov.fif'
pattern = 'sample_audvis-no-filter-ave.fif'
evoked = mne.read_evokeds(sample_dir / pattern)[0]
report = mne.Report(
    title='parse_folder example 4', baseline=baseline, cov_fname=cov_fname
)
with tempfile.TemporaryDirectory() as path:
    evoked.save(Path(path) / pattern)
    report.parse_folder(
        path, pattern=pattern, render_bem=False, n_time_points_evokeds=5
    )
report.save('report_parse_folder_evoked.html', overwrite=True)

# %%
# If you want to actually *view* the noise covariance in the report, make sure
# it is captured by the pattern passed to :meth:`~mne.Report.parse_folder`, and
# also include a source for an :class:`~mne.Info` object (any of the
# :class:`~mne.io.Raw`, :class:`~mne.Epochs` or :class:`~mne.Evoked`
# :file:`.fif` files that contain subject data also contain the measurement
# information and should work):

pattern = 'sample_audvis-cov.fif'
info_fname = sample_dir / 'sample_audvis-ave.fif'
report = mne.Report(title='parse_folder example 5', info_fname=info_fname)
report.parse_folder(
    data_path, pattern=pattern, render_bem=False, n_time_points_evokeds=5
)
report.save('report_parse_folder_cov.html', overwrite=True)

# %%
#
# Adding custom HTML (e.g., a description text)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The :meth:`~mne.Report.add_html` method allows you to add custom HTML to
# your report. This feature can be very convenient to add short descriptions,
# lists, or reminders to your report (among many other things you can think
# of encoding in HTML).

report = mne.Report(title='Report on hypothesis 1')

my_html = """
<p>We have the following hypothesis:</p>
<ol>
<li>There is a difference between images showing man-made vs. natural
environments</li>
<li>This difference manifests itself most strongly in the amplitude of the
N1 ERP component</li>
</ol>
<p>Below we show several plots and tests of the data.</p>
"""

report.add_html(title='Hypothesis', html=my_html)
report.save('report_add_html.html', overwrite=True)
