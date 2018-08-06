"""
What's a SourceEstimate?
========================

Source estimations in MNE-Python are represented using two different types of
objects:

    - :class:`mne.SourceEstimate` or :class:`mne.VectorSourceEstimate` (which
        will in the following be substituted by 'SourceEstimate')

    - :class:`mne.SourceSpaces`


:class:`mne.SourceEstimate` is the result of a source analysis. MNE-Python
provides different methods for solving the so called 'inverse problem'. Thereby
sensor level data will be projected into a 3-dimensional 'source space'
representing the individual subject's brain anatomy. Hence the data is
transformed such that the recorded time series at each sensor location maps to
a time series at each spatial location of the brain representation.

Knowing this the :class:`mne.SourceEstimate` (within the MNE ecosystem mostly
called **stc**) represents the carrier of the new time series data, whereas
:class:`mne.SourceSpaces` (mostly called **src**) the mapping towards the
anatomical representation.

For an example on how to compute different kinds of source estimates see
:ref:`sphx_glr_auto_tutorials_plot_mne_dspm_source_localization.py`.
"""
import os

from mne import read_source_estimate
from mne.datasets import sample

print(__doc__)

# Paths to example data
sample_dir_raw = sample.data_path()
sample_dir = os.path.join(sample_dir_raw, 'MEG', 'sample')
subjects_dir = os.path.join(sample_dir_raw, 'subjects')

fname_stc = os.path.join(sample_dir, 'sample_audvis-meg')

###############################################################################
# Load example data
#
# This data set contains source estimation data from an audio visual task. It
# has been mapped towards the inflated cortical surface representation obtained
# from
# :ref:`FreeSurfer <sphx_glr_auto_tutorials_plot_background_freesurfer.py>` and
# exposes a noticeable peak at auditory cortex locations.
#
# Let's see how it looks like.

stc = read_source_estimate(fname_stc, subject='sample')

# Define plotting parameters
surfer_kwargs = dict(
    hemi='lh', subjects_dir=subjects_dir,
    clim=dict(kind='value', lims=[8, 12, 15]), views='lateral',
    initial_time=0.09, time_unit='s', size=(800, 800),
    smoothing_steps=5)

# Plot surface
brain = stc.plot(**surfer_kwargs)

# Add title
brain.add_text(0.1, 0.9, 'SourceEstimate', 'title', font_size=16)

###############################################################################
# About SourceEstimate (stc)
# --------------------------
#
# We know that source estimate data is represented as a time series of a signal
# at a spatial location. In the context of a FreeSurfer surface, we could call
# each data point on the inflated brain representation a *vertex*, since it
# resembles a 3D mesh. If every vertex represents the spatial location of a
# time series, the time series and spatial location can be written into a
# matrix, where to each vertex (rows) at multiple time points (columns) a value
# can be assigned. This value is the strength of our signal at a given point in
# space and time. Exactly this matrix is stored in ``stc.data``.
#
# Let's have a look at the shape

shape = stc.data.shape

print('The data has %s vertex locations with %s sample points each.' % shape)

###############################################################################
# We see that stc carries 7498 time series of 25 samples length. Those time
# series belong to 7498 vertices, which in turn represent cortical locations
# on the inflated brain surface. So where do those vertex values come from?
#
# FreeSurfer separates both hemispheres and creates an inflated surface
# representation for left and right hemisphere. Indices to surface locations
# are stored in ``stc.vertices``. This is a list with two arrays of integers,
# that index a perticular vertex of the FreeSurfer mesh. A value of 42 would
# hence map to the 42nd set of x,y,z coordinates of the mesh.
#
# Since both hemispheres are always represented separately, both attributes
# introduced above, can also be obtained by selecting the respective
# hemisphere. This is done by adding the correct prefix ('lh_' or 'rh_').

shape_lh = stc.lh_data.shape

print('The left hemisphere has %s vertex locations with %s sample points each.'
      % shape_lh)

###############################################################################
# Since we did not change the time representation, only the selected subset of
# vertices and hence the size of the matrix changed. We can check if the rows
# of ``stc.lh_data`` and ``stc.rh_data`` sum up to the value we had before.

is_equal = stc.lh_data.shape[0] + stc.rh_data.shape[0] == stc.data.shape[0]

print('\nThe number of vertices in stc.lh_data and stc.rh_data do ' +
      ('not ' if not is_equal else '') +
      'sum up to the number of rows in stc.data')

###############################################################################
# Indeed and as the mindful reader already suspected, the same can be said
# about vertices. ``stc.lh_vertno`` thereby maps to the left and
# ``stc.rh_vertno`` to the right inflated surface representation of
# FreeSurfer.
#
# About SourceSpaces (src)
# ------------------------
#
# As mentioned above, :class:`src <mne.SourceSpaces>` carries the mapping from
# stc to the surface. The surface is built up from a triangulated mesh for each
# hemisphere. Each triangle building up a face consists of 3 vertices. Since
# src is a list of two source spaces (left and right hemisphere), we can access
# the respective data by selecting the source space first. Faces building up
# the left hemisphere can be accessed via ``src[0]['tris']``, where the index
# :math:`0` stands for the left and :math:`1` for the right hemisphere.
#
# The values in src[0]['tris'] refer to row indices in ``src[0]['rr']``.
# Here we find the actual coordinates of the surface mesh. Hence every index
# value for vertices will select a coordinate from here. Furthermore
# ``src[0]['vertno']`` stores the same data as ``stc.lh_vertno``.
#
# In other words ``stc.lh_vertno`` equals ``src[0]['vertno']``, whereas
# ``stc.rh_vertno`` equals ``src[1]['vertno']``. Thus the Nth time series in
# stc.lh_data corresponds to the Nth value in stc.lh_vertno and
# src[0]['vertno'] respectively, which in turn map the time series to a
# specific location on the surface, represented as the set of cartesian
# coordinates stc.lh_vertno[N] in ``src[0]['rr']``.
#
# Let's obtain the peak amplitude of the data as vertex and time point index

peak_vertex, peak_time = stc.get_peak(hemi='lh', vert_as_index=True,
                                      time_as_index=True)

###############################################################################
# The first value thereby indicates which vertex and the second which time
# point index from within stc.lh_vertno stc.lh_data is used. We can use the
# respective information to get the index of the surface vertex resembling the
# peak and its value.

peak_vertex_surf = stc.lh_vertno[peak_vertex]

peak_value = stc.lh_data[peak_vertex, peak_time]

###############################################################################
# Let's visualize this as well using the same 'surfer_kwargs' in the beginning.

brain = stc.plot(**surfer_kwargs)

# We add the new peak coordinate (as vertex index) as an annotation dot
brain.add_foci(peak_vertex_surf, coords_as_verts=True, hemi='lh', color='blue')

# We add a title as well, stating the amplitude at this time and location
brain.add_text(0.1, 0.9, 'Peak coordinate', 'title', font_size=14)

###############################################################################
# Summary
# -------
#
# :class:`stc <mne.SourceEstimate>` is a class of MNE-Python, representing the
# transformed time series obtained from source estimation. For both hemispheres
# the data is stored separately in ``stc.lh_data`` and ``stc.rh_data`` in form
# of a :math:`m \times n` matrix, where :math:`m` is the number of spatial
# locations belonging to that hemishpere and :math:`n` the number of time
# points.
#
# ``stc.lh_vertno`` and ``stc.rh_vertno`` correspond to ``src[0]['vertno']``
# and ``src[1]['vertno']``. Those are the indices of locations on the surface
# representation.
#
# The surface's mesh coordinates are stored in ``src[0]['rr']`` and
# ``src[1]['rr']`` for left and right hemisphere. 3D coordinates can be
# accessed by the above logic:
#
#   >>> lh_coordinates = src[0]['rr'][stc.lh_vertno]
#   >>> lh_data = stc.lh_data
# or
#   >>> rh_coordinates = src[1]['rr'][src[1]['vertno']]
#   >>> rh_data = stc.rh_data
