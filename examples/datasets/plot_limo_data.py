"""
.. _ex-limo-data:

=============================================================
Single trial linear regression analysis with the LIMO dataset
=============================================================

Here we explore the structure of the data contained in the
`LIMO dataset`_.
Furthermore, the code replicates and extends some of the main analysis
and tools integrated in `LIMO MEEG`_, a MATLAB toolbox originally designed
to interface with EEGLAB_.

In summary, the code allows to:

- Fetch epoched data from single subject files of the LIMO dataset [1]_.
  If the LIMO files are not found on disk, the
  fetcher :func:`mne.datasets.limo.load_data()` will automatically download
  the files from a remote repository.

- During import, the epochs information (i.e., sampling rate, number of epochs
  per condition, number and name of EEG channels per subject, etc.) is
  extracted from the LIMO .mat-files stored on disk and added to the epochs
  structure as metadata.

- In addition, the code shows how to to fit linear models on single subject
  data and derive inferential measures to evaluate the significance of the
  estimated effects using bootstrap and spatio-temporal clustering techniques.

References
----------
.. [1] Guillaume, Rousselet. (2016). LIMO EEG Dataset, [dataset].
       University of Edinburgh, Centre for Clinical Brain Sciences.
       https://doi.org/10.7488/ds/1556.
.. [2] Rousselet, G. A., Gaspar, C. M., Pernet, C. R., Husk, J. S.,
       Bennett, P. J., & Sekuler, A. B. (2010). Healthy aging delays scalp EEG
       sensitivity to noise in a face discrimination task.
       Frontiers in psychology, 1, 19. https://doi.org/10.3389/fpsyg.2010.00019
.. [3] Rousselet, G. A., Pernet, C. R., Bennett, P. J., & Sekuler, A. B.
       (2008). Parametric study of EEG sensitivity to phase noise during face
       processing. BMC neuroscience, 9(1), 98.
       https://doi.org/10.1186/1471-2202-9-98
.. _LIMO dataset: https://datashare.is.ed.ac.uk/handle/10283/2189?show=full
.. _LIMO MEEG: https://github.com/LIMO-EEG-Toolbox
.. _EEGLAB: https://sccn.ucsd.edu/eeglab/index.php
.. _Fig 1: https://bmcneurosci.biomedcentral.com/articles/10.1186/1471-2202-9-98/figures/1
.. _least squares: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lstsq.html
"""  # noqa: E501
# Authors: Jose C. Garcia Alanis <alanis.jcg@gmail.com>
#
# License: BSD (3-clause)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from mne.datasets.limo import load_data
from mne.decoding import Vectorizer, get_coef
from mne.evoked import EvokedArray
from mne.stats import linear_regression
from mne.viz import plot_events, plot_compare_evokeds
from mne import combine_evoked


print(__doc__)

# subject to use
subj = 1

###############################################################################
# About the data
# --------------
#
# In the original LIMO experiment (see [2]_), participants performed a
# two-alternative forced choice task, discriminating between two face stimuli.
# Subjects discriminated the same two faces during the whole experiment.
# However, the critical manipulation consisted in the level of noise
# added to the face-stimuli during the task, making the faces more or less
# discernible to the observer (see `Fig 1`_ in [3]_ for instance).
#
# The presented faces varied across a noise-signal (or phase-coherence)
# continuum spanning from 0 to 100% in increasing steps of 10%.
# In other words, faces with high phase-coherence (e.g., 90%) were easy to
# identify, while faces with low phase-coherence (e.g., 10%) were hard to
# identify and by extension very hard to discriminate.

###############################################################################
# Load the data
# -------------
#
# We'll begin by loading the data from subject 1 of the LIMO dataset.

# This step can take a little while if you're loading the data for the
# first time.
limo_epochs = load_data(subject=subj)

###############################################################################
# Note that the result of the loading process is an
# :class:`mne.EpochsArray` containing the data ready to interface
# with MNE-Python.

print(limo_epochs)

###############################################################################
# Visualize events
# ----------------
#
# We can visualise the distribution of the face events contained in the
# ``limo_epochs`` structure. Events should appear clearly grouped, as they
# were ordered during the import process.

fig, ax = plt.subplots(figsize=(7, 4))
plot_events(limo_epochs.events, event_id=limo_epochs.event_id, axes=ax)
ax.set(title="Distribution of events in LIMO epochs")
plt.legend(loc='lower left', borderaxespad=1.)
plt.tight_layout()
plt.show()

###############################################################################
# As it can be seen above, conditions are coded as ``Face/A`` and ``Face/B``.
# Information about the phase-coherence of the presented faces is stored in the
# epochs metadata. These information can be easily accessed by calling
# ``limo_epochs.metadata``. As shown below, the epochs metadata also contains
# information about the presented faces for convenience.

print(limo_epochs.metadata.head())

###############################################################################
# Now we can take a closer look at the information entailed in the epochs
# metadata.

# We want include all columns in the summary table
epochs_summary = limo_epochs.metadata.describe(include='all').round(3)
print(epochs_summary)

###############################################################################
# The first column of the summary table above provides more or less the same
# information as the ``print(limo.epochs)`` command we ran before. There are
# 1055 faces (i.e., epochs), subdivided in 2 conditions (i.e., Face A and
# Face B) and, for this particular subject, there are more epochs for the
# condition Face B.
#
# In addition, we can see in the second column that the values for the
# phase-coherence variable range from -1.619 to 1.642. This is because the
# phase-coherence values are provided as a z-scored variable in the LIMO
# dataset. Note that they have a mean of zero and a standard deviation of 1.

###############################################################################
# Visualize condition ERPs
# ------------------------
#
# We can go ahead and plot the ERPs evoked by Face A and Face B

# only show -250 to 500 ms
ts_args = dict(xlim=(-.25, 0.5))

# plot evoked response for face A
limo_epochs['Face/A'].average().plot_joint(times=[.15],
                                           title='Evoked response: Face A',
                                           ts_args=ts_args)
# and face B
limo_epochs['Face/B'].average().plot_joint(times=[.15],
                                           title='Evoked response: Face B',
                                           ts_args=ts_args)

###############################################################################
# We can also compute the difference wave contrasting Face A and Face B.
# Although, looking at the evoked responses above, we shouldn't expect great
# differences among these face-stimuli.

# Face A minus Face B
difference_wave = combine_evoked([limo_epochs['Face/A'].average(),
                                  -limo_epochs['Face/B'].average()],
                                 weights='equal')

# plot difference wave
difference_wave.plot_joint(times=[.15], title='Difference Face A - Face B')

###############################################################################
# As expected, no see clear differential pattern appears when contrasting
# Face A and Face B. However, we could narrow our search a little bit more.
# Since this is a "visual paradigm" it might be best to look at electrodes
# located over the occipital lobe, as differences between stimuli (if any)
# might easier to spot over visual areas.

# Create a dictionary containing the evoked responses
conditions = ["Face/A", "Face/B"]
evokeds = {condition: limo_epochs[condition].average()
           for condition in conditions}

# concentrate analysis an occipital electrodes (e.g. B11)
pick = evokeds["Face/A"].ch_names.index('B11')

# compare evoked responses
plot_compare_evokeds(evokeds, picks=pick, ylim=dict(eeg=(-15, 5)))

###############################################################################
# As expected, the difference between Face A and Face B are very small.

###############################################################################
# Visualize effect of stimulus phase-coherence
# --------------------------------------------
#
# For visualization purposes, we can transform the phase-coherence
# variable back to fit roughly to it's original scale. Phase-coherence
# determined whether a face stimulus could be identified as such. Thus,
# one could expect that faces with high phase-coherence should evoke stronger
# activation patterns along occipital electrodes.
#
# As phase-coherence variable is a continuous variable, we'll need to
# split the values of face coherence to percentiles to match roughly their
# original scale (i.e, 0 to 100 % signal-to-noise ratio in 10% steps).
# **Note:** We'll only do this for visualization purposes,

name = "phase-coherence"
factor = 'factor_' + name

# color scheme for percentile plot
limo_epochs.metadata[factor] = pd.cut(limo_epochs.metadata[name], 11,
                                      labels=False) / 10

# color scheme for percentile plot
colors = {str(val): val
          for val in np.sort(limo_epochs.metadata[factor].unique())}
# compute evoked for each phase-coherence percentile
evokeds = {str(val): limo_epochs[limo_epochs.metadata[factor] == val].average()
           for val in colors.values()}

# pick channel to plot
electrodes = ['C22', 'B11']
# create figures
for electrode in electrodes:
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_compare_evokeds(evokeds,
                         axes=ax,
                         ylim=dict(eeg=(-20, 20)),
                         colors=colors,
                         split_legend=True,
                         picks=electrode,
                         cmap=(name + " Percentile", "magma"))
    fig.show()

###############################################################################
# As shown above, there are some considerable differences between the
# activation patterns evoked by stimuli with low vs. high phase-coherence.
#
# In particular, differences appear to be most pronounced along fronto-central
# and occipital electrodes.

###############################################################################
# Prepare data for linear regression analysis
# --------------------------------------------
#
# Next, we can test the significance of these differences using linear
# regression.But before we go on, we'll start by interpolating any missing
# channels in the LIMO epochs structure. Some subjects in the datasets contain
# missing channels (stored in ``limo_epochs.info[‘bads’]``) as these were
# dropped during preprocessing of the data.

limo_epochs.interpolate_bads(reset_bads=True)

# Furthermore, we'll drop the EOG channels (marked by the "EXG" prefix)
# present in the LIMO epochs structure.
limo_epochs.drop_channels(['EXG1', 'EXG2', 'EXG3', 'EXG4'])

###############################################################################
# Define predictor variables and design matrix
# --------------------------------------------
#
# First, we need to create a design matrix containing information about the
# variables (i.e., predictors) we want to use for prediction of brain
# activity patterns. For this purpose, we'll use the information contained in
# the ``limo_epochs.metadata``. Here, we'll explore the effect of
# phase-coherence as well as the effect of Face A vs. Face B.

# name of predictors + intercept
predictor_vars = ['face a - face b', 'phase-coherence', 'intercept']

# create design matrix
design = limo_epochs.metadata[['phase-coherence', 'face']]
design['face a - face b'] = np.where(design['face'] == 'A', 1, -1)
design['intercept'] = 1
design = design[predictor_vars]

###############################################################################
# Now we can set up the linear model to be used in the analysis.
# For the purpose of demonstration, we'll use scikit-learn's LinearRegression
# (see :class:`sklearn.linear_model.LinearRegression`) in this example.
#
# Basically, we're creating a wrapper, ``STLinearRegression``, for
# sklearn's ``LinearRegression``, which computes a
# `least squares`_ solution for our data given the provided design matrix. The
# and results are stored within ``LinearRegression`` object for convenience.


class STLinearRegression(LinearRegression):
    """
    Create linear model object.

    Notes
    -----
    Currently, the input data has to have  a shape of samples by channels by
    time points. The data will be automatically vectorized for easier / faster
    handling. Thus the vectorized data (Y) within STLinearRegression
    has shape of samples by channels * time points.
    """
    def __init__(self, predictors, design_matrix, data, weights=None,
                 fit_intercept=True, normalize=False,
                 n_jobs=None):

        # store model parameters
        super().__init__(fit_intercept=fit_intercept, normalize=normalize,
                         n_jobs=n_jobs)
        self.predictors = predictors
        self.design = design_matrix
        self.orig_shape = data.shape[1:]
        self.Y = Vectorizer().fit_transform(data)
        self.weights = weights

        # automatically fit the linear model
        self.fit(X=self.design, y=self.Y, sample_weight=self.weights)

    # compute beta coefficients
    def compute_beta_coefs(self, predictor, output='evoked'):
        # extract coefficients from linear model estimator
        beta_coefs = get_coef(self, 'coef_')
        # select predictor in question
        pred_col = predictor_vars.index(predictor)

        if output == 'evoked':
            # coefficients are projected back to a channels x time points
            betas = beta_coefs[:, pred_col]
            betas = betas.reshape(self.orig_shape)
            # create evoked object containing the back projected coefficients
            betas = EvokedArray(betas,
                                comment=predictor,
                                info=limo_epochs.info,
                                tmin=limo_epochs.tmin)
        else:
            # return raw values
            betas = beta_coefs[:, pred_col]

        # return beta coefficients
        return betas

    # compute model predictions
    def compute_predictions(self):
        # compute predicted values
        predictions = self.predict(X=self.design)
        # return beta coefficients
        return predictions


###############################################################################
# Set up the model
# ----------------
#
# We already have an intercept column in the design matrix,
# thus we'll call STLinearRegression with fit_intercept=False

linear_model = STLinearRegression(fit_intercept=False,
                                  predictors=predictor_vars,
                                  design_matrix=design,
                                  data=limo_epochs.get_data())

###############################################################################
# Extract regression coefficients
# -------------------------------
#
# As described above, the results stored within the object ``linear_model``.
# We can extract the coefficients (i.e., the betas) from the
# linear model estimator by calling ``linear_model.compute_beta_coefs()``.
# This will automatically create an evoked object that can used later

pc_betas = linear_model.compute_beta_coefs(predictor='phase-coherence')
face_betas = linear_model.compute_beta_coefs(predictor='face a - face b')

###############################################################################
# Plot model results
# ------------------
#
# Now we can plot results of the linear regression analysis.
# Below we can see a clear effect of phase-coherence, with higher
# phase-coherence (i.e., better "face visibility") having a negative effect on
# the activity measured at occipital electrodes around 200 to 250 ms following
# stimulus onset.

# only show -250 to 500 ms
ts_args = dict(xlim=(-.25, 0.5))

pc_betas.plot_joint(ts_args=ts_args,
                    title='Phase-coherence - sklearn betas',
                    times=[.23])

###############################################################################
# Conversely, there appears to be no (or very small) systematic effects when
# constraining Face A and Face B stimuli. This is largely consistent with the
# difference wave approach presented above.

face_betas.plot_joint(title='Face A - Face B - sklearn betas',
                      ts_args=ts_args,
                      times=[.15])

###############################################################################
# Finally we can compare the output to MNE's linear_regression function
# see func:`mne.stats.linear_regression`.

mne_reg = linear_regression(limo_epochs,
                            design_matrix=design,
                            names=predictor_vars)

mne_reg['phase-coherence'].beta.plot_joint(ts_args=ts_args,
                                           title='Phase-coherence - MNE',
                                           times=[.23])
