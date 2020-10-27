"""
.. _ex-megnoise_processing:

====================================
Find MEG reference channel artifacts
====================================

Use ICA decompositions of MEG reference channels to remove intermittent noise.

Many MEG systems have an array of reference channels which are used to detect
external magnetic noise. However, standard techniques that use reference
channels to remove noise from standard channels often fail when noise is
intermittent. The technique described here (using ICA on the reference
channels) often succeeds where the standard techniques do not.

There are two algorithms to choose from: separate and together (default). In
the "separate" algorithm, two ICA decompositions are made: one on the reference
channels, and one on reference + standard channels. The reference + standard
channel components which correlate with the reference channel components are
removed.

In the "together" algorithm, a single ICA decomposition is made on reference +
standard channels, and those components whose weights are particularly heavy
on the reference channels are removed.

This technique is fully described and validated in :footcite:`HannaEtAl2020`

"""
# Authors: Jeff Hanna <jeff.hanna@gmail.com>
#
# License: BSD (3-clause)

import mne
from mne import io
from mne.datasets import refmeg_noise
from mne.preprocessing import ICA
import numpy as np

print(__doc__)

data_path = refmeg_noise.data_path()

###############################################################################
# Read raw data, cropping to 5 minutes to save memory

raw_fname = data_path + '/sample_reference_MEG_noise-raw.fif'
raw = io.read_raw_fif(raw_fname).crop(300, 600).load_data()

###############################################################################
# Note that even though standard noise removal has already
# been applied to these data, much of the noise in the reference channels
# (bottom of the plot) can still be seen in the standard channels.
select_picks = np.concatenate(
    (mne.pick_types(raw.info, meg=True)[-32:],
     mne.pick_types(raw.info, meg=False, ref_meg=True)))
plot_kwargs = dict(
    duration=100, order=select_picks, n_channels=len(select_picks),
    scalings={"mag": 8e-13, "ref_meg": 2e-11})
raw.plot(**plot_kwargs)

###############################################################################
# The PSD of these data show the noise as clear peaks.
raw.plot_psd(fmax=30)

###############################################################################
# Run the "together" algorithm.
raw_tog = raw.copy()
ica_kwargs = dict(
    method='picard',
    fit_params=dict(tol=1e-4),  # use a high tol here for speed
)
all_picks = mne.pick_types(raw_tog.info, meg=True, ref_meg=True)
ica_tog = ICA(n_components=60, allow_ref_meg=True, **ica_kwargs)
ica_tog.fit(raw_tog, picks=all_picks)
# low threshold (2.0) here because of cropped data, entire recording can use
# a higher threshold (2.5)
bad_comps, scores = ica_tog.find_bads_ref(raw_tog, threshold=2.0)

# Plot scores with bad components marked.
ica_tog.plot_scores(scores, bad_comps)

# Examine the properties of removed components. It's clear from the time
# courses and topographies that these components represent external,
# intermittent noise.
ica_tog.plot_properties(raw_tog, picks=bad_comps)

# Remove the components.
raw_tog = ica_tog.apply(raw_tog, exclude=bad_comps)

###############################################################################
# Cleaned data:
raw_tog.plot_psd(fmax=30)

###############################################################################
# Now try the "separate" algorithm.
raw_sep = raw.copy()

# Do ICA only on the reference channels.
ref_picks = mne.pick_types(raw_sep.info, meg=False, ref_meg=True)
ica_ref = ICA(n_components=2, allow_ref_meg=True, **ica_kwargs)
ica_ref.fit(raw_sep, picks=ref_picks)

# Do ICA on both reference and standard channels. Here, we can just reuse
# ica_tog from the section above.
ica_sep = ica_tog.copy()

# Extract the time courses of these components and add them as channels
# to the raw data. Think of them the same way as EOG/EKG channels, but instead
# of giving info about eye movements/cardiac activity, they give info about
# external magnetic noise.
ref_comps = ica_ref.get_sources(raw_sep)
for c in ref_comps.ch_names:  # they need to have REF_ prefix to be recognised
    ref_comps.rename_channels({c: "REF_" + c})
raw_sep.add_channels([ref_comps])

# Now that we have our noise channels, we run the separate algorithm.
bad_comps, scores = ica_sep.find_bads_ref(raw_sep, method="separate")

# Plot scores with bad components marked.
ica_sep.plot_scores(scores, bad_comps)

# Examine the properties of removed components.
ica_sep.plot_properties(raw_sep, picks=bad_comps)

# Remove the components.
raw_sep = ica_sep.apply(raw_sep, exclude=bad_comps)

###############################################################################
# Cleaned raw data traces:

raw_sep.plot(**plot_kwargs)

###############################################################################
# Cleaned raw data PSD:

raw_sep.plot_psd(fmax=30)

##############################################################################
# References
# ----------
#
# .. footbibliography::
