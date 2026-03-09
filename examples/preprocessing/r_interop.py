"""
.. _r-interop:

==============================================
Mass-univariate t-test: Python and R compared
==============================================

This example shows how to run a mass-univariate 2-sample t-test on
:class:`~mne.Epochs` data in Python using :func:`scipy.stats.ttest_ind`,
then run the equivalent test in R via :mod:`rpy2`, and confirm that both
approaches give identical results.

This is useful when you want to leverage R's statistical ecosystem (e.g.,
``lme4``, ``EMMeans``) while keeping your data loading and visualization
in Python.

.. note::
    This example requires ``rpy2`` to be installed (``pip install rpy2``)
    and a working R installation with the ``stats`` package (included by
    default in R).
"""
# Authors: Aman Srivastava <amansri345@gmail.com>
#
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

# %%
# Load sample data and create Epochs
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We use the MNE sample dataset and create epochs for two conditions:
# auditory/left and auditory/right.

import rpy2.robjects as ro
from rpy2.robjects import default_converter, numpy2ri
from rpy2.robjects.conversion import localconverter
from scipy import stats

import mne

data_path = mne.datasets.sample.data_path()
raw_fname = data_path / "MEG" / "sample" / "sample_audvis_filt-0-40_raw.fif"
event_fname = data_path / "MEG" / "sample" / "sample_audvis_filt-0-40_raw-eve.fif"

raw = mne.io.read_raw_fif(raw_fname, preload=True)
events = mne.read_events(event_fname)

event_id = {"auditory/left": 1, "auditory/right": 2}
tmin, tmax = -0.2, 0.5

epochs = mne.Epochs(
    raw,
    events,
    event_id=event_id,
    tmin=tmin,
    tmax=tmax,
    baseline=(None, 0),
    preload=True,
)

# %%
# Extract data and run t-test in Python
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We average over channels and time to get one value per epoch, then run
# a 2-sample t-test comparing the two conditions.

epochs_left = epochs["auditory/left"].get_data(picks="eeg").mean(axis=(1, 2))
epochs_right = epochs["auditory/right"].get_data(picks="eeg").mean(axis=(1, 2))

t_python, p_python = stats.ttest_ind(epochs_left, epochs_right)
print(f"Python  →  t = {t_python:.4f},  p = {p_python:.4f}")

# %%
# Run the same t-test in R via rpy2
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We pass the same NumPy arrays to R using :mod:`rpy2` and call R's
# built-in ``t.test()``.

with localconverter(default_converter + numpy2ri.converter):
    r_left = ro.FloatVector(epochs_left)
    r_right = ro.FloatVector(epochs_right)

r_ttest = ro.r["t.test"]
result = r_ttest(r_left, r_right, **{"var.equal": True})

t_r = float(result.rx2("statistic")[0])
p_r = float(result.rx2("p.value")[0])
print(f"R       →  t = {t_r:.4f},  p = {p_r:.4f}")

# %%
# Compare results
# ^^^^^^^^^^^^^^^^
#
# Both approaches give identical t and p values (up to floating point
# precision), confirming that R and Python produce equivalent results.

print(f"\nt difference: {abs(t_python - t_r):.2e}")
print(f"p difference: {abs(p_python - p_r):.2e}")
