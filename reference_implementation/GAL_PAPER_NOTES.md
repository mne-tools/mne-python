# GAL decoding research notes

Sources: Santos-Mayo et al., *Decoding in the Fourth Dimension* ([Human
Brain Mapping, 2025](https://doi.org/10.1002/hbm.70152)); the
[Time-GAL MATLAB toolbox](https://github.com/csea-lab/time-GAL); and its
[OSF project](https://osf.io/q56ns/). The article is open access under CC BY;
these are original-worded implementation notes, not a transcription.

## What GAL decodes

Conventional M/EEG temporal decoding treats sensors as features and fits one
model per time point. Time-GAL reverses those roles. At a single sensor, every
time sample in one trial is a feature of an LDA classifier. A classifier fitted
at one sensor is evaluated at all other sensors. Thus, a GAL matrix has
training-sensor rows and test-sensor columns, and a diagonal value is the
held-out accuracy when training and testing at the same sensor.

For the ERP example, the two classes are pleasant and unpleasant pictures. The
inputs produced by `groupCondition.m` are preprocessed arrays of shape
`channels × time × trials`, plus one participant identifier per trial. The
aggregate files currently contain 129 channels and 601 samples. The published
analysis uses E1--E124 and Cz, omitting E125--E128 facial electrodes.

## MATLAB-to-MNE correspondence

| Time-GAL operation | MNE/scikit-learn operation |
| --- | --- |
| One LDA per channel, temporal waveform as features | transpose arrays to `trials × time × sensors` and use `GeneralizingEstimator` |
| Leave one participant out | `LeaveOneGroupOut` with participant IDs as `groups` |
| Train one channel, test every channel | final estimator axis is sensor; score is training sensor × test sensor |
| Correct classifications / test trials | `scoring="accuracy"` |
| MATLAB `fitcdiscr(..., "Prior", "empirical")` | `LinearDiscriminantAnalysis(solver="lsqr", priors=None)` |

No feature scaler is inserted. This is intentional: the MATLAB implementation
fits LDA directly to the supplied single-trial waveforms.

## Group inference reproduced here

The toolbox runs a one-sample t-test against accuracy 0.5 at each GAL cell,
separately above and below chance. It corrects with `alpha / n_channels`, so
the ERP run uses `0.05 / 125`. This project reproduces that setting solely for
MATLAB parity and labels it as a study-specific legacy choice, not a general
connectivity-inference prescription.

The correlation-based temporal weighting, Time-GAL masks, circular graphs,
source projection, and CSD/Laplacian preprocessing are intentionally excluded.

## Replication checks and caveats

The archive file named `resultsTimeGAL_IAPS_ERP.mat` is mislabeled: its
1,501-sample input averages exactly match Dataset 2 (Habituation vs.
Extinction), which has 31 participants. Dataset 1 instead contains the
39-participant, 601-sample Pleasant vs. Unpleasant ERP analysis described
above. The documentation tutorial uses Dataset 2 so that its raw-data MNE run
can be compared to the supplied MATLAB tensor. Score comparison permits one
held-out trial per fold and also compares positive and negative masks exactly.
Current MNE and MATLAB outputs still differ, so the tutorial reports the
observed agreement rather than claiming numerical parity.
