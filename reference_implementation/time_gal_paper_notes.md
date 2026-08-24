# Time-GAL paper notes

Source: Santos-Mayo et al., *Decoding in the Fourth Dimension:
Classification of Temporal Patterns and Their Generalization Across Locations*,
Human Brain Mapping (2025), DOI
[10.1002/hbm.70152](https://doi.org/10.1002/hbm.70152). The supplied paper is
open access under CC BY; this Markdown note is an attributed, tutorial-focused
conversion rather than a replacement for the article.

## Central idea

Time-GAL uses all time points from one sensor's trial waveform as features for
a binary classifier. Repeating that fit for each sensor and testing each model
at every sensor produces a generalization-across-location matrix. The source
paper calls this a backward model: it extracts information about the
experimental condition from recorded signals.

## Data contract

The MATLAB workflow expects one three-dimensional array per condition with
shape `sensors x time x trials`, plus trial-to-participant labels. In MNE's
decoding convention the equivalent GAL input is `trials x time x sensors`:
time is the classifier feature axis and sensors are the
`GeneralizingEstimator` generalization axis.

## Paper datasets

The paper demonstrates the method with (1) affective-picture ERPs from 39
participants and (2) fear-conditioning ssVEPs from 31 participants. Both were
recorded with a HydroCel net at 500 Hz and processed with a current-source
density transform. The MNE tutorial instead uses the CI-supported
`visual_92_categories` MEG dataset, so that it is runnable without the OSF
archive and does not claim MATLAB numerical equivalence.

## Interpretation

The GAL diagonal is within-sensor temporal decoding. Off-diagonal values show
whether a temporal decoder learned at one location transfers to another. High
off-diagonal accuracy is therefore evidence of shared decodable temporal
information, rather than a direct measure of anatomical connectivity.
