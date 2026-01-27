# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.


def _check_n_features_3d(estimator, X, reset):  # pragma: no cover
    n_features = X.shape[-1]
    if reset:
        estimator.n_features_in_ = n_features
        return

    if not hasattr(estimator, "n_features_in_"):
        # Skip this check if the expected number of expected input features
        # was not recorded by calling fit first. This is typically the case
        # for stateless transformers.
        return

    if n_features != estimator.n_features_in_:
        raise ValueError(
            f"X has {n_features} features, but {estimator.__class__.__name__} "
            f"is expecting {estimator.n_features_in_} features as input."
        )
