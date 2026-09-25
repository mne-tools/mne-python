.. _array_api:

Array API support (experimental)
================================

:class:`mne.decoding.ReceptiveField` can use Array API inputs, such as PyTorch
tensors, with compatible estimators such as :class:`sklearn.linear_model.Ridge`
with ``solver="svd"``. Fitting, prediction, inverse patterns, and both scoring
methods keep arrays in the input backend and on its device. This experimental
support does not make other MNE-Python operations GPU-compatible.

Installation and configuration
------------------------------

Install ``array-api-compat >= 1.12``, ``scikit-learn >= 1.5``, and your array
backend separately. NumPy workflows do not require these optional array packages.

Set ``SCIPY_ARRAY_API=1`` in the environment **before** importing SciPy,
scikit-learn, or MNE-Python. Enable scikit-learn's ``array_api_dispatch`` during
fitting, prediction, and scoring. For supported estimators, devices, and
dependency versions, see the
`scikit-learn Array API guide <https://scikit-learn.org/stable/modules/array_api.html>`_.

Example
-------

After setting the environment variable, the following runs on the CPU using
PyTorch tensors. It requires no downloaded data:

.. code-block:: python

    import numpy as np
    import torch
    from sklearn import config_context
    from sklearn.linear_model import Ridge
    from mne.decoding import ReceptiveField

    rng = np.random.default_rng(0)
    X = torch.asarray(rng.standard_normal((100, 3)), dtype=torch.float64)
    y = X[:, 0] - 0.5 * X[:, 1]
    rf = ReceptiveField(
        -0.2, 0.4, sfreq=10,
        estimator=Ridge(solver="svd", random_state=0), patterns=True,
    )
    with config_context(array_api_dispatch=True):
        rf.fit(X[:70], y[:70])
        predicted = rf.predict(X[70:])
        scores = rf.score(X[70:], y[70:])

To use a GPU, create your input tensors on that device. The backend, solver,
hardware, and chosen dtype must support the required operations.

Limitations
-----------

- Use matching backend/device inputs throughout the workflow. Prediction
  rejects inputs on a different backend or device from the fitted model.
  Multi-output ``array-api-strict`` inputs require scikit-learn 1.9 or newer.
- The default :class:`mne.decoding.TimeDelayingRidge`, including ``None`` and
  numeric ``estimator`` shortcuts, requires NumPy. Its ``n_jobs="cuda"``
  option is separate from Array API support.
- Arrays must be mutable: delay construction uses indexed assignment.
- Use float32 or float64 features with ``Ridge(solver="svd")``. Integer tensor
  features are converted to float64. During fitting, tensor targets are cast
  to the feature dtype. The legacy NumPy delay buffer promotes float32
  features to float64 and is unchanged.
- ``score`` returns one value per output, as in the NumPy API. Correlation
  requires at least two samples; complex-valued tensor targets are unsupported.
- The explicit delayed design uses memory proportional to samples, features,
  and delays, plus the solver's workspace. It is not the FFT-based
  ``TimeDelayingRidge`` algorithm; make sure the design fits in device memory.
