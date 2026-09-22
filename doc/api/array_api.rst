.. _array_api:

Array API support (experimental)
================================

The `Python Array API standard <https://data-apis.org/array-api/latest/>`_
provides a common interface to array libraries. Selected MNE-Python operations
can use this interface through
`array-api-compat <https://data-apis.org/array-api-compat/>`_, keeping arrays in
their original backend rather than converting them to NumPy.
Support is experimental and limited to the operations listed below; it does
not make all MNE-Python functions or data containers GPU-compatible.

Installation and configuration
------------------------------

Install the optional dependencies with ``pip install 'mne[array-api]'`` and
install the array backend separately, for example PyTorch. NumPy workflows
do not require this extra or PyTorch.

Set ``SCIPY_ARRAY_API=1`` in the environment **before** importing SciPy,
scikit-learn, or MNE-Python. Enable scikit-learn's ``array_api_dispatch`` during
fitting, prediction, and scoring, as in the example below. Without dispatch,
scikit-learn uses its usual NumPy conversion path, which cannot accept arrays
on a GPU. See also the
`scikit-learn Array API guide <https://scikit-learn.org/stable/modules/array_api.html>`_.

Supported operations
--------------------

:class:`mne.decoding.ReceptiveField` supports ``fit``, ``predict``, and ``score``
when its estimator supports Array API inputs, for example
:class:`sklearn.linear_model.Ridge` with ``solver="svd"``.
This includes continuous and epoched data, positive and negative lags,
inverse patterns, and both ``"r2"`` and ``"corrcoef"`` scoring.
The fitted coefficients, patterns, delays, predictions, and score arrays use
the input backend and device. The estimator's ``array_api_support`` tag follows
the configured base estimator; its supported solvers and other restrictions
still apply.

The default :class:`mne.decoding.TimeDelayingRidge`, including the ``None`` and
numeric ``estimator`` shortcuts, requires NumPy inputs. The separate
``n_jobs="cuda"`` option for that estimator is not Array API dispatch.

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

Use arrays on the desired device when fitting and predicting. GPU execution
also requires that the backend, solver, and hardware support the necessary
operations; CPU tests alone do not establish GPU compatibility or speedups.
PyTorch CPU and the strict reference Array API implementation are exercised
with the minimum and current scikit-learn versions. Strict multi-output tests
require scikit-learn 1.9 or newer because older Ridge implementations have
restrictions on these inputs.

Limitations
-----------

- Use matching backend/device inputs throughout the workflow. Prediction
  rejects inputs on a different backend or device from the fitted model.
- Arrays must be mutable: delay construction uses indexed assignment.
- Use float32 or float64 features with ``Ridge(solver="svd")``. Integer tensor
  features are converted to float64. During fitting, tensor targets are cast
  to the feature dtype; scoring retains target precision. The legacy NumPy
  delay buffer promotes float32 features to float64 and is unchanged.
- ``score`` returns one value per output, as in the NumPy API, not the scalar
  expected by some generic scikit-learn checks. Correlation is NaN for
  constant or non-finite columns; complex-valued targets are unsupported.
- The explicit delayed design uses memory proportional to samples, features,
  and delays, plus the solver's workspace. It is not the FFT-based
  ``TimeDelayingRidge`` algorithm; make sure the design fits in device memory.
