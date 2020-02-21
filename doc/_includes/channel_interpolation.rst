:orphan:

Bad channel repair via interpolation
====================================

Spherical spline interpolation (EEG)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. NOTE: part of this file is included in doc/overview/implementation.rst.
   Changes here are reflected there. If you want to link to this content, link
   to :ref:`channel-interpolation` to link to that section of the
   implementation.rst page. The next line is a target for :start-after: so we
   can omit the title from the include:
   channel-interpolation-begin-content

In short, data repair using spherical spline interpolation :footcite:`PerrinEtAl1989` consists of the following steps:

* Project the good and bad electrodes onto a unit sphere
* Compute a mapping matrix that maps :math:`N` good channels to :math:`M` bad channels
* Use this mapping matrix to compute interpolated data in the bad channels

Spherical splines assume that the potential :math:`V(\boldsymbol{r_i})` at any point :math:`\boldsymbol{r_i}` on the surface of the sphere can be represented by:

.. math:: V(\boldsymbol{r_i}) = c_0 + \sum_{j=1}^{N}c_{i}g_{m}(cos(\boldsymbol{r_i}, \boldsymbol{r_{j}}))
   :label: model

where the :math:`C = (c_{1}, ..., c_{N})^{T}` are constants which must be estimated. The function :math:`g_{m}(\cdot)` of order :math:`m` is given by:

.. math:: g_{m}(x) = \frac{1}{4 \pi}\sum_{n=1}^{\infty} \frac{2n + 1}{(n(n + 1))^m}P_{n}(x)
   :label: legendre

where :math:`P_{n}(x)` are `Legendre polynomials`_ of order `n`.

.. _Legendre polynomials: https://en.wikipedia.org/wiki/Legendre_polynomials

To estimate the constants :math:`C`, we must solve the following two equations simultaneously:

.. math:: G_{ss}C + T_{s}c_0 = X
   :label: matrix_form

.. math:: {T_s}^{T}C = 0
   :label: constraint

where :math:`G_{ss} \in R^{N \times N}` is a matrix whose entries are :math:`G_{ss}[i, j] = g_{m}(cos(\boldsymbol{r_i}, \boldsymbol{r_j}))` and :math:`X \in R^{N \times 1}` are the potentials :math:`V(\boldsymbol{r_i})` measured at the good channels. :math:`T_{s} = (1, 1, ..., 1)^\top` is a column vector of dimension :math:`N`. Equation :eq:`matrix_form` is the matrix formulation of Equation :eq:`model` and equation :eq:`constraint` is like applying an average reference to the data. From equation :eq:`matrix_form` and :eq:`constraint`, we get:

.. math:: \begin{bmatrix} c_0 \\ C \end{bmatrix} = {\begin{bmatrix} {T_s}^{T} && 0 \\ T_s && G_{ss} \end{bmatrix}}^{-1} \begin{bmatrix} 0 \\ X \end{bmatrix} = C_{i}X
   :label: estimate_constant

:math:`C_{i}` is the same as matrix :math:`{\begin{bmatrix} {T_s}^{T} && 0 \\ T_s && G_{ss} \end{bmatrix}}^{-1}` but with its first column deleted, therefore giving a matrix of dimension :math:`(N + 1) \times N`.

Now, to estimate the potentials :math:`\hat{X} \in R^{M \times 1}` at the bad channels, we have to do:

.. math:: \hat{X} = G_{ds}C + T_{d}c_0
   :label: estimate_data

where :math:`G_{ds} \in R^{M \times N}` computes :math:`g_{m}(\boldsymbol{r_i}, \boldsymbol{r_j})` between the bad and good channels. :math:`T_{d} = (1, 1, ..., 1)^\top` is a column vector of dimension :math:`M`. Plugging in equation :eq:`estimate_constant` in :eq:`estimate_data`, we get

.. math:: \hat{X} = \begin{bmatrix} T_d && G_{ds} \end{bmatrix} \begin{bmatrix} c_0 \\ C \end{bmatrix} = \underbrace{\begin{bmatrix} T_d && G_{ds} \end{bmatrix} C_{i}}_\text{mapping matrix}X


To interpolate bad channels, one can simply do:

	>>> evoked.interpolate_bads(reset_bads=False)  # doctest: +SKIP

and the bad channel will be fixed.

.. target for :end-before: channel-interpolation-end-content

.. topic:: Examples:

	* :ref:`sphx_glr_auto_examples_preprocessing_plot_interpolate_bad_channels.py`


References
~~~~~~~~~~

.. footbibliography::
