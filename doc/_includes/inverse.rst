.. _ch_mne:

The minimum-norm current estimates
==================================

This page describes the mathematical concepts and the computation of the
minimum-norm estimates needed in order to obtain the linear inverse operator.

.. contents:: Page contents
   :local:
   :depth: 2

Overview
~~~~~~~~

.. NOTE: part of this file is included in doc/overview/implementation.rst.
   Changes here are reflected there. If you want to link to this content, link
   to :ref:`ch_mne` to link to that section of the implementation.rst page.
   The next line is a target for :start-after: so we can omit the title from
   the include:
   inverse-begin-content


Computing the inverse operator is accomplished using
:func:`mne.minimum_norm.make_inverse_operator` and
:func:`mne.minimum_norm.apply_inverse`. The use of these functions is presented
in the tutorial :ref:`tut-inverse-methods`.

.. _minimum_norm_estimates:

Minimum-norm estimates
~~~~~~~~~~~~~~~~~~~~~~

This section describes the mathematical details of the calculation of
minimum-norm estimates. In Bayesian sense, the ensuing current distribution is
the maximum a posteriori (MAP) estimate under the following assumptions:

- The viable locations of the currents are constrained to the cortex.
  Optionally, the current orientations can be fixed to be normal to the
  cortical mantle.

- The amplitudes of the currents have a Gaussian prior distribution with a
  known source covariance matrix.

- The measured data contain additive noise with a Gaussian distribution with a
  known covariance matrix. The noise is not correlated over time.

The linear inverse operator
---------------------------

The measured data in the source estimation procedure consists of MEG and EEG
data, recorded on a total of N channels. The task is to estimate a total of M
strengths of sources located on the cortical mantle. If the number of source
locations is P, M = P for fixed-orientation sources and M = 3P if the source
orientations are unconstrained. The regularized linear inverse operator
following from the Bayesian approach is given by the :math:`M \times N` matrix

.. math::    M = R' G^T (G R' G^T + C)^{-1}\ ,

.. sidebar:: Inverse operators in MNE-Python

   For computational convenience, in MNE-Python the linear inverse operator is
   not computed explicitly. See :ref:`mne_solution` for mathematical
   details, and :ref:`CIHCFJEI` for a detailed example.

where G is the gain matrix relating the source strengths to the measured
MEG/EEG data, :math:`C` is the data noise-covariance matrix and :math:`R'` is
the source covariance matrix. The dimensions of these matrices are :math:`N
\times M`, :math:`N \times N`, and :math:`M \times M`, respectively. The
:math:`M \times 1` source-strength vector is obtained by multiplying the
:math:`N \times 1` data vector by :math:`M`.

The expected value of the current amplitudes at time *t* is then given by
:math:`\hat{j}(t) = Mx(t)`, where :math:`x(t)` is a vector containing the
measured MEG and EEG data values at time *t*.

.. _mne_regularization:

Regularization
--------------

The a priori variance of the currents is, in practise, unknown. We can express
this by writing :math:`R' = R/ \lambda^2`, which yields the inverse operator

.. math::    M = R G^T (G R G^T + \lambda^2 C)^{-1}\ ,

where the unknown current amplitude is now interpreted in terms of the
regularization parameter :math:`\lambda^2`. Small :math:`\lambda^2` corresponds
to large current amplitudes and complex estimate current patterns while a large
:math:`\lambda^2` means the amplitude of the current is limited and a simpler,
smooth, current estimate is obtained.

We can arrive in the regularized linear inverse operator
also by minimizing the cost function

.. math::    S = \tilde{e}^T \tilde{e} + \lambda^2 j^T R^{-1} j\ ,

where the first term consists of the difference between the whitened measured
data (see :ref:`whitening_and_scaling`) and those predicted by the model while the
second term is a weighted-norm of the current estimate. It is seen that, with
increasing :math:`\lambda^2`, the source term receive more weight and larger
discrepancy between the measured and predicted data is tolerable.

.. _whitening_and_scaling:

Whitening and scaling
---------------------

The MNE software employs data whitening so that a 'whitened' inverse operator
assumes the form

.. math::    \tilde{M} = R \tilde{G}^T (\tilde{G} R \tilde{G}^T + I)^{-1}\ ,

where :math:`\tilde{G} = C^{-^1/_2}G` is the spatially whitened gain matrix.
The expected current values are :math:`\hat{j} = Mx(t)`, where :math:`x(t) =
C^{-^1/_2}x(t)` is a the whitened measurement vector at *t*. The spatial
whitening operator is obtained with the help of the eigenvalue decomposition
:math:`C = U_C \Lambda_C^2 U_C^T` as :math:`C^{-^1/_2} = \Lambda_C^{-1} U_C^T`.
In the MNE software the noise-covariance matrix is stored as the one applying
to raw data. To reflect the decrease of noise due to averaging, this matrix,
:math:`C_0`, is scaled by the number of averages, :math:`L`, *i.e.*, :math:`C =
C_0 / L`.

As shown above, regularization of the inverse solution is equivalent to a
change in the variance of the current amplitudes in the Bayesian *a priori*
distribution.

A convenient choice for the source-covariance matrix :math:`R` is such that
:math:`\text{trace}(\tilde{G} R \tilde{G}^T) / \text{trace}(I) = 1`. With this
choice we can approximate :math:`\lambda^2 \sim 1/SNR`, where SNR is the
(power) signal-to-noise ratio of the whitened data.

.. note::
   The definition of the signal to noise-ratio/ :math:`\lambda^2` relationship
   given above works nicely for the whitened forward solution. In the
   un-whitened case scaling with the trace ratio :math:`\text{trace}(GRG^T) /
   \text{trace}(C)` does not make sense, since the diagonal elements summed
   have, in general, different units of measure. For example, the MEG data are
   expressed in T or T/m whereas the unit of EEG is Volts.

See :ref:`tut_compute_covariance` for example of noise covariance computation
and whitening.

.. _cov_regularization_math:

Regularization of the noise-covariance matrix
---------------------------------------------

Since finite amount of data is usually available to compute an estimate of the
noise-covariance matrix :math:`C`, the smallest eigenvalues of its estimate are
usually inaccurate and smaller than the true eigenvalues. Depending on the
seriousness of this problem, the following quantities can be affected:

- The model data predicted by the current estimate,

- Estimates of signal-to-noise ratios, which lead to estimates of the required
  regularization, see :ref:`mne_regularization`,

- The estimated current values, and

- The noise-normalized estimates, see :ref:`noise_normalization`.

Fortunately, the latter two are least likely to be affected due to
regularization of the estimates. However, in some cases especially the EEG part
of the noise-covariance matrix estimate can be deficient, *i.e.*, it may
possess very small eigenvalues and thus regularization of the noise-covariance
matrix is advisable.

Historically, the MNE software accomplishes the regularization by replacing a
noise-covariance matrix estimate :math:`C` with

.. math::    C' = C + \sum_k {\varepsilon_k \bar{\sigma_k}^2 I^{(k)}}\ ,

where the index :math:`k` goes across the different channel groups (MEG planar
gradiometers, MEG axial gradiometers and magnetometers, and EEG),
:math:`\varepsilon_k` are the corresponding regularization factors,
:math:`\bar{\sigma_k}` are the average variances across the channel groups, and
:math:`I^{(k)}` are diagonal matrices containing ones at the positions
corresponding to the channels contained in each channel group.

See :ref:`plot_compute_covariance_howto` for details on computing and
regularizing the channel covariance matrix.

.. _mne_solution:

Computation of the solution
---------------------------

The most straightforward approach to calculate the MNE is to employ expression
for the original or whitened inverse operator directly. However, for
computational convenience we prefer to take another route, which employs the
singular-value decomposition (SVD) of the matrix

.. math::    A = \tilde{G} R^{^1/_2} = U \Lambda V^T

where the superscript :math:`^1/_2` indicates a square root of :math:`R`. For a
diagonal matrix, one simply takes the square root of :math:`R` while in the
more general case one can use the Cholesky factorization :math:`R = R_C R_C^T`
and thus :math:`R^{^1/_2} = R_C`.

With the above SVD it is easy to show that

.. math::    \tilde{M} = R^{^1/_2} V \Gamma U^T

where the elements of the diagonal matrix :math:`\Gamma` are

.. math::    \gamma_k = \frac{1}{\lambda_k} \frac{\lambda_k^2}{\lambda_k^2 + \lambda^2}\ .

With :math:`w(t) = U^T C^{-^1/_2} x(t)` the expression for the expected current
is

.. math::    \hat{j}(t) = R^C V \Gamma w(t) = \sum_k {\bar{v_k} \gamma_k w_k(t)}\ ,

where :math:`\bar{v_k} = R^C v_k`, :math:`v_k` being the :math:`k` th column of
:math:`V`. It is thus seen that the current estimate is a weighted sum of the
'modified' eigenleads :math:`v_k`.

It is easy to see that :math:`w(t) \propto \sqrt{L}`. To maintain the relation
:math:`(\tilde{G} R \tilde{G}^T) / \text{trace}(I) = 1` when :math:`L` changes
we must have :math:`R \propto 1/L`. With this approach, :math:`\lambda_k` is
independent of  :math:`L` and, for fixed :math:`\lambda`, we see directly that
:math:`j(t)` is independent of :math:`L`.

.. sidebar:: Computing the solution in MNE-Python

   In MNE-Python the minimum-norm estimate is computed using
   :func:`mne.minimum_norm.make_inverse_operator` and its usage is illustrated
   in :ref:`CIHCFJEI`.


.. _noise_normalization:

Noise normalization
-------------------

The noise-normalized linear estimates introduced by Dale et al.
:footcite:`DaleEtAl1999` require division of the expected current amplitude by
its variance. Noise normalization serves three purposes:

- It converts the expected current value into a dimensionless statistical test
  variable. Thus the resulting time and location dependent values are often
  referred to as dynamic statistical parameter maps (dSPM).

- It reduces the location bias of the estimates. In particular, the tendency of
  the MNE to prefer superficial currents is eliminated.

- The width of the point-spread function becomes less dependent on the source
  location on the cortical mantle. The point-spread is defined as the MNE
  resulting from the signals coming from a point current source (a current
  dipole) located at a certain point on the cortex.

In practice, noise normalization requires the computation of the diagonal
elements of the matrix

.. math::    M C M^T = \tilde{M} \tilde{M}^T\ .

With help of the singular-value decomposition approach we see directly that

.. math::    \tilde{M} \tilde{M}^T\ = \bar{V} \Gamma^2 \bar{V}^T\ .

Under the conditions expressed at the end of :ref:`mne_solution`, it
follows that the *t*-statistic values associated with fixed-orientation
sources) are thus proportional to :math:`\sqrt{L}` while the *F*-statistic
employed with free-orientation sources is proportional to :math:`L`,
correspondingly.

.. note::
   The MNE software usually computes the *square roots* of the F-statistic to
   be displayed on the inflated cortical surfaces. These are also proportional
   to :math:`\sqrt{L}`.

Predicted data
--------------

Under noiseless conditions the SNR is infinite and thus leads to
:math:`\lambda^2 = 0` and the minimum-norm estimate explains the measured data
perfectly. Under realistic conditions, however, :math:`\lambda^2 > 0` and there
is a misfit between measured data and those predicted by the MNE. Comparison of
the predicted data, here denoted by :math:`x(t)`, and measured one can give
valuable insight on the correctness of the regularization applied.

In the SVD approach we easily find

.. math::    \hat{x}(t) = G \hat{j}(t) = C^{^1/_2} U \Pi w(t)\ ,

where the diagonal matrix :math:`\Pi` has elements :math:`\pi_k = \lambda_k
\gamma_k` The predicted data is thus expressed as the weighted sum of the
'recolored eigenfields' in :math:`C^{^1/_2} U`.

Cortical patch statistics
-------------------------

.. sidebar:: Cortical patch statistics in MNE-Python

   In MNE-Python, the ``use_cps`` parameter in
   :func:`mne.convert_forward_solution`, and
   :func:`mne.minimum_norm.make_inverse_operator` controls whether to use
   cortical patch statistics (CPS) to define normal orientations or not (see
   :ref:`CHDBBCEJ`).

If the ``add_dists=True`` option was used in source space creation,
the source space file will contain
Cortical Patch Statistics (CPS) for each vertex of the cortical surface. The
CPS provide information about the source space point closest to it as well as
the distance from the vertex to this source space point. The vertices for which
a given source space point is the nearest one define the cortical patch
associated with with the source space point. Once these data are available, it
is straightforward to compute the following cortical patch statistics for each
source location :math:`d`:

- The average over the normals of at the vertices in a patch,
  :math:`\bar{n_d}`,

- The areas of the patches, :math:`A_d`, and

- The average deviation of the vertex normals in a patch from their average,
  :math:`\sigma_d`, given in degrees.

.. _inverse_orientation_constrains:

The orientation constraints
---------------------------

.. sidebar:: Orientation constraints in MNE-Python

   In MNE-Python, rigid orientation is employed by specifying ``fixed=True`` in
   :func:`mne.minimum_norm.make_inverse_operator` (forcing dipole orientation
   to be orthogonal to the cortical surface, pointing outwards). If cortical
   patch statistics are available the average normal over each patch,
   :math:`\bar{n_d}`, are used to define the source orientation. Otherwise, the
   vertex normal at the source space location is employed. See
   :ref:`plot_dipole_orientations_fixed_orientations`.

   The *fLOC* is employed by specifying ``fixed=False`` and ``loose=1.0`` when
   calling :func:`mne.minimum_norm.make_inverse_operator`. See
   :ref:`plot_dipole_orientations_fLOC_orientations`.

   The *vLOC* is employed by specifying ``fixed=False`` and ``loose``
   parameters when calling :func:`mne.minimum_norm.make_inverse_operator`. This
   is similar to *fLOC* except that the value given with the ``loose``
   parameter will be multiplied by :math:`\sigma_d`, defined above. See
   :ref:`plot_dipole_orientations_vLOC_orientations`.

The principal sources of MEG and EEG signals are generally believed to be
postsynaptic currents in the cortical pyramidal neurons. Since the net primary
current associated with these microscopic events is oriented normal to the
cortical mantle, it is reasonable to use the cortical normal orientation as a
constraint in source estimation. In addition to allowing completely free source
orientations, the MNE software implements three orientation constraints based
of the surface normal data:

- Source orientation can be rigidly fixed to the surface normal direction (the
  ``--fixed`` option). If cortical patch statistics are available the average
  normal over each patch, :math:`\bar{n_d}`, are used to define the source
  orientation. Otherwise, the vertex normal at the source space location is
  employed.

- A *location independent or fixed loose orientation constraint* (fLOC) can be
  employed (the ``--loose`` option). In this approach, a source coordinate
  system based on the local surface orientation at the source location is
  employed. By default, the three columns of the gain matrix G, associated with
  a given source location, are the fields of unit dipoles pointing to the
  directions of the :math:`x`, :math:`y`, and :math:`z` axis of the coordinate
  system employed in the forward calculation (usually the :ref:`MEG head
  coordinate frame <head_device_coords>`). For LOC the orientation is changed so
  that the first two source components lie in the plane normal to the surface
  normal at the source location and the third component is aligned with it.
  Thereafter, the variance of the source components tangential to the cortical
  surface are reduced by a factor defined by the ``--loose`` option.

- A *variable loose orientation constraint* (vLOC) can be employed (the
  ``--loosevar`` option). This is similar to fLOC except that the value given
  with the ``--loosevar`` option will be multiplied by :math:`\sigma_d`,
  defined above.

Depth weighting
---------------

.. sidebar:: Adjusting depth weighting in MNE-Python

   The maximal amount of depth weighting can be adjusted with ``depth``
   parameter in :func:`mne.minimum_norm.make_inverse_operator`.

The minimum-norm estimates have a bias towards superficial currents. This
tendency can be alleviated by adjusting the source covariance matrix :math:`R`
to favor deeper source locations. In the depth weighting scheme employed in MNE
analyze, the elements of :math:`R` corresponding to the :math:`p` th source
location are be scaled by a factor

.. math::    f_p = (g_{1p}^T g_{1p} + g_{2p}^T g_{2p} + g_{3p}^T g_{3p})^{-\gamma}\ ,

where :math:`g_{1p}`, :math:`g_{2p}`, and :math:`g_{3p}` are the three columns
of :math:`G` corresponding to source location :math:`p` and :math:`\gamma` is
the order of the depth weighting, which is specified via the ``depth`` option.

Effective number of averages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is often the case that the epoch to be analyzed is a linear combination over
conditions rather than one of the original averages computed. As stated above,
the noise-covariance matrix computed is originally one corresponding to raw
data. Therefore, it has to be scaled correctly to correspond to the actual or
effective number of epochs in the condition to be analyzed. In general, we have

.. math::    C = C_0 / L_{eff}

where :math:`L_{eff}` is the effective number of averages. To calculate
:math:`L_{eff}` for an arbitrary linear combination of conditions

.. math::    y(t) = \sum_{i = 1}^n {w_i x_i(t)}

we make use of the the fact that the noise-covariance matrix

.. math::    C_y = \sum_{i = 1}^n {w_i^2 C_{x_i}} = C_0 \sum_{i = 1}^n {w_i^2 / L_i}

which leads to

.. math::    1 / L_{eff} = \sum_{i = 1}^n {w_i^2 / L_i}

An important special case  of the above is a weighted average, where

.. math::    w_i = L_i / \sum_{i = 1}^n {L_i}

and, therefore

.. math::    L_{eff} = \sum_{i = 1}^n {L_i}

Instead of a weighted average, one often computes a weighted sum, a simplest
case being a difference or sum of two categories. For a difference :math:`w_1 =
1` and :math:`w_2 = -1` and thus

.. math::    1 / L_{eff} = 1 / L_1 + 1 / L_2

or

.. math::    L_{eff} = \frac{L_1 L_2}{L_1 + L_2}

Interestingly, the same holds for a sum, where :math:`w_1 = w_2 = 1`.
Generalizing, for any combination of sums and differences, where :math:`w_i =
1` or :math:`w_i = -1`, :math:`i = 1 \dotso n`, we have

.. math::    1 / L_{eff} = \sum_{i = 1}^n {1/{L_i}}

.. target for :end-before: inverse-end-content

References
~~~~~~~~~~

.. footbibliography::
