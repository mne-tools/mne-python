.. _ch_mne:

The minimum-norm current estimates
==================================

.. contents:: Page contents
   :local:
   :depth: 2

.. NOTE: part of this file is included in doc/overview/implementation.rst.
   Changes here are reflected there. If you want to link to this content, link
   to :ref:`ch_mne` to link to that section of the implementation.rst page.
   The next line is a target for :start-after: so we can omit the title from
   the include:
   inverse-begin-content


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

Computing the inverse operator is accomplished using
:func:`mne.minimum_norm.make_inverse_operator` and
:func:`mne.minimum_norm.apply_inverse`. The use of these functions is presented
in the tutorial :ref:`tut-inverse-methods`.

The linear inverse operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The measured data in the source estimation procedure consists of MEG and EEG
data, recorded on a total of N channels. The task is to estimate a total of
:math:`Q`
strengths of sources located on the cortical mantle. If the number of source
locations is :math:`P`, :math:`Q = P` for fixed-orientation sources and
:math:`Q = 3P` if the source
orientations are unconstrained. The regularized linear inverse operator
following from regularized maximal likelihood of the above probabilistic model
is given by the :math:`Q \times N` matrix

.. math::    M = R' G^\top (G R' G^\top + C)^{-1}\ ,

where :math:`G` is the gain matrix relating the source strengths to the measured
MEG/EEG data, :math:`C` is the data noise-covariance matrix and :math:`R'` is
the source covariance matrix. The dimensions of these matrices are :math:`N
\times Q`, :math:`N \times N`, and :math:`Q \times Q`, respectively. The
:math:`Q \times 1` source-strength vector is obtained by multiplying the
:math:`Q \times 1` data vector by :math:`Q`.

The expected value of the current amplitudes at time *t* is then given by
:math:`\hat{j}(t) = Mx(t)`, where :math:`x(t)` is a vector containing the
measured MEG and EEG data values at time *t*.

For computational convenience, the linear inverse operator is
not computed explicitly. See :ref:`mne_solution` for mathematical
details, and :ref:`CIHCFJEI` for a detailed example.

.. _mne_regularization:

Regularization
~~~~~~~~~~~~~~

The a priori variance of the currents is, in practice, unknown. We can express
this by writing :math:`R' = R/ \lambda^2 = R \lambda^{-2}`, which yields the
inverse operator

.. math::
   :label: inv_m

    M &= R' G^\top (G R' G^\top + C)^{-1} \\
      &= R \lambda^{-2} G^\top (G R \lambda^{-2} G^\top + C)^{-1} \\
      &= R \lambda^{-2} G^\top \lambda^2 (G R G^\top + \lambda^2 C)^{-1} \\
      &= R G^\top (G R G^\top + \lambda^2 C)^{-1}\ ,

where the unknown current amplitude is now interpreted in terms of the
regularization parameter :math:`\lambda^2`. Larger :math:`\lambda^2` values
correspond to spatially smoother and weaker current amplitudes, whereas smaller
:math:`\lambda^2` values lead to the opposite.

We can arrive at the regularized linear inverse operator also by minimizing a
cost function :math:`S` with respect to the estimated current :math:`\hat{j}`
(given the measurement vector :math:`x` at any given time :math:`t`) as

.. math::

    \min_\hat{j} \Bigl\{ S \Bigr\} &= \min_\hat{j} \Bigl\{ \tilde{e}^\top \tilde{e} + \lambda^2 \hat{j}^\top R^{-1} \hat{j} \Bigr\} \\
                                   &= \min_\hat{j} \Bigl\{ (x - G\hat{j})^\top C^{-1} (x - G\hat{j}) + \lambda^2 \hat{j}^\top R^{-1} \hat{j} \Bigr\} \,

where the first term consists of the difference between the whitened measured
data (see :ref:`whitening_and_scaling`) and those predicted by the model while the
second term is a weighted-norm of the current estimate. It is seen that, with
increasing :math:`\lambda^2`, the source term receive more weight and larger
discrepancy between the measured and predicted data is tolerable.

.. _whitening_and_scaling:

Whitening and scaling
~~~~~~~~~~~~~~~~~~~~~

The MNE software employs data whitening so that a 'whitened' inverse operator
assumes the form

.. math::    \tilde{M} = M C^{^1/_2} = R \tilde{G}^\top (\tilde{G} R \tilde{G}^\top + \lambda^2 I)^{-1}\ ,
   :label: inv_m_tilde

where

.. math:: \tilde{G} = C^{-^1/_2}G
   :label: inv_g_tilde

is the spatially whitened gain matrix. We arrive at the whitened inverse
operator equation :eq:`inv_m_tilde` by making the substitution for
`G` from :eq:`inv_g_tilde` in :eq:`inv_m` as

.. math::

    \tilde{M} = M C^{^1/_2} &= R G^\top (G R G^\top + \lambda^2 C)^{-1} C^{^1/_2} \\
                             &= R \tilde{G}^\top C^{^1/_2} (C^{^1/_2} \tilde{G} R \tilde{G}^\top C^{^1/_2} + \lambda^2 C)^{-1} C^{^1/_2} \\
                             &= R \tilde{G}^\top C^{^1/_2} (C^{^1/_2} (\tilde{G} R \tilde{G}^\top + \lambda^2 I) C^{^1/_2})^{-1} C^{^1/_2} \\
                             &= R \tilde{G}^\top C^{^1/_2} C^{-^1/_2} (\tilde{G} R \tilde{G}^\top + \lambda^2 I)^{-1} C^{-^1/_2} C^{^1/_2} \\
                             &= R \tilde{G}^\top (\tilde{G} R \tilde{G}^\top + \lambda^2 I)^{-1}\ .

The expected current values are

.. math::
   :label: inv_j_hat_t

    \hat{j}(t) &= Mx(t) \\
               &= M C^{^1/_2} C^{-^1/_2} x(t) \\
               &= \tilde{M} \tilde{x}(t)

knowing :eq:`inv_m_tilde` and taking

.. math::
   :label: inv_tilde_x_t

    \tilde{x}(t) = C^{-^1/_2}x(t)

as the whitened measurement vector at time *t*. The spatial
whitening operator :math:`C^{-^1/_2}` is obtained with the help of the
eigenvalue decomposition
:math:`C = U_C \Lambda_C^2 U_C^\top` as :math:`C^{-^1/_2} = \Lambda_C^{-1} U_C^\top`.
In the MNE software the noise-covariance matrix is stored as the one applying
to raw data. To reflect the decrease of noise due to averaging, this matrix,
:math:`C_0`, is scaled by the number of averages, :math:`L`, *i.e.*, :math:`C =
C_0 / L`.

As shown above, regularization of the inverse solution is equivalent to a
change in the variance of the current amplitudes in the Bayesian *a priori*
distribution.

A convenient choice for the source-covariance matrix :math:`R` is such that
:math:`\text{trace}(\tilde{G} R \tilde{G}^\top) / \text{trace}(I) = 1`. With this
choice we can approximate :math:`\lambda^2 \sim 1/SNR`, where SNR is the
(power) signal-to-noise ratio of the whitened data.

.. note::
   The definition of the signal to noise-ratio/ :math:`\lambda^2` relationship
   given above works nicely for the whitened forward solution. In the
   un-whitened case scaling with the trace ratio :math:`\text{trace}(GRG^\top) /
   \text{trace}(C)` does not make sense, since the diagonal elements summed
   have, in general, different units of measure. For example, the MEG data are
   expressed in T or T/m whereas the unit of EEG is Volts.

See :ref:`tut_compute_covariance` for example of noise covariance computation
and whitening.

.. _cov_regularization_math:

Regularization of the noise-covariance matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The most straightforward approach to calculate the MNE is to employ the
expression of the original or whitened inverse operator directly. However, for
computational convenience we prefer to take another route, which employs the
singular-value decomposition (SVD) of the matrix

.. math::
   :label: inv_a

    A &= \tilde{G} R^{^1/_2} \\
      &= U \Lambda V^\top

where the superscript :math:`^1/_2` indicates a square root of :math:`R`. For a
diagonal matrix, one simply takes the square root of :math:`R` while in the
more general case one can use the Cholesky factorization :math:`R = R_C R_C^\top`
and thus :math:`R^{^1/_2} = R_C`.

Combining the SVD from :eq:`inv_a` with the inverse equation :eq:`inv_m` it is
easy to show that

.. math::
   :label: inv_m_tilde_svd

    \tilde{M} &= R \tilde{G}^\top (\tilde{G} R \tilde{G}^\top + \lambda^2 I)^{-1} \\
              &= R^{^1/_2} A^\top (A A^\top + \lambda^2 I)^{-1} \\
              &= R^{^1/_2} V \Lambda U^\top (U \Lambda V^\top V \Lambda U^\top + \lambda^2 I)^{-1} \\
              &= R^{^1/_2} V \Lambda U^\top (U (\Lambda^2 + \lambda^2 I) U^\top)^{-1} \\
              &= R^{^1/_2} V \Lambda U^\top U (\Lambda^2 + \lambda^2 I)^{-1} U^\top \\
              &= R^{^1/_2} V \Lambda (\Lambda^2 + \lambda^2 I)^{-1} U^\top \\
              &= R^{^1/_2} V \Gamma U^\top

where the elements of the diagonal matrix :math:`\Gamma` are simply

.. `reginv` in our code:

.. math::
   :label: inv_gamma_k

    \gamma_k = \frac{\lambda_k}{\lambda_k^2 + \lambda^2}\ .

From our expected current equation :eq:`inv_j_hat_t` and our whitened
measurement equation :eq:`inv_tilde_x_t`, if we take

.. math::
   :label: inv_w_t

    w(t) &= U^\top \tilde{x}(t) \\
         &= U^\top C^{-^1/_2} x(t)\ ,

we can see that the expression for the expected current is just

.. math::
   :label: inv_j_hat_t_svd

    \hat{j}(t) &= R^{^1/_2} V \Gamma w(t) \\
               &= \sum_k {\bar{v_k} \gamma_k w_k(t)}\ ,

where :math:`\bar{v_k} = R^{^1/_2} v_k`, with :math:`v_k` being the
:math:`k` th column of :math:`V`. It is thus seen that the current estimate is
a weighted sum of the "weighted" eigenleads :math:`v_k`.

It is easy to see that :math:`w(t) \propto \sqrt{L}`. To maintain the relation
:math:`(\tilde{G} R \tilde{G}^\top) / \text{trace}(I) = 1` when :math:`L` changes
we must have :math:`R \propto 1/L`. With this approach, :math:`\lambda_k` is
independent of  :math:`L` and, for fixed :math:`\lambda`, we see directly that
:math:`j(t)` is independent of :math:`L`.

The minimum-norm estimate is computed using this procedure in
:func:`mne.minimum_norm.make_inverse_operator`, and its usage is illustrated
in :ref:`CIHCFJEI`.


.. _noise_normalization:

Noise normalization
~~~~~~~~~~~~~~~~~~~

Noise normalization serves three purposes:

- It converts the expected current value into a dimensionless statistical test
  variable. Thus the resulting time and location dependent values are often
  referred to as dynamic statistical parameter maps (dSPM).

- It reduces the location bias of the estimates. In particular, the tendency of
  the MNE to prefer superficial currents is eliminated.

- The width of the point-spread function becomes less dependent on the source
  location on the cortical mantle. The point-spread is defined as the MNE
  resulting from the signals coming from a point current source (a current
  dipole) located at a certain point on the cortex.

In practice, noise normalization is implemented as a division by the square
root of the estimated variance of each voxel. In computing these noise
normalization factors, it's convenient to reuse our "weighted eigenleads"
definition from equation :eq:`inv_j_hat_t` in matrix form as

.. math::
   :label: inv_eigenleads_weighted

    \bar{V} = R^{^1/_2} V\ .

dSPM
----

Noise-normalized linear estimates introduced by Dale et al.
:footcite:`DaleEtAl1999` require division of the expected current amplitude by
its variance. In practice, this requires the computation of the diagonal
elements of the following matrix, using SVD equation :eq:`inv_m_tilde` and
:eq:`inv_eigenleads_weighted`:

.. math::

    M C M^\top &= M C^{^1/_2} C^{^1/_2} M^\top \\
            &= \tilde{M} \tilde{M}^\top \\
            &= R^{^1/_2} V \Gamma U^\top U \Gamma V^\top R^{^1/_2} \\
            &= \bar{V} \Gamma^2 \bar{V}^\top\ .

Because we only care about the diagonal entries here, we can find the
variances for each source as

.. math::

    \sigma_k^2 = \gamma_k^2

Under the conditions expressed at the end of :ref:`mne_solution`, it
follows that the *t*-statistic values associated with fixed-orientation
sources) are thus proportional to :math:`\sqrt{L}` while the *F*-statistic
employed with free-orientation sources is proportional to :math:`L`,
correspondingly.

.. note::
   The MNE software usually computes the *square roots* of the F-statistic to
   be displayed on the inflated cortical surfaces. These are also proportional
   to :math:`\sqrt{L}`.

sLORETA
-------
sLORETA :footcite:`Pascual-Marqui2002` estimates the current variances as the
diagonal entries of the
resolution matrix, which is the product of the inverse and forward operators.
In other words, the diagonal entries of (using :eq:`inv_m_tilde_svd`,
:eq:`inv_g_tilde`, and :eq:`inv_a`)

.. math::

    M G &= M C^{^1/_2} C^{-^1/_2} G \\
        &= \tilde{M} \tilde{G} \\
        &= R^{^1/_2} V \Gamma U^\top \tilde{G} R^{^1/_2} R^{-^1/_2} \\
        &= R^{^1/_2} V \Gamma U^\top U \Lambda V^\top R^{-^1/_2} \\
        &= R^{^1/_2} V \Gamma U^\top U \Lambda V^\top R^{^1/_2} R^{-1} \\
        &= \bar{V} \Gamma U^\top U \Lambda \bar{V}^\top R^{-1} \\
        &= \bar{V} \Gamma \Lambda \bar{V}^\top R^{-1}\ .

Because :math:`R` is diagonal and we only care about the diagonal entries,
we can find our variance estimates as

.. math::

    \sigma_k^2 &= \gamma_k \lambda_k R_{k,k}^{-1} \\
               &= \left(\frac{\lambda_k}{(\lambda_k^2 + \lambda^2)}\right) \left(\frac{\lambda_k}{1}\right) \left(\frac{1}{\lambda^2}\right) \\
               &= \frac{\lambda_k^2}{(\lambda_k^2 + \lambda^2) \lambda^2} \\
               &= \left(\frac{\lambda_k^2}{(\lambda_k^2 + \lambda^2)^2}\right) \left(\frac{\lambda^2 + \lambda_k^2}{\lambda^2}\right) \\
               &= \left(\frac{\lambda_k}{\lambda_k^2 + \lambda^2}\right)^2 \left(1 + \frac{\lambda_k^2}{\lambda^2}\right) \\
               &= \gamma_k^2 \left(1 + \frac{\lambda_k^2}{\lambda^2}\right)\ .

eLORETA
~~~~~~~
While dSPM and sLORETA solve for noise normalization weights
:math:`\sigma^2_k` that are applied to standard minimum-norm estimates
:math:`\hat{j}(t)`, eLORETA :footcite:`Pascual-Marqui2011` instead solves for
a source covariance
matrix :math:`R` that achieves zero localization bias. For fixed-orientation
solutions the resulting matrix :math:`R` will be a diagonal matrix, and for
free-orientation solutions it will be a block-diagonal matrix with
:math:`3 \times 3` blocks.

.. In https://royalsocietypublishing.org/doi/full/10.1098/rsta.2011.0081
.. eq. 2.10 (classical min norm), their values map onto our values as:
..
.. - α=λ²
.. - W=R⁻¹ (pos semidef weight matrix)
.. - K=G
.. - ϕ=x
.. - C=H
..

In :footcite:`Pascual-Marqui2011` eq. 2.13 states that the following system
of equations can be used to find the weights, :math:`\forall i \in {1, ..., P}`
(note that here we represent the equations from that paper using our notation):

.. math:: r_i = \left[ G_i^\top \left( GRG^\top + \lambda^2C \right)^{-1} G_i \right] ^{-^1/_2}

And an iterative algorithm can be used to find the values for the weights
:math:`r_i` that satisfy these equations as:

1. Initialize identity weights.
2. Compute :math:`N= \left( GRG^\top + \lambda^2C \right)^{-1}`.
3. Holding :math:`N` fixed, compute new weights :math:`r_i = \left[ G_i^\top N G_i \right]^{-^1/_2}`.
4. Using new weights, go to step (2) until convergence.

In particular, for step (2) we can use our substitution from :eq:`inv_g_tilde`
as:

.. math::

    N &= (G R G^\top + \lambda^2 C)^{-1} \\
      &= (C^{^1/_2} \tilde{G} R \tilde{G}^\top C^{^1/_2} + \lambda^2 C)^{-1} \\
      &= (C^{^1/_2} (\tilde{G} R \tilde{G}^\top + \lambda^2 I) C^{^1/_2})^{-1} \\
      &= C^{-^1/_2} (\tilde{G} R \tilde{G}^\top + \lambda^2 I)^{-1} C^{-^1/_2} \\
      &= C^{-^1/_2} (\tilde{G} R \tilde{G}^\top + \lambda^2 I)^{-1} C^{-^1/_2}\ .

Then defining :math:`\tilde{N}` as the whitened version of :math:`N`, i.e.,
the regularized pseudoinverse of :math:`\tilde{G}R\tilde{G}^\top`, we can
compute :math:`N` as:

.. math::

    N &= C^{-^1/_2} (U_{\tilde{G}R\tilde{G}^\top} \Lambda_{\tilde{G}R\tilde{G}^\top} V_{\tilde{G}R\tilde{G}^\top}^\top + \lambda^2 I)^{-1} C^{-^1/_2} \\
      &= C^{-^1/_2} (U_{\tilde{G}R\tilde{G}^\top} (\Lambda_{\tilde{G}R\tilde{G}^\top} + \lambda^2 I) V_{\tilde{G}R\tilde{G}^\top}^\top)^{-1} C^{-^1/_2} \\
      &= C^{-^1/_2} V_{\tilde{G}R\tilde{G}^\top} (\Lambda_{\tilde{G}R\tilde{G}^\top} + \lambda^2 I)^{-1} U_{\tilde{G}R\tilde{G}^\top}^\top C^{-^1/_2} \\
      &= C^{-^1/_2} \tilde{N} C^{-^1/_2}\ .

In step (3) we left and right multiply with subsets of :math:`G`, but making
the substitution :eq:`inv_g_tilde` we see that we equivalently compute:

.. math::

    r_i &= \left[ G_i^\top N G_i \right]^{-^1/_2} \\
        &= \left[ (C^{^1/_2} \tilde{G}_i)^\top N C^{^1/_2} \tilde{G}_i \right]^{-^1/_2} \\
        &= \left[ \tilde{G}_i^\top C^{^1/_2} N C^{^1/_2} \tilde{G}_i \right]^{-^1/_2} \\
        &= \left[ \tilde{G}_i^\top C^{^1/_2} C^{-^1/_2} \tilde{N} C^{-^1/_2} C^{^1/_2} \tilde{G}_i \right]^{-^1/_2} \\
        &= \left[ \tilde{G}_i^\top \tilde{N} \tilde{G}_i \right]^{-^1/_2}\ .

For convenience, we thus never need to compute :math:`N` itself but can instead
compute the whitened version :math:`\tilde{N}`.

Predicted data
~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~

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

``use_cps`` parameter in :func:`mne.convert_forward_solution`, and
:func:`mne.minimum_norm.make_inverse_operator` controls whether to use
cortical patch statistics (CPS) to define normal orientations or not (see
:ref:`CHDBBCEJ`).

.. _inverse_orientation_constrains:

Orientation constraints
~~~~~~~~~~~~~~~~~~~~~~~

The principal sources of MEG and EEG signals are generally believed to be
postsynaptic currents in the cortical pyramidal neurons. Since the net primary
current associated with these microscopic events is oriented normal to the
cortical mantle, it is reasonable to use the cortical normal orientation as a
constraint in source estimation. In addition to allowing completely free source
orientations, the MNE software implements three orientation constraints based
of the surface normal data:

- Source orientation can be rigidly fixed to the surface normal direction by
  specifying ``fixed=True`` in :func:`mne.minimum_norm.make_inverse_operator`.
  If cortical patch statistics are available the average
  normal over each patch, :math:`\bar{n_d}`, are used to define the source
  orientation. Otherwise, the vertex normal at the source space location is
  employed.

- A *location independent or fixed loose orientation constraint* (fLOC) can be
  employed by specifying ``fixed=False`` and ``loose=1.0`` when
  calling :func:`mne.minimum_norm.make_inverse_operator` (see
  :ref:`plot_dipole_orientations_fLOC_orientations`).
  In this approach, a source coordinate
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

- A *variable loose orientation constraint* (vLOC) can be employed by
  specifying ``fixed=False`` and ``loose`` parameters when calling
  :func:`mne.minimum_norm.make_inverse_operator` (see
  :ref:`plot_dipole_orientations_vLOC_orientations`). This
  is similar to *fLOC* except that the value given with the ``loose``
  parameter will be multiplied by :math:`\sigma_d`, defined above.

Depth weighting
~~~~~~~~~~~~~~~

The minimum-norm estimates have a bias towards superficial currents. This
tendency can be alleviated by adjusting the source covariance matrix :math:`R`
to favor deeper source locations. In the depth weighting scheme employed in MNE
analyze, the elements of :math:`R` corresponding to the :math:`p` th source
location are be scaled by a factor

.. math::    f_p = (g_{1p}^\top g_{1p} + g_{2p}^\top g_{2p} + g_{3p}^\top g_{3p})^{-\gamma}\ ,

where :math:`g_{1p}`, :math:`g_{2p}`, and :math:`g_{3p}` are the three columns
of :math:`G` corresponding to source location :math:`p` and :math:`\gamma` is
the order of the depth weighting, which is specified via the ``depth`` option
in :func:`mne.minimum_norm.make_inverse_operator`.

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
