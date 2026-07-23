:orphan:

Generalized eigendecomposition in decoding
==========================================

.. NOTE: part of this file is included in doc/overview/implementation.rst.
   Changes here are reflected there. If you want to link to this content, link
   to :ref:`ged` to link to that section of the implementation.rst page.
   The next line is a target for :start-after: so we can omit the title from
   the include:
   ged-begin-content

This section describes the mathematical formulation and application of
Generalized Eigendecomposition (GED), often used in spatial filtering
and source separation algorithms, such as :class:`mne.decoding.CSP`, 
:class:`mne.decoding.SPoC`, :class:`mne.decoding.SSD` and 
:class:`mne.decoding.XdawnTransformer`.

The core principle of GED is to find a set of channel weights (spatial filter) 
that maximizes the ratio of signal power between two data features. 
These features are defined by the researcher and are represented by two covariance matrices: 
a "signal" matrix :math:`S` and a "reference" matrix :math:`R`. 
For example, :math:`S` could be the covariance of data from a task time interval, 
and :math:`S` could be the covariance from a baseline time interval. For more details see :footcite:`Cohen2022`.

Algebraic formulation of GED
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A few definitions first: 
Let :math:`n \in \mathbb{N}^+` be a number of channels.
Let :math:`\text{Symm}_n(\mathbb{R}) \subset M_n(\mathbb{R})` be a vector space of real symmetric matrices.
Let :math:`S^n_+, S^n_{++} \subset \text{Symm}_n(\mathbb{R})` be sets of real positive semidefinite and positive definite matrices, respectively.
Let :math:`S, R \in S^n_+` be covariance matrices estimated from electrophysiological data :math:`X_S \in M_{n \times t_S}(\mathbb{R})` and :math:`X_R \in M_{n \times t_R}(\mathbb{R})`.

GED (or simultaneous diagonalization by congruence) of :math:`S` and :math:`R` 
is possible when :math:`R` is full rank (and thus :math:`R \in S^n_{++}`):

.. math::

   SW = RWD,

where :math:`W \in M_n(\mathbb{R})` is an invertible matrix of eigenvectors 
of :math:`(S, R)` and :math:`D` is a diagonal matrix of eigenvalues :math:`\lambda_i`.

Each eigenvector :math:`\mathbf{w} \in W` is a spatial filter that solves 
an optimization problem of the form:

.. math::

   \operatorname{argmax}_{\mathbf{w}} \frac{\mathbf{w}^t S \mathbf{w}}{\mathbf{w}^t R \mathbf{w}}

That is, using spatial filters :math:`W` on time-series :math:`X \in M_{n \times t}(\mathbb{R})`:

.. math::

   \mathbf{A} = W^t X,

results in "activation" time-series :math:`A` of the estimated "sources", 
such that the ratio of their variances, 
:math:`\frac{\text{Var}(\mathbf{w}^T X_S)}{\text{Var}(\mathbf{w}^T X_R)} = \frac{\mathbf{w}^T S \mathbf{w}}{\mathbf{w}^T R \mathbf{w}}`, 
is sequentially maximized spatial filters :math:`\mathbf{w}_i`, sorted according to :math:`\lambda_i`.

GED in the principal subspace
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unfortunately, :math:`R` might not be full rank depending on the data :math:`X_R` (for example due to average reference, removed PCA/ICA components, etc.). 
In such cases, GED can be performed on :math:`S` and :math:`R` in the principal subspace :math:`Q = \operatorname{Im}(C_{ref}) \subset \mathbb{R}^n` of some reference 
covariance :math:`C_{ref}` (in Common Spatial Pattern (CSP) algorithm, for example, :math:`C_{ref}=\frac{1}{2}(S+R)` and GED is performed on S and R'=S+R). 

More formally: 
Let :math:`r \leq n` be a rank of :math:`C \in S^n_+`. 
Let :math:`Q=\operatorname{Im}(C_{ref})` be a principal subspace of :math:`C_{ref}`. 
Let :math:`P \in M_{n \times r}(\mathbb{R})` be an isometry formed by orthonormal basis of :math:`Q`.
Let :math:`f:S^n_+ \to S^r_+`, :math:`A \mapsto P^t A P` be a "restricting" map, that restricts quadratic form 
:math:`q_A:\mathbb{R}^n \to \mathbb{R}` to :math:`q_{A|_Q}:\mathbb{R}^n \to \mathbb{R}` (in practical terms, :math:`q_A` maps 
spatial filters to variance of the spatially filtered data :math:`X_A`).

Then, the GED of :math:`S` and :math:`R` in the principal subspace :math:`Q` of :math:`C_{ref}` is performed as follows:

1. :math:`S` and :math:`R` are transformed to :math:`S_Q = f(S) = P^t S P` and :math:`R_Q = f(R) = P^t R P`, 
   such that :math:`S_Q` and :math:`R_Q` are matrix representations of restricted :math:`q_{S|_Q}` and :math:`q_{R|_Q}`.
2. GED is performed on :math:`S_Q` and :math:`R_Q`: :math:`S_Q W_Q = R_Q W_Q D`.
3. Eigenvectors :math:`W_Q` of :math:`(S_Q, R_Q)` are transformed back to :math:`\mathbb{R}^n` 
   by :math:`W = P W_Q \in \mathbb{R}^{n \times r}` to obtain :math:`r` spatial filters.

Note that the solution to the original optimization problem is preserved:

.. math::

   \frac{\mathbf{w_Q}^t S_Q \mathbf{w_Q}}{\mathbf{w_Q}^t R_Q \mathbf{w_Q}}= \frac{\mathbf{w_Q}^t (P^t S P) \mathbf{w_Q}}{\mathbf{w_Q}^t (P^t R P) 
   \mathbf{w_Q}} = \frac{\mathbf{w}^t S \mathbf{w}}{\mathbf{w}^t R \mathbf{w}} = \lambda


In addition to restriction, :math:`q_S` and :math:`q_R` can be rescaled based on the whitened :math:`C_{ref}`. 
In this case the whitening map :math:`f_{wh}:S^n_+ \to S^r_+`, 
:math:`A \mapsto P_{wh}^t A P_{wh}` transforms :math:`A` into matrix representation of :math:`q_{A|Q}` rescaled according to :math:`\Lambda^{-1/2}`, 
where :math:`\Lambda` is a diagonal matrix of eigenvalues of :math:`C_{ref}` and so :math:`P_{wh} = P \Lambda^{-1/2}`.

In MNE-Python, the matrix :math:`P` of the restricting map can be obtained using
::

    _, ref_evecs, mask = mne.cov._smart_eigh(C_ref, ..., proj_subspace=True, ...)
    restr_mat = ref_evecs[mask]

while :math:`P_{wh}` using:
::

    restr_mat = compute_whitener(C_ref, ..., pca=True, ...)