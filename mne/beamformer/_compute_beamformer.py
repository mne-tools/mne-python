"""Functions shared between different beamformer types."""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from copy import deepcopy

import numpy as np

from .._fiff.proj import Projection, make_projector
from ..cov import Covariance, make_ad_hoc_cov
from ..forward.forward import _restrict_forward_to_src_sel, is_fixed_orient
from ..minimum_norm.inverse import _get_vertno, _prepare_forward
from ..source_space._source_space import label_src_vertno_sel
from ..time_frequency.csd import CrossSpectralDensity
from ..utils import (
    _check_option,
    _check_src_normal,
    _import_h5io_funcs,
    _pl,
    _reg_pinv,
    _sym_mat_pow,
    check_fname,
    logger,
    verbose,
    warn,
)


def _check_proj_match(proj, filters):
    """Check whether SSP projections in data and spatial filter match."""
    proj_data, _, _ = make_projector(proj, filters["ch_names"])
    if not np.allclose(
        proj_data, filters["proj"], atol=np.finfo(float).eps, rtol=1e-13
    ):
        raise ValueError(
            "The SSP projections present in the data "
            "do not match the projections used when "
            "calculating the spatial filter."
        )


def _check_src_type(filters):
    """Check whether src_type is in filters and set custom warning."""
    if "src_type" not in filters:
        filters["src_type"] = None
    warn_text = (
        "The spatial filter does not contain src_type and a robust "
        "guess of src_type is not possible without src. Consider "
        "recomputing the filter."
    )
    return filters, warn_text


def _prepare_beamformer_input(
    info,
    forward,
    label=None,
    pick_ori=None,
    noise_cov=None,
    rank=None,
    pca=False,
    loose=None,
    combine_xyz="fro",
    exp=None,
    limit=None,
    allow_fixed_depth=True,
    limit_depth_chs=False,
):
    """Input preparation common for LCMV, DICS, and RAP-MUSIC."""
    _check_option("pick_ori", pick_ori, ("normal", "max-power", "vector", None))

    # Restrict forward solution to selected vertices
    if label is not None:
        _, src_sel = label_src_vertno_sel(label, forward["src"])
        forward = _restrict_forward_to_src_sel(forward, src_sel)

    if loose is None:
        loose = 0.0 if is_fixed_orient(forward) else 1.0
    # TODO: Deduplicate with _check_one_ch_type, should not be necessary
    # (DICS hits this code path, LCMV does not)
    if noise_cov is None:
        noise_cov = make_ad_hoc_cov(info, std=1.0)
    (
        forward,
        info_picked,
        gain,
        _,
        orient_prior,
        _,
        trace_GRGT,
        noise_cov,
        whitener,
    ) = _prepare_forward(
        forward,
        info,
        noise_cov,
        "auto",
        loose,
        rank=rank,
        pca=pca,
        use_cps=True,
        exp=exp,
        limit_depth_chs=limit_depth_chs,
        combine_xyz=combine_xyz,
        limit=limit,
        allow_fixed_depth=allow_fixed_depth,
    )
    is_free_ori = not is_fixed_orient(forward)  # could have been changed
    nn = forward["source_nn"]
    if is_free_ori:  # take Z coordinate
        nn = nn[2::3]
    nn = nn.copy()
    vertno = _get_vertno(forward["src"])
    if forward["surf_ori"]:
        nn[...] = [0, 0, 1]  # align to local +Z coordinate
    if pick_ori is not None and not is_free_ori:
        raise ValueError(
            f"Normal or max-power orientation (got {pick_ori!r}) can only be picked "
            "when a forward operator with free orientation is used."
        )
    if pick_ori == "normal" and not forward["surf_ori"]:
        raise ValueError(
            "Normal orientation can only be picked when a forward operator oriented in "
            "surface coordinates is used."
        )
    _check_src_normal(pick_ori, forward["src"])
    del forward, info

    # Undo the scaling that MNE prefers
    scale = np.sqrt((noise_cov["eig"] > 0).sum() / trace_GRGT)
    gain /= scale
    if orient_prior is not None:
        orient_std = np.sqrt(orient_prior)
    else:
        orient_std = np.ones(gain.shape[1])

    # Get the projector
    proj, _, _ = make_projector(info_picked["projs"], info_picked["ch_names"])
    return (is_free_ori, info_picked, proj, vertno, gain, whitener, nn, orient_std)


def _reduce_leadfield_rank(G):
    """Reduce the rank of the leadfield."""
    # decompose lead field
    u, s, v = np.linalg.svd(G, full_matrices=False)

    # backproject, omitting one direction (equivalent to setting the smallest
    # singular value to zero)
    G = np.matmul(u[:, :, :-1], s[:, :-1, np.newaxis] * v[:, :-1, :])

    return G


def _sym_inv_sm(x, reduce_rank, inversion, sk):
    """Symmetric inversion with single- or matrix-style inversion."""
    if x.shape[1:] == (1, 1):
        with np.errstate(divide="ignore", invalid="ignore"):
            x_inv = 1.0 / x
        x_inv[~np.isfinite(x_inv)] = 1.0
    else:
        assert x.shape[1:] == (3, 3)
        if inversion == "matrix":
            x_inv = _sym_mat_pow(x, -1, reduce_rank=reduce_rank)
            # Reapply source covariance after inversion
            x_inv *= sk[:, :, np.newaxis]
            x_inv *= sk[:, np.newaxis, :]
        else:
            # Invert for each dipole separately using plain division
            diags = np.diagonal(x, axis1=1, axis2=2)
            assert not reduce_rank  # guaranteed earlier
            with np.errstate(divide="ignore"):
                diags = 1.0 / diags
            # set the diagonal of each 3x3
            x_inv = np.zeros_like(x)
            for k in range(x.shape[0]):
                this = diags[k]
                # Reapply source covariance after inversion
                this *= sk[k] * sk[k]
                x_inv[k].flat[::4] = this
    return x_inv


def _compute_beamformer(
    G,
    Cm,
    reg,
    n_orient,
    weight_norm,
    pick_ori,
    reduce_rank,
    rank,
    inversion,
    nn,
    orient_std,
    whitener,
):
    """Compute a spatial beamformer filter (LCMV or DICS).

    For more detailed information on the parameters, see the docstrings of
    `make_lcmv` and `make_dics`.

    Parameters
    ----------
    G : ndarray, shape (n_dipoles, n_channels)
        The leadfield.
    Cm : ndarray, shape (n_channels, n_channels)
        The data covariance matrix.
    reg : float
        Regularization parameter.
    n_orient : int
        Number of dipole orientations defined at each source point
    weight_norm : None | 'unit-noise-gain' | 'nai'
        The weight normalization scheme to use.
    pick_ori : None | 'normal' | 'max-power'
        The source orientation to compute the beamformer in.
    reduce_rank : bool
        Whether to reduce the rank by one during computation of the filter.
    rank : dict | None | 'full' | 'info'
        See compute_rank.
    inversion : 'matrix' | 'single'
        The inversion scheme to compute the weights.
    nn : ndarray, shape (n_dipoles, 3)
        The source normals.
    orient_std : ndarray, shape (n_dipoles,)
        The std of the orientation prior used in weighting the lead fields.
    whitener : ndarray, shape (n_channels, n_channels)
        The whitener.

    Returns
    -------
    W : ndarray, shape (n_dipoles, n_channels)
        The beamformer filter weights.
    """
    _check_option(
        "weight_norm",
        weight_norm,
        ["unit-noise-gain-invariant", "unit-noise-gain", "nai", None],
    )

    # Whiten the data covariance
    Cm = whitener @ Cm @ whitener.T.conj()
    # Restore to properly Hermitian as large whitening coefs can have bad
    # rounding error
    Cm[:] = (Cm + Cm.T.conj()) / 2.0

    assert Cm.shape == (G.shape[0],) * 2
    s, _ = np.linalg.eigh(Cm)
    if not (s >= -s.max() * 1e-7).all():
        # This shouldn't ever happen, but just in case
        warn(
            "data covariance does not appear to be positive semidefinite, "
            "results will likely be incorrect"
        )
    # Tikhonov regularization using reg parameter to control for
    # trade-off between spatial resolution and noise sensitivity
    # eq. 25 in Gross and Ioannides, 1999 Phys. Med. Biol. 44 2081
    Cm_inv, loading_factor, rank = _reg_pinv(Cm, reg, rank)

    assert orient_std.shape == (G.shape[1],)
    n_sources = G.shape[1] // n_orient
    assert nn.shape == (n_sources, 3)

    logger.info(f"Computing beamformer filters for {n_sources} source{_pl(n_sources)}")
    n_channels = G.shape[0]
    assert n_orient in (3, 1)
    Gk = np.reshape(G.T, (n_sources, n_orient, n_channels)).transpose(0, 2, 1)
    assert Gk.shape == (n_sources, n_channels, n_orient)
    sk = np.reshape(orient_std, (n_sources, n_orient))
    del G, orient_std

    _check_option("reduce_rank", reduce_rank, (True, False))

    # inversion of the denominator
    _check_option("inversion", inversion, ("matrix", "single"))
    if (
        inversion == "single"
        and n_orient > 1
        and pick_ori == "vector"
        and weight_norm == "unit-noise-gain-invariant"
    ):
        raise ValueError(
            'Cannot use pick_ori="vector" with inversion="single" and '
            'weight_norm="unit-noise-gain-invariant"'
        )
    if reduce_rank and inversion == "single":
        raise ValueError(
            'reduce_rank cannot be used with inversion="single"; '
            'consider using inversion="matrix" if you have a '
            "rank-deficient forward model (i.e., from a sphere "
            "model with MEG channels), otherwise consider using "
            "reduce_rank=False"
        )
    if n_orient > 1:
        _, Gk_s, _ = np.linalg.svd(Gk, full_matrices=False)
        assert Gk_s.shape == (n_sources, n_orient)
        if not reduce_rank and (Gk_s[:, 0] > 1e6 * Gk_s[:, 2]).any():
            raise ValueError(
                "Singular matrix detected when estimating spatial filters. "
                "Consider reducing the rank of the forward operator by using "
                "reduce_rank=True."
            )
        del Gk_s

    #
    # 1. Reduce rank of the lead field
    #
    if reduce_rank:
        Gk = _reduce_leadfield_rank(Gk)

    def _compute_bf_terms(Gk, Cm_inv):
        bf_numer = np.matmul(Gk.swapaxes(-2, -1).conj(), Cm_inv)
        bf_denom = np.matmul(bf_numer, Gk)
        return bf_numer, bf_denom

    #
    # 2. Reorient lead field in direction of max power or normal
    #
    if pick_ori == "max-power":
        assert n_orient == 3
        _, bf_denom = _compute_bf_terms(Gk, Cm_inv)
        if weight_norm is None:
            ori_numer = np.eye(n_orient)[np.newaxis]
            ori_denom = bf_denom
        else:
            # compute power, cf Sekihara & Nagarajan 2008, eq. 4.47
            ori_numer = bf_denom
            # Cm_inv should be Hermitian so no need for .T.conj()
            ori_denom = np.matmul(
                np.matmul(Gk.swapaxes(-2, -1).conj(), Cm_inv @ Cm_inv), Gk
            )
        ori_denom_inv = _sym_inv_sm(ori_denom, reduce_rank, inversion, sk)
        ori_pick = np.matmul(ori_denom_inv, ori_numer)
        assert ori_pick.shape == (n_sources, n_orient, n_orient)

        # pick eigenvector that corresponds to maximum eigenvalue:
        eig_vals, eig_vecs = np.linalg.eig(ori_pick.real)  # not Hermitian!
        # sort eigenvectors by eigenvalues for picking:
        order = np.argsort(np.abs(eig_vals), axis=-1)
        # eig_vals = np.take_along_axis(eig_vals, order, axis=-1)
        max_power_ori = eig_vecs[np.arange(len(eig_vecs)), :, order[:, -1]]
        assert max_power_ori.shape == (n_sources, n_orient)

        # set the (otherwise arbitrary) sign to match the normal
        signs = np.sign(np.sum(max_power_ori * nn, axis=1, keepdims=True))
        signs[signs == 0] = 1.0
        max_power_ori *= signs

        # Compute the lead field for the optimal orientation,
        # and adjust numer/denom
        Gk = np.matmul(Gk, max_power_ori[..., np.newaxis])
        n_orient = 1
    else:
        max_power_ori = None
        if pick_ori == "normal":
            Gk = Gk[..., 2:3]
            n_orient = 1

    #
    # 3. Compute numerator and denominator of beamformer formula (unit-gain)
    #

    bf_numer, bf_denom = _compute_bf_terms(Gk, Cm_inv)
    assert bf_denom.shape == (n_sources,) + (n_orient,) * 2
    assert bf_numer.shape == (n_sources, n_orient, n_channels)
    del Gk  # lead field has been adjusted and should not be used anymore

    #
    # 4. Invert the denominator
    #

    # Here W is W_ug, i.e.:
    # G.T @ Cm_inv / (G.T @ Cm_inv @ G)
    bf_denom_inv = _sym_inv_sm(bf_denom, reduce_rank, inversion, sk)
    assert bf_denom_inv.shape == (n_sources, n_orient, n_orient)
    W = np.matmul(bf_denom_inv, bf_numer)
    assert W.shape == (n_sources, n_orient, n_channels)
    del bf_denom_inv, sk

    #
    # 5. Re-scale filter weights according to the selected weight_norm
    #

    # Weight normalization is done by computing, for each source::
    #
    #     W_ung = W_ug / sqrt(W_ug @ W_ug.T)
    #
    # with W_ung referring to the unit-noise-gain (weight normalized) filter
    # and W_ug referring to the above-calculated unit-gain filter stored in W.

    if weight_norm is not None:
        # Three different ways to calculate the normalization factors here.
        # Only matters when in vector mode, as otherwise n_orient == 1 and
        # they are all equivalent.
        #
        # In MNE < 0.21, we just used the Frobenius matrix norm:
        #
        #    noise_norm = np.linalg.norm(W, axis=(1, 2), keepdims=True)
        #    assert noise_norm.shape == (n_sources, 1, 1)
        #    W /= noise_norm
        #
        # Sekihara 2008 says to use sqrt(diag(W_ug @ W_ug.T)), which is not
        # rotation invariant:
        if weight_norm in ("unit-noise-gain", "nai"):
            noise_norm = np.matmul(W, W.swapaxes(-2, -1).conj()).real
            noise_norm = np.reshape(  # np.diag operation over last two axes
                noise_norm, (n_sources, -1, 1)
            )[:, :: n_orient + 1]
            np.sqrt(noise_norm, out=noise_norm)
            noise_norm[noise_norm == 0] = np.inf
            assert noise_norm.shape == (n_sources, n_orient, 1)
            W /= noise_norm
        else:
            assert weight_norm == "unit-noise-gain-invariant"
            # Here we use sqrtm. The shortcut:
            #
            #    use = W
            #
            # ... does not match the direct route (it is rotated!), so we'll
            # use the direct one to match FieldTrip:
            use = bf_numer
            inner = np.matmul(use, use.swapaxes(-2, -1).conj())
            W = np.matmul(_sym_mat_pow(inner, -0.5), use)
            noise_norm = 1.0

        if weight_norm == "nai":
            # Estimate noise level based on covariance matrix, taking the
            # first eigenvalue that falls outside the signal subspace or the
            # loading factor used during regularization, whichever is largest.
            if rank > len(Cm):
                # Covariance matrix is full rank, no noise subspace!
                # Use the loading factor as noise ceiling.
                if loading_factor == 0:
                    raise RuntimeError(
                        "Cannot compute noise subspace with a full-rank "
                        "covariance matrix and no regularization. Try "
                        "manually specifying the rank of the covariance "
                        "matrix or using regularization."
                    )
                noise = loading_factor
            else:
                noise, _ = np.linalg.eigh(Cm)
                noise = noise[-rank]
                noise = max(noise, loading_factor)
            W /= np.sqrt(noise)

    W = W.reshape(n_sources * n_orient, n_channels)
    logger.info("Filter computation complete")
    return W, max_power_ori


def _compute_power(Cm, W, n_orient):
    """Use beamformer filters to compute source power.

    Parameters
    ----------
    Cm : ndarray, shape (n_channels, n_channels)
        Data covariance matrix or CSD matrix.
    W : ndarray, shape (nvertices*norient, nchannels)
        Beamformer weights.

    Returns
    -------
    power : ndarray, shape (nvertices,)
        Source power.
    """
    n_sources = W.shape[0] // n_orient

    Wk = W.reshape(n_sources, n_orient, W.shape[1])
    source_power = np.trace(
        (Wk @ Cm @ Wk.conj().transpose(0, 2, 1)).real, axis1=1, axis2=2
    )

    return source_power


class Beamformer(dict):
    """A computed beamformer.

    Notes
    -----
    .. versionadded:: 0.17
    """

    def copy(self):
        """Copy the beamformer.

        Returns
        -------
        beamformer : instance of Beamformer
            A deep copy of the beamformer.
        """
        return deepcopy(self)

    def __repr__(self):  # noqa: D105
        n_verts = sum(len(v) for v in self["vertices"])
        n_channels = len(self["ch_names"])
        if self["subject"] is None:
            subject = "unknown"
        else:
            subject = f'"{self["subject"]}"'
        out = "<Beamformer | {}, subject {}, {} vert, {} ch".format(
            self["kind"],
            subject,
            n_verts,
            n_channels,
        )
        if self["pick_ori"] is not None:
            out += f', {self["pick_ori"]} ori'
        if self["weight_norm"] is not None:
            out += f', {self["weight_norm"]} norm'
        if self.get("inversion") is not None:
            out += f', {self["inversion"]} inversion'
        if "rank" in self:
            out += f', rank {self["rank"]}'
        out += ">"
        return out

    @verbose
    def save(self, fname, overwrite=False, verbose=None):
        """Save the beamformer filter.

        Parameters
        ----------
        fname : path-like
            The filename to use to write the HDF5 data.
            Should end in ``'-lcmv.h5'`` or ``'-dics.h5'``.
        %(overwrite)s
        %(verbose)s
        """
        _, write_hdf5 = _import_h5io_funcs()

        ending = f'-{self["kind"].lower()}.h5'
        check_fname(fname, self["kind"], (ending,))
        csd_orig = None
        try:
            if "csd" in self:
                csd_orig = self["csd"]
                self["csd"] = self["csd"].__getstate__()
            write_hdf5(fname, self, overwrite=overwrite, title="mnepython")
        finally:
            if csd_orig is not None:
                self["csd"] = csd_orig


def read_beamformer(fname):
    """Read a beamformer filter.

    Parameters
    ----------
    fname : path-like
        The filename of the HDF5 file.

    Returns
    -------
    filter : instance of Beamformer
        The beamformer filter.
    """
    read_hdf5, _ = _import_h5io_funcs()
    beamformer = read_hdf5(fname, title="mnepython")
    if "csd" in beamformer:
        beamformer["csd"] = CrossSpectralDensity(**beamformer["csd"])
    # h5io seems to cast `bool` to `int` on round-trip, probably a bug
    # we should fix at some point (if possible -- could be HDF5 limitation)
    for key in ("normalize_fwd", "is_free_ori", "is_ssp"):
        if key in beamformer:
            beamformer[key] = bool(beamformer[key])
    for key in ("data_cov", "noise_cov"):
        if beamformer.get(key) is not None:
            for pi, p in enumerate(beamformer[key]["projs"]):
                p = Projection(**p)
                p["active"] = bool(p["active"])
                beamformer[key]["projs"][pi] = p
            beamformer[key] = Covariance(
                *[
                    beamformer[key].get(arg)
                    for arg in (
                        "data",
                        "names",
                        "bads",
                        "projs",
                        "nfree",
                        "eig",
                        "eigvec",
                        "method",
                        "loglik",
                    )
                ]
            )
    return Beamformer(beamformer)


def _proj_whiten_data(M, proj, filters):
    if filters.get("is_ssp", True):
        # check whether data and filter projs match
        _check_proj_match(proj, filters)
        if filters["whitener"] is None:
            M = np.dot(filters["proj"], M)

    if filters["whitener"] is not None:
        M = np.dot(filters["whitener"], M)
    return M
