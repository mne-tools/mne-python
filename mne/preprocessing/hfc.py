# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np

from .._fiff.pick import _picks_to_idx, pick_info
from .._fiff.proj import Projection
from ..utils import verbose
from .maxwell import _prep_mf_coils, _sss_basis


@verbose
def compute_proj_hfc(
    info, order=1, picks="meg", exclude="bads", *, accuracy="accurate", verbose=None
):
    """Generate projectors to perform homogeneous/harmonic correction to data.

    Remove environmental fields from magnetometer data by assuming it is
    explained as a homogeneous :footcite:`TierneyEtAl2021` or harmonic field
    :footcite:`TierneyEtAl2022`. Useful for arrays of OPMs.

    Parameters
    ----------
    %(info)s
    order : int
        The order of the spherical harmonic basis set to use. Set to 1 to use
        only the homogeneous field component (default), 2 to add gradients, 3
        to add quadrature terms, etc.
    picks : str | array_like | slice | None
        Channels to include. Default of ``'meg'`` (same as None) will select
        all non-reference MEG channels. Use ``('meg', 'ref_meg')`` to include
        reference sensors as well.
    exclude : list | 'bads'
        List of channels to exclude from HFC, only used when picking
        based on types (e.g., exclude="bads" when picks="meg").
        Specify ``'bads'`` (the default) to exclude all channels marked as bad.
    accuracy : str
        Can be ``"point"``, ``"normal"`` or ``"accurate"`` (default), defines
        which level of coil definition accuracy is used to generate model.
    %(verbose)s

    Returns
    -------
    %(projs)s

    See Also
    --------
    mne.io.Raw.add_proj
    mne.io.Raw.apply_proj

    Notes
    -----
    To apply the projectors to a dataset, use
    ``inst.add_proj(projs).apply_proj()``.

    .. versionadded:: 1.4

    References
    ----------
    .. footbibliography::
    """
    picks = _picks_to_idx(info, picks, none="meg", exclude=exclude, with_ref_meg=False)
    info = pick_info(info, picks)
    del picks
    exp = dict(origin=(0.0, 0.0, 0.0), int_order=0, ext_order=order)
    coils = _prep_mf_coils(info, ignore_ref=False, accuracy=accuracy)
    n_chs = len(coils[5])
    if n_chs != info["nchan"]:
        raise ValueError(
            f'Only {n_chs}/{info["nchan"]} picks could be interpreted '
            "as MEG channels."
        )
    S = _sss_basis(exp, coils)
    del coils
    bad_chans = [
        info["ch_names"][pick] for pick in np.where((~np.isfinite(S)).any(axis=1))[0]
    ]
    if bad_chans:
        raise ValueError(
            "The following channel(s) generate non-finite projectors:\n"
            f"    {bad_chans}\nPlease exclude from picks!"
        )
    S /= np.linalg.norm(S, axis=0)
    labels = _label_basis(order)
    assert len(labels) == S.shape[1]
    projs = []
    for label, vec in zip(labels, S.T):
        proj_data = dict(
            col_names=info["ch_names"],
            row_names=None,
            data=vec[np.newaxis, :],
            ncol=info["nchan"],
            nrow=1,
        )
        projs.append(Projection(active=False, data=proj_data, desc=label))
    return projs


def _label_basis(order):
    """Give basis vectors names for Projection() class."""
    return [
        f"HFC: l={L} m={m}"
        for L in np.arange(1, order + 1)
        for m in np.arange(-1 * L, L + 1)
    ]
