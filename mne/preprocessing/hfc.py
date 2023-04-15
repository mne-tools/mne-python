# Authors: George O'Neill <g.o'neill@ucl.ac.uk>
#
# License: BSD-3-Clause

import numpy as np
from .maxwell import _prep_mf_coils, _sss_basis
from ..io.pick import _picks_to_idx
from ..io.proj import Projection
from ..utils import verbose


@verbose
def compute_proj_hfc(info, order=1, picks='meg', exclude='bads',
                     ignore_ref=True, accuracy='accurate', verbose=None):
    """Generate projectors to perform homogeneous/harmonic correction to data.

    Remove evironmental fields from magentometer data by assuming it is
    explained as a homogeneous :footcite:`TierneyEtAl2021` or harmonic field
    :footcite:`TierneyEtAl2022`. Useful for arrays of OPMs.

    Parameters
    ----------
    info : instance of Info
        Info from an instance of Raw (preferable).
    order : int
        The order of the spherical harmonic basis set to use. Set to 1 to use
        only the homogeneous field component (default), 2 to add gradients, 3
        to add quadrature terms etc.
    picks : str | array_like | slice | None
        Channels to include. Default of ``'meg'`` will select all non-reference
        MEC channels.
    exclude : list | 'bads'
        List of channels to exclude from HFC, only used when picking
        based on types (e.g., exclude="bads" when picks="meg").
        Specify ``'bads'`` (the default) to exclude all channels marked as bad.
    ignore_ref : bool
        Specify whether reference MEG channels should be ignored when
        calculating the basis set (default: ``True``).
    accuracy : str
        Can be ``"point"``, ``"normal"`` or ``"accurate"`` (default), defines
        which level of coil definition accuracy is used to generate model.
    %(verbose)s

    Returns
    -------
    projs : list of Projection
        List of projection vectors.

    Notes
    -----
    To apply the projectors to a dataset, use ``inst.add_proj(projs).apply
    _proj()``.

    References
    ----------
    .. footbibliography::
    """
    idx = _picks_to_idx(info, picks, exclude=exclude)
    if not ignore_ref:
        idx = np.union1d(idx,
                         _picks_to_idx(info, picks='ref_meg', exclude=exclude))
    basis, channels = _generate_basis_set(info, idx, order,
                                          accuracy, ignore_ref)
    labels = _label_basis(order)
    _assert_isfinite(basis, channels)
    projs = []
    for ii, label in enumerate(labels):
        data = basis[:, ii]
        proj_data = dict(col_names=channels, row_names=None,
                         data=data[np.newaxis, :], ncol=len(channels), nrow=1)
        proj = Projection(active=False, data=proj_data, desc=label)
        projs.append(proj)

    return projs


def _generate_basis_set(info, picks=None, order=1, accuracy='accurate',
                        ignore_ref=True, origin=(0, 0, 0)):
    """Generate the basis set used for HFC."""
    exp = dict(origin=origin, int_order=0, ext_order=order)
    mf_coils = _prep_mf_coils(info, ignore_ref=ignore_ref,
                              accuracy=accuracy, return_names=True)
    coils = mf_coils[:-1]
    mf_names = mf_coils[-1]
    S = _sss_basis(exp, coils)
    if picks is not None:
        pick_names = [info['chs'][i]['ch_name'] for i in picks]
        basis_chans, basis_picks = _filter_list(pick_names, mf_names)
        S = S[basis_picks]
    return S, basis_chans


def _label_basis(order):
    """Give basis vectors names for Projection() class."""
    labels = list()
    for L in np.arange(1, order + 1):
        for m in np.arange(-1 * L, L + 1):
            labels.append(f"HFC: l={L} m={m}")
    return labels


def _filter_list(A_list, B_list):
    """Locate where one list matches another."""
    hit_inds = list()
    hit_list = list()
    for ii in A_list:
        hit = False
        if ii in B_list:
            hit_inds.append(B_list.index(ii))
            hit_list.append(ii)
            hit = True
        if not hit:
            string = '\t' + ii + ' not in basis set, ignoring.'
            print(string)
    return hit_list, hit_inds


def _assert_isfinite(basis, channels):
    """Check all basis values are finite, error and report which are not."""
    infs = np.isfinite(basis)
    bad_chans = np.where(~infs.any(axis=1))[0]
    if bad_chans.size > 0:
        string = "The following channels generate non-finite projectors"
        string += " for HFC:"
        for b in bad_chans:
            string += "\n\t" + channels[b]
        string += "\nPlease exclude from selection!"
        raise ValueError(string)
