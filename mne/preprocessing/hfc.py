# Authors: George O'Neill <g.o'neill@ucl.ac.uk>
#
# License: BSD-3-Clause

import numpy as np
from .maxwell import _prep_mf_coils, _sss_basis
from ..forward import _prep_meg_channels
from ..io.pick import pick_types
from ..io.proj import Projection
from ..utils import verbose


@verbose
def compute_proj_hfc(info, order=1, picks=None, exclude_bads=True,
                     accuracy='accurate', verbose=None):
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
    picks : int | array-like
        List of channels indices to compute projeciton on. Default behaviour
        is to select meg (non-reference) with known positions.
    exclude_bads : bool
        Do not include bad channels in the projection, (default: True).
    accuracy : str
        Can be ``"point"``, ``"normal"`` or ``"accurate"`` (default), defines
        which level of coil definition accuracy is used to generate model.
    %(verbose)s

    Returns
    -------
    projs: list
        List of projection vectors.

    Notes
    -----
    To apply the projectors to a dataset, use
    ``[DATA].add_proj(projs).apply_proj()``.

    References
    ----------
    .. footbibliography::
    """
    if picks is None:
        if exclude_bads is True:
            picks = pick_types(info, meg=True)
        else:
            picks = pick_types(info, meg=True, exclude=None)
    picks = _filter_channels_with_positions(info, picks)
    basis, channels = _generate_basis_set(info, picks, order, accuracy)
    labels = _label_basis(order)
    projs = []
    for ii in range(len(labels)):
        data = basis[:, ii]
        proj_data = dict(col_names=channels, row_names=None,
                         data=data[np.newaxis, :], ncol=len(channels), nrow=1)
        proj = Projection(active=False, data=proj_data, desc=labels[ii])
        projs.append(proj)

    return projs


def _filter_channels_with_positions(info, indsin):
    """Keep indices of channels with position information."""
    ch_inds = list()
    for ii in indsin:
        if not (any(np.isnan(info['chs'][ii]['loc']))):
            ch_inds.append(ii)
    return ch_inds


def _generate_basis_set(info, picks=None, order=1,
                        accuracy='accurate', origin=(0, 0, 0)):
    """Generate the basis set used for HFC."""
    exp = dict(origin=(0, 0, 0), int_order=0, ext_order=order)
    coils = _prep_mf_coils(info, accuracy=accuracy)
    mf_names = _get_mf_names(info, accuracy=accuracy)
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
            labels.append("HFC: l=%d m=%d" % (L, m))
    return labels


def _pick_sensors_auto(info):
    """Pick of good magnetometors with known positions."""
    mags_idx = pick_types(info, meg=True)
    picks = _filter_channels_with_positions(info, mags_idx)
    return picks


def _get_mf_names(info, ignore_ref=True, accuracy='accurate', verbose=None):
    """Get names of coils used in MaxFilter basis set generation."""
    meg_sensors = _prep_meg_channels(
        info, head_frame=False, ignore_ref=ignore_ref, accuracy=accuracy,
        verbose=False)
    coils = meg_sensors['defs']
    names = [coil['chname'] for coil in coils]
    return names


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
