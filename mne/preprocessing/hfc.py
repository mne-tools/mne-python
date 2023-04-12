# Authors: George O'Neill <g.o'neill@ucl.ac.uk>
#
# License: BSD-3-Clause

import numpy as np
from .maxwell import _prep_mf_coils, _sss_basis
from ..forward import _prep_meg_channels
from ..io.pick import pick_types
from ..io.proj import Projection
from ..utils import fill_doc, verbose


@verbose
def apply_hfc(
        raw, order=1, exclude_bads=True, accuracy='accurate', verbose=None):
    """Apply homgenous/harmonic field correction to magnetometer data.

    Remove evironmental fields from magentometer data by assuming it is
    explained as a homogeneous or harmonic field. Useful for arrays of OPMs.

    Parameters
    ----------
    raw : instance of Raw
        The data instance to process.
    order : int
        The order of the spherical harmonic basis set to use. Set to 1 to use
        only the homogeneous field component (default), 2 to add gradients, 3
        to add quadrature terms etc.
    exclude_bads : bool
        Do not include bad channels in the projection, or alter their data
        (default: True).
    accuracy : str
        Can be ``"point"``, ``"normal"`` or ``"accurate"`` (default), defines
        which level of coil definition accuracy is used to generate model.
    %(verbose)s

    Returns
    -------
    proc : Raw
        The processed data.
    model: FieldCorrector
        Field correction model used.
    """
    if exclude_bads is True:
        picks_idx = pick_types(raw.info, meg=True)
    else:
        picks_idx = pick_types(raw.info, meg=True, exclude=None)
    sens_idx = _filter_channels_with_positions(raw.info, picks_idx)
    model = FieldCorrector(picks=sens_idx, order=order,
                           accuracy=accuracy).fit(raw)
    proc = model.apply(raw)

    return proc, model


@fill_doc
class FieldCorrector():
    """Constructor for homogeneous/harmonic field correction.

    FieldCorrector contains assets required for offline subtraction of
    the modelled environmental interference, and forward model compensation.

    Parameters
    ----------
    picks : list
        List of sensor indices to build model with. Default of None will
        automatically select only good magnetometers with a known location and
        orientation.
    order : int
        The order of the spherical harmonic basis set to use. Set to 1 to use
        only the homogeneous field component (default), 2 to add gradients, 3
        to add quadrature terms etc.
    accuracy : str
        Can be ``"point"``, ``"normal"`` or ``"accurate"`` (default), defines
        which level of coil definition accuracy is used to generate model.

    Attributes
    ----------
    picks : list | array-like
        Channels to perform the correction on.
    order : int
        Selected model order
    basis : ndarray, shape (n, order ** 2 + 2 * order)
        The channels designated as containing the artifacts of interest.
    channels : list
        List of channels basis vectors were generated with, may be different
        to the initial picks (e.g. if picked sensor is classed as a refmag).
    labels: list
        Names of the basis vectors, based on their sphrical harmonic content.
    """

    def __init__(self, picks=None, order=1, accuracy='accurate'):
        self.picks = picks
        self.order = order
        self.accuracy = accuracy

    def fit(self, inst):
        """Fit HFC model to MEG sensors.

        Parameters
        ----------
        inst : Raw
            The data on which the HFC should be applied.

        Returns
        -------
        self : FieldCorrector
            FieldCorrector instance updated with
            fitted basis set and projector.
        """
        if self.picks is None:
            self.picks = _pick_sensors_auto(inst.info)
        # self.picks = _remove_missing_picks(inst.info, self.picks)
        self.basis, self.channels = _generate_basis_set(inst.info,
                                                        self.picks,
                                                        self.order,
                                                        accuracy=self.accuracy)
        self.labels = _label_basis(self.order)
        # self.proj = _generate_projector(self.basis)
        # self.rank = estimate_rank(self.proj, verbose=False)
        return self

    @fill_doc
    def apply(self, inst, copy=False, activate=True):
        """Apply basis set as series of projectors.

        Parameters
        ----------
        inst : Raw
            The data on which the HFC should be applied.
        %(copy_df)s
        activate : bool
            Turn on the correction now (default: True) or wait until later.

        Returns
        -------
        inst : raw
            Processed instance of data.

        Notes
        -----
        Only works after ``.fit()`` has been used.
        """
        if copy:
            inst = inst.copy()
        projs = []
        for ii in range(len(self.labels)):
            data = self.basis[:, ii]
            proj_data = dict(col_names=self.channels, row_names=None,
                             data=data[np.newaxis, :], ncol=len(self.channels),
                             nrow=1)
            proj = Projection(active=False, data=proj_data,
                              desc=self.labels[ii])
            projs.append(proj)
        inst.add_proj(projs, remove_existing=False)
        if activate:
            inst.apply_proj(verbose=False)
        return inst

    # @fill_doc
    # def apply(self, inst, copy=True, memsize=100):
    #     """Apply HFC to MEG data based on generated model.

    #     Parameters
    #     ----------
    #     inst : Raw
    #         The data on which the HFC should be applied.
    #     %(copy_df)s
    #     memsize : int
    #         Size in MB of block of data to be corrected in one go
    #         (default 100).

    #     Returns
    #     -------
    #     inst : raw
    #         Processed instance of data.

    #     Notes
    #     -----
    #     Only works after ``.fit()`` has been used.
    #     """
    #     if copy:
    #         inst = inst.copy()
    #     _check_preload(inst, 'field correction')
    #     tis, tfs = _get_chunks(inst, memsize)
    #     _, picks = _filter_list(self.channels, inst.ch_names)
    #     for start, stop in zip(tis, tfs):
    #         this_data = inst._data[picks, start:stop]
    #         inst._data[picks, start:stop] = self.proj @ this_data
    #     return inst

    # def custom(self, basis=None, proj=None):
    #     """Use a custom basis set or projector, useful for debugging.

    #     Parameters
    #     ----------
    #     basis : ndarray, shape (n, nvectors)
    #         The channels designated as containing the artifacts of interest.
    #     proj : ndarray, shape (n, n)
    #         Projection matrix, which data will be multiplied with to subtract
    #         environmental signal.

    #     Returns
    #     -------
    #     self : FieldCorrector
    #         FieldCorrector instance updated with basis set or projector.
    #     """
    #     self.order = None
    #     if basis is not None:
    #         self.basis = basis
    #         self.proj = _generate_projector(self.basis)
    #     elif proj is not None:
    #         self.proj = proj
    #     self.ranks = estimate_rank(self.proj, verbose=False)
    #     return self

    # @fill_doc
    # def save(self, fname, overwrite=False):
    #     """Save the regression model to an HDF5 file.

    #     Parameters
    #     ----------
    #     fname : path-like
    #         The file to write the regression weights to.
    #         Should end in ``.h5``.
    #     %(overwrite)s
    #     """
    #     _, write_hdf5 = _import_h5io_funcs()
    #     _validate_type(fname, 'path-like', 'fname')
    #     fname = _check_fname(fname, overwrite=overwrite, name='fname')
    #     write_hdf5(fname, self.__dict__, overwrite=overwrite)


# def read_field_corrector(fname):
#     """Read a field correction from an HDF5 file.

#     Parameters
#     ----------
#     fname : path-like
#         The file to read the regression model from. Should end in ``.h5``.

#     Returns
#     -------
#     model : FieldCorrector
#         The field correction model read from the file.
#     """
#     read_hdf5, _ = _import_h5io_funcs()
#     _validate_type(fname, 'path-like', 'fname')
#     fname = _check_fname(fname, overwrite='read', must_exist=True,
#                          name='fname')
#     model = FieldCorrector()
#     model.__dict__.update(read_hdf5(fname))
#     return model


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

# def _generate_projector(S):
#     """Generate a projection matrix used for HFC."""
#     if np.any(np.isnan(S)):
#         warn('\nFound NaNs in basis set, setting these to 0')
#         S = np.nan_to_num(S)
#     M = _make_projection_matrix(S)
#     return M


# def _make_projection_matrix(S):
#     """Turn basis set into a projector matrix"""
#     M = np.eye(len(S)) - S @ np.linalg.pinv(S)
#     return M


# def _get_chunks(raw, size=100):
#     """Get slices for blocks of data for a given amount of memory"""
#     chunk_samples = round(size * 1e6 / (8 * len(raw.info['chs'])))

#     start = np.arange(0, len(raw), chunk_samples)
#     stop = np.zeros(np.shape(start), dtype=int)
#     if len(start) == 1:
#         pass
#     else:
#         for ii in range(1, len(start)):
#             stop[ii - 1] = start[ii]
#     stop[-1] = len(raw)

#     return start, stop


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
