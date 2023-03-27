# Authors: George O'Neill <g.o'neill@ucl.ac.uk>
#
# License: BSD-3-Clause

import numpy as np
from .maxwell import _prep_mf_coils, _sss_basis
from ..forward import _read_coil_defs
from ..io.constants import FWD
from ..io.pick import pick_types, pick_info
from ..rank import estimate_rank
from ..utils import (_check_preload, warn, _import_h5io_funcs, _check_fname,
                     _validate_type, fill_doc, verbose)


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
    pm : bool
        Assume the sensors are point magnetometers, False (default) will use
        accurate sensor definitions when generating model, whilst True will use
        the simpler point-sensor definition.

    Attributes
    ----------
    picks : list | array-like
        Channels to perform the correction on.
    order : int
        Selected model order
    basis : ndarray, shape (n, order ** 2 + 2 * order)
        The channels designated as containing the artifacts of interest.
    proj : ndarray, shape (n, n)
        Projection matrix, which data will be multiplied with to subtract
        environmental signal.
    rank: int
        Rank of the projection matrix
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
        self.picks = _remove_missing_picks(inst.info, self.picks)
        self.basis = _generate_basis_set(inst.info, self.picks,
                                         self.order, accuracy=self.accuracy)
        self.proj = _generate_projector(self.basis)
        self.rank = estimate_rank(self.proj, verbose=False)
        return self

    @fill_doc
    def apply(self, inst, copy=True, memsize=100):
        """Apply HFC to MEG data based on generated model.

        Parameters
        ----------
        inst : Raw
            The data on which the HFC should be applied.
        %(copy_df)s
        memsize : int
            Size in MB of block of data to be corrected in one go
            (default 100).

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
        _check_preload(inst, 'field correction')
        tis, tfs = _get_chunks(inst, memsize)
        for start, stop in zip(tis, tfs):
            this_data = inst._data[self.picks, start:stop]
            inst._data[self.picks, start:stop] = self.proj @ this_data
        return inst

    def custom(self, basis=None, proj=None):
        """Use a custom basis set or projector, useful for debugging.

        Parameters
        ----------
        basis : ndarray, shape (n, nvectors)
            The channels designated as containing the artifacts of interest.
        proj : ndarray, shape (n, n)
            Projection matrix, which data will be multiplied with to subtract
            environmental signal.

        Returns
        -------
        self : FieldCorrector
            FieldCorrector instance updated with basis set or projector.
        """
        self.order = None
        if basis is not None:
            self.basis = basis
            self.proj = _generate_projector(self.basis)
        elif proj is not None:
            self.proj = proj
        self.ranks = estimate_rank(self.proj, verbose=False)
        return self

    @fill_doc
    def save(self, fname, overwrite=False):
        """Save the regression model to an HDF5 file.

        Parameters
        ----------
        fname : path-like
            The file to write the regression weights to. Should end in ``.h5``.
        %(overwrite)s
        """
        _, write_hdf5 = _import_h5io_funcs()
        _validate_type(fname, 'path-like', 'fname')
        fname = _check_fname(fname, overwrite=overwrite, name='fname')
        write_hdf5(fname, self.__dict__, overwrite=overwrite)


def read_field_corrector(fname):
    """Read a field correction from an HDF5 file.

    Parameters
    ----------
    fname : path-like
        The file to read the regression model from. Should end in ``.h5``.

    Returns
    -------
    model : FieldCorrector
        The field correction model read from the file.
    """
    read_hdf5, _ = _import_h5io_funcs()
    _validate_type(fname, 'path-like', 'fname')
    fname = _check_fname(fname, overwrite='read', must_exist=True,
                         name='fname')
    model = FieldCorrector()
    model.__dict__.update(read_hdf5(fname))
    return model


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
    S = _sss_basis(exp, coils)
    if picks is not None:
        picks2, _ = _reduce_pick_list(info, picks)
        S = S[picks2]
    return S


def _generate_projector(S):
    """Generate the projection matrix used for HFC."""
    if np.any(np.isnan(S)):
        warn('\nFound NaNs in basis set, setting these to 0')
        S = np.nan_to_num(S)
    M = _make_projector(S)
    return M


def _make_projector(S):
    """Turn basis set into a projector with magic, or linear algebra"""
    M = np.eye(len(S)) - S @ np.linalg.pinv(S)
    return M


def _get_chunks(raw, size=100):
    """Get slices for blocks of data for a given amount of memory"""
    chunk_samples = round(size * 1e6 / (8 * len(raw.info['chs'])))

    start = np.arange(0, len(raw), chunk_samples)
    stop = np.zeros(np.shape(start), dtype=int)
    if len(start) == 1:
        pass
    else:
        for ii in range(1, len(start)):
            stop[ii - 1] = start[ii]
    stop[-1] = len(raw)

    return start, stop


def _pick_sensors_auto(info):
    """Default pick of good magnetometeors with known positions"""
    mags_idx = pick_types(info, meg=True)
    picks = _filter_channels_with_positions(info, mags_idx)
    return picks


def _prep_hfc_pm(info):
    """Get all point magnetometer information loaded and sorted."""
    picks = pick_types(info, meg=True, ref_meg=False)
    chs = pick_info(info, sel=picks)['chs']

    # Now coils is a sorted list of coils. Time to do some vectorization.
    n_chs = len(chs)
    mag_mask = np.array([_mag_mask(ch) for ch in chs])
    rmags = np.concatenate([ch['loc'][:3] for ch in chs])
    rmags = np.reshape(rmags, (n_chs, 3))
    cosmags = np.concatenate([ch['loc'][-3:] for ch in chs])
    cosmags = np.reshape(cosmags, (n_chs, 3))
    n_int = np.array([1 for ch in chs])
    bins = np.repeat(np.arange(len(n_int)), n_int)
    bd = np.concatenate(([0], np.cumsum(n_int)))
    slice_map = {ii: slice(start, stop)
                 for ii, (start, stop) in enumerate(zip(bd[:-1], bd[1:]))}
    return rmags, cosmags, bins, n_chs, mag_mask, slice_map


def _mag_mask(ch):
    defs = _read_coil_defs(verbose=False)
    coil_id = list([d['coil_type'] for d in defs])
    coil_class = list([d['coil_class'] for d in defs])
    # find the first index where coil def matches
    idx = [i for i in range(len(coil_id)) if coil_id[i] == ch['coil_type']][0]
    mag = coil_class[idx] == FWD.COILC_MAG
    return mag


def _match_str(A_list, B_list):
    """Locate where in a list matches another."""
    B_inds = list()
    for ii in A_list:
        hit = False
        if ii in B_list:
            B_inds.append(B_list.index(ii))
            hit = True
        if not hit:
            B_inds.append(None)
    return B_inds


def _reduce_pick_list(info, picks):
    """Update pick list to be compatible with _prep_mf_coils"""
    picks_reduced = pick_types(info, meg=True, ref_meg=False)
    ch_list_full = [info['chs'][i]['ch_name'] for i in picks]
    ch_list_reduced = [info['chs'][i]['ch_name'] for i in picks_reduced]
    picks2 = _match_str(ch_list_full, ch_list_reduced)
    if None in picks2:
        missing = [i for i, p in enumerate(picks2) if p is None]
        missing_chans = [ch_list_full[m] for m in missing]
        string = '\nThe following channels are missing from the basis set:\n'
        for m in missing_chans:
            string = string + m + '\n'
        string = string + 'Will ignore!'
        warn(string)
        picks2 = [p for p in picks2 if p is not None]
        picks = [value for i, value in enumerate(picks) if i not in missing]
        
    return picks2, picks

def _remove_missing_picks(info, picks):
    _, picks_filtered = _reduce_pick_list(info, picks)
    return picks_filtered