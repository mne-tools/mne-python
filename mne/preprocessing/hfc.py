# Authors: George O'Neill <g.o'neill@ucl.ac.uk>
#
# License: BSD-3-Clause

from numpy import linalg, arange, isnan, zeros, eye, shape, nan_to_num
from numpy import any as npany
from mne.preprocessing.maxwell import _prep_mf_coils, _sss_basis
from mne.io.pick import pick_types
from ..utils import _check_preload, warn
from ..rank import estimate_rank


# @verbose
def apply_hfc(raw, order=1, exclude_bads=True, verbose=None):
    """Remove evironmental fields from magentometer data by assuming it is
    explained as a homogeneous or harmonic field. Useful for arrays of OPMs.

    Parameters
    ----------
    raw : instance of Raw
        The data instance to process.
    order: int
        The order of the spherical harmonic basis set to use. Set to 1 to use
        only the homogeneous field component (default), 2 to add gradients, 3
        to add quadrature terms etc.
    exclude_bads: bool
        Do not include bad channels in the projection, or alter their data
        (default: True).
    %(verbose)s

    Returns
    -------
    proc : Raw
        The processed data.
    """

    if exclude_bads is True:
        picks_idx = pick_types(raw.info, meg=True)
    else:
        picks_idx = pick_types(raw.info, meg=True, exclude=None)
    sens_idx = _filter_channels_with_positions(raw.info, picks_idx)
    model = FieldCorrector(raw, picks=sens_idx, order=order)
    proc = model.apply(raw)

    return proc


# @fill_doc
class FieldCorrector():
    """Constructor for homogeneous/harmonic field correction, contains all
    assests required for offline subtraction of the modelled environmental
    interference, and forward model compensation.

    Parameters
    ----------
    inst : instance of Raw
        The instance to generate model from.
    picks: list of sensor indices to build model with. Default of None will
        automatically select only good magnetometers with a known location
    order: int
        The order of the spherical harmonic basis set to use. Set to 1 to use
        only the homogeneous field component (default), 2 to add gradients, 3
        to add quadrature terms etc.

    Attributes
    ----------
    picks : list | array-like
        Channels to perform the regression on.
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
    def __init__(self, inst, picks=None, order=1):
        if picks is None:
            self.picks = _pick_sensors_auto(inst.info)
        else:
            self.picks = picks
        self.order = order
        self.basis = _generate_basis_set(inst.info, self.picks, order)
        self.proj = _generate_projector(inst.info, self.picks, order)
        self.ranks = estimate_rank(self.proj, verbose=False)

    def apply(self, inst, copy=True, memsize=100):
        """Apply HFC to MEG data based on generated model.

        Parameters
        ----------
        inst : Raw
            The data on which the HFC should be applied.
        memsize: int
            Size in MB of block of data to be corrected in one go (default 100)

        Returns
        -------
        inst : EOGRegression
            Processed instance of data
        """
        if copy:
            inst = inst.copy()
        _check_preload(inst, 'field correction')
        tis, tfs = _get_chunks(inst, memsize)
        for start, stop in zip(tis, tfs):
            this_data = inst._data[self.picks, start:stop]
            inst._data[self.picks, start:stop] = self.proj @ this_data
        return inst


def _filter_channels_with_positions(info, indsin):
    """Keep indices of channels with position information."""
    ch_inds = list()
    for ii in indsin:
        if not (any(isnan(info['chs'][ii]['loc']))):
            ch_inds.append(ii)
    return ch_inds


def _generate_basis_set(info, picks=None, order=1, origin=(0, 0, 0)):
    """Generate the basis set used for HFC."""
    exp = dict(origin=(0, 0, 0), int_order=0, ext_order=order)
    coils = _prep_mf_coils(info)
    S = _sss_basis(exp, coils)
    if picks is not None:
        S = S[picks]
    return S


def _generate_projector(info, picks=None, order=1):
    """Generate the projection matrix used for HFC."""
    S = _generate_basis_set(info, picks, order)
    if npany(isnan(S)):
        warn('\nFound NaNs in basis set, setting these to 0')
        S = nan_to_num(S)
    M = eye(len(S)) - S @ linalg.pinv(S)
    return M


def _get_chunks(raw, size=100):
    """Get slices for blocks of data for a given amount of memory"""
    chunk_samples = round(size * 1e6 / (8 * len(raw.info['chs'])))

    start = arange(0, len(raw), chunk_samples)
    stop = zeros(shape(start), dtype=int)
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
