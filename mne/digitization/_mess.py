"""Do not look at this !! or blame Teon !!!."""
from ..utils import warn
import numpy as np


##########################################
# Things that should be common to every reader

# from ..io.meas_info import _empty_info
from ._utils import _read_dig_points
from ._utils import _make_dig_points
from ..transforms import apply_trans
from ..transforms import als_ras_trans
from ..transforms import get_ras_to_neuromag_trans
from ..transforms import Transform
from ..transforms import combine_transforms
from ..transforms import invert_transform
from ..coreg import fit_matched_points, _decimate_points

##########################################
# Things that might be specific

# From mne.io.kit
from ..io.kit.constants import KIT
from ..io.kit.coreg import read_mrk


def _bar(nasion, lpa, rpa, hpi, extra, convert=True):

    if convert:
        neuromag_trans = get_ras_to_neuromag_trans(nasion, lpa, rpa)
        nasion = apply_trans(neuromag_trans, nasion)
        lpa = apply_trans(neuromag_trans, lpa)
        rpa = apply_trans(neuromag_trans, rpa)

        if hpi is not None:
            hpi = apply_trans(neuromag_trans, hpi)

        extra = apply_trans(neuromag_trans, extra).astype(np.float32)
    else:
        neuromag_trans = None

    ctf_head_t = Transform(fro='ctf_head', to='head', trans=neuromag_trans)

    info_dig = _make_dig_points(nasion=nasion,
                                lpa=lpa,
                                rpa=rpa,
                                hpi=hpi,
                                extra_points=extra)

    return info_dig, ctf_head_t


##############################################################################
# From mne.io.kit
def _set_dig_kit(mrk, elp, hsp):
    """Add landmark points and head shape data to the KIT instance.

    Digitizer data (elp and hsp) are represented in [mm] in the Polhemus
    ALS coordinate system. This is converted to [m].

    Parameters
    ----------
    mrk : None | str | array_like, shape = (5, 3)
        Marker points representing the location of the marker coils with
        respect to the MEG Sensors, or path to a marker file.
    elp : None | str | array_like, shape = (8, 3)
        Digitizer points representing the location of the fiducials and the
        marker coils with respect to the digitized head shape, or path to a
        file containing these points.
    hsp : None | str | array, shape = (n_points, 3)
        Digitizer head shape points, or path to head shape file. If more
        than 10`000 points are in the head shape, they are automatically
        decimated.

    Returns
    -------
    dig_points : list
        List of digitizer points for info['dig'].
    dev_head_t : dict
        A dictionary describe the device-head transformation.
    """
    if isinstance(hsp, str):
        hsp = _read_dig_points(hsp)
    n_pts = len(hsp)
    if n_pts > KIT.DIG_POINTS:
        hsp = _decimate_points(hsp, res=0.005)
        n_new = len(hsp)
        warn("The selected head shape contained {n_in} points, which is "
             "more than recommended ({n_rec}), and was automatically "
             "downsampled to {n_new} points. The preferred way to "
             "downsample is using FastScan.".format(
                 n_in=n_pts, n_rec=KIT.DIG_POINTS, n_new=n_new))

    if isinstance(elp, str):
        elp_points = _read_dig_points(elp)
        if len(elp_points) != 8:
            raise ValueError("File %r should contain 8 points; got shape "
                             "%s." % (elp, elp_points.shape))
        elp = elp_points
    elif len(elp) != 8:
        raise ValueError("ELP should contain 8 points; got shape "
                         "%s." % (elp.shape,))
    if isinstance(mrk, str):
        mrk = read_mrk(mrk)

    # hsp = apply_trans(als_ras_trans, hsp)
    # elp = apply_trans(als_ras_trans, elp)
    mrk = apply_trans(als_ras_trans, mrk)

    nasion, lpa, rpa = elp[:3]
    nmtrans = get_ras_to_neuromag_trans(nasion, lpa, rpa)
    elp = apply_trans(nmtrans, elp)
    hsp = apply_trans(nmtrans, hsp)

    # device head transform
    trans = fit_matched_points(tgt_pts=elp[3:], src_pts=mrk, out='trans')

    nasion, lpa, rpa = elp[:3]
    elp = elp[3:]

    dig_points = _make_dig_points(nasion, lpa, rpa, elp, hsp)
    dev_head_t = Transform('meg', 'head', trans)

    return dig_points, dev_head_t


##############################################################################
# From artemis123 (we have modified the function a bit)
def _foo_read_pos(nas, lpa, rpa, hpi, extra):
    # move into MNE head coords
    dig_points, _ = _bar(nas, lpa, rpa, hpi, extra)
    return dig_points


##############################################################################
# From bti
def _make_bti_dig_points(nasion, lpa, rpa, hpi, extra,
                         convert=False, use_hpi=False,
                         bti_dev_t=False, dev_ctf_t=False):

    _hpi = hpi if use_hpi else None
    info_dig, ctf_head_t = _bar(nasion, lpa, rpa, _hpi, extra,
                                convert)

    if convert:
        t = combine_transforms(invert_transform(bti_dev_t), dev_ctf_t,
                               'meg', 'ctf_head')
        dev_head_t = combine_transforms(t, ctf_head_t, 'meg', 'head')
    else:
        dev_head_t = Transform('meg', 'head', trans=None)

    return info_dig, dev_head_t, ctf_head_t  # ctf_head_t should not be needed
