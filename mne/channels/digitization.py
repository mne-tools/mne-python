import os.path as op

import numpy as np

from ..io.meas_info import (_make_dig_points, _read_dig_points, _read_dig_fif,
                            write_dig)

from .layout import _pol_to_cart, _cart_to_sph


class Digitization(object):
    def __init__(self, dig_list=None):
        self.foo = dig_list

MONTAGES_PATH = op.join(op.dirname(__file__), 'data', 'montages')
SUPPORTED_MONTAGES = {
    'EGI_256': op.join(MONTAGES_PATH, 'EGI_256.csd'),
    'GSN-HydroCel-128': op.join(MONTAGES_PATH, 'GSN-HydroCel-128.sfp'),
    'GSN-HydroCel-129': op.join(MONTAGES_PATH, 'GSN-HydroCel-129.sfp'),
    'GSN-HydroCel-256': op.join(MONTAGES_PATH, 'GSN-HydroCel-256.sfp'),
    'GSN-HydroCel-257': op.join(MONTAGES_PATH, 'GSN-HydroCel-257.sfp'),
    'GSN-HydroCel-32': op.join(MONTAGES_PATH, 'GSN-HydroCel-32.sfp'),
    'GSN-HydroCel-64_1.0': op.join(MONTAGES_PATH, 'GSN-HydroCel-64_1.0.sfp'),
    'GSN-HydroCel-65_1.0': op.join(MONTAGES_PATH, 'GSN-HydroCel-65_1.0.sfp'),
    'biosemi128': op.join(MONTAGES_PATH, 'biosemi128.txt'),
    'biosemi16': op.join(MONTAGES_PATH, 'biosemi16.txt'),
    'biosemi160': op.join(MONTAGES_PATH, 'biosemi160.txt'),
    'biosemi256': op.join(MONTAGES_PATH, 'biosemi256.txt'),
    'biosemi32': op.join(MONTAGES_PATH, 'biosemi32.txt'),
    'biosemi64': op.join(MONTAGES_PATH, 'biosemi64.txt'),
    'easycap-M1': op.join(MONTAGES_PATH, 'easycap-M1.txt'),
    'easycap-M10': op.join(MONTAGES_PATH, 'easycap-M10.txt'),
    'mgh60': op.join(MONTAGES_PATH, 'mgh60.elc'),
    'mgh70': op.join(MONTAGES_PATH, 'mgh70.elc'),
    'standard_1005': op.join(MONTAGES_PATH, 'standard_1005.elc'),
    'standard_1020': op.join(MONTAGES_PATH, 'standard_1020.elc'),
    'standard_alphabetic': op.join(MONTAGES_PATH, 'standard_alphabetic.elc'),
    'standard_postfixed': op.join(MONTAGES_PATH, 'standard_postfixed.elc'),
    'standard_prefixed': op.join(MONTAGES_PATH, 'standard_prefixed.elc'),
    'standard_primed': op.join(MONTAGES_PATH, 'standard_primed.elc'),
}


def _get_sph_data_from_biosemi_file(fname):
    # This should not be done with pandas, but while tinkering..
    import pandas as pd
    FIDUTIAL_NAMES = ['Nz', 'LPA', 'RPA']
    records = pd.read_csv(fname, sep='\t', index_col='Site').apply(np.deg2rad)
    fidutials = {'lpa': records.loc['LPA'].values,
                 'nz': records.loc['Nz'].values,
                 'rpa': records.loc['RPA'].values}
    records = records.drop(index=FIDUTIAL_NAMES)

    return fidutials, (records.index.values,
                       records['Phi'].values,
                       records[' Theta'].values)


# this should end up being read_montage
def read_foo(kind, ch_names=None, path=None, unit='m', transform=False):
    """Read a generic (built-in) montage.

    """
    # import pdb; pdb.set_trace()
    if kind not in ['biosemi16']:  # SUPPORTED_MONTAGES:
        raise NotImplementedError
    else:
        fname = SUPPORTED_MONTAGES[kind]

    fid, (ch_names, phi, theta) = _get_sph_data_from_biosemi_file(fname)
    fid_names = ['lpa', 'nz', 'rpa']
    #     # easycap
    #     try:  # newer version
    #         data = np.genfromtxt(fname, dtype='str', skip_header=1)
    #     except TypeError:
    #         data = np.genfromtxt(fname, dtype='str', skiprows=1)
    #     ch_names_ = data[:, 0].tolist()
    #     az = np.deg2rad(data[:, 2].astype(float))
    #     pol = np.deg2rad(data[:, 1].astype(float))
    #     pos = _sph_to_cart(np.array([np.ones(len(az)) * 85., az, pol]).T)
    # selection = np.arange(len(pos))

    # if unit == 'mm':
    #     pos /= 1e3
    # elif unit == 'cm':
    #     pos /= 1e2
    # elif unit != 'm':
    #     raise ValueError("'unit' should be either 'm', 'cm', or 'mm'.")
    # names_lower = [name.lower() for name in list(ch_names_)]
    # fids = {key: pos[names_lower.index(fid_names[ii])]
    #         if fid_names[ii] in names_lower else None
    #         for ii, key in enumerate(['lpa', 'nasion', 'rpa'])}
    # if transform:
    #     missing = [name for name, val in fids.items() if val is None]
    #     if missing:
    #         raise ValueError("The points %s are missing, but are needed "
    #                          "to transform the points to the MNE coordinate "
    #                          "system. Either add the points, or read the "
    #                          "montage with transform=False. " % missing)
    #     neuromag_trans = get_ras_to_neuromag_trans(
    #         fids['nasion'], fids['lpa'], fids['rpa'])
    #     pos = apply_trans(neuromag_trans, pos)
    # fids = {key: pos[names_lower.index(fid_names[ii])]
    #         if fid_names[ii] in names_lower else None
    #         for ii, key in enumerate(['lpa', 'nasion', 'rpa'])}

    # if ch_names is not None:
    #     # Ensure channels with differing case are found.
    #     upper_names = [ch_name.upper() for ch_name in ch_names]
    #     sel, ch_names_ = zip(*[(i, ch_names[upper_names.index(e)]) for i, e in
    #                            enumerate([n.upper() for n in ch_names_])
    #                            if e in upper_names])
    #     sel = list(sel)
    #     pos = pos[sel]
    #     selection = selection[sel]
    # kind = op.split(kind)[-1]
    # return Montage(pos=pos, ch_names=ch_names_, kind=kind, selection=selection,
    #                lpa=fids['lpa'], nasion=fids['nasion'], rpa=fids['rpa'])
    return False


from mne import __file__ as mne_init_path
from mne.channels.montage import _read_dig_points

KIT_PATH = op.join(op.dirname(mne_init_path), 'io', 'kit', 'tests', 'data')
KIT_HSP = op.join(KIT_PATH, 'test.hsp')
KIT_ELP = op.join(KIT_PATH, 'test.elp')
KIT_SQD = op.join(KIT_PATH, 'test.sqd')


def read_foobar(hsp_fname=KIT_HSP, elp_fname=KIT_ELP, sqd_fname=KIT_SQD):
    from mne.io.kit.kit import _set_dig_kit

    # If you had to load them, thats how it's done, but kit has some helper
    # functions
    #
    # hsp = _read_dig_points(hsp_fname, unit=unit)
    # elp = _read_dig_points(elp_fname, unit=unit)
    # hpi = read_mrk(sqd_fname)

    dig, dev_head_t = _set_dig_kit(mrk=sqd_fname, elp=elp_fname, hsp=hsp_fname)
    return Digitization(dig_list=dig)


def _read_pos_file_reader(fname):
    HEADER_LENGHT = 8
    from itertools import islice

    with open(fname) as my_file:
        head = [_.split() for _ in list(islice(my_file, 0, HEADER_LENGHT))]
        points = [_.split() for _ in list(islice(my_file, 0, None))]

    head_idx, head_name, head_xs, head_ys, head_zs = zip(*head)

    # This should not be hard-coded ('cos who grants the order?)
    fidutials = {'lpa': np.array([head_xs[1], head_ys[1], head_zs[1]], dtype=float),
                 'nz': np.array([head_xs[0], head_ys[0], head_zs[0]], dtype=float),
                 'rpa': np.array([head_xs[2], head_ys[2], head_zs[2]], dtype=float)}

    reference_points = np.column_stack([head_xs[3:], head_ys[3:], head_zs[3:]])
    reference_points = dict(zip(head_name[3:], reference_points.astype(float)))

    points = np.array(points, dtype=float)

    return fidutials, reference_points, points


def _my_set_dig_kit(mrk, elp, hsp):
    """copy-paste of Internal parts of _set_dig_kit but allow for mrk=None"""
    from mne.transforms import (apply_trans, als_ras_trans,
                                get_ras_to_neuromag_trans, Transform)
    from mne.coreg import fit_matched_points

    hsp = apply_trans(als_ras_trans, hsp)
    elp = apply_trans(als_ras_trans, elp)
    mrk = None if mrk is None else apply_trans(als_ras_trans, mrk)

    nasion, lpa, rpa = elp[:3]
    nmtrans = get_ras_to_neuromag_trans(nasion, lpa, rpa)
    elp = apply_trans(nmtrans, elp)
    hsp = apply_trans(nmtrans, hsp)

    # device head transform
    trans = None if mrk is None else fit_matched_points(tgt_pts=elp[3:], src_pts=mrk, out='trans')

    nasion, lpa, rpa = elp[:3]
    elp = elp[3:]

    dig_points = _make_dig_points(nasion, lpa, rpa, elp, hsp)
    dev_head_t = Transform('meg', 'head', trans)

    return dig_points, dev_head_t


POS_FNAME = op.join(op.dirname(mne_init_path), 'channels', 'data', 'test.pos')
def read_pos(fname=POS_FNAME):
    fidutials, reference_points, hsp = _read_pos_file_reader(fname)

    # create a valid elp: nz, lpa, rpa, 0-RED, 1-YELLOW, .. , 4-BLACK
    elp = np.vstack([fidutials[_] for _ in ('nz', 'lpa', 'rpa')] +
                    list(reference_points.values()))

    # dig, dev_head_t = _set_dig_kit(mrk=None, elp=elp, hsp=hsp)
    dig, dev_head_t = _my_set_dig_kit(mrk=None, elp=elp, hsp=hsp)

    return Digitization(dig_list=dig)
