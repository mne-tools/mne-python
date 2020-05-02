# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Denis Engemann <denis.engemann@gmail.com>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Eric Larson <larson.eric.d@gmail.com>
#          Marijn van Vliet <w.m.vanvliet@gmail.com>
#          Jona Sassenhagen <jona.sassenhagen@gmail.com>
#          Teon Brooks <teon.brooks@gmail.com>
#          Christian Brodbeck <christianbrodbeck@nyu.edu>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#          Joan Massich <mailsik@gmail.com>
#
# License: Simplified BSD

import xml.etree.ElementTree as ElementTree

import numpy as np


from ..utils import _check_fname, Bunch, warn


# XXX: to split as _parse like bvct
def _read_dig_montage_egi(
        fname,
        _scaling,
        _all_data_kwargs_are_none,
):

    if not _all_data_kwargs_are_none:
        raise ValueError('hsp, hpi, elp, point_names, fif must all be '
                         'None if egi is not None')
    _check_fname(fname, overwrite='read', must_exist=True)

    root = ElementTree.parse(fname).getroot()
    ns = root.tag[root.tag.index('{'):root.tag.index('}') + 1]
    sensors = root.find('%ssensorLayout/%ssensors' % (ns, ns))
    fids = dict()
    dig_ch_pos = dict()

    fid_name_map = {'Nasion': 'nasion',
                    'Right periauricular point': 'rpa',
                    'Left periauricular point': 'lpa'}

    for s in sensors:
        name, number, kind = s[0].text, int(s[1].text), int(s[2].text)
        coordinates = np.array([float(s[3].text), float(s[4].text),
                                float(s[5].text)])

        coordinates *= _scaling

        # EEG Channels
        if kind == 0:
            dig_ch_pos['EEG %03d' % number] = coordinates
        # Reference
        elif kind == 1:
            dig_ch_pos['EEG %03d' %
                       (len(dig_ch_pos.keys()) + 1)] = coordinates
            # XXX: The EGI reader needs to be fixed with this code here.
            # As a reference channel it should be called EEG000 or
            # REF to follow the conventions. I should be:
            # dig_ch_pos['REF'] = coordinates

        # Fiducials
        elif kind == 2:
            fid_name = fid_name_map[name]
            fids[fid_name] = coordinates
        # Unknown
        else:
            warn('Unknown sensor type %s detected. Skipping sensor...'
                 'Proceed with caution!' % kind)

    return Bunch(
        # EGI stuff
        nasion=fids['nasion'], lpa=fids['lpa'], rpa=fids['rpa'],
        dig_ch_pos=dig_ch_pos, coord_frame='unknown',
        # not EGI stuff
        hsp=None, hpi=None, elp=None, point_names=None,
    )


def _parse_brainvision_dig_montage(fname, scale):
    FID_NAME_MAP = {'Nasion': 'nasion', 'RPA': 'rpa', 'LPA': 'lpa'}

    root = ElementTree.parse(fname).getroot()
    sensors = root.find('CapTrakElectrodeList')

    fids, dig_ch_pos = dict(), dict()

    for s in sensors:
        name = s.find('Name').text

        is_fid = name in FID_NAME_MAP
        coordinates = scale * np.array([float(s.find('X').text),
                                        float(s.find('Y').text),
                                        float(s.find('Z').text)])

        # Fiducials
        if is_fid:
            fids[FID_NAME_MAP[name]] = coordinates
        # EEG Channels
        else:
            dig_ch_pos[name] = coordinates

    return dict(
        # BVCT stuff
        nasion=fids['nasion'], lpa=fids['lpa'], rpa=fids['rpa'],
        ch_pos=dig_ch_pos, coord_frame='unknown'
    )
