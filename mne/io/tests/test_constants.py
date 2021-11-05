# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD-3-Clause

import os.path as op
import re
import shutil
import zipfile

import numpy as np
import pytest

from mne.io.constants import (FIFF, FWD, _coord_frame_named, _ch_kind_named,
                              _ch_unit_named, _ch_unit_mul_named,
                              _ch_coil_type_named, _dig_kind_named,
                              _dig_cardinal_named)
from mne.forward._make_forward import _read_coil_defs
from mne.utils import requires_good_network
from mne.utils.check import _soft_import

# import pooch library for handling the dataset downloading
pooch = _soft_import('pooch', 'dataset downloading', strict=True)


# https://github.com/mne-tools/fiff-constants/commits/master
REPO = 'mne-tools'
COMMIT = 'aae5960007ee8a67dfc07535ea37d421d37dfe1b'

# These are oddities that we won't address:
iod_dups = (355, 359)  # these are in both MEGIN and MNE files
tag_dups = (3501,)  # in both MEGIN and MNE files

_dir_ignore_names = ('clear', 'copy', 'fromkeys', 'get', 'items', 'keys',
                     'pop', 'popitem', 'setdefault', 'update', 'values',
                     'has_key', 'iteritems', 'iterkeys', 'itervalues',  # Py2
                     'viewitems', 'viewkeys', 'viewvalues',  # Py2
                     )
_tag_ignore_names = (
)  # for fiff-constants pending updates
_ignore_incomplete_enums = (  # XXX eventually we could complete these
    'bem_surf_id', 'cardinal_point_cardiac', 'cond_model', 'coord',
    'dacq_system', 'diffusion_param', 'gantry_type', 'map_surf',
    'mne_lin_proj', 'mne_ori', 'mri_format', 'mri_pixel', 'proj_by',
    'tags', 'type', 'iod', 'volume_type', 'vol_type',
)
# not in coil_def.dat but in DictionaryTypes:enum(coil)
_missing_coil_def = (
    0,      # The location info contains no data
    1,      # EEG electrode position in r0
    3,      # Old 24 channel system in HUT
    4,      # The axial devices in the HUCS MCG system
    5,      # Bipolar EEG electrode position
    6,      # CSD-transformed EEG electrodes
    200,    # Time-varying dipole definition
    300,    # fNIRS oxyhemoglobin
    301,    # fNIRS deoxyhemoglobin
    302,    # fNIRS continuous wave
    303,    # fNIRS optical density
    304,    # fNIRS frequency domain AC amplitude
    305,    # fNIRS frequency domain phase
    1000,   # For testing the MCG software
    2001,   # Generic axial gradiometer
    3011,   # VV prototype wirewound planar sensor
    3014,   # Vectorview SQ20950N planar gradiometer
    3021,   # VV prototype wirewound magnetometer
)
# explicit aliases in constants.py
_aliases = dict(
    FIFFV_COIL_MAGNES_R_MAG='FIFFV_COIL_MAGNES_REF_MAG',
    FIFFV_COIL_MAGNES_R_GRAD='FIFFV_COIL_MAGNES_REF_GRAD',
    FIFFV_COIL_MAGNES_R_GRAD_OFF='FIFFV_COIL_MAGNES_OFFDIAG_REF_GRAD',
    FIFFV_COIL_FNIRS_RAW='FIFFV_COIL_FNIRS_CW_AMPLITUDE',
    FIFFV_MNE_COORD_CTF_HEAD='FIFFV_MNE_COORD_4D_HEAD',
    FIFFV_MNE_COORD_KIT_HEAD='FIFFV_MNE_COORD_4D_HEAD',
    FIFFV_MNE_COORD_DIGITIZER='FIFFV_COORD_ISOTRAK',
    FIFFV_MNE_COORD_SURFACE_RAS='FIFFV_COORD_MRI',
    FIFFV_MNE_SENSOR_COV='FIFFV_MNE_NOISE_COV',
    FIFFV_POINT_EEG='FIFFV_POINT_ECG',
    FIFF_DESCRIPTION='FIFF_COMMENT',
    FIFF_REF_PATH='FIFF_MRI_SOURCE_PATH',
)


@requires_good_network
def test_constants(tmp_path):
    """Test compensation."""
    tmp_path = str(tmp_path)  # old pytest...
    fname = 'fiff.zip'
    dest = op.join(tmp_path, fname)
    pooch.retrieve(
        url='https://codeload.github.com/'
            f'{REPO}/fiff-constants/zip/{COMMIT}',
        path=tmp_path,
        fname=fname,
        known_hash=None
    )
    names = list()
    with zipfile.ZipFile(dest, 'r') as ff:
        for name in ff.namelist():
            if 'Dictionary' in name:
                ff.extract(name, tmp_path)
                names.append(op.basename(name))
                shutil.move(op.join(tmp_path, name),
                            op.join(tmp_path, names[-1]))
    names = sorted(names)
    assert names == ['DictionaryIOD.txt', 'DictionaryIOD_MNE.txt',
                     'DictionaryStructures.txt',
                     'DictionaryTags.txt', 'DictionaryTags_MNE.txt',
                     'DictionaryTypes.txt', 'DictionaryTypes_MNE.txt']
    # IOD (MEGIN and MNE)
    fif = dict(iod=dict(), tags=dict(), types=dict(), defines=dict())
    con = dict(iod=dict(), tags=dict(), types=dict(), defines=dict())
    fiff_version = None
    for name in ['DictionaryIOD.txt', 'DictionaryIOD_MNE.txt']:
        with open(op.join(tmp_path, name), 'rb') as fid:
            for line in fid:
                line = line.decode('latin1').strip()
                if line.startswith('# Packing revision'):
                    assert fiff_version is None
                    fiff_version = line.split()[-1]
                if (line.startswith('#') or line.startswith('alias') or
                        len(line) == 0):
                    continue
                line = line.split('"')
                assert len(line) in (1, 2, 3)
                desc = '' if len(line) == 1 else line[1]
                line = line[0].split()
                assert len(line) in (2, 3)
                if len(line) == 2:
                    kind, id_ = line
                else:
                    kind, id_, tagged = line
                    assert tagged in ('tagged',)
                id_ = int(id_)
                if id_ not in iod_dups:
                    assert id_ not in fif['iod']
                fif['iod'][id_] = [kind, desc]
    # Tags (MEGIN)
    with open(op.join(tmp_path, 'DictionaryTags.txt'), 'rb') as fid:
        for line in fid:
            line = line.decode('ISO-8859-1').strip()
            if (line.startswith('#') or line.startswith('alias') or
                    line.startswith(':') or len(line) == 0):
                continue
            line = line.split('"')
            assert len(line) in (1, 2, 3), line
            desc = '' if len(line) == 1 else line[1]
            line = line[0].split()
            assert len(line) == 4, line
            kind, id_, dtype, unit = line
            id_ = int(id_)
            val = [kind, dtype, unit]
            assert id_ not in fif['tags'], (fif['tags'].get(id_), val)
            fif['tags'][id_] = val
    # Tags (MNE)
    with open(op.join(tmp_path, 'DictionaryTags_MNE.txt'), 'rb') as fid:
        for li, line in enumerate(fid):
            line = line.decode('ISO-8859-1').strip()
            # ignore continuation lines (*)
            if (line.startswith('#') or line.startswith('alias') or
                    line.startswith(':') or line.startswith('*') or
                    len(line) == 0):
                continue
            # weird syntax around line 80:
            if line in ('/*', '"'):
                continue
            line = line.split('"')
            assert len(line) in (1, 2, 3), line
            if len(line) == 3 and len(line[2]) > 0:
                l2 = line[2].strip()
                assert l2.startswith('/*') and l2.endswith('*/'), l2
            desc = '' if len(line) == 1 else line[1]
            line = line[0].split()
            assert len(line) == 3, (li + 1, line)
            kind, id_, dtype = line
            unit = '-'
            id_ = int(id_)
            val = [kind, dtype, unit]
            if id_ not in tag_dups:
                assert id_ not in fif['tags'], (fif['tags'].get(id_), val)
            fif['tags'][id_] = val

    # Types and enums
    in_ = None
    re_prim = re.compile(r'^primitive\((.*)\)\s*(\S*)\s*"(.*)"$')
    re_enum = re.compile(r'^enum\((\S*)\)\s*".*"$')
    re_enum_entry = re.compile(r'\s*(\S*)\s*(\S*)\s*"(.*)"$')
    re_defi = re.compile(r'#define\s*(\S*)\s*(\S*)\s*"(.*)"$')
    used_enums = list()
    for extra in ('', '_MNE'):
        with open(op.join(tmp_path, 'DictionaryTypes%s.txt'
                          % (extra,)), 'rb') as fid:
            for li, line in enumerate(fid):
                line = line.decode('ISO-8859-1').strip()
                if in_ is None:
                    p = re_prim.match(line)
                    e = re_enum.match(line)
                    d = re_defi.match(line)
                    if p is not None:
                        t, s, d = p.groups()
                        s = int(s)
                        assert s not in fif['types']
                        fif['types'][s] = [t, d]
                    elif e is not None:
                        # entering an enum
                        this_enum = e.group(1)
                        if this_enum not in fif:
                            used_enums.append(this_enum)
                            fif[this_enum] = dict()
                            con[this_enum] = dict()
                        in_ = fif[this_enum]
                    elif d is not None:
                        t, s, d = d.groups()
                        s = int(s)
                        fif['defines'][t] = [s, d]
                    else:
                        assert not line.startswith('enum(')
                else:  # in an enum
                    if line == '{':
                        continue
                    elif line == '}':
                        in_ = None
                        continue
                    t, s, d = re_enum_entry.match(line).groups()
                    s = int(s)
                    if t != 'ecg' and s != 3:  # ecg defined the same way
                        assert s not in in_
                    in_[s] = [t, d]

    #
    # Assertions
    #

    # Version
    mne_version = '%d.%d' % (FIFF.FIFFC_MAJOR_VERSION,
                             FIFF.FIFFC_MINOR_VERSION)
    assert fiff_version == mne_version
    unknowns = list()

    # Assert that all our constants are in the FIF def
    assert 'FIFFV_SSS_JOB_NOTHING' in dir(FIFF)
    for name in sorted(dir(FIFF)):
        if name.startswith('_') or name in _dir_ignore_names:
            continue
        check = None
        val = getattr(FIFF, name)
        if name in fif['defines']:
            assert fif['defines'][name][0] == val
        elif name.startswith('FIFFC_'):
            # Checked above
            assert name in ('FIFFC_MAJOR_VERSION', 'FIFFC_MINOR_VERSION',
                            'FIFFC_VERSION')
        elif name.startswith('FIFFB_'):
            check = 'iod'
        elif name.startswith('FIFFT_'):
            check = 'types'
        elif name.startswith('FIFFV_'):
            if name.startswith('FIFFV_MNE_') and name.endswith('_ORI'):
                check = 'mne_ori'
            elif name.startswith('FIFFV_MNE_') and name.endswith('_COV'):
                check = 'covariance_type'
            elif name.startswith('FIFFV_MNE_COORD'):
                check = 'coord'  # weird wrapper
            elif name.endswith('_CH') or '_QUAT_' in name or name in \
                    ('FIFFV_DIPOLE_WAVE', 'FIFFV_GOODNESS_FIT',
                     'FIFFV_HPI_ERR', 'FIFFV_HPI_G', 'FIFFV_HPI_MOV'):
                check = 'ch_type'
            elif name.startswith('FIFFV_SUBJ_'):
                check = name.split('_')[2].lower()
            elif name in ('FIFFV_POINT_LPA', 'FIFFV_POINT_NASION',
                          'FIFFV_POINT_RPA', 'FIFFV_POINT_INION'):
                check = 'cardinal_point'
            else:
                for check in used_enums:
                    if name.startswith('FIFFV_' + check.upper()):
                        break
                else:
                    if name not in _tag_ignore_names:
                        raise RuntimeError('Could not find %s' % (name,))
            assert check in used_enums, name
            if 'SSS' in check:
                raise RuntimeError
        elif name.startswith('FIFF_UNIT'):  # units and multipliers
            check = name.split('_')[1].lower()
        elif name.startswith('FIFF_'):
            check = 'tags'
        else:
            unknowns.append((name, val))
        if check is not None and name not in _tag_ignore_names:
            assert val in fif[check], '%s: %s, %s' % (check, val, name)
            if val in con[check]:
                msg = "%s='%s'  ?" % (name, con[check][val])
                assert _aliases.get(name) == con[check][val], msg
            else:
                con[check][val] = name
    unknowns = '\n\t'.join('%s (%s)' % u for u in unknowns)
    assert len(unknowns) == 0, 'Unknown types\n\t%s' % unknowns

    # Assert that all the FIF defs are in our constants
    assert set(fif.keys()) == set(con.keys())
    for key in sorted(set(fif.keys()) - {'defines'}):
        this_fif, this_con = fif[key], con[key]
        assert len(set(this_fif.keys())) == len(this_fif)
        assert len(set(this_con.keys())) == len(this_con)
        missing_from_con = sorted(set(this_con.keys()) - set(this_fif.keys()))
        assert missing_from_con == [], key
        if key not in _ignore_incomplete_enums:
            missing_from_fif = sorted(set(this_fif.keys()) -
                                      set(this_con.keys()))
            assert missing_from_fif == [], key

    # Assert that `coil_def.dat` has accurate descriptions of all enum(coil)
    coil_def = _read_coil_defs()
    coil_desc = np.array([c['desc'] for c in coil_def])
    coil_def = np.array([(c['coil_type'], c['accuracy'])
                         for c in coil_def], int)
    mask = (coil_def[:, 1] == FWD.COIL_ACCURACY_ACCURATE)
    coil_def = coil_def[mask, 0]
    coil_desc = coil_desc[mask]
    bad_list = []
    for key in fif['coil']:
        if key not in _missing_coil_def and key not in coil_def:
            bad_list.append(('    %s,' % key).ljust(10) +
                            '  # ' + fif['coil'][key][1])
    assert len(bad_list) == 0, \
        '\nIn fiff-constants, missing from coil_def:\n' + '\n'.join(bad_list)
    # Assert that enum(coil) has all `coil_def.dat` entries
    for key, desc in zip(coil_def, coil_desc):
        if key not in fif['coil']:
            bad_list.append(('    %s,' % key).ljust(10) + '  # ' + desc)
    assert len(bad_list) == 0, \
        'In coil_def, missing  from fiff-constants:\n' + '\n'.join(bad_list)


@pytest.mark.parametrize('dict_, match, extras', [
    ({**_dig_kind_named, **_dig_cardinal_named}, 'FIFFV_POINT_', ()),
    (_ch_kind_named, '^FIFFV_.*_CH$',
     (FIFF.FIFFV_DIPOLE_WAVE, FIFF.FIFFV_GOODNESS_FIT)),
    (_coord_frame_named, 'FIFFV_COORD_', ()),
    (_ch_unit_named, 'FIFF_UNIT_', ()),
    (_ch_unit_mul_named, 'FIFF_UNITM_', ()),
    (_ch_coil_type_named, 'FIFFV_COIL_', ()),
])
def test_dict_completion(dict_, match, extras):
    """Test readable dict completions."""
    regex = re.compile(match)
    got = set(FIFF[key] for key in FIFF if regex.search(key) is not None)
    for e in extras:
        got.add(e)
    want = set(dict_)
    assert got == want, match
