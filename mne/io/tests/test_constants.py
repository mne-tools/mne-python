# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import os.path as op
import re
import shutil
import zipfile

from mne.io.constants import FIFF
from mne.utils import _fetch_file, requires_good_network


# These are oddities that we won't address:
iod_dups = (355, 359)  # these are in both MEGIN and MNE files
tag_dups = (3501, 3507)  # in both MEGIN and MNE files

_dir_ignore_names = ('clear', 'copy', 'fromkeys', 'get', 'items', 'keys',
                     'pop', 'popitem', 'setdefault', 'update', 'values')

# XXX These should all probably be added to the FIFF constants
_missing_names = (
    'FIFFV_NEXT_SEQ',
    'FIFFV_NEXT_NONE',
    'FIFFV_COIL_ARTEMIS123_GRAD',
    'FIFFV_COIL_ARTEMIS123_REF_GRAD',
    'FIFFV_COIL_ARTEMIS123_REF_MAG',
    'FIFFV_COIL_BABY_REF_MAG',
    'FIFFV_COIL_BABY_REF_MAG2',
    'FIFFV_COIL_KRISS_GRAD',
    'FIFFV_COIL_POINT_MAGNETOMETER_X',
    'FIFFV_COIL_POINT_MAGNETOMETER_Y',
    'FIFFV_COIL_SAMPLE_TMS_PLANAR',
    'FIFF_UNIT_AM_M2',
    'FIFF_UNIT_AM_M3',
    'FIFFV_MNE_COORD_4D_HEAD',
    'FIFFV_MNE_COORD_CTF_DEVICE',
    'FIFFV_MNE_COORD_CTF_HEAD',
    'FIFFV_MNE_COORD_FS_TAL',
    'FIFFV_MNE_COORD_FS_TAL_GTZ',
    'FIFFV_MNE_COORD_FS_TAL_LTZ',
    'FIFFV_MNE_COORD_KIT_HEAD',
    'FIFFV_MNE_COORD_MNI_TAL',
    'FIFFV_MNE_COORD_MRI_VOXEL',
    'FIFFV_MNE_COORD_RAS',
    'FIFFV_MNE_COORD_TUFTS_EEG',
)


@requires_good_network
def test_constants(tmpdir):
    """Test compensation."""
    tmpdir = str(tmpdir)  # old pytest...
    dest = op.join(tmpdir, 'fiff.zip')
    _fetch_file('https://codeload.github.com/mne-tools/fiff-constants/zip/'
                'master', dest)
    names = list()
    with zipfile.ZipFile(dest, 'r') as ff:
        for name in ff.namelist():
            if 'Dictionary' in name:
                ff.extract(name, tmpdir)
                names.append(op.basename(name))
                shutil.move(op.join(tmpdir, name), op.join(tmpdir, names[-1]))
    names = sorted(names)
    assert names == ['DictionaryIOD.txt', 'DictionaryIOD_MNE.txt',
                     'DictionaryStructures.txt',
                     'DictionaryTags.txt', 'DictionaryTags_MNE.txt',
                     'DictionaryTypes.txt', 'DictionaryTypes_MNE.txt']
    # IOD (MEGIN and MNE)
    iod = dict()
    fiff_version = None
    for name in ['DictionaryIOD.txt', 'DictionaryIOD_MNE.txt']:
        with open(op.join(tmpdir, name), 'rb') as fid:
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
                    assert id_ not in iod
                iod[id_] = [kind, desc]
    # Tags (MEGIN)
    tags = dict()
    with open(op.join(tmpdir, 'DictionaryTags.txt'), 'rb') as fid:
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
            assert id_ not in tags, (tags.get(id_), val)
            tags[id_] = val
    # Tags (MNE)
    with open(op.join(tmpdir, 'DictionaryTags_MNE.txt'), 'rb') as fid:
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
                assert id_ not in tags, (tags.get(id_), val)
            tags[id_] = val

    # Types and enums
    defines = dict()  # maps the other way (name->val)
    types = dict()
    used_enums = ('unit', 'unitm', 'coil', 'aspect', 'bem_surf_id',
                  'ch_type', 'coord', 'mri_pixel', 'point', 'role',
                  'hand', 'sex', 'proj_item', 'bem_approx',
                  'mne_cov_ch', 'mne_ori', 'mne_map', 'covariance_type',
                  'mne_priors', 'mne_space', 'mne_surf')
    enums = dict((k, dict()) for k in used_enums)
    in_ = None
    re_prim = re.compile(r'^primitive\((.*)\)\s*(\S*)\s*"(.*)"$')
    re_enum = re.compile(r'^enum\((\S*)\)\s*".*"$')
    re_enum_entry = re.compile(r'\s*(\S*)\s*(\S*)\s*"(.*)"$')
    re_defi = re.compile(r'#define\s*(\S*)\s*(\S*)\s*"(.*)"$')
    for extra in ('', '_MNE'):
        with open(op.join(tmpdir, 'DictionaryTypes%s.txt'
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
                        assert s not in types
                        types[s] = [t, d]
                    elif e is not None:
                        # entering an enum
                        this_enum = e.group(1)
                        if this_enum in enums:
                            in_ = enums[e.group(1)]
                    elif d is not None:
                        t, s, d = d.groups()
                        s = int(s)
                        defines[t] = [s, d]
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

    # Assert that all our constants are in the dict
    # (we are not necessarily complete the other way)
    for name in sorted(dir(FIFF)):
        if name.startswith('_') or name in _dir_ignore_names or \
                name in _missing_names:
            continue
        val = getattr(FIFF, name)
        if name in defines:
            assert defines[name][0] == val
        elif name.startswith('FIFFC_'):
            # Checked above
            assert name in ('FIFFC_MAJOR_VERSION', 'FIFFC_MINOR_VERSION',
                            'FIFFC_VERSION')
        elif name.startswith('FIFFB_'):
            assert val in iod, (val, name)
        elif name.startswith('FIFFT_'):
            assert val in types, (val, name)
        elif name.startswith('FIFFV_'):
            if name.startswith('FIFFV_MNE_') and name.endswith('_ORI'):
                this_enum = 'mne_ori'
            elif name.startswith('FIFFV_MNE_') and name.endswith('_COV'):
                this_enum = 'covariance_type'
            elif name.startswith('FIFFV_MNE_COORD'):
                this_enum = 'coord'  # weird wrapper
            elif name.endswith('_CH') or '_QUAT_' in name or name in \
                    ('FIFFV_DIPOLE_WAVE', 'FIFFV_GOODNESS_FIT',
                     'FIFFV_HPI_ERR', 'FIFFV_HPI_G', 'FIFFV_HPI_MOV'):
                this_enum = 'ch_type'
            elif name.startswith('FIFFV_SUBJ_'):
                this_enum = name.split('_')[2].lower()
            else:
                for this_enum in used_enums:
                    if name.startswith('FIFFV_' + this_enum.upper()):
                        break
                else:
                    raise RuntimeError('Could not find %s' % (name,))
            assert this_enum in used_enums, name
            assert val in enums[this_enum], (val, name)
        elif name.startswith('FIFF_UNIT'):  # units and multipliers
            this_enum = name.split('_')[1].lower()
            assert val in enums[this_enum], (name, val)
        elif name.startswith('FIFF_'):
            assert val in tags, (name, val)
        else:
            unknowns.append((name, val))
    unknowns = '\n\t'.join('%s (%s)' % u for u in unknowns)
    assert len(unknowns) == 0, 'Unknown types\n\t%s' % unknowns
