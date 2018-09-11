# Author: Eric Larson <larson.eric.d@gmail.com>
#
# License: BSD (3-clause)

import os.path as op
import shutil
import zipfile

from mne.io.constants import FIFF
from mne.utils import _fetch_file, requires_good_network


# These are oddities that we won't address:
iod_dups = (355, 359)  # these are in both MEGIN and MNE files
tag_dups = (3501, 3507)  # in both MEGIN and MNE files
# The tests should also be improved, see several XXX below.

_dir_ignore_names = ('clear', 'copy', 'fromkeys', 'get', 'items', 'keys',
                     'pop', 'popitem', 'setdefault', 'update', 'values')


@requires_good_network
def test_constants(tmpdir):
    """Test compensation."""
    dest = op.join(tmpdir, 'fiff.zip')
    _fetch_file('https://api.github.com/repos/mne-tools/fiff-constants'
                '/zipball/master', dest)
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
                line = line.decode('ISO-8859-1').strip()
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
        if name.startswith('_') or name in _dir_ignore_names:
            continue
        val = getattr(FIFF, name)
        if name.startswith('FIFFC_'):
            # Checked above
            assert name in ('FIFFC_MAJOR_VERSION', 'FIFFC_MINOR_VERSION',
                            'FIFFC_VERSION')
        elif name.startswith('FIFFB_'):
            assert val in iod, (val, name)
        elif name.startswith('FIFFT_'):
            continue  # XXX add check for this
        elif name.startswith('FIFFV_'):
            continue  # XXX add check for this
        elif name.startswith('FIFF_UNIT_'):
            continue  # XXX add check for this
        elif name.startswith('FIFF_UNITM_'):  # multipliers
            continue  # XXX add check for this
        elif name.startswith('FWD_'):
            # These are not FIFF constants really
            # XXX remove from FIFF to forward.py namespace
            continue
        elif name.startswith('FIFF_'):
            assert val in tags, (name, val)
        else:
            unknowns.append((name, val))
    unknowns = '\n\t'.join('%s (%s)' % u for u in unknowns)
    assert len(unknowns) == 0, 'Unknown types\n\t%s' % unknowns
