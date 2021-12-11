# -*- coding: utf-8 -*-
# Authors: Federico Raimondo  <federaimondo@gmail.com>
#          simplified BSD-3 license
from pathlib import Path

import pytest
from numpy.testing import assert_array_almost_equal

from mne.io import read_raw_nihon, read_raw_edf
from mne.io.tests.test_raw import _test_raw_reader
from mne.datasets.testing import data_path, requires_testing_data
from mne.io.nihon.nihon import (_read_nihon_header, _read_nihon_metadata,
                                _read_nihon_annotations)


@requires_testing_data
def test_nihon_eeg():
    """Test reading Nihon Kohden EEG files."""
    fname = Path(data_path()) / 'NihonKohden' / 'MB0400FU.EEG'
    raw = read_raw_nihon(fname.as_posix(), preload=True)
    assert 'RawNihon' in repr(raw)
    _test_raw_reader(read_raw_nihon, fname=fname, test_scaling=False)
    fname_edf = Path(data_path()) / 'NihonKohden' / 'MB0400FU.EDF'
    raw_edf = read_raw_edf(fname_edf, preload=True)

    assert raw._data.shape == raw_edf._data.shape
    assert raw.info['sfreq'] == raw.info['sfreq']
    # ch names and order are switched in the EDF
    edf_ch_names = {x: x.split(' ')[1].replace('-Ref', '')
                    for x in raw_edf.ch_names}
    raw_edf.rename_channels(edf_ch_names)
    assert raw.ch_names == raw_edf.ch_names

    for i, an1 in enumerate(raw.annotations):
        # EDF has some weird annotations, which are not in the LOG file
        an2 = raw_edf.annotations[i * 2 + 1]
        assert an1['onset'] == an2['onset']
        assert an1['duration'] == an2['duration']
        # Also, it prepends 'Segment: ' to some annotations
        t_desc = an2['description'].replace('Segment: ', '')
        assert an1['description'] == t_desc

    assert_array_almost_equal(raw._data, raw_edf._data)

    with pytest.raises(ValueError, match='Not a valid Nihon Kohden EEG file'):
        raw = read_raw_nihon(fname_edf, preload=True)

    with pytest.raises(ValueError, match='Not a valid Nihon Kohden EEG file'):
        raw = _read_nihon_header(fname_edf)

    bad_fname = Path(data_path()) / 'eximia' / 'text_eximia.nxe'

    msg = 'No PNT file exists. Metadata will be blank'
    with pytest.warns(RuntimeWarning, match=msg):
        meta = _read_nihon_metadata(bad_fname)
        assert len(meta) == 0

    msg = 'No LOG file exists. Annotations will not be read'
    with pytest.warns(RuntimeWarning, match=msg):
        annot = _read_nihon_annotations(bad_fname)
        assert all(len(x) == 0 for x in annot.values())

    # the nihon test file has $A1 and $A2 in it, which are not EEG
    assert '$A1' in raw.ch_names

    # assert that channels with $ are 'misc'
    picks = [ch for ch in raw.ch_names if ch.startswith('$')]
    ch_types = raw.get_channel_types(picks=picks)
    assert all(ch == 'misc' for ch in ch_types)
