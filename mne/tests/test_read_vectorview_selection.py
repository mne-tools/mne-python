import os.path as op

import pytest

from mne import read_vectorview_selection
from mne.io import read_raw_fif

test_path = op.join(op.split(__file__)[0], '..', 'io', 'tests', 'data')
raw_fname = op.join(test_path, 'test_raw.fif')
raw_new_fname = op.join(test_path, 'test_chpi_raw_sss.fif')


def test_read_vectorview_selection():
    """Test reading of Neuromag Vector View channel selections."""
    # test one channel for each selection
    ch_names = ['MEG 2211', 'MEG 0223', 'MEG 1312', 'MEG 0412', 'MEG 1043',
                'MEG 2042', 'MEG 2032', 'MEG 0522', 'MEG 1031']
    sel_names = ['Vertex', 'Left-temporal', 'Right-temporal', 'Left-parietal',
                 'Right-parietal', 'Left-occipital', 'Right-occipital',
                 'Left-frontal', 'Right-frontal']

    raw = read_raw_fif(raw_fname)
    for i, name in enumerate(sel_names):
        sel = read_vectorview_selection(name)
        assert ch_names[i] in sel
        sel_info = read_vectorview_selection(name, info=raw.info)
        assert sel == sel_info

    # test some combinations
    all_ch = read_vectorview_selection(['L', 'R'])
    left = read_vectorview_selection('L')
    right = read_vectorview_selection('R')

    assert len(all_ch) == len(left) + len(right)
    assert len(set(left).intersection(set(right))) == 0

    frontal = read_vectorview_selection('frontal')
    occipital = read_vectorview_selection('Right-occipital')
    assert len(set(frontal).intersection(set(occipital))) == 0

    ch_names_new = [ch.replace(' ', '') for ch in ch_names]
    raw_new = read_raw_fif(raw_new_fname)
    for i, name in enumerate(sel_names):
        sel = read_vectorview_selection(name, info=raw_new.info)
        assert ch_names_new[i] in sel

    with pytest.raises(TypeError, match='must be an instance of Info or None'):
        read_vectorview_selection(name, info='foo')
