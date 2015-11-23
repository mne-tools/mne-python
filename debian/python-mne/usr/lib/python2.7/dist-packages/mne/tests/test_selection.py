from mne import read_selection


def test_read_selection():
    """Test reading of selections"""
    # test one channel for each selection
    ch_names = ['MEG 2211', 'MEG 0223', 'MEG 1312', 'MEG 0412', 'MEG 1043',
                'MEG 2042', 'MEG 2032', 'MEG 0522', 'MEG 1031']
    sel_names = ['Vertex', 'Left-temporal', 'Right-temporal', 'Left-parietal',
                 'Right-parietal', 'Left-occipital', 'Right-occipital',
                 'Left-frontal', 'Right-frontal']

    for i, name in enumerate(sel_names):
        sel = read_selection(name)
        assert(ch_names[i] in sel)

    # test some combinations
    all_ch = read_selection(['L', 'R'])
    left = read_selection('L')
    right = read_selection('R')

    assert(len(all_ch) == len(left) + len(right))
    assert(len(set(left).intersection(set(right))) == 0)

    frontal = read_selection('frontal')
    occipital = read_selection('Right-occipital')
    assert(len(set(frontal).intersection(set(occipital))) == 0)
