import os.path as op
import warnings

from nose.tools import assert_true
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_equal

from mne.datasets import sample
from mne.fiff import Raw, Evoked, pick_types_forward
from mne import read_forward_solution, apply_forward, apply_forward_raw
from mne import SourceEstimate
from mne import read_label
from mne.forward import restrict_foward_to_stc, restrict_foward_to_label


data_path = sample.data_path()
fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis-meg-oct-6-fwd.fif')

fname_raw = op.join(op.dirname(__file__), '..', 'fiff', 'tests', 'data',
                    'test_raw.fif')

fname_evoked = op.join(op.dirname(__file__), '..', 'fiff', 'tests', 'data',
                       'test-ave.fif')


def test_io_forward():
    """Test IO for forward solutions
    """
    fwd = read_forward_solution(fname)
    fwd = read_forward_solution(fname, surf_ori=True)
    leadfield = fwd['sol']['data']
    assert_equal(leadfield.shape, (306, 22494))
    assert_equal(len(fwd['sol']['row_names']), 306)

    fwd = read_forward_solution(fname, force_fixed=True)
    leadfield = fwd['sol']['data']
    assert_equal(leadfield.shape, (306, 22494 / 3))
    assert_equal(len(fwd['sol']['row_names']), 306)
    assert_equal(len(fwd['info']['chs']), 306)
    assert_true('dev_head_t' in fwd['info'])
    assert_true('mri_head_t' in fwd)


def test_apply_forward():
    """Test projection of source space data to sensor space
    """
    start = 0
    stop = 5
    n_times = stop - start - 1
    sfreq = 10.0
    t_start = 0.123

    fwd = read_forward_solution(fname, force_fixed=True)
    fwd = pick_types_forward(fwd, meg=True)

    vertno = [fwd['src'][0]['vertno'], fwd['src'][1]['vertno']]
    stc_data = np.ones((len(vertno[0]) + len(vertno[1]), n_times))
    stc = SourceEstimate(stc_data, vertno, tmin=t_start, tstep=1.0 / sfreq)

    gain_sum = np.sum(fwd['sol']['data'], axis=1)

    # Evoked
    with warnings.catch_warnings(record=True) as w:
        evoked = Evoked(fname_evoked, setno=0)
        evoked = apply_forward(fwd, stc, evoked, start=start, stop=stop)
        assert_equal(len(w), 2)
        data = evoked.data
        times = evoked.times

        # do some tests
        assert_array_almost_equal(evoked.info['sfreq'], sfreq)
        assert_array_almost_equal(np.sum(data, axis=1), n_times * gain_sum)
        assert_array_almost_equal(times[0], t_start)
        assert_array_almost_equal(times[-1], t_start + (n_times - 1) / sfreq)

        # Raw
        raw = Raw(fname_raw)
        raw_proj = apply_forward_raw(fwd, stc, raw, start=start, stop=stop)
        data, times = raw_proj[:, :]

        # do some tests
        assert_array_almost_equal(raw_proj.info['sfreq'], sfreq)
        assert_array_almost_equal(np.sum(data, axis=1), n_times * gain_sum)
        assert_array_almost_equal(times[0], t_start)
        assert_array_almost_equal(times[-1], t_start + (n_times - 1) / sfreq)


def test_restrict_foward_to_stc():
    """Test restriction of source space to source SourceEstimate
    """
    start = 0
    stop = 5
    n_times = stop - start - 1
    sfreq = 10.0
    t_start = 0.123

    fwd = read_forward_solution(fname, force_fixed=True)
    fwd = pick_types_forward(fwd, meg=True)

    vertno = [fwd['src'][0]['vertno'][0:15], fwd['src'][1]['vertno'][0:5]]
    stc_data = np.ones((len(vertno[0]) + len(vertno[1]), n_times))
    stc = SourceEstimate(stc_data, vertno, tmin=t_start, tstep=1.0 / sfreq)

    fwd_out = restrict_foward_to_stc(fwd, stc)

    assert_equal(fwd_out['sol']['ncol'], 20)
    assert_equal(fwd_out['src'][0]['nuse'], 15)
    assert_equal(fwd_out['src'][1]['nuse'], 5)
    assert_equal(fwd_out['src'][0]['vertno'], fwd['src'][0]['vertno'][0:15])
    assert_equal(fwd_out['src'][1]['vertno'], fwd['src'][1]['vertno'][0:5])


def test_restrict_foward_to_label():
    """Test restriction of source space to source SourceEstimate
    """
    fwd = read_forward_solution(fname, force_fixed=True)
    fwd = pick_types_forward(fwd, meg=True)

    label_path = op.join(data_path, 'MEG', 'sample', 'labels')
    labels = ['Aud-lh', 'Vis-rh']

    fwd_out = restrict_foward_to_label(fwd, labels, label_path)

    label_lh = read_label(op.join(label_path, labels[0] + '.label'))
    label_rh = read_label(op.join(label_path, labels[1] + '.label'))

    src_sel_lh = np.intersect1d(fwd['src'][0]['vertno'], label_lh.vertices)
    src_sel_lh = np.searchsorted(fwd['src'][0]['vertno'], src_sel_lh)

    src_sel_rh = np.intersect1d(fwd['src'][1]['vertno'], label_rh.vertices)
    src_sel_rh = np.searchsorted(fwd['src'][1]['vertno'], src_sel_rh)\
                 + len(fwd['src'][0]['vertno'])

    assert_equal(fwd_out['sol']['ncol'], len(src_sel_lh) + len(src_sel_rh))
    assert_equal(fwd_out['src'][0]['nuse'], len(src_sel_lh))
    assert_equal(fwd_out['src'][1]['nuse'], len(src_sel_rh))
    assert_equal(fwd_out['src'][0]['vertno'], src_sel_lh)
    assert_equal(fwd_out['src'][1]['vertno'], src_sel_rh)
