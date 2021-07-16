# -*- coding: utf-8 -*-
"""Test reading of NEDF format."""
# Author: Tristan Stenner <nedf@nicht.dienstli.ch>
#
# License: BSD-3-Clause

import os.path as op

import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mne import find_events
from mne.io.constants import FIFF
from mne.io.nedf import read_raw_nedf, _parse_nedf_header
from mne.datasets import testing
from mne.io.tests.test_raw import _test_raw_reader

eeg_path = testing.data_path(download=False, verbose=True)
eegfile = op.join(eeg_path, 'nedf', 'testdata.nedf')

stimhdr = b"""
<nedf>
    <NEDFversion>1.3</NEDFversion>
    <NumberOfChannelsOfAccelerometer>%d</NumberOfChannelsOfAccelerometer>
    <EEGSettings>
        <TotalNumberOfChannels>4</TotalNumberOfChannels>
        <EEGSamplingRate>500</EEGSamplingRate>
        <EEGMontage><C>A</C><C>B</C><C>C</C><C>D</C></EEGMontage>
        <NumberOfRecordsOfEEG>11</NumberOfRecordsOfEEG>
    </EEGSettings>
    <STIMSettings/>
</nedf>\x00"""


@pytest.mark.parametrize('nacc', (0, 3))
def test_nedf_header_parser(nacc):
    """Test NEDF header parsing and dtype extraction."""
    with pytest.warns(RuntimeWarning, match='stim channels.*ignored'):
        info, dt, dt_last, n_samples, n_full = _parse_nedf_header(
            stimhdr % nacc)
    assert n_samples == 11
    assert n_full == 2
    nchan = 4
    assert info['nchan'] == nchan
    assert dt.itemsize == 200 + nacc * 2
    if nacc:
        assert dt.names[0] == 'acc'
        assert dt['acc'].shape == (nacc,)

    assert dt['data'].shape == (5,)  # blocks of 5 EEG samples each
    assert dt_last['data'].shape == (1,)  # plus one last extra one

    eegsampledt = dt['data'].subdtype[0]
    assert eegsampledt.names == ('eeg', 'stim', 'trig')
    assert eegsampledt['eeg'].shape == (nchan, 3)
    assert eegsampledt['stim'].shape == (2, nchan, 3)


def test_invalid_headers():
    """Test that invalid headers raise exceptions."""
    tpl = b"""<nedf>
        <NEDFversion>1.3</NEDFversion>
        <EEGSettings>
            %s
            <EEGMontage><C>A</C><C>B</C><C>C</C><C>D</C></EEGMontage>
        </EEGSettings>
    </nedf>\x00"""
    nchan = b'<TotalNumberOfChannels>4</TotalNumberOfChannels>'
    sr = b'<EEGSamplingRate>500</EEGSamplingRate>'
    hdr = {
        'null':
            b'No null terminator',
        'Unknown additional':
            (b'<a><NEDFversion>1.3</NEDFversion>' +
             b'<AdditionalChannelStatus>???</AdditionalChannelStatus></a>\x00'),  # noqa: E501
        'No EEG channels found':
            b'<a><NEDFversion>1.3</NEDFversion></a>\x00',
        'TotalNumberOfChannels not found':
            tpl % b'No nchan.',
        '!= channel count':
            tpl % (sr + b'<TotalNumberOfChannels>52</TotalNumberOfChannels>'),
        'EEGSamplingRate not found':
            tpl % nchan,
        'NumberOfRecordsOfEEG not found':
            tpl % (sr + nchan),
    }
    for match, invalid_hdr in hdr.items():
        with pytest.raises(RuntimeError, match=match):
            _parse_nedf_header(invalid_hdr)

    sus_hdrs = {
        'unsupported': b'<a><NEDFversion>25</NEDFversion></a>\x00',
        'tested': (
            b'<a><NEDFversion>1.3</NEDFversion><stepDetails>' +
            b'<DeviceClass>STARSTIM</DeviceClass></stepDetails></a>\x00'),
    }
    for match, sus_hdr in sus_hdrs.items():
        with pytest.warns(RuntimeWarning, match=match):
            with pytest.raises(RuntimeError, match='No EEG channels found'):
                _parse_nedf_header(sus_hdr)


@testing.requires_testing_data
def test_nedf_data():
    """Test reading raw NEDF files."""
    raw = read_raw_nedf(eegfile)
    nsamples = len(raw)
    assert nsamples == 32538

    events = find_events(raw, shortest_event=1)
    assert len(events) == 4
    assert_array_equal(events[:, 2], [1, 1, 1, 1])
    onsets = events[:, 0] / raw.info['sfreq']
    assert raw.info['sfreq'] == 500

    data_end = raw.get_data('Fp1', nsamples - 100, nsamples).mean()
    assert_allclose(data_end, .0176, atol=.01)
    assert_allclose(raw.get_data('Fpz', 0, 100).mean(), .0185, atol=.01)

    assert_allclose(onsets, [22.384, 38.238, 49.496, 63.15])
    assert raw.info['meas_date'].year == 2019
    assert raw.ch_names[2] == 'AF7'

    for ch in raw.info['chs'][:-1]:
        assert ch['kind'] == FIFF.FIFFV_EEG_CH
        assert ch['unit'] == FIFF.FIFF_UNIT_V
    assert raw.info['chs'][-1]['kind'] == FIFF.FIFFV_STIM_CH
    assert raw.info['chs'][-1]['unit'] == FIFF.FIFF_UNIT_V

    # full tests
    _test_raw_reader(read_raw_nedf, filename=eegfile)
