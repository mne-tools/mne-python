# -*- coding: utf-8 -*-
"""Test reading of NEDF format."""
# Author: Tristan Stenner <nedf@nicht.dienstli.ch>
#
# License: BSD (3-clause)

import os.path as op

import pytest
from numpy.testing import (assert_allclose, assert_equal, assert_raises)

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
    </EEGSettings>
    <STIMSettings/>
</nedf>\x00"""


@pytest.mark.parametrize('nacc', (0, 3))
def test_nedf_header_parser(nacc):
    """Test NEDF header parsing and dtype extraction."""
    info, dt = _parse_nedf_header(stimhdr % nacc)
    nchan = 4
    assert_equal(info['nchan'], nchan)
    assert_equal(dt.itemsize, 200 + nacc * 2)
    if nacc:
        assert_equal(dt.names[0], 'acc')
        assert_equal(dt['acc'].shape, (nacc,))

    assert_equal(dt['data'].shape, (5,))  # blocks of 5 EEG samples each

    eegsampledt = dt['data'].subdtype[0]
    assert_equal(eegsampledt.names, ('eeg', 'stim', 'trig'))
    assert_equal(eegsampledt['eeg'].shape, (nchan, 3))
    assert_equal(eegsampledt['stim'].shape, (2, nchan, 3))


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
    hdr = [b'No null terminator',
           # AdditionalChannelStatus set
           b'<a><NEDFversion>1.3</NEDFversion>' +
           b'<AdditionalChannelStatus>???</AdditionalChannelStatus></a>\x00',
           # missing <EEGSettings>
           b'<a><NEDFversion>1.3</NEDFversion></a>\x00',
           tpl % b'No nchan.',
           # wrong TotalNumberOfChannels
           tpl % (sr + b'<TotalNumberOfChannels>52</TotalNumberOfChannels>'),
           # missing EEGSamplingRate
           tpl % nchan,
           ]

    for invalid_hdr in hdr:
        assert_raises(RuntimeError, _parse_nedf_header, invalid_hdr)

    sus_hdrs = [b'<a><NEDFversion>25</NEDFversion></a>\x00',
                b'<a><NEDFversion>1.3</NEDFversion><stepDetails>' +
                b'<DeviceClass>STARSTIM</DeviceClass></stepDetails></a>\x00',
                ]
    for sus_hdr in sus_hdrs:
        assert_raises(RuntimeWarning, _parse_nedf_header, sus_hdr)


@testing.requires_testing_data
def test_nedf_data():
    """Test reading raw NEDF files."""
    raw = read_raw_nedf(eegfile)

    assert_equal(len(raw.annotations), 4)
    assert_equal(raw.info['sfreq'], 500)

    nsamples = len(raw)
    assert_equal(nsamples, 32535)
    data_end = raw.get_data('Fp1', nsamples - 100, nsamples).mean()
    assert_allclose(data_end, .0176, atol=.01)
    assert_allclose(raw.get_data('Fpz', 0, 100).mean(), .0185, atol=.01)

    assert_allclose(raw.annotations.onset, [22.384, 38.238, 49.496, 63.15])
    assert_equal(raw.info['meas_date'].year, 2019)
    assert_equal(raw.ch_names[2], 'AF7')

    for ch in raw.info['chs']:
        assert_equal(ch['kind'], FIFF.FIFFV_EEG_CH)
        assert_equal(ch['unit'], FIFF.FIFF_UNIT_V)

    # full tests
    _test_raw_reader(read_raw_nedf, filename=eegfile, test_preloading=False)
