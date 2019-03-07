
# Author: Jaakko Leppakangas <jaeilepp@student.jyu.fi>
#
# License: BSD (3-clause)

import os.path as op
import pytest
import socket

from mne import pick_types
from mne import __file__ as _mne_file
from mne.utils import run_tests_if_main
from mne.datasets import testing
from mne.io.tests.test_raw import _test_raw_reader
from mne.io.cnt import read_raw_cnt

data_path = testing.data_path(download=False)
fname = op.join(data_path, 'CNT', 'scan41_short.cnt')

if socket.gethostname() == 'toothless':
    regular_path = op.join(op.dirname(_mne_file), '..', 'sandbox', 'data')
    confidential_path = op.join(regular_path, 'confidential', 'cnt')
    test_files = [fname,
                  op.join(regular_path, '914flankers.cnt'),
                  # op.join(confidential_path, 'BoyoAEpic1_16bit.cnt'),
                  # op.join(confidential_path, 'cont_67chan_resp_32bit.cnt'),
                  # op.join(confidential_path, 'cont_68chan_32bit.cnt'),
                  # op.join(confidential_path, 'SampleCNTFile_16bit.cnt')]
                  ]
else:
    test_files = [fname]


@testing.requires_testing_data
def test_data():
    """Test reading raw cnt files."""
    with pytest.warns(RuntimeWarning, match='number of bytes'):
        raw = _test_raw_reader(read_raw_cnt, montage=None, input_fname=fname,
                               eog='auto', misc=['NA1', 'LEFT_EAR'])
    eog_chs = pick_types(raw.info, eog=True, exclude=[])
    assert len(eog_chs) == 2  # test eog='auto'
    assert raw.info['bads'] == ['LEFT_EAR', 'VEOGR']  # test bads

    # XXX: the data has "05/10/200 17:35:31" so it is set to None
    assert raw.info['meas_date'] is None


@testing.requires_testing_data
@pytest.mark.parametrize('fname', test_files, ids=op.basename)
def test_explore_stuff_for_is5493(fname):
    SETUP_MEAN_AGE_OFFSET = 247

    # float  mean_age;       /* Mean age (Group files only)         */
    # float  stdev;          /* Std dev of age (Group files only)   */
    # short int n;           /* Number in group file                */
    # char   compfile[38];   /* Path and name of comparison file    */
    # float  SpectWinComp;   /* Spectral window compensation factor */
    # float  MeanAccuracy;   /* Average respose accuracy            */
    # float  MeanLatency;    /* Average response latency            */
    # char   sortfile[46];   /* Path and name of sort file          */
    # long   NumEvents;      /* Number of events in eventable       */
    # char   compoper;       /* Operation used in comparison        */
    # char   avgmode;        /* Set during online averaging         */
    # char   review;         /* Set during review of EEG data       */

    from struct import Struct
    from collections import namedtuple

    My_Struct = namedtuple('My_Struct',
                           ('mean_age stdev n compfile SpectWinComp '
                            'MeanAccuracy MeanLatency sortfile NumEvents '
                            'compoper avgmode review'))

    my_struct_parser = Struct('<ffh38sfff46slBBB')

    with open(fname, 'rb') as file:
        file.seek(SETUP_MEAN_AGE_OFFSET)
        data = my_struct_parser.unpack_from(file.read(my_struct_parser.size))
        records = My_Struct._make(data)

    # for name, val in records._asdict().items():
    #     print('{}: {}'.format(name, val))
    print('{} NumEvents: {}'.format(op.basename(fname), records.NumEvents))

    assert True


@testing.requires_testing_data
@pytest.mark.parametrize('fname', test_files, ids=op.basename)
def test_read_bunch_of_files(fname, recwarn):
    raw = read_raw_cnt(input_fname=fname, montage=None, preload=False)
    assert raw is not None


def _read_teeg(file_name):
    from struct import Struct, unpack, calcsize
    from collections import namedtuple

    # My_Struct = namedtuple('My_Struct',
    #                     ('mean_age stdev n compfile SpectWinComp '
    #                         'MeanAccuracy MeanLatency sortfile NumEvents '
    #                         'compoper avgmode review'))
    teeg_reader = Struct('<Bll')
    with open(file_name, 'rb') as fid:
        SETUP_EVENTTABLEPOS_OFFSET = 886
        fid.seek(SETUP_EVENTTABLEPOS_OFFSET)
        (event_table_pos,) = unpack('<l', fid.read(calcsize('<l')))

        print('mine :', event_table_pos)
        fid.seek(event_table_pos)
        data = teeg_reader.unpack(fid.read(teeg_reader.size))

    (event_type, total_lenght, xx) = data
    print(data)
    if event_type == 1:
        event_reader = Struct('<HBcl')  # EVENT type 1
    elif event_type == 2:
        event_reader = Struct('<HBclhhfccc')  # EVENT type 2
    else:
        raise RuntimeError('Event type can only be 1 or 2')

    return (event_reader, event_table_pos, total_lenght, xx)


@testing.requires_testing_data
# @pytest.mark.parametrize('fname', test_files, ids=op.basename)
# def test_read_events(fname, recwarn):
def test_read_events(recwarn):

    def _translating_function(offset, n_channels, n_bytes=2):
        # n_bytes is related to _get_cnt_info's data_format parameter
        # 'auto', 'int16', and 'int32'
        event_time = offset - 900 - (75 * n_channels)
        event_time //= n_channels * n_bytes
        return event_time - 1

    from mne import find_events
    raw = read_raw_cnt(input_fname=fname, montage=None, preload=False)
    events = find_events(raw)

    print(events)

    event_reader, event_table_pos, total_length, _ = _read_teeg(fname)
    with open(fname, 'rb') as fid:
        fid.seek(event_table_pos + 9)  # the real table stats at +9
        buffer = fid.read(total_length)

    # unsigned short StimType; /* range 0-65535                           */
    # unsigned char  KeyBoard; /* range 0-11 corresponding to fcn keys +1 */
    # char KeyPad_Accept;      /* 0->3 range 0-15 bit coded response pad  */
    #                          /* 4->7 values 0xd=Accept 0xc=Reject       */
    # long Offset;             /* file offset of event                    */
    # short Type;
    # short Code;
    # float Latency;
    # char EpochEvent;
    # char Accept2;
    # char Accuracy;

    from collections import namedtuple
    event_maker = namedtuple('Eevent_Type_2',
                             ('StimType KeyBoard KeyPad_Accept Offset Type Code '
                              'Latency EpochEvent Accept2 Accuracy'))

    my_events = [event_maker._make(data)
                 for data in event_reader.iter_unpack(buffer)]

    import numpy as np
    from numpy.testing import assert_array_equal

    print('description: ', [e.StimType for e in my_events])
    read_onset = np.array([e.Offset for e in my_events])
    print('read onset:', read_onset)
    transformed_onset = _translating_function(offset=read_onset,
                                              n_channels=raw.info['nchan'] - 1)
    print('transformed onset: ', transformed_onset)
    print('duration: ', [e.Latency for e in my_events])

    assert_array_equal(transformed_onset[:-1], events[:,0])


run_tests_if_main()
