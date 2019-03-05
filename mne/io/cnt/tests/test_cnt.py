
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
    test_files = [fname, op.join(op.dirname(_mne_file), '..',
                                 'sandbox', 'data', '914flankers.cnt')]
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

    from struct import Struct
    from collections import namedtuple

    My_Struct = namedtuple('My_Struct',
                           ('mean_age stdev n compfile SpectWinComp '
                            'MeanAccuracy MeanLatency sortfile NumEvents'))

    my_struct_parser = Struct('ffh38sfff46sl')

    with open(fname, 'rb') as file:
        file.seek(SETUP_MEAN_AGE_OFFSET)
        records = My_Struct._make(my_struct_parser.unpack_from(
            file.read(my_struct_parser.size)))

    print(records)
    assert True


run_tests_if_main()
