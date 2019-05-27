# Author: Joan Massich <mailsik@gmail.com>
#
# License: BSD (3-clause)

from struct import Struct
from collections import namedtuple
from math import modf
from datetime import datetime

from ...utils import warn


def _read_teeg(f, teeg_offset):
    """
    Read TEEG structure from an open CNT file.

    # from TEEG structure in http://paulbourke.net/dataformats/eeg/
    typedef struct {
       char Teeg;       /* Either 1 or 2                    */
       long Size;       /* Total length of all the events   */
       long Offset;     /* Hopefully always 0               */
    } TEEG;
    """
    # we use a more descriptive names based on TEEG doc comments
    Teeg = namedtuple('Teeg', 'event_type total_length offset')
    teeg_parser = Struct('<Bll')

    f.seek(teeg_offset)
    return Teeg(*teeg_parser.unpack(f.read(teeg_parser.size)))


CNTEventType1 = namedtuple('CNTEventType1',
                           ('StimType KeyBoard KeyPad_Accept Offset'))
# typedef struct {
#    unsigned short StimType;     /* range 0-65535                           */
#    unsigned char  KeyBoard;     /* range 0-11 corresponding to fcn keys +1 */
#    char KeyPad_Accept;          /* 0->3 range 0-15 bit coded response pad  */
#                                 /* 4->7 values 0xd=Accept 0xc=Reject       */
#    long Offset;                 /* file offset of event                    */
# } EVENT1;


CNTEventType2 = namedtuple('CNTEventType2',
                           ('StimType KeyBoard KeyPad_Accept Offset Type '
                            'Code Latency EpochEvent Accept2 Accuracy'))
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


# needed for backward compat: EVENT type 3 has the same structure as type 2
CNTEventType3 = namedtuple('CNTEventType3',
                           ('StimType KeyBoard KeyPad_Accept Offset Type '
                            'Code Latency EpochEvent Accept2 Accuracy'))


def _get_event_parser(event_type):
    if event_type == 1:
        event_maker = CNTEventType1
        struct_pattern = '<HBcl'
    elif event_type == 2:
        event_maker = CNTEventType2
        struct_pattern = '<HBclhhfccc'
    elif event_type == 3:
        event_maker = CNTEventType3
        struct_pattern = '<HBclhhfccc'  # Same as event type 2
    else:
        raise ValueError('unknown CNT even type %s' % event_type)

    def parser(buffer):
        struct = Struct(struct_pattern)
        for chunk in struct.iter_unpack(buffer):
            yield event_maker(*chunk)

    return parser


def _session_date_2_meas_date(session_date, date_format):
    try:
        frac_part, int_part = modf(datetime
                                   .strptime(session_date, date_format)
                                   .timestamp())
    except ValueError:
        warn('  Could not parse meas date from the header. Setting to None.')
        return None
    else:
        return (int_part, frac_part)
