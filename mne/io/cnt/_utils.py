# Author: Joan Massich <mailsik@gmail.com>
#
# License: BSD (3-clause)

from struct import Struct
from collections import namedtuple


def _read_TEEG(f, teeg_offset):
    """
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
    return Teeg._make(teeg_parser.unpack(f.read(teeg_parser.size)))
    # XXX: maybe add the warning if offset is not 0


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


CNTEventType3 = CNTEventType2


def iter_parse_event_type_1(buffer):
    event_1_parser = Struct('<HBcl')  # EVENT type 1
    for chunk in event_1_parser.iter_unpack(buffer):
        yield CNTEventType1._make(chunk)


def iter_parse_event_type_2(buffer):
    event_2_parser = Struct('<HBclhhfccc')  # EVENT type 2
    for chunk in event_2_parser.iter_unpack(buffer):
        yield CNTEventType2._make(chunk)


def iter_parse_event_type_3(buffer):
    event_3_parser = Struct('<HBclhhfccc')  # EVENT type 3 is the same as 2
    for chunk in event_3_parser.iter_unpack(buffer):
        yield CNTEventType3._make(chunk)


def _get_event_parser(event_type):
    if event_type == 1:
        return iter_parse_event_type_1
    elif event_type == 2:
        return iter_parse_event_type_2
    elif event_type == 3:
        return iter_parse_event_type_3
    else:
        raise ValueError('unknown CNT even type')
