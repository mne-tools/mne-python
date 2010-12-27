import struct
import numpy as np

from .bunch import Bunch
from .constants import FIFF

class Tag(object):
    """docstring for Tag"""
    def __init__(self, kind, type, size, next):
        self.kind = kind
        self.type = type
        self.size = size
        self.next = next

    def __repr__(self):
        out = "kind: %s - type: %s - size: %s - next: %s" % (
                self.kind, self.type, self.size, self.next)
        if hasattr(self, 'data'):
            out += " - data: %s\n" % self.data
        else:
            out += "\n"
        return out

    @property
    def pos(self):
        return self.next

def read_tag_info(fid):
    s = fid.read(4*4)
    tag = Tag(*struct.unpack(">iiii", s))
    if tag.next == 0:
        fid.seek(tag.size, 1)
    else:
        fid.seek(tag.next, 0)
    return tag


def read_tag(fid, pos=None):
    if pos is not None:
        fid.seek(pos, 0)

    s = fid.read(4*4)
    tag = Tag(*struct.unpack(">iIii", s))

    #
    #   The magic hexadecimal values
    #
    is_matrix           = 4294901760 # ffff0000
    matrix_coding_dense = 16384      # 4000
    matrix_coding_CCS   = 16400      # 4010
    matrix_coding_RCS   = 16416      # 4020
    data_type           = 65535      # ffff
    #
    if tag.size > 0:
        matrix_coding = is_matrix & tag.type
        if matrix_coding != 0:
            raise ValueError, "matrix_coding not implemented"
            # XXX : todo
            pass
        else:
            #   All other data types

            #   Simple types

            if tag.type == FIFF.FIFFT_BYTE:
                tag.data = np.fromfile(fid, dtype=">B1", count=tag.size)
            elif tag.type == FIFF.FIFFT_SHORT:
                tag.data = np.fromfile(fid, dtype=">h2", count=tag.size/2)
            elif tag.type ==  FIFF.FIFFT_INT:
                tag.data = np.fromfile(fid, dtype=">i4", count=tag.size/4)
            elif tag.type ==  FIFF.FIFFT_USHORT:
                tag.data = np.fromfile(fid, dtype=">H2", count=tag.size/2)
            elif tag.type ==  FIFF.FIFFT_UINT:
                tag.data = np.fromfile(fid, dtype=">I4", count=tag.size/4)
            elif tag.type ==  FIFF.FIFFT_FLOAT:
                tag.data = np.fromfile(fid, dtype=">f4", count=tag.size/4)
            elif tag.type ==  FIFF.FIFFT_DOUBLE:
                tag.data = np.fromfile(fid, dtype=">f8", count=tag.size/8)
            elif tag.type ==  FIFF.FIFFT_STRING:
                tag.data = np.fromfile(fid, dtype=">c1", count=tag.size)
            elif tag.type ==  FIFF.FIFFT_DAU_PACK16:
                tag.data = np.fromfile(fid, dtype=">h2", count=tag.size/2)
            elif tag.type ==  FIFF.FIFFT_COMPLEX_FLOAT:
                tag.data = np.fromfile(fid, dtype=">f4", count=tag.size/4)
                tag.data = tag.data[::2] + 1j * tag.data[1::2]
            elif tag.type ==  FIFF.FIFFT_COMPLEX_DOUBLE:
                tag.data = np.fromfile(fid, dtype=">f8", count=tag.size/8)
                tag.data = tag.data[::2] + 1j * tag.data[1::2]
            #
            #   Structures
            #
            elif tag.type == FIFF.FIFFT_ID_STRUCT:
                tag.data = dict()
                tag.data['version'] = np.fromfile(fid, dtype=">i4", count=1)
                tag.data['version'] = np.fromfile(fid, dtype=">i4", count=1)
                tag.data['machid'] = np.fromfile(fid, dtype=">i4", count=2)
                tag.data['secs'] = np.fromfile(fid, dtype=">i4", count=1)
                tag.data['usecs'] = np.fromfile(fid, dtype=">i4", count=1)
            elif tag.type == FIFF.FIFFT_DIG_POINT_STRUCT:
                tag.data = dict()
                tag.data['kind'] = np.fromfile(fid, dtype=">i4", count=1)
                tag.data['ident'] = np.fromfile(fid, dtype=">i4", count=1)
                tag.data['r'] = np.fromfile(fid, dtype=">i4", count=3)
                tag.data['coord_frame'] = 0
            elif tag.type == FIFF.FIFFT_COORD_TRANS_STRUCT:
                tag.data = Bunch()
                tag.data['from'] = np.fromfile(fid, dtype=">i4", count=1)
                tag.data['to'] = np.fromfile(fid, dtype=">i4", count=1)
                rot = np.fromfile(fid, dtype=">f4", count=9).reshape(3, 3)
                move = np.fromfile(fid, dtype=">f4", count=3)
                tag.data['trans'] = np.r_[ np.c_[rot, move], [0, 0, 0, 1]]
                #
                # Skip over the inverse transformation
                # It is easier to just use inverse of trans in Matlab
                #
                fid.seek(12*4,1)
            elif tag.type == FIFF.FIFFT_CH_INFO_STRUCT:
                tag.data = Bunch()
                tag.data['scanno'] = np.fromfile(fid, dtype=">i4", count=1)
                tag.data['logno'] = np.fromfile(fid, dtype=">i4", count=1)
                tag.data['kind'] = np.fromfile(fid, dtype=">i4", count=1)
                tag.data['range'] = np.fromfile(fid, dtype=">i4", count=1)
                tag.data['cal'] = np.fromfile(fid, dtype=">i4", count=1)
                tag.data['coil_type'] = np.fromfile(fid, dtype=">i4", count=1)
                #
                #   Read the coil coordinate system definition
                #
                tag.data['loc'] = np.fromfile(fid, dtype=">f4", count=12)
                tag.data['coil_trans'] = None
                tag.data['eeg_loc'] = None
                tag.data['coord_frame'] = FIFF.FIFFV_COORD_UNKNOWN
                #
                #   Convert loc into a more useful format
                #
                loc = tag.data.loc
                kind = tag.data.kind
                if kind == FIFF.FIFFV_MEG_CH or kind == FIFF.FIFFV_REF_MEG_CH:
                    tag.data.coil_trans  = np.r_[ np.c_[loc[3:5], loc[6:8],
                                                        loc[9:11], loc[0:2] ],
                                                  [0, 0, 0, 1 ] ]
                    tag.data.coord_frame = FIFF.FIFFV_COORD_DEVICE
                elif tag.data.kind == FIFF.FIFFV_EEG_CH:
                    if np.norm(loc[3:5]) > 0:
                       tag.data.eeg_loc = np.c_[ loc[0:2], loc[3:5] ]
                    else:
                       tag.data.eeg_loc = loc[1:3]
                    tag.data.coord_frame = FIFF.FIFFV_COORD_HEAD
                #
                #   Unit and exponent
                #
                tag.data['unit'] = np.fromfile(fid, dtype=">i4", count=1)
                tag.data['unit_mul'] = np.fromfile(fid, dtype=">i4", count=1)
                #
                #   Handle the channel name
                #
                ch_name   = np.fromfile(fid, dtype=">c", count=16)
                #
                # Omit nulls
                #
                length = 16
                for k in range(16):
                    if ch_name(k) == 0:
                        length = k-1
                        break
                tag.data['ch_name'] = ch_name[1:length]
                import pdb; pdb.set_trace()

            elif tag.type == FIFF.FIFFT_OLD_PACK:
                 offset = np.fromfile(fid, dtype=">f4", count=1)
                 scale = np.fromfile(fid, dtype=">f4", count=1)
                 tag.data = np.fromfile(fid, dtype=">h2", count=(tag.size-8)/2)
                 tag.data = scale*tag.data + offset
            elif tag.type == FIFF.FIFFT_DIR_ENTRY_STRUCT:
                 tag.data = list()
                 for _ in range(tag.size/16-1):
                     s = fid.read(4*4)
                     tag.data.append(Tag(*struct.unpack(">iIii", s)))
            else:
                raise ValueError, 'Unimplemented tag data type %s' % tag.type

    if tag.next != FIFF.FIFFV_NEXT_SEQ:
        # f.seek(tag.next,0)
        fid.seek(tag.next, 1) # XXX : fix? pb when tag.next < 0

    return tag
