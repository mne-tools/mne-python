# Authors: Christoph Dinh <chdinh@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

import socket
import sys
import time
import struct
import numpy as np

from scipy import linalg
from .constants import FIFF
from .tag import Tag

class ClientSocket(object):
    """Define Class ClientSocket."""
    def __init__(self, host, port, timeout = 0.5):
        """Method __init__."""
        # Create a TCP/IP socket
        self._client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._timeout = timeout
        self.connect(host, port)

    def connect(self, host, port):
        """Method connect."""
        server_address = (host, port)
        sys.stdout.write("Connecting to %s port %s\n" % server_address)
        self._client_sock.connect(server_address)

    def close(self):
        """Method close."""
        self._client_sock.close();


class CmdClientSocket(ClientSocket):
    """Define Class CmdClientSocket."""     
    def __init__(self, host, port):
        """Method __init__."""
        super(CmdClientSocket, self).__init__(host, port)
  
    def send_command(self, command):
        """Method send_command."""
        sys.stdout.write('%s\n' % command)
        command += '\n'
        self._client_sock.sendall(command)
        
        self._client_sock.setblocking(0)
        buf, chunk, begin = [], '', time.time()
        while True:
            #if we got some data, then break after wait sec
            if buf and time.time() - begin > self._timeout:
                break
            #if we got no data at all, wait a little longer
            elif time.time() - begin > self._timeout * 2:
                break
            try:
                chunk = self._client_sock.recv(8192)
                if chunk:
                    buf.append(chunk)
                    begin=time.time()
                else:
                    time.sleep(0.1)
            except:
                pass
        
        return ''.join(buf)

    def request_meas_info(self, alias_or_id):
        """request_meas_info."""
        # ToDo parse whether alias_or_id is string or num
        cmd = 'measinfo %d' % alias_or_id
        self.send_command(cmd)

    def request_meas(self, alias_or_id):
        """request_meas."""
        cmd = 'meas %d' % alias_or_id
        self.send_command(cmd)

    def stop_all(self):
        """stop_all."""
        self.send_command('stop-all')


class DataClientSocket(ClientSocket):
    """Define Class CmdClientSocket."""     
    def __init__(self, host, port):
        """Method __init__."""
        super(DataClientSocket, self).__init__(host, port)
        self._client_id = -1

    def read_info(self):
        """Method readInfo reads the measurement info."""
        
        block_start_read = False
        block_end_read = False
        #
        # Find the start
        #
        #ToDo Time Out
        while block_start_read != True:
            tag = self.read_tag()
            if tag.kind == FIFF.FIFF_BLOCK_START and tag.data == FIFF.FIFFB_MEAS_INFO:
                sys.stdout.write('FIFF_BLOCK_START FIFFB_MEAS_INFO\n')
                block_start_read = True

        #
        # Parse until the endblock
        #
        info = dict()
        
        info['dev_head_t'] = None
        info['ctf_head_t'] = None
        info['dev_ctf_t'] = None
        info['dig'] = None
        info['bads'] = None
        info['projs'] = None
        info['comps'] = None
        info['acq_pars'] = None
        info['acq_stim'] = None
        info['chs'] = None
        chs = list()

        dev_head_t_read = False;
        ctf_head_t_read = False;
        
        while block_end_read != True:
            tag = self.read_tag();
            #
            #  megacq parameters
            #
            if tag.kind == FIFF.FIFF_BLOCK_START and tag.data == FIFF.FIFFB_DACQ_PARS:
                while tag.kind != FIFF.FIFF_BLOCK_END or tag.data != FIFF.FIFFB_DACQ_PARS:
                    tag = self.read_tag()
                    if tag.kind == FIFF.FIFF_DACQ_PARS:
                        info['acq_pars'] = tag.data
                    elif tag.kind == FIFF.FIFF_DACQ_STIM:
                        info['acq_stim'] = tag.data
            #
            #    Coordinate transformations if the HPI result block was not there
            #
            if tag.kind == FIFF.FIFF_COORD_TRANS:
                if dev_head_t_read == False:
                    info['dev_head_t'] = tag.data
                    dev_head_t_read = True;
                elif ctf_head_t_read == False:
                    info['ctf_head_t'] = tag.data;
                    ctf_head_t_read = True;
            #
            #    Polhemus data
            #
            if tag.kind == FIFF.FIFF_BLOCK_START and tag.data == FIFF.FIFFB_ISOTRAK:
                dig = [];
                while tag.kind != FIFF.FIFF_BLOCK_END or tag.data != FIFF.FIFFB_ISOTRAK:
                    tag = self.read_tag();
                    if tag.kind == FIFF.FIFF_DIG_POINT:
                        dig.append(tag.data)
                        dig[-1]['coord_frame'] = FIFF.FIFFV_COORD_HEAD
                info['dig'] = dig;
            #
            #    Projectors
            #
            if tag.kind == FIFF.FIFF_BLOCK_START and tag.data == FIFF.FIFFB_PROJ:
                projs = list()
                while tag.kind != FIFF.FIFF_BLOCK_END or tag.data != FIFF.FIFFB_PROJ:
                    tag = self.read_tag();
                    if tag.kind == FIFF.FIFF_BLOCK_START and tag.data == FIFF.FIFFB_PROJ_ITEM:
                        proj = dict();
                        data = dict();
                        while tag.kind != FIFF.FIFF_BLOCK_END or tag.data != FIFF.FIFFB_PROJ_ITEM:
                            tag = self.read_tag()
                            if tag.kind == FIFF.FIFF_NAME:
                                proj['desc'] = tag.data
                            elif tag.kind == FIFF.FIFF_PROJ_ITEM_KIND:
                                proj['kind'] = int(tag.data)
                            elif tag.kind == FIFF.FIFF_NCHAN:
                                data['ncol'] = int(tag.data)
                            elif tag.kind == FIFF.FIFF_PROJ_ITEM_NVEC:
                                data['nrow'] = int(tag.data)
                            elif tag.kind == FIFF.FIFF_MNE_PROJ_ITEM_ACTIVE:
                                proj['active'] = bool(tag.data)
                            elif tag.kind == FIFF.FIFF_PROJ_ITEM_CH_NAME_LIST:
                                data['col_names'] = tag.data.split(':')
                            elif tag.kind == FIFF.FIFF_PROJ_ITEM_VECTORS:
                                data['data'] = tag.data
                        
                        proj['data'] = data
                        projs.append(proj)
                    
                info['projs'] = projs
                
            #
            #    CTF compensation info
            #
            if tag.kind == FIFF.FIFF_BLOCK_START and tag.data == FIFF.FIFFB_MNE_CTF_COMP:
                comps = list()
                while tag.kind != FIFF.FIFF_BLOCK_END or tag.data != FIFF.FIFFB_MNE_CTF_COMP:
                    tag = self.read_tag()
                    if tag.kind == FIFF.FIFF_BLOCK_START and tag.data == FIFF.FIFFB_MNE_CTF_COMdata:
                        comp = [];
                        while tag.kind != FIFF.FIFF_BLOCK_END or tag.data != FIFF.FIFFB_MNE_CTF_COMdata:
                            tag = self.read_tag()
                            if tag.kind == FIFF.FIFF_MNE_CTF_COMP_KIND:
                                comp['ctfkind'] = tag.data
                            elif tag.kind == FIFF.FIFF_MNE_CTF_COMP_CALIBRATED:
                                comp['save_calibrated'] = tag.data
                            elif tag.kind == FIFF.FIFF_MNE_CTF_COMdata:
                                comp['data'] = tag.data
                    comps.append(comp)
                info['comps'] = comps
            #
            #    Bad channels
            #
            if tag.kind == FIFF.FIFF_BLOCK_START and tag.data == FIFF.FIFFB_MNE_BAD_CHANNELS:
                bads = []
                while tag.kind != FIFF.FIFF_BLOCK_END or tag.data != FIFF.FIFFB_MNE_BAD_CHANNELS:
                    tag = self.read_tag()
                    if tag.kind == FIFF.FIFF_MNE_CH_NAME_LIST:
                        bads = tag.data.split(':')
                        
                info['bads'] = bads
            #
            #    General
            #
            if tag.kind == FIFF.FIFF_SFREQ:
                info['sfreq'] = float(tag.data)
            elif tag.kind == FIFF.FIFF_HIGHPASS:
                info['highpass'] = float(tag.data)
            elif tag.kind == FIFF.FIFF_LOWPASS:
                info['lowpass'] = float(tag.data)
            elif tag.kind == FIFF.FIFF_NCHAN:
                info['nchan'] = int(tag.data)
            elif tag.kind == FIFF.FIFF_MEAS_DATE:
                info['highpass'] = tag.data
            
            if tag.kind == FIFF.FIFF_CH_INFO:
                chs.append(tag.data);
            
            # END MEAS INFO
            if tag.kind == FIFF.FIFF_BLOCK_END and tag.data == FIFF.FIFFB_MEAS_INFO:
                sys.stdout.write('FIFF_BLOCK_END FIFFB_MEAS_INFO\n'); 
                block_end_read = True
            
        info['chs'] = chs
        
        return info
    
    def read_raw_buffer(self, nchan):
        """Method read_raw_buffer."""
        data = []
        kind = []

        tag = self.read_tag();
            
        kind = tag.kind
        
        if tag.kind == FIFF.FIFF_DATA_BUFFER:
            nsamples = (tag.size)/4/nchan
 #           sys.stdout.write("size %d\n" % len(tag.data))
            data = tag.data.reshape(nchan, nsamples)
        else:
            data = tag.data
            
        return (kind, data)


    def set_client_alias(self, alias):
        """Method set_client_alias."""
        self.send_fiff_command(2, alias) # MNE_RT.MNE_RT_SET_CLIENT_ALIAS == 2
    
    def get_client_id(self):
        """Method set_client_alias."""
            
        if self._client_id == -1:

            self.send_fiff_command(1) # MNE_RT.MNE_RT_GET_CLIENT_ID == 1

            # ID is send as answer
            tag = self.read_tag();
            if tag.kind == 3701:  #FIFF.FIFF_MNE_RT_CLIENT_ID):
                self._client_id = tag.data
                
        return self._client_id

    def send_fiff_command(self, command, data = None):
        """Method send_fiff_command."""

        kind = 3700 #FIFF.FIFF_MNE_RT_COMMAND            = 3700;    	% Fiff Real-Time Command
        type = 0 #FIFF.FIFFT_VOID;
        size = 4
        if data is not None:
            size += len(data) # first 4 bytes are the command code
        next = 0
        
        
        #msg = "%d%d%d%d" % (kind, type, size, next)#unfortunately with this its not working: 
        msg = np.array(kind, dtype='>i4').tostring()
        msg += np.array(type, dtype='>i4').tostring()
        msg += np.array(size, dtype='>i4').tostring()
        msg += np.array(next, dtype='>i4').tostring()
        
        msg += np.array(command, dtype='>i4').tostring()
        if data is not None:
            msg += np.array(data, dtype='>c').tostring()

        self._client_sock.sendall(msg)
        
    def read_tag(self):
        """Method read_tag."""
        #set socket to blocking mode
        self._client_sock.setblocking(1)
        #
        # read the tag info
        #
        tag = self.read_tag_info()

        #
        # read the tag data
        #
        tag = self.read_tag_data(tag)
        
        return tag
        
    def read_tag_data(self, p_tagInfo, pos = None):
        """Method read_tag."""

        if pos is not None:
            self._client_sock.recv(pos);

        tag = p_tagInfo;

        #
        #   The magic hexadecimal values
        #
        is_matrix           = 4294901760; # ffff0000
        matrix_coding_dense = 16384;      # 4000
        matrix_coding_CCS   = 16400;      # 4010
        matrix_coding_RCS   = 16416;      # 4020
        data_type           = 65535;      # ffff
        #
        if tag.size > 0:
            matrix_coding = is_matrix & tag.type
            if matrix_coding != 0:
                matrix_coding = matrix_coding >> 16
                #
                #   Matrices
                #
                if matrix_coding == matrix_coding_dense:
                    #
                    # Find dimensions and return to the beginning of tag data
                    #
                    
# Check can't be done in real-time --> moved to the end for reshape
#                         pos = ftell(fid);
#                         fseek(fid,tag.size-4,'cof');
#                         ndim = fread(fid,1,'int32');
#                         fseek(fid,-(ndim+1)*4,'cof');
#                         dims = fread(fid,ndim,'int32');
#                         %
#                         % Back to where the data start
#                         %
#                         fseek(fid,pos,'bof');

                    matrix_type = data_type & tag.type;

                    el_size = tag.size - 3*4; # 3*4 --> case 2D matrix; ToDo calculate el_size through

                    
#                    if matrix_type == FIFF.FIFFT_INT
#                                 tag.data = zeros(el_size/4, 1);
#                                 for i = 1:el_size/4
#                                     tag.data(i) = dataInputStream.readInt;%idata = fread(fid,dims(1)*dims(2),'int32=>int32');
#                                 end
#                    elif matrix_type == FIFF.FIFFT_JULIAN
#                                 tag.data = zeros(el_size/4, 1);
#                                 for i = 1:el_size/4
#                                     tag.data(i) = dataInputStream.readInt;%idata = fread(fid,dims(1)*dims(2),'int32=>int32');
#                                 end
                    if matrix_type == FIFF.FIFFT_FLOAT:
                        tmp = self._client_sock.recv(el_size)
                        tag.data = np.fromstring(tmp, dtype='>f4') #fdata = fread(fid,dims(1)*dims(2),'single=>double');
#                        tag.data = swapbytes(tag.data);
                    else:#Raise no exception during real-time acquisition
                        sys.stdout.write('Cannot handle a matrix of type %d yet\n' % matrix_type)
                    
                    
                    # ToDo consider 3D case --> do that by using tag->size
                    dims = list()
                    dims.append(np.fromstring(self._client_sock.recv(4), dtype='>i4'))
                    dims.append(np.fromstring(self._client_sock.recv(4), dtype='>i4'))
                    
                    ndim = np.fromstring(self._client_sock.recv(4), dtype='>i4')
                    
                    tag.data = tag.data.reshape(dims)
                else:#Raise no exception during real-time acquisition
                    sys.stdout.write('Cannot handle other than dense or sparse matrices yet\n')
            else:
                #
                #   All other data types
                #
                
                #
                #   Simple types
                #
                if tag.type == FIFF.FIFFT_INT:
                    tag.data = np.fromstring(self._client_sock.recv(tag.size), dtype=">i4")
                elif tag.type == FIFF.FIFFT_FLOAT:
                    if tag.size < 50000:
                        tag.data = np.fromstring(self._client_sock.recv(tag.size), dtype=">f4")
                    else:
                        total_len=0;total_data=''
                        sock_data='';recv_size=8192
                        while total_len < tag.size:
                            if tag.size - total_len < recv_size:
                                recv_size = tag.size - total_len;
                            sock_data = self._client_sock.recv(recv_size)
                            total_data += sock_data
                            total_len=len(total_data)
                        tag.data = np.fromstring(total_data, dtype=">f4")
                        
                elif tag.type == FIFF.FIFFT_STRING:
                    tag.data = np.fromstring(self._client_sock.recv(tag.size), dtype=">c")
                    tag.data = ''.join(tag.data)                          
                elif tag.type == FIFF.FIFFT_ID_STRUCT:
                    tag.data = dict()
                    tag.data['version'] = int(np.fromstring(self._client_sock.recv(4), dtype=">i4"))
                    tag.data['version'] = int(np.fromstring(self._client_sock.recv(4), dtype=">i4"))
                    tag.data['machid'] = np.fromstring(self._client_sock.recv(8), dtype=">i4")
                    tag.data['secs'] = int(np.fromstring(self._client_sock.recv(4), dtype=">i4"))
                    tag.data['usecs'] = int(np.fromstring(self._client_sock.recv(4), dtype=">i4"))
                elif tag.type == FIFF.FIFFT_DIG_POINT_STRUCT:
                    tag.data = dict()
                    tag.data['kind'] = int(np.fromstring(self._client_sock.recv(4), dtype=">i4"))
                    tag.data['ident'] = int(np.fromstring(self._client_sock.recv(4), dtype=">i4"))
                    tag.data['r'] = np.fromstring(self._client_sock.recv(12), dtype=">f4")
                    tag.data['coord_frame'] = 0
                elif tag.type == FIFF.FIFFT_COORD_TRANS_STRUCT:
                    tag.data = dict()
                    tag.data['from'] = int(np.fromstring(self._client_sock.recv(4), dtype=">i4"))
                    tag.data['to'] = int(np.fromstring(self._client_sock.recv(4), dtype=">i4"))
                    rot = np.fromstring(self._client_sock.recv(36), dtype=">f4").reshape(3, 3)
                    move = np.fromstring(self._client_sock.recv(12), dtype=">f4")
                    tag.data['trans'] = np.r_[np.c_[rot, move], np.array([[0], [0], [0], [1]]).T]
                    #
                    # Skip over the inverse transformation
                    # It is easier to just use inverse of trans in Matlab
                    #
                    self._client_sock.recv(12 * 4) #fseek(fid,12*4,'cof');
                elif tag.type == FIFF.FIFFT_CH_INFO_STRUCT:
                    d = dict()
                    d['scanno'] = int(np.fromstring(self._client_sock.recv(4), dtype=">i4"))
                    d['logno'] = int(np.fromstring(self._client_sock.recv(4), dtype=">i4"))
                    d['kind'] = int(np.fromstring(self._client_sock.recv(4), dtype=">i4"))
                    d['range'] = float(np.fromstring(self._client_sock.recv(4), dtype=">f4"))
                    d['cal'] = float(np.fromstring(self._client_sock.recv(4), dtype=">f4"))
                    d['coil_type'] = int(np.fromstring(self._client_sock.recv(4), dtype=">i4"))
                    #
                    #   Read the coil coordinate system definition
                    #
                    d['loc'] = np.fromstring(self._client_sock.recv(48), dtype=">f4")
                    d['coil_trans'] = None
                    d['eeg_loc'] = None
                    d['coord_frame'] = FIFF.FIFFV_COORD_UNKNOWN
                    tag.data = d
                    #
                    #   Convert loc into a more useful format
                    #
                    loc = tag.data['loc']
                    kind = tag.data['kind']
                    if kind == FIFF.FIFFV_MEG_CH or kind == FIFF.FIFFV_REF_MEG_CH:
                        tag.data['coil_trans'] = np.r_[np.c_[loc[3:6], loc[6:9],
                                                            loc[9:12], loc[0:3]],
                                            np.array([0, 0, 0, 1]).reshape(1, 4)]
                        tag.data['coord_frame'] = FIFF.FIFFV_COORD_DEVICE
                    elif tag.data['kind'] == FIFF.FIFFV_EEG_CH:
                        if linalg.norm(loc[3:6]) > 0.:
                            tag.data['eeg_loc'] = np.c_[loc[0:3], loc[3:6]]
                        else:
                            tag.data['eeg_loc'] = loc[0:3]
                        tag.data['coord_frame'] = FIFF.FIFFV_COORD_HEAD
                    #
                    #   Unit and exponent
                    #
                    tag.data['unit'] = int(np.fromstring(self._client_sock.recv(4), dtype=">i4"))
                    tag.data['unit_mul'] = int(np.fromstring(self._client_sock.recv(4), dtype=">i4"))
                    #
                    #   Handle the channel name
                    #
                    ch_name = np.fromstring(self._client_sock.recv(16), dtype=">c")
                    #
                    # Omit nulls
                    #
                    tag.data['ch_name'] = ''.join(
                                        ch_name[:np.where(ch_name == '')[0][0]])

                else:#Raise no exception during real-time acquisition
                    sys.stdout.write('Unimplemented tag data type %s\n' % tag.type)

        # if tag.next ~= FIFF.FIFFV_NEXT_SEQ
        #     fseek(fid,tag.next,'bof');
        # end

        return tag

    def read_tag_info(self, pos = None):
        """Read Tag info"""
        
        if pos is not None:
            self._client_sock.recv(pos);
        
        s = self._client_sock.recv(4 * 4)
        tag = Tag(*struct.unpack(">iiii", s))
        
        return tag