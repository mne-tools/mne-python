#===============================================================================
# Authors: Christoph Dinh <chdinh@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

#===============================================================================
# Imports
#===============================================================================

import socket
import sys
import time

import numpy as np

from .tag import FIFF
from .constants import FIFF


#===============================================================================
class ClientSocket(object):
    """Define Class ClientSocket."""
    #===========================================================================
    def __init__(self, p_Host, p_iPort, p_fTimeout = 0.5):
        """Method __init__."""
        # Create a TCP/IP socket
        self.m_ClientSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.m_fTimeout = p_fTimeout
        self.connect(p_Host, p_iPort)

    #===========================================================================
    def connect(self, p_Host, p_iPort):
        """Method connect."""
        t_serverAddress = (p_Host, p_iPort)
        sys.stdout.write("Connecting to %s port %s\n" % t_serverAddress)
        self.m_ClientSock.connect(t_serverAddress)

    #===========================================================================
    def close(self):
        """Method close."""
        self.m_ClientSock.close();


#===============================================================================
class CmdClientSocket(ClientSocket):
    """Define Class CmdClientSocket."""
    #===========================================================================
    def __init__(self, p_Host, p_iPort):
        """Method __init__."""
        super(CmdClientSocket, self).__init__(p_Host, p_iPort)

    #===========================================================================
    def sendCommand(self, p_sCommand):
        """Method sendCommand."""
        sys.stdout.write('%s\n' % p_sCommand)
        p_sCommand += '\n'
        self.m_ClientSock.sendall(p_sCommand)

        self.m_ClientSock.setblocking(0)
        t_Buf = []; chunk = ''; begin = time.time()
        while True:
            #if we got some data, then break after wait sec
            if t_Buf and time.time() - begin > self.m_fTimeout:
                break
            #if we got no data at all, wait a little longer
            elif time.time() - begin > self.m_fTimeout * 2:
                break
            try:
                chunk = self.m_ClientSock.recv(8192)
                if chunk:
                    t_Buf.append(chunk)
                    begin=time.time()
                else:
                    time.sleep(0.1)
            except:
                pass

        return ''.join(t_Buf)

    #===========================================================================
    def requestMeasInfo(self, p_aliasOrId):
        """requestMeasInfo."""
        p_aliasOrId = 'measinfo ' + p_aliasOrId
        self.sendCommand(p_aliasOrId)

    #===========================================================================
    def requestMeas(self, p_aliasOrId):
        """requestMeas."""
        p_aliasOrId = 'meas ' + p_aliasOrId
        self.sendCommand(p_aliasOrId)

    #===========================================================================
    def stopAll(self, p_aliasOrId):
        """stopAll."""
        self.sendCommand('stop-all')








#===============================================================================
class DataClientSocket(ClientSocket):
    """Define Class CmdClientSocket."""
    #===========================================================================
    def __init__(self, p_Host, p_iPort):
        """Method __init__."""
        super(DataClientSocket, self).__init__(p_Host, p_iPort)

        self.m_clientID = -1

    #===========================================================================
    def readInfo(self, p_sCommand):
        """Method readInfo reads the measurement info."""

        t_bReadMeasBlockStart = False
        t_bReadMeasBlockEnd = False
        #
        # Find the start
        #
        while t_bReadMeasBlockStart != True:
            tag = mne_rt_data_client.read_tag(obj.m_DataInputStream);

            if tag.kind == FIFF.FIFF_BLOCK_START and tag.data == FIFF.FIFFB_MEAS_INFO:
                sys.stdout.write('FIFF_BLOCK_START FIFFB_MEAS_INFO\n');
                t_bReadMeasBlockStart = true;

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

        while t_bReadMeasBlockEnd != True:
            tag = self.readTag();
            #
            #  megacq parameters
            #
            if tag.kind == FIFF.FIFF_BLOCK_START and tag.data == FIFF.FIFFB_DACQ_PARS:
                while tag.kind != FIFF.FIFF_BLOCK_END or tag.data != FIFF.FIFFB_DACQ_PARS:
                    tag = self.readTag()
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
                    tag = self.m_ClientSock.readTag();
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
                    tag = self.m_ClientSock.readTag();
                    if tag.kind == FIFF.FIFF_BLOCK_START and tag.data == FIFF.FIFFB_PROJ_ITEM:
                        proj = [];
                        while tag.kind != FIFF.FIFF_BLOCK_END or tag.data != FIFF.FIFFB_PROJ_ITEM:
                            tag = self.m_ClientSock.readTag()
                            if tag.kind == FIFF.FIFF_NAME:
                                proj['desc'] = tag.data
                            elif tag.kind == FIFF.FIFF_PROJ_ITEM_KIND:
                                proj['kind'] = int(tag.data)
                            elif tag.kind == FIFF.FIFF_NCHAN:
                                proj['data']['ncol'] = int(tag.data)
                            elif tag.kind == FIFF.FIFF_PROJ_ITEM_NVEC:
                                proj['data']['nrow'] = int(tag.data)
                            elif tag.kind == FIFF.FIFF_MNE_PROJ_ITEM_ACTIVE:
                                proj['active'] = bool(tag.data)
                            elif tag.kind == FIFF.FIFF_PROJ_ITEM_CH_NAME_LIST:
                                proj['data']['col_names'] = tag.data.split(':')
                            elif tag.kind == FIFF.FIFF_PROJ_ITEM_VECTORS:
                                proj['data']['data'] = tag.data

                        projs.append(proj)

                info['projs'] = projs

            #
            #    CTF compensation info
            #
            if tag.kind == FIFF.FIFF_BLOCK_START and tag.data == FIFF.FIFFB_MNE_CTF_COMP:
                comps = list()
                while tag.kind != FIFF.FIFF_BLOCK_END or tag.data != FIFF.FIFFB_MNE_CTF_COMP:
                    tag = self.readTag()
                    if tag.kind == FIFF.FIFF_BLOCK_START and tag.data == FIFF.FIFFB_MNE_CTF_COMP_DATA:
                        comp = [];
                        while tag.kind != FIFF.FIFF_BLOCK_END or tag.data != FIFF.FIFFB_MNE_CTF_COMP_DATA:
                            tag = self.readTag()
                            if tag.kind == FIFF.FIFF_MNE_CTF_COMP_KIND:
                                comp['ctfkind'] = tag.data
                            elif tag.kind == FIFF.FIFF_MNE_CTF_COMP_CALIBRATED:
                                comp['save_calibrated'] = tag.data
                            elif tag.kind == FIFF.FIFF_MNE_CTF_COMP_DATA:
                                comp['data'] = tag.data
                    comps.append(comp)
                info['comps'] = comps
            #
            #    Bad channels
            #
            if tag.kind == FIFF.FIFF_BLOCK_START and tag.data == FIFF.FIFFB_MNE_BAD_CHANNELS:
                bads = []
                while tag.kind != FIFF.FIFF_BLOCK_END or tag.data != FIFF.FIFFB_MNE_BAD_CHANNELS:
                    tag = self.readTag()
                    if tag.kind == FIFF.FIFF_MNE_CH_NAME_LIST:
                        info.bads = tag.data.split(':')

                info.bads = bads
            #
            #    General
            #
            if tag.kind == FIFF.FIFF_SFREQ:
                info.sfreq = float(tag.data)
            elif tag.kind == FIFF.FIFF_HIGHPASS:
                info.highpass = float(tag.data)
            elif tag.kind == FIFF.FIFF_LOWPASS:
                info.lowpass = float(tag.data)
            elif tag.kind == FIFF.FIFF_NCHAN:
                info.nchan = int(tag.data)
            elif tag.kind == FIFF.FIFF_MEAS_DATE:
                info.highpass = tag.data



            if tag.kind == FIFF.FIFF_CH_INFO:
                chs.append(tag.data);

            # END MEAS INFO
            if tag.kind == FIFF.FIFF_BLOCK_END and tag.data == FIFF.FIFFB_MEAS_INFO:
                sys.stdout.write('FIFF_BLOCK_END FIFFB_MEAS_INFO\n');
                t_bReadMeasBlockEnd = True

        info['chs'] = chs

        return info


    #===========================================================================
    def setClientAlias(self, alias):
        """Method setClientAlias."""

        self.sendFiffCommand(2, alias)#MNE_RT.MNE_RT_SET_CLIENT_ALIAS == 2


    #===========================================================================
    def getClientId(self):
        """Method setClientAlias."""

        if self.m_clientID == -1:

            self.sendFiffCommand(1) # MNE_RT.MNE_RT_GET_CLIENT_ID == 1

            # ID is send as answer
            tag = self.readTag();
            if tag.kind == 3701:  #FIFF.FIFF_MNE_RT_CLIENT_ID):
                self.m_clientID = tag.data

        return self.m_clientID









    #===========================================================================
    def sendFiffCommand(self, p_Cmd, p_data = None):
        """Method sendFiffCommand."""

        kind = 3700 #FIFF.FIFF_MNE_RT_COMMAND            = 3700;    	% Fiff Real-Time Command
        type = 0 #FIFF.FIFFT_VOID;
        size = 4
        if p_data is not None:
            size += len(p_data) # first 4 bytes are the command code
        next = 0


        msg = np.array(kind, dtype='>i4').tostring()
        msg += np.array(type, dtype='>i4').tostring()
        msg += np.array(size, dtype='>i4').tostring()
        msg += np.array(next, dtype='>i4').tostring()

        msg += np.array(p_Cmd, dtype='>i4').tostring()
        if p_data is not None:
            msg += np.array(p_data, dtype='>c').tostring()

        self.m_ClientSock.sendall(msg)

    #===========================================================================
    def readTag(self):
        """Method readTag."""
        #
        # read the tag info
        #
        tag = self.readTagInfo()

        #
        # read the tag data
        #
        tag = self.readTagData(tag)

        return tag


    def readTagData(self, p_tagInfo, pos = None):
        """Method readTag."""

        if pos is not None:
            self.m_ClientSock.recv(pos);

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
#                                     tag.data(i) = p_DataInputStream.readInt;%idata = fread(fid,dims(1)*dims(2),'int32=>int32');
#                                 end
#                    elif matrix_type == FIFF.FIFFT_JULIAN
#                                 tag.data = zeros(el_size/4, 1);
#                                 for i = 1:el_size/4
#                                     tag.data(i) = p_DataInputStream.readInt;%idata = fread(fid,dims(1)*dims(2),'int32=>int32');
#                                 end
                    if matrix_type == FIFF.FIFFT_FLOAT:
                        tmp = self.m_ClientSock.recv(el_size)
                        tag.data = np.fromstring(tmp, dtype='>f4') #fdata = fread(fid,dims(1)*dims(2),'single=>double');
#                        tag.data = swapbytes(tag.data);
                    else:
                        raise Exception('Cannot handle a matrix of type %d yet' % matrix_type)


                    # ToDo consider 3D case --> do that by using tag->size

                    dims[0] = np.fromstring(self.m_ClientSock.recv(4), dtype='>i4')
                    dims[1] = np.fromstring(self.m_ClientSock.recv(4), dtype='>i4')

                    ndim = np.fromstring(self.m_ClientSock.recv(4), dtype='>i4')

                    tag.data = tag.data.reshape(dims)
                else:
                    raise Exception('Cannot handle other than dense or sparse matrices yet')
            else:
                #
                #   All other data types
                #

                #
                #   Simple types
                #
                if tag.type == FIFF.FIFFT_INT:
                    tag.data = np.fromstring(self.m_ClientSock.recv(tag.size), dtype=">i4")
                elif tag.type == FIFF.FIFFT_FLOAT:
                    tag.data = np.fromstring(self.m_ClientSock.recv(tag.size), dtype=">f4")
                elif tag.type == FIFF.FIFFT_STRING:
                    tag.data = np.fromstring(self.m_ClientSock.recv(tag.size), dtype=">c")
                    tag.data = ''.join(tag.data)
                elif tag.type == FIFF.FIFFT_ID_STRUCT:
                    tag.data = dict()
                    tag.data['version'] = int(np.fromstring(self.m_ClientSock.recv(4), dtype=">i4"))
                    tag.data['version'] = int(np.fromstring(self.m_ClientSock.recv(4), dtype=">i4"))
                    tag.data['machid'] = np.fromstring(self.m_ClientSock.recv(8), dtype=">i4")
                    tag.data['secs'] = int(np.fromstring(self.m_ClientSock.recv(4), dtype=">i4"))
                    tag.data['usecs'] = int(np.fromstring(self.m_ClientSock.recv(4), dtype=">i4"))
                elif tag.type == FIFF.FIFFT_DIG_POINT_STRUCT:
                    tag.data = dict()
                    tag.data['kind'] = int(np.fromstring(self.m_ClientSock.recv(4), dtype=">i4"))
                    tag.data['ident'] = int(np.fromstring(self.m_ClientSock.recv(4), dtype=">i4"))
                    tag.data['r'] = np.fromstring(self.m_ClientSock.recv(12), dtype=">f4")
                    tag.data['coord_frame'] = 0
                elif tag.type == FIFF.FIFFT_COORD_TRANS_STRUCT:
                    tag.data = dict()
                    tag.data['from'] = int(np.fromstring(self.m_ClientSock.recv(4), dtype=">i4"))
                    tag.data['to'] = int(np.fromstring(self.m_ClientSock.recv(4), dtype=">i4"))
                    rot = np.fromstring(self.m_ClientSock.recv(36), dtype=">f4").reshape(3, 3)
                    move = np.fromstring(self.m_ClientSock.recv(12), dtype=">f4")
                    tag.data['trans'] = np.r_[np.c_[rot, move], np.array([[0], [0], [0], [1]]).T]
                    #
                    # Skip over the inverse transformation
                    # It is easier to just use inverse of trans in Matlab
                    #
                    self.m_ClientSock.recv(12 * 4) #fseek(fid,12*4,'cof');
                elif tag.type == FIFF.FIFFT_CH_INFO_STRUCT:
                    d = dict()
                    d['scanno'] = int(np.fromstring(self.m_ClientSock.recv(4), dtype=">i4"))
                    d['logno'] = int(np.fromstring(self.m_ClientSock.recv(4), dtype=">i4"))
                    d['kind'] = int(np.fromstring(self.m_ClientSock.recv(4), dtype=">i4"))
                    d['range'] = float(np.fromstring(self.m_ClientSock.recv(4), dtype=">f4"))
                    d['cal'] = float(np.fromstring(self.m_ClientSock.recv(4), dtype=">f4"))
                    d['coil_type'] = int(np.fromstring(self.m_ClientSock.recv(4), dtype=">i4"))
                    #
                    #   Read the coil coordinate system definition
                    #
                    d['loc'] = np.fromstring(self.m_ClientSock.recv(48), dtype=">f4")
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
                    tag.data['unit'] = int(np.fromstring(self.m_ClientSock.recv(4), dtype=">i4"))
                    tag.data['unit_mul'] = int(np.fromstring(self.m_ClientSock.recv(4), dtype=">i4"))
                    #
                    #   Handle the channel name
                    #
                    ch_name = np.fromstring(self.m_ClientSock.recv(16), dtype=">c")
                    #
                    # Omit nulls
                    #
                    tag.data['ch_name'] = ''.join(
                                        ch_name[:np.where(ch_name == '')[0][0]])

                else:
                    raise Exception('Unimplemented tag data type %s' % tag.type)

        # if tag.next ~= FIFF.FIFFV_NEXT_SEQ
        #     fseek(fid,tag.next,'bof');
        # end

        return tag

    def readTagInfo(self, pos = None):
        """Read Tag info"""

        if pos is not None:
            self.m_ClientSock.recv(pos);

        s = self.m_ClientSock.recv(4 * 4)
        tag = Tag(*struct.unpack(">iiii", s))

        return tag



#*******************************************************************************
# Here we go: Cmd Client

# create command client
t_cmdClient = CmdClientSocket('localhost', 4217)

# create data client
t_dataClient = DataClientSocket('localhost', 4218)


# set data client alias -> for convinience (optional)
t_dataClient.setClientAlias('mne_ex_python') # used in option 2 later on

# example commands
t_helpInfo = t_cmdClient.sendCommand('help')
sys.stdout.write('### Help ###\n%s' % t_helpInfo)
t_clistInfo = t_cmdClient.sendCommand('clist')
sys.stdout.write('### Client List ###\n%s' % t_clistInfo)
t_conInfo = t_cmdClient.sendCommand('conlist')
sys.stdout.write('### Connector List ###\n%s' % t_conInfo)

t_aliasOrId = t_dataClient.getClientId()


t_cmdClient.close()
t_dataClient.close()

