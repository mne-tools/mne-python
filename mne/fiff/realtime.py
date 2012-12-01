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
        while 1:
            #if you got some data, then break after wait sec
            if t_Buf and time.time() - begin > self.m_fTimeout:
                break
            #if you got no data at all, wait a little longer
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




#*******************************************************************************
# Here we go: Cmd Client
t_cmdClientSocket = CmdClientSocket('localhost', 10034)

data = t_cmdClientSocket.sendCommand('meas')
sys.stdout.write('Server reply: %s\n' % data)

t_cmdClientSocket.close()
