# Authors: Christoph Dinh <chdinh@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#          Alexandre Gramfort <gramfort@nmr.mgh.harvard.edu>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

import mne
import sys

from mne.fiff.realtime import *

# create command client
cmd_client = CmdClientSocket('localhost', 4217)

# create data client
data_client = DataClientSocket('localhost', 4218)

# set data client alias -> for convinience (optional)
data_client.set_client_alias('mne_ex_python')

# example commands
help_info = cmd_client.send_command('help')
sys.stdout.write('### Help ###\n%s' % help_info)

clist_info = cmd_client.send_command('clist')
sys.stdout.write('### Client List ###\n%s' % clist_info)

con_info = cmd_client.send_command('conlist')
sys.stdout.write('### Connector List ###\n%s' % con_info)

#get the ID actual not necessary, we can use the set alias 'mne_ex_python'
data_client_id = data_client.get_client_id()
sys.stdout.write('ID %d\n' % data_client_id)

#read info
cmd_client.request_meas_info(data_client_id)
info = data_client.read_info()

#start measurement
cmd_client.request_meas(data_client_id);

is_running = True
max_nbuffers = 10
count = 0
while is_running:
    sys.stdout.write("read buffer...\n")
    
    kind, raw_buffer = data_client.read_raw_buffer(info['nchan'])
    
    if kind == FIFF.FIFF_DATA_BUFFER:
        #do processing here
        sys.stdout.write('buffer available\n')
        
        import pylab as pl
        pl.figure(1)
        pl.plot(raw_buffer.T)
        pl.show()
        #ToDo this figure is blocking -> replace it by non-blocking display
        
    elif kind == FIFF.FIFF_BLOCK_END and raw_buffer == FIFF.FIFFB_RAW_DATA:
        is_running = False

    count += 1;
    if count >= max_nbuffers: 
        cmd_client.stop_all()
        is_running = False

# close command and data socket
cmd_client.close()
data_client.close()
