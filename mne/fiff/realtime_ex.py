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



# close command and data socket
cmd_client.close()
data_client.close()

