# Released under The MIT License (MIT)
# http://opensource.org/licenses/MIT
# Copyright (c) 2013 SCoT Development Team

from __future__ import print_function

from uuid import uuid4
import os
import sys
import subprocess

import numpy as np

if not hasattr(__builtins__, 'FileNotFoundError'):
    # PY27: subprocess.Popen raises OSError instead of FileNotFoundError
    FileNotFoundError = OSError


binica_binary = os.path.dirname(os.path.abspath(__file__)) + '/binica/ica_linux'

#noinspection PyNoneFunctionAssignment,PyTypeChecker
def binica(data, binary=binica_binary):
    """ Simple wrapper for the BINICA binary.
    
    This function calculates the ICA transformation using the Infomax algorithm implemented in BINICA.

    BINICA is bundled with EEGLAB, or can be downloaded from here:
        http://sccn.ucsd.edu/eeglab/binica/
        
    This function attempts to automatically download and extract the BINICA binary.
    
    By default the binary is expected to be "binica/ica_linux" relative
    to the directory where this module lies (typically scot/binica/ica_linux)
    
    Parameters
    ----------
    data : array-like, shape = [n_samples, n_channels]
        EEG data set
    binary : str
        Full path to the binica binary
        
    Returns
    -------
    w : array, shape = [n_channels, n_channels]
        ICA weights matrix
    s : array, shape = [n_channels, n_channels]
        Sphering matrix
        
    Notes
    -----
    The unmixing matrix is obtained by multiplying U = dot(s, w)
    """
    
    check_binary_(binary)
    
    data = np.array(data, dtype=np.float32)
    
    nframes, nchans = data.shape
    
    uid = uuid4()

    scriptfile = 'binica-%s.sc' % uid
    datafile = 'binica-%s.fdt' % uid
    weightsfile = 'binica-%s.wts' % uid
    #weightstmpfile = 'binicatmp-%s.wts' % uid
    spherefile = 'binica-%s.sph' % uid
    
    config = {'DataFile': datafile,
              'WeightsOutFile': weightsfile,
              'SphereFile': spherefile,
              'chans': nchans,
              'frames': nframes,
              'extended': 1}
    #    config['WeightsTempFile'] = weightstmpfile

    # create data file
    f = open(datafile, 'wb')
    data.tofile(f)
    f.close()

    # create script file        
    f = open(scriptfile, 'wt')
    for h in config:
        print(h, config[h], file=f)
    f.close()
    
    # flush output streams otherwise things printed before might appear after the ICA output.
    sys.stdout.flush()
    sys.stderr.flush()

    if os.path.exists(binary):
        with open(scriptfile) as sc:
            try:
                proc = subprocess.Popen(binary, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=sc)
                print('waiting for binica to finish...')
                proc.wait()
                #print('binica output:')
                #print(proc.stdout.read().decode())
                proc.stdout.close()
            except FileNotFoundError:
                raise RuntimeError('The BINICA binary ica_linux exists in the file system but could not be executed. '
                                   'This indicates that 32 bit libraries are not installed on the system.')
    else:
        raise RuntimeError('the binary is not there!?')
    
    os.remove(scriptfile)
    os.remove(datafile)    
    
    # read weights
    f = open(weightsfile, 'rb')
    weights = np.fromfile(f, dtype=np.float32)
    f.close()
    weights = np.reshape(weights, (nchans,nchans))
    
#    os.remove(weightstmpfile)
    os.remove(weightsfile)
    
    # read sphering matrix
    f = open( spherefile, 'rb' )
    sphere = np.fromfile(f, dtype=np.float32)
    f.close()
    sphere = np.reshape(sphere, (nchans,nchans))    
    
    os.remove(spherefile)
    
    return weights, sphere
    

def check_binary_(binary):
    """check if binary is available, and try to download it if not"""
    
    if os.path.exists(binary):
        print(binary, 'found')
        return

    url = 'http://sccn.ucsd.edu/eeglab/binica/binica.zip'
    print(binary+' not found. Trying to download from '+url)

    path = os.path.dirname(binary)
        
    if not os.path.exists(path):
        os.makedirs(path)
   
    try: 
        # Python 3
        from urllib.request import urlretrieve as urlretrieve
    except ImportError:
        # Python 2.7
        from urllib import urlretrieve as urlretrieve
    import zipfile
    import stat

    urlretrieve(url, path + '/binica.zip')

    if not os.path.exists(path + '/binica.zip'):
        raise RuntimeError('Error downloading binica.zip.')

    print('unzipping', path + '/binica.zip')

    with zipfile.ZipFile(path + '/binica.zip') as tgz:
        tgz.extractall(path + '/..')
    
    if not os.path.exists(binary):
        raise RuntimeError(binary + ' not found, even after extracting binica.zip.')

    mode = os.stat(binary).st_mode
    os.chmod(binary, mode | stat.S_IXUSR)
