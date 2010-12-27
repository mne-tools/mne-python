import numpy as np

from .bunch import Bunch
from .constants import FIFF
from .open import fiff_open
from .ctf import read_ctf_comp
from .tag import read_tag, find_tag
from .tree import dir_tree_find


def read_proj(fid, node):
    """
    [ projdata ] = fiff_read_proj(fid,node)

     Read the SSP data under a given directory node

    """

    projdata = []

    #   Locate the projection data
    nodes = dir_tree_find(node, FIFF.FIFFB_PROJ)
    if len(nodes) == 0:
       return projdata

    tag = find_tag(fid, nodes[0], FIFF.FIFF_NCHAN)
    if tag is not None:
        global_nchan = tag.data

    items = dir_tree_find(nodes[0], FIFF.FIFFB_PROJ_ITEM)
    for i in range(len(items)):

        #   Find all desired tags in one item
        item = items[i]
        tag = find_tag(fid, item, FIFF.FIFF_NCHAN)
        if tag is not None:
            nchan = tag.data
        else:
            nchan = global_nchan

        tag = find_tag(fid, item, FIFF.FIFF_DESCRIPTION)
        if tag is not None:
            desc = tag.data
        else:
            tag = find_tag(fid, item, FIFF.FIFF_NAME)
            if tag is not None:
                desc = tag.data
            else:
                raise ValueError, 'Projection item description missing'

        tag = find_tag(fid, item, FIFF.FIFF_PROJ_ITEM_CH_NAME_LIST)
        if tag is not None:
            namelist = tag.data;
        else:
            raise ValueError, 'Projection item channel list missing'

        tag = find_tag(fid, item,FIFF.FIFF_PROJ_ITEM_KIND);
        if tag is not None:
            kind = tag.data;
        else:
            raise ValueError, 'Projection item kind missing'

        tag = find_tag(fid, item, FIFF.FIFF_PROJ_ITEM_NVEC)
        if tag is not None:
            nvec = tag.data
        else:
            raise ValueError, 'Number of projection vectors not specified'

        tag = find_tag(fid, item, FIFF.FIFF_PROJ_ITEM_CH_NAME_LIST)
        if tag is not None:
            names = tag.data.split(':')
        else:
            raise ValueError, 'Projection item channel list missing'

        tag = find_tag(fid, item, FIFF.FIFF_PROJ_ITEM_VECTORS);
        if tag is not None:
            data = tag.data;
        else:
            raise ValueError, 'Projection item data missing'

        tag = find_tag(fid, item, FIFF.FIFF_MNE_PROJ_ITEM_ACTIVE);
        if tag is not None:
            active = tag.data;
        else:
            active = False;

        if data.shape[1] != len(names):
            raise ValueError, 'Number of channel names does not match the size of data matrix'

        #   Use exactly the same fields in data as in a named matrix
        one = Bunch(kind=kind, active=active, desc=desc,
                    data=Bunch(nrow=nvec, ncol=nchan, row_names=None,
                              col_names=names, data=data))

        projdata.append(one)

    if len(projdata) > 0:
        print '\tRead a total of %d projection items:\n', len(projdata)
        for k in range(len(projdata)):
            print '\t\t%s (%d x %d)' % (projdata[k].desc,
                                        projdata[k].data.nrow,
                                        projdata[k].data.ncol)
            if projdata[k].active:
                print ' active\n'
            else:
                print ' idle\n'

    return projdata


def read_bad_channels(fid, node):
    """
    %
    % [bads] = fiff_read_bad_channels(fid,node)
    %
    % Reas the bad channel list from a node if it exists
    %
    % fid      - The file id
    % node     - The node of interes
    %
    """

    nodes = dir_tree_find(node, FIFF.FIFFB_MNE_BAD_CHANNELS)

    bads = [];
    if len(nodes) > 0:
        for node in nodes:
            tag = find_tag(fid, node, FIFF.FIFF_MNE_CH_NAME_LIST)
            if tag is not None:
                bads = tag.data.split(':')
    return bads


def read_meas_info(source, tree=None):
    """[info,meas] = fiff_read_meas_info(source,tree)

     Read the measurement info

     If tree is specified, source is assumed to be an open file id,
     otherwise a the name of the file to read. If tree is missing, the
     meas output argument should not be specified.
    """
    if tree is None:
       fid, tree, _ = fiff_open(source)
       open_here = True
    else:
       fid = source
       open_here = False

    #   Find the desired blocks
    meas = dir_tree_find(tree, FIFF.FIFFB_MEAS)
    if len(meas) == 0:
        if open_here:
            fid.close()
        raise ValueError, 'Could not find measurement data'
    if len(meas) > 1:
        if open_here:
            fid.close()
        raise ValueError, 'Cannot read more that 1 measurement data'
    meas = meas[0]

    meas_info = dir_tree_find(meas, FIFF.FIFFB_MEAS_INFO)
    if len(meas_info) == 0:
        if open_here:
            fid.close()
        raise ValueError, 'Could not find measurement info'
    if len(meas_info) > 1:
        if open_here:
            fid.close()
        raise ValueError, 'Cannot read more that 1 measurement info'
    meas_info = meas_info[0]

    #   Read measurement info
    dev_head_t = None
    ctf_head_t = None
    meas_date = None
    highpass = None
    lowpass = None
    nchan = None
    sfreq = None
    chs = []
    p = 0
    for k in range(meas_info.nent):
        kind = meas_info.directory[k].kind
        pos  = meas_info.directory[k].pos
        if kind == FIFF.FIFF_NCHAN:
            tag = read_tag(fid, pos)
            nchan = tag.data
        elif kind == FIFF.FIFF_SFREQ:
            tag = read_tag(fid, pos)
            sfreq = tag.data
        elif kind == FIFF.FIFF_CH_INFO:
            tag = read_tag(fid, pos)
            chs.append(tag.data)
            p += 1
        elif kind == FIFF.FIFF_LOWPASS:
            tag = read_tag(fid, pos)
            lowpass = tag.data
        elif kind == FIFF.FIFF_HIGHPASS:
            tag = read_tag(fid, pos)
            highpass = tag.data
        elif kind == FIFF.FIFF_MEAS_DATE:
            tag = read_tag(fid, pos)
            meas_date = tag.data
        elif kind == FIFF.FIFF_COORD_TRANS:
            tag = read_tag(fid, pos)
            cand = tag.data
            if cand.from_ == FIFF.FIFFV_COORD_DEVICE and \
                                cand.to == FIFF.FIFFV_COORD_HEAD: # XXX : from
                dev_head_t = cand
            elif cand.from_ == FIFF.FIFFV_MNE_COORD_CTF_HEAD and \
                                cand.to == FIFF.FIFFV_COORD_HEAD:
                ctf_head_t = cand

    #  XXX : fix
    #   Check that we have everything we need
    # if ~exist('nchan','var')
    #    if open_here
    #       fclose(fid);
    #    end
    #    error(me,'Number of channels in not defined');
    # end
    # if ~exist('sfreq','var')
    #    if open_here
    #       fclose(fid);
    #    end
    #    error(me,'Sampling frequency is not defined');
    # end
    # if ~exist('chs','var')
    #    if open_here
    #       fclose(fid);
    #    end
    #    error(me,'Channel information not defined');
    # end
    # if length(chs) ~= nchan
    #    if open_here
    #       fclose(fid);
    #    end
    #    error(me,'Incorrect number of channel definitions found');
    # end

    if dev_head_t is None or ctf_head_t is None:
        hpi_result = dir_tree_find(meas_info, FIFF.FIFFB_HPI_RESULT)
        if len(hpi_result) == 1:
            hpi_result = hpi_result[0]
            for k in range(hpi_result.nent):
               kind = hpi_result.directory[k].kind
               pos  = hpi_result.directory[k].pos
               if kind == FIFF.FIFF_COORD_TRANS:
                    tag = read_tag(fid, pos)
                    cand = tag.data;
                    if cand.from_ == FIFF.FIFFV_COORD_DEVICE and \
                                cand.to == FIFF.FIFFV_COORD_HEAD: # XXX: from
                        dev_head_t = cand;
                    elif cand.from_ == FIFF.FIFFV_MNE_COORD_CTF_HEAD and \
                                cand.to == FIFF.FIFFV_COORD_HEAD:
                        ctf_head_t = cand;

    #   Locate the Polhemus data
    isotrak = dir_tree_find(meas_info,FIFF.FIFFB_ISOTRAK)
    if len(isotrak):
        isotrak = isotrak[0]
    else:
        if len(isotrak) == 0:
            if open_here:
                fid.close()
            raise ValueError, 'Isotrak not found'
        if len(isotrak) > 1:
            if open_here:
                fid.close()
            raise ValueError, 'Multiple Isotrak found'

    dig = []
    if len(isotrak) == 1:
        for k in range(isotrak.nent):
            kind = isotrak.directory[k].kind;
            pos  = isotrak.directory[k].pos;
            if kind == FIFF.FIFF_DIG_POINT:
                tag = read_tag(fid,pos);
                dig.append(tag.data)
                dig[-1]['coord_frame'] = FIFF.FIFFV_COORD_HEAD

    #   Locate the acquisition information
    acqpars = dir_tree_find(meas_info, FIFF.FIFFB_DACQ_PARS);
    acq_pars = []
    acq_stim = []
    if len(acqpars) == 1:
        for k in range(acqpars.nent):
            kind = acqpars.directory[k].kind
            pos  = acqpars.directory[k].pos
            if kind == FIFF.FIFF_DACQ_PARS:
                tag = read_tag(fid, pos)
                acq_pars = tag.data
            elif kind == FIFF.FIFF_DACQ_STIM:
               tag = read_tag(fid, pos)
               acq_stim = tag.data

    #   Load the SSP data
    projs = read_proj(fid, meas_info)

    #   Load the CTF compensation data
    comps = read_ctf_comp(fid, meas_info, chs)

    #   Load the bad channel list
    bads = read_bad_channels(fid, meas_info)

    #
    #   Put the data together
    #
    if tree.id is not None:
       info = dict(file_id=tree.id)
    else:
       info = dict(file_id=None)

    #  Make the most appropriate selection for the measurement id
    if meas_info.parent_id is None:
        if meas_info.id is None:
            if meas.id is None:
                if meas.parent_id is None:
                    info['meas_id'] = info.file_id
                else:
                    info['meas_id'] = meas.parent_id
            else:
                info['meas_id'] = meas.id
        else:
            info['meas_id'] = meas_info.id
    else:
       info['meas_id'] = meas_info.parent_id;

    if meas_date is None:
       info['meas_date'] = [info['meas_id']['secs'], info['meas_id']['usecs']]
    else:
       info['meas_date'] = meas_date

    info['nchan'] = nchan
    info['sfreq'] = sfreq
    info['highpass'] = highpass if highpass is not None else 0
    info['lowpass'] = lowpass if lowpass is not None else info['sfreq']/2.0

    #   Add the channel information and make a list of channel names
    #   for convenience
    info['chs'] = chs;
    info['ch_names'] = [ch.ch_name for ch in chs]

    #
    #  Add the coordinate transformations
    #
    info['dev_head_t'] = dev_head_t
    info['ctf_head_t'] = ctf_head_t
    if dev_head_t is not None and ctf_head_t is not None:
       info['dev_ctf_t'] = info['dev_head_t']
       info['dev_ctf_t'].to = info['ctf_head_t'].from_ # XXX : see if better name
       info['dev_ctf_t'].trans = np.dot(np.inv(ctf_head_t.trans), info.dev_ctf_t.trans)
    else:
       info['dev_ctf_t'] = []

    #   All kinds of auxliary stuff
    info['dig'] = dig
    info['bads'] = bads
    info['projs'] = projs
    info['comps'] = comps
    info['acq_pars'] = acq_pars
    info['acq_stim'] = acq_stim

    if open_here:
       fid.close()

    return info, meas


def read_evoked(fname, setno=0):
    """
    [data] = fiff_read_evoked(fname,setno)

    Read one evoked data set

    """

    if setno < 0:
       raise ValueError, 'Data set selector must be positive'

    print 'Reading %s ...\n' % fname
    fid, tree, _ = fiff_open(fname);

    #   Read the measurement info
    info, meas = read_meas_info(fid, tree)
    info['filename'] = fname

    #   Locate the data of interest
    processed = dir_tree_find(meas,FIFF.FIFFB_PROCESSED_DATA);
    if len(processed) == 0:
        fid.close()
        raise ValueError, 'Could not find processed data'

    evoked = dir_tree_find(meas,FIFF.FIFFB_EVOKED)
    if len(evoked) == 0:
        fid.close(fid)
        raise ValueError, 'Could not find evoked data'

    #   Identify the aspects
    #
    naspect = 0
    is_smsh = None
    sets = []
    for k in range(len(evoked)):
        aspects_k = dir_tree_find(evoked[k], FIFF.FIFFB_ASPECT)
        set_k = dict(aspects=aspects_k,
                     naspect=len(aspects_k))
        sets.append(set_k)

        if sets[k]['naspect'] > 0:
            if is_smsh is None:
                is_smsh = np.zeros((1, sets[k]['naspect']))
            else:
                is_smsh = np.c_[is_smsh, np.zeros((1, sets[k]['naspect']))]
            naspect += sets[k]['naspect']

        saspects  = dir_tree_find(evoked[k], FIFF.FIFFB_SMSH_ASPECT)
        nsaspects = len(saspects)
        if nsaspects > 0:
            sets[k]['naspect'] += nsaspects
            sets[k]['naspect'] = [sets[k]['naspect'], saspects] # XXX : potential bug
            is_smsh = np.c_[is_smsh, np.ones(1, sets[k]['naspect'])]
            naspect += nsaspects;

    print '\t%d evoked data sets containing a total of %d data aspects' \
          ' in %s\n' % (len(evoked), naspect, fname)

    if setno >= naspect or setno < 0:
        fid.close()
        raise ValueError, 'Data set selector out of range'

    #   Next locate the evoked data set
    #
    p = 0
    goon = True
    for k in range(len(evoked)):
        for a in range(sets[k]['naspect']):
            if p == setno:
                my_evoked = evoked[k]
                my_aspect = sets[k]['aspects'][a]
                goon = False
                break
            p += 1
        if not goon:
            break
    else:
        #   The desired data should have been found but better to check
        fid.close(fid)
        raise ValueError, 'Desired data set not found'

    #   Now find the data in the evoked block
    nchan = 0
    sfreq = -1
    q = 0
    chs = []
    comment = None
    for k in range(my_evoked.nent):
        kind = my_evoked.directory[k].kind
        pos  = my_evoked.directory[k].pos
        if kind == FIFF.FIFF_COMMENT:
            tag = read_tag(fid, pos)
            comment = tag.data
        elif kind == FIFF.FIFF_FIRST_SAMPLE:
            tag = read_tag(fid, pos)
            first = tag.data
        elif kind == FIFF.FIFF_LAST_SAMPLE:
            tag = read_tag(fid, pos)
            last = tag.data
        elif kind == FIFF.FIFF_NCHAN:
            tag = read_tag(fid, pos)
            nchan = tag.data
        elif kind == FIFF.FIFF_SFREQ:
            tag = read_tag(fid, pos)
            sfreq = tag.data
        elif kind ==FIFF.FIFF_CH_INFO:
            tag = read_tag(fid, pos)
            chs.append(tag.data)
            q += 1

    if comment is None:
       comment = 'No comment'

    #   Local channel information?
    if nchan > 0:
        if chs is None:
            fid.close()
            raise ValueError, 'Local channel information was not found when it was expected.'

        if len(chs) != nchan:
            fid.close()
            raise ValueError, 'Number of channels and number of channel definitions are different'

        info.chs = chs
        info.nchan = nchan
        print '\tFound channel information in evoked data. nchan = %d\n' % nchan
        if sfreq > 0:
            info['sfreq'] = sfreq

    nsamp = last - first + 1
    print '\tFound the data of interest:\n'
    print '\t\tt = %10.2f ... %10.2f ms (%s)\n' % (
         1000*first/info['sfreq'], 1000*last/info['sfreq'], comment)
    if info['comps'] is not None:
        print '\t\t%d CTF compensation matrices available\n' % len(info['comps'])

    # Read the data in the aspect block
    nave = 1
    epoch = []
    for k in range(my_aspect.nent):
        kind = my_aspect.directory[k].kind
        pos  = my_aspect.directory[k].pos
        if kind == FIFF.FIFF_COMMENT:
            tag = read_tag(fid, pos)
            comment = tag.data
        elif kind == FIFF.FIFF_ASPECT_KIND:
            tag = read_tag(fid, pos)
            aspect_kind = tag.data
        elif kind == FIFF.FIFF_NAVE:
            tag = read_tag(fid, pos)
            nave = tag.data
        elif kind == FIFF.FIFF_EPOCH:
            tag = read_tag(fid, pos)
            epoch.append(tag)

    print '\t\tnave = %d aspect type = %d\n' % (nave, aspect_kind)

    nepoch = len(epoch)
    if nepoch != 1 and nepoch != info.nchan:
        fid.close()
        raise ValueError, 'Number of epoch tags is unreasonable '\
                          '(nepoch = %d nchan = %d)' % (nepoch, info.nchan)

    if nepoch == 1:
       #   Only one epoch
       all_data = epoch[0].data
       #   May need a transpose if the number of channels is one
       if all_data.shape[1] == 1 and info.nchan == 1:
          all_data = all_data.T

    else:
        #   Put the old style epochs together
        all_data = epoch[0].data.T
        for k in range(1, nepoch):
            all_data = np.r_[all_data, epoch[k].data.T]

    if all_data.shape[1] != nsamp:
        fid.close()
        raise ValueError, 'Incorrect number of samples (%d instead of %d)' % (
                                    all_data.shape[1], nsamp)

    #   Calibrate
    cals = np.array([info['chs'][k].cal for k in range(info['nchan'])])
    all_data = np.dot(np.diag(cals.ravel()), all_data)

    #   Put it all together
    data = dict(info=info, evoked=dict(aspect_kind=aspect_kind,
                                   is_smsh=is_smsh[setno],
                                   nave=nave, first=first,
                                   last=last, comment=comment,
                                   times=np.arange(first, last+1,
                                            dtype=np.float) / info['sfreq'],
                                   epochs=all_data))

    fid.close()

    return data
