from .tree import dir_tree_find
from .constants import FIFF
from .tag import find_tag
from .bunch import Bunch


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
        global_nchan = int(tag.data)

    items = dir_tree_find(nodes[0], FIFF.FIFFB_PROJ_ITEM)
    for i in range(len(items)):

        #   Find all desired tags in one item
        item = items[i]
        tag = find_tag(fid, item, FIFF.FIFF_NCHAN)
        if tag is not None:
            nchan = int(tag.data)
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
        print '\tRead a total of %d projection items:' % len(projdata)
        for k in range(len(projdata)):
            if projdata[k].active:
                misc = 'active'
            else:
                misc = ' idle'
            print '\t\t%s (%d x %d) %s' % (projdata[k].desc,
                                        projdata[k].data.nrow,
                                        projdata[k].data.ncol,
                                        misc)

    return projdata
