from .read_tag import read_tag_info, read_tag
from .tree import make_dir_tree
from .constants import FIFF

def fiff_open(fname, verbose=False):

    fid = open(fname, "rb") # Open in binary mode

    tag = read_tag_info(fid)

    #
    #   Check that this looks like a fif file
    #
    if tag.kind != FIFF.FIFF_FILE_ID:
        raise ValueError, 'file does not start with a file id tag'

    if tag.type != FIFF.FIFFT_ID_STRUCT:
        raise ValueError, 'file does not start with a file id tag'

    if tag.size != 20:
        raise ValueError, 'file does not start with a file id tag'

    tag = read_tag(fid)

    if tag.kind != FIFF.FIFF_DIR_POINTER:
        raise ValueError, 'file does have a directory pointer'

    #
    #   Read or create the directory tree
    #
    if verbose:
        print '\tCreating tag directory for %s...' % fname

    dirpos = int(tag.data)
    if dirpos > 0:
        tag = read_tag(fid, dirpos)
        directory = tag.data
    else:
        fid.seek(0, 0)
        directory = list()
        while tag.next >= 0:
            pos = fid.tell()
            directory.append(read_tag_info(fid))

    tree, _ = make_dir_tree(fid, directory)

    if verbose:
       print '[done]\n'

    #
    #   Back to the beginning
    #
    fid.seek(0)
    # fid.close()

    return fid, tree, directory
