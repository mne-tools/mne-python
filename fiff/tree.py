from .bunch import Bunch
from .tag import read_tag


def dir_tree_find(tree, kind):
    """[nodes] = dir_tree_find(tree,kind)

       Find nodes of the given kind from a directory tree structure
       
       Returns a list of matching nodes
    """
    nodes = []

    if isinstance(tree, list):
        for t in tree:
            nodes += dir_tree_find(t, kind)
    else:
        #   Am I desirable myself?
        if tree.block == kind:
            nodes.append(tree)

        #   Search the subtrees
        for child in tree.children:
            nodes += dir_tree_find(child, kind)
    return nodes


def make_dir_tree(fid, directory, start=0, indent=0, verbose=False):
    """Create the directory tree structure
    """
    FIFF_BLOCK_START = 104
    FIFF_BLOCK_END = 105
    FIFF_FILE_ID = 100
    FIFF_BLOCK_ID = 103
    FIFF_PARENT_BLOCK_ID = 110

    if directory[start].kind == FIFF_BLOCK_START:
        tag = read_tag(fid, directory[start].pos)
        block = tag.data
    else:
        block = 0

    if verbose:
        for k in range(indent):
            print '\t'
        print 'start { %d\n' % block

    this = start

    tree = Bunch()
    tree['block'] = block
    tree['id'] = None
    tree['parent_id'] = None
    tree['nent'] = 0
    tree['nchild'] = 0
    tree['directory'] = directory[this]
    tree['children'] = []

    while this < len(directory):
        if directory[this].kind == FIFF_BLOCK_START:
            if this != start:
                child, this = make_dir_tree(fid, directory, this, indent+1)
                tree.nchild += 1
                tree.children.append(child)
        elif directory[this].kind == FIFF_BLOCK_END:
            tag = read_tag(fid, directory[start].pos)
            if tag.data == block:
                break
        else:
            tree.nent += 1
            if tree.nent == 1:
                tree.directory = list()
            tree.directory.append(directory[this])
        #
        #  Add the id information if available
        #
        if block == 0:
           if directory[this].kind == FIFF_FILE_ID:
              tag = read_tag(fid, directory[this].pos)
              tree.id = tag.data
        else:
           if directory[this].kind == FIFF_BLOCK_ID:
              tag = read_tag(fid, directory[this].pos)
              tree.id = tag.data
           elif directory[this].kind == FIFF_PARENT_BLOCK_ID:
              tag = read_tag(fid, directory[this].pos)
              tree.parent_id = tag.data
        this += 1
    #
    # Eliminate the empty directory
    #
    if tree.nent == 0:
       tree.directory = None

    if verbose:
        for k in range(indent+1):
            print '\t'
        print 'block = %d nent = %d nchild = %d\n' % (tree.block, tree.nent,
                                                      tree.nchild)
        for k in range(indent):
            print '\t'
        print 'end } %d\n' % block

    last = this
    return tree, last
