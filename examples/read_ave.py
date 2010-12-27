import fiff

fname = 'sm02a1-ave.fif'

# fid, tree, directory = fiff.fiff_open(fname, verbose=True)
# meas = fiff.tree.dir_tree_find(tree, fiff.FIFF.FIFFB_MEAS)
# meas_info = fiff.tree.dir_tree_find(meas, fiff.FIFF.FIFFB_MEAS_INFO)

# meas = fiff.evoked.read_meas_info(fname)
# def is_tree(tree):
#     assert isinstance(tree, dict)
#     tree.block
#     for child in tree.children:
#         is_tree(child)
# 
# is_tree(tree)
# meas = fiff.tree.dir_tree_find(tree, fiff.FIFF.FIFFB_MEAS)
# is_tree(meas)
# meas_info = fiff.tree.dir_tree_find(meas, fiff.FIFF.FIFFB_MEAS_INFO)

data = fiff.read_evoked(fname, setno=0)

