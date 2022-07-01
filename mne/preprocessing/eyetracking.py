def find_blinks(inst, from_nan=True, pad_sec=.01):
    # find eyetrack channels
    et_data = inst.pick_types(eyetrack=True)


    annot = None
    return annot


def find_saccades(inst):
    raise NotImplementedError()
