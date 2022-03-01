import os
import os.path as op
import mne

docdict = mne.utils.docs.docdict

entry_args = {entry: desc.split(' ')[0].strip().replace('*', '')
              for entry, desc in docdict.items()}
mappings = dict()

# modify entries with backward format
for entry, arg in entry_args.items():
    if len(docdict[entry].split(' ')) <= 1 or \
            docdict[entry].split(' ')[1] != ':':
        continue  # skip purely narrative
    if not entry.startswith(arg):
        entry2 = entry
        # specific fix
        entry2 = entry2.replace('thresh', 'threshold')
        entry2 = entry2.replace('adj', 'adjacency')
        entry2 = entry2.replace('longform', 'long_format')
        # remove arg
        for word in arg.split('_'):
            if len(word) > 1:  # don't remove single letters
                entry2 = entry2.replace(word, '')
        # add arg at the begining
        entry2 = entry2.lower()
        # fix underscores
        while '__' in entry2:
            entry2 = entry2.replace('__', '_')
        while entry2.startswith('_'):
            entry2 = entry2.removeprefix('_')
        while entry2.endswith('_'):
            entry2 = entry2.removesuffix('_')
        entry2 = arg + '_' + entry2
        # specific fixes
        entry2 = entry2.replace('db_', 'dB_')
        if entry == 'applyfun_fun':
            entry2 = 'fun_applyfun'
        if entry == 'kwarg_fun':
            entry2 = 'kwargs_fun'
        if entry == 'df_tf':
            entry2 = 'time_format_df'
        if entry == 'clust_nperm_all':
            entry2 = 'n_permutations_clust_all'
        if entry == 'clust_nperm_int':
            entry2 = 'n_permutations_clust_int'
        if entry == 'clust_power_t':
            entry2 = 't_power_clust'
        if entry == 'clust_power_f':
            entry2 = 'f_power_clust'
        if entry == 'by_event_type_returns_average':
            entry2 = 'evoked_by_event_type_returns'
        if entry == 'ecg_filter_freqs':
            entry2 = 'l_freq_ecg_filter'
        print(f'Changing {entry} to {entry2}')
        mappings[entry] = entry2
        os.system('find . -type f -name "*.py" -print0 | xargs -0 '
                  f'sed -i \'\' -e \'s/%({entry})s/%({entry2})s/g\'')


# set mappings
for entry, entry2 in mappings.items():
    arg = entry_args.pop(entry)
    entry_args[entry2] = arg

mappings_r = {v: k for k, v in mappings.items()}

# write a new docdict
with open('docdict.txt', 'w') as fid:
    letter = ''
    for entry in sorted(entry_args.keys()):
        if entry[0] != letter:
            letter = entry[0]
            fid.write(f'# %%\n# {letter.upper()}\n\n')
        if entry in docdict:
            desc = docdict[entry]
        else:
            desc = docdict[mappings_r[entry]]
        fid.write(
            f'docdict[\'{entry}\'] = """'
            f'{desc}'
            '"""\n\n')
