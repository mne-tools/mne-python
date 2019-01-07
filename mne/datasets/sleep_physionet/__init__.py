"""EEG Sleep Polysomnography (PSG) Dataset."""

from .sleep_physionet import (data_path, fetch_data)

# now the numpy records files that contains info on dataset:
# It was obtained with:
# base_url = "https://physionet.org/pn4/sleep-edfx/"
# sha1sums_url = base_url + "SHA1SUMS"
# sha1sums_fname = "SHA1SUMS"
# _fetch_file(sha1sums_url, sha1sums_fname)
# df = pd.read_csv(sha1sums_fname, sep='  ', header=None,
#                  names=['sha', 'fname'], engine='python')
# df[['subject', 'type']] = df.fname.str.split('-', expand=True)
# df = df[df['type'].str.endswith('.edf') == True].copy()
# df['type'] = df['type'].apply(lambda x: x.split(".")[0])
# df['subject'] = df['subject'].str[:-1]
# df.set_index(pd.factorize(df.subject)[0], inplace=True)
