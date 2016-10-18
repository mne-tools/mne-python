# Author: Eric Larson <larson.eric.d@gmail.com>
# License: BSD Style.

import numpy as np

url_root = 'http://cobre.mrn.org/megsim'

urls = ['/empdata/neuromag/visual/subject1_day1_vis_raw.fif',
        '/empdata/neuromag/visual/subject1_day2_vis_raw.fif',
        '/empdata/neuromag/visual/subject3_day1_vis_raw.fif',
        '/empdata/neuromag/visual/subject3_day2_vis_raw.fif',
        '/empdata/neuromag/aud/subject1_day1_aud_raw.fif',
        '/empdata/neuromag/aud/subject1_day2_aud_raw.fif',
        '/empdata/neuromag/aud/subject3_day1_aud_raw.fif',
        '/empdata/neuromag/aud/subject3_day2_aud_raw.fif',
        '/empdata/neuromag/somato/subject1_day1_median_raw.fif',
        '/empdata/neuromag/somato/subject1_day2_median_raw.fif',
        '/empdata/neuromag/somato/subject3_day1_median_raw.fif',
        '/empdata/neuromag/somato/subject3_day2_median_raw.fif',

        '/simdata/neuromag/visual/M87174545_vis_sim1A_4mm_30na_neuro_rn.fif',
        '/simdata/neuromag/visual/M87174545_vis_sim1B_20mm_50na_neuro_rn.fif',
        '/simdata/neuromag/visual/M87174545_vis_sim2_4mm_30na_neuro_rn.fif',
        '/simdata/neuromag/visual/M87174545_vis_sim3A_4mm_30na_neuro_rn.fif',
        '/simdata/neuromag/visual/M87174545_vis_sim3B_20mm_50na_neuro_rn.fif',
        '/simdata/neuromag/visual/M87174545_vis_sim4_4mm_30na_neuro_rn.fif',
        '/simdata/neuromag/visual/M87174545_vis_sim5_4mm_30na_neuro_rn.fif',

        '/simdata_singleTrials/subject1_singleTrials_VisWorkingMem_fif.zip',
        '/simdata_singleTrials/subject1_singleTrials_VisWorkingMem_withOsc_fif.zip',  # noqa: E501
        '/simdata_singleTrials/4545_sim_oscOnly_v1_IPS_ILOG_30hzAdded.fif',

        '/index.html',
        ]

data_formats = ['raw',
                'raw',
                'raw',
                'raw',
                'raw',
                'raw',
                'raw',
                'raw',
                'raw',
                'raw',
                'raw',
                'raw',

                'evoked',
                'evoked',
                'evoked',
                'evoked',
                'evoked',
                'evoked',
                'evoked',

                'single-trial',
                'single-trial',
                'single-trial',

                'text']

subjects = ['subject_1',
            'subject_1',
            'subject_3',
            'subject_3',
            'subject_1',
            'subject_1',
            'subject_3',
            'subject_3',
            'subject_1',
            'subject_1',
            'subject_3',
            'subject_3',

            'subject_1',
            'subject_1',
            'subject_1',
            'subject_1',
            'subject_1',
            'subject_1',
            'subject_1',

            'subject_1',
            'subject_1',
            'subject_1',

            '']

data_types = ['experimental',
              'experimental',
              'experimental',
              'experimental',
              'experimental',
              'experimental',
              'experimental',
              'experimental',
              'experimental',
              'experimental',
              'experimental',
              'experimental',

              'simulation',
              'simulation',
              'simulation',
              'simulation',
              'simulation',
              'simulation',
              'simulation',

              'simulation',
              'simulation',
              'simulation',

              'text']

conditions = ['visual',
              'visual',
              'visual',
              'visual',
              'auditory',
              'auditory',
              'auditory',
              'auditory',
              'somatosensory',
              'somatosensory',
              'somatosensory',
              'somatosensory',

              'visual',
              'visual',
              'visual',
              'visual',
              'visual',
              'visual',
              'visual',

              'visual',
              'visual',
              'visual',

              'index']

valid_data_types = list(set(data_types))
valid_data_formats = list(set(data_formats))
valid_conditions = list(set(conditions))

# turn them into arrays for ease of use
urls = np.atleast_1d(urls)
data_formats = np.atleast_1d(data_formats)
subjects = np.atleast_1d(subjects)
data_types = np.atleast_1d(data_types)
conditions = np.atleast_1d(conditions)

# Useful for testing
# assert len(conditions) == len(data_types) == len(subjects) \
#     == len(data_formats) == len(urls)


def url_match(condition, data_format, data_type):
    """Function to match MEGSIM data files."""
    inds = np.logical_and(conditions == condition, data_formats == data_format)
    inds = np.logical_and(inds, data_types == data_type)
    inds = np.logical_and(inds, data_formats == data_format)
    good_urls = list(urls[inds])
    for gi, g in enumerate(good_urls):
        good_urls[gi] = url_root + g
    if len(good_urls) == 0:
        raise ValueError('No MEGSIM dataset found with condition="%s",\n'
                         'data_format="%s", data_type="%s"'
                         % (condition, data_format, data_type))
    return good_urls


def _load_all_data():
    """Helper for downloading all megsim datasets."""
    from .megsim import data_path
    for url in urls:
        data_path(url_root + url)
