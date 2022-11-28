# Authors: Adam Li <adam2392@gmail.com>
#          Daniel McCloy <dan@mccloy.info>
#
# License: BSD Style.


_bst_license_text = """
License
-------
This tutorial dataset (EEG and MRI data) remains a property of the MEG Lab,
McConnell Brain Imaging Center, Montreal Neurological Institute,
McGill University, Canada. Its use and transfer outside the Brainstorm
tutorial, e.g. for research purposes, is prohibited without written consent
from the MEG Lab.

If you reference this dataset in your publications, please:

    1) acknowledge its authors: Elizabeth Bock, Esther Florin, Francois Tadel
       and Sylvain Baillet, and
    2) cite Brainstorm as indicated on the website:
       http://neuroimage.usc.edu/brainstorm

For questions, please contact Francois Tadel (francois.tadel@mcgill.ca).
"""

_hcp_mmp_license_text = """
License
-------
I request access to data collected by the Washington University - University
of Minnesota Consortium of the Human Connectome Project (WU-Minn HCP), and
I agree to the following:

1. I will not attempt to establish the identity of or attempt to contact any
   of the included human subjects.

2. I understand that under no circumstances will the code that would link
   these data to Protected Health Information be given to me, nor will any
   additional information about individual human subjects be released to me
   under these Open Access Data Use Terms.

3. I will comply with all relevant rules and regulations imposed by my
   institution. This may mean that I need my research to be approved or
   declared exempt by a committee that oversees research on human subjects,
   e.g. my IRB or Ethics Committee. The released HCP data are not considered
   de-identified, insofar as certain combinations of HCP Restricted Data
   (available through a separate process) might allow identification of
   individuals.  Different committees operate under different national, state
   and local laws and may interpret regulations differently, so it is
   important to ask about this. If needed and upon request, the HCP will
   provide a certificate stating that you have accepted the HCP Open Access
   Data Use Terms.

4. I may redistribute original WU-Minn HCP Open Access data and any derived
   data as long as the data are redistributed under these same Data Use Terms.

5. I will acknowledge the use of WU-Minn HCP data and data derived from
   WU-Minn HCP data when publicly presenting any results or algorithms
   that benefitted from their use.

   1. Papers, book chapters, books, posters, oral presentations, and all
      other printed and digital presentations of results derived from HCP
      data should contain the following wording in the acknowledgments
      section: "Data were provided [in part] by the Human Connectome
      Project, WU-Minn Consortium (Principal Investigators: David Van Essen
      and Kamil Ugurbil; 1U54MH091657) funded by the 16 NIH Institutes and
      Centers that support the NIH Blueprint for Neuroscience Research; and
      by the McDonnell Center for Systems Neuroscience at Washington
      University."

   2. Authors of publications or presentations using WU-Minn HCP data
      should cite relevant publications describing the methods used by the
      HCP to acquire and process the data. The specific publications that
      are appropriate to cite in any given study will depend on what HCP
      data were used and for what purposes. An annotated and appropriately
      up-to-date list of publications that may warrant consideration is
      available at http://www.humanconnectome.org/about/acknowledgehcp.html

   3. The WU-Minn HCP Consortium as a whole should not be included as an
      author of publications or presentations if this authorship would be
      based solely on the use of WU-Minn HCP data.

6. Failure to abide by these guidelines will result in termination of my
   privileges to access WU-Minn HCP data.
"""

# To update the `testing` or `misc` datasets, push or merge commits to their
# respective repos, and make a new release of the dataset on GitHub. Then
# update the checksum in the MNE_DATASETS dict below, and change version
# here:                  ↓↓↓↓↓         ↓↓↓
RELEASES = dict(testing='0.140', misc='0.23')
TESTING_VERSIONED = f'mne-testing-data-{RELEASES["testing"]}'
MISC_VERSIONED = f'mne-misc-data-{RELEASES["misc"]}'

# To update any other dataset besides `testing` or `misc`, upload the new
# version of the data archive itself (e.g., to https://osf.io or wherever) and
# then update the corresponding checksum in the MNE_DATASETS dict entry below.
MNE_DATASETS = dict()

# MANDATORY KEYS:
# - archive_name : the name of the compressed file that is downloaded
# - hash : the checksum type followed by a colon and then the checksum value
#          (examples: "sha256:19uheid...", "md5:upodh2io...")
# - url : URL from which the file can be downloaded
# - folder_name : the subfolder within the MNE data folder in which to save and
#                 uncompress (if needed) the file(s)
#
# OPTIONAL KEYS:
# - config_key : key to use with `mne.set_config` to store the on-disk location
#                of the downloaded dataset (ex: "MNE_DATASETS_EEGBCI_PATH").

# Testing and misc are at the top as they're updated most often
MNE_DATASETS['testing'] = dict(
    archive_name=f'{TESTING_VERSIONED}.tar.gz',
    hash='md5:f4377b017867f58a7c490b568764f44a',
    url=('https://codeload.github.com/mne-tools/mne-testing-data/'
         f'tar.gz/{RELEASES["testing"]}'),
    # In case we ever have to resort to osf.io again...
    # archive_name='mne-testing-data.tar.gz',
    # hash='md5:c805a5fed8ca46f723e7eec828d90824',
    # url='https://osf.io/dqfgy/download?version=1',  # 0.136
    folder_name='MNE-testing-data',
    config_key='MNE_DATASETS_TESTING_PATH',
)
MNE_DATASETS['misc'] = dict(
    archive_name=f'{MISC_VERSIONED}.tar.gz',  # 'mne-misc-data',
    hash='md5:01e409d82ff11ca8b19a27c4f7ee6794',
    url=('https://codeload.github.com/mne-tools/mne-misc-data/tar.gz/'
         f'{RELEASES["misc"]}'),
    folder_name='MNE-misc-data',
    config_key='MNE_DATASETS_MISC_PATH'
)

MNE_DATASETS['fnirs_motor'] = dict(
    archive_name='MNE-fNIRS-motor-data.tgz',
    hash='md5:c4935d19ddab35422a69f3326a01fef8',
    url='https://osf.io/dj3eh/download?version=1',
    folder_name='MNE-fNIRS-motor-data',
    config_key='MNE_DATASETS_FNIRS_MOTOR_PATH',
)

MNE_DATASETS['kiloword'] = dict(
    archive_name='MNE-kiloword-data.tar.gz',
    hash='md5:3a124170795abbd2e48aae8727e719a8',
    url='https://osf.io/qkvf9/download?version=1',
    folder_name='MNE-kiloword-data',
    config_key='MNE_DATASETS_KILOWORD_PATH',
)

MNE_DATASETS['multimodal'] = dict(
    archive_name='MNE-multimodal-data.tar.gz',
    hash='md5:26ec847ae9ab80f58f204d09e2c08367',
    url='https://ndownloader.figshare.com/files/5999598',
    folder_name='MNE-multimodal-data',
    config_key='MNE_DATASETS_MULTIMODAL_PATH',
)

MNE_DATASETS['opm'] = dict(
    archive_name='MNE-OPM-data.tar.gz',
    hash='md5:370ad1dcfd5c47e029e692c85358a374',
    url='https://osf.io/p6ae7/download?version=2',
    folder_name='MNE-OPM-data',
    config_key='MNE_DATASETS_OPM_PATH',
)

MNE_DATASETS['phantom_4dbti'] = dict(
    archive_name='MNE-phantom-4DBTi.zip',
    hash='md5:938a601440f3ffa780d20a17bae039ff',
    url='https://osf.io/v2brw/download?version=2',
    folder_name='MNE-phantom-4DBTi',
    config_key='MNE_DATASETS_PHANTOM_4DBTI_PATH',
)

MNE_DATASETS['sample'] = dict(
    archive_name='MNE-sample-data-processed.tar.gz',
    hash='md5:e8f30c4516abdc12a0c08e6bae57409c',
    url='https://osf.io/86qa2/download?version=6',
    folder_name='MNE-sample-data',
    config_key='MNE_DATASETS_SAMPLE_PATH',
)

MNE_DATASETS['somato'] = dict(
    archive_name='MNE-somato-data.tar.gz',
    hash='md5:32fd2f6c8c7eb0784a1de6435273c48b',
    url='https://osf.io/tp4sg/download?version=7',
    folder_name='MNE-somato-data',
    config_key='MNE_DATASETS_SOMATO_PATH'
)

MNE_DATASETS['spm'] = dict(
    archive_name='MNE-spm-face.tar.gz',
    hash='md5:9f43f67150e3b694b523a21eb929ea75',
    url='https://osf.io/je4s8/download?version=2',
    folder_name='MNE-spm-face',
    config_key='MNE_DATASETS_SPM_FACE_PATH',
)

# Visual 92 categories has the dataset split into 2 files.
# We define a dictionary holding the items with the same
# value across both files: folder name and configuration key.
MNE_DATASETS['visual_92_categories'] = dict(
    folder_name='MNE-visual_92_categories-data',
    config_key='MNE_DATASETS_VISUAL_92_CATEGORIES_PATH',
)
MNE_DATASETS['visual_92_categories_1'] = dict(
    archive_name='MNE-visual_92_categories-data-part1.tar.gz',
    hash='md5:74f50bbeb65740903eadc229c9fa759f',
    url='https://osf.io/8ejrs/download?version=1',
    folder_name='MNE-visual_92_categories-data',
    config_key='MNE_DATASETS_VISUAL_92_CATEGORIES_PATH',
)
MNE_DATASETS['visual_92_categories_2'] = dict(
    archive_name='MNE-visual_92_categories-data-part2.tar.gz',
    hash='md5:203410a98afc9df9ae8ba9f933370e20',
    url='https://osf.io/t4yjp/download?version=1',
    folder_name='MNE-visual_92_categories-data',
    config_key='MNE_DATASETS_VISUAL_92_CATEGORIES_PATH',
)

MNE_DATASETS['mtrf'] = dict(
    archive_name='mTRF_1.5.zip',
    hash='md5:273a390ebbc48da2c3184b01a82e4636',
    url='https://osf.io/h85s2/download?version=1',
    folder_name='mTRF_1.5',
    config_key='MNE_DATASETS_MTRF_PATH'
)
MNE_DATASETS['refmeg_noise'] = dict(
    archive_name='sample_reference_MEG_noise-raw.zip',
    hash='md5:779fecd890d98b73a4832e717d7c7c45',
    url='https://osf.io/drt6v/download?version=1',
    folder_name='MNE-refmeg-noise-data',
    config_key='MNE_DATASETS_REFMEG_NOISE_PATH'
)

MNE_DATASETS['ssvep'] = dict(
    archive_name='ssvep_example_data.zip',
    hash='md5:af866bbc0f921114ac9d683494fe87d6',
    url='https://osf.io/z8h6k/download?version=5',
    folder_name='ssvep-example-data',
    config_key='MNE_DATASETS_SSVEP_PATH'
)

MNE_DATASETS['erp_core'] = dict(
    archive_name='MNE-ERP-CORE-data.tar.gz',
    hash='md5:5866c0d6213bd7ac97f254c776f6c4b1',
    url='https://osf.io/rzgba/download?version=1',
    folder_name='MNE-ERP-CORE-data',
    config_key='MNE_DATASETS_ERP_CORE_PATH',
)

MNE_DATASETS['epilepsy_ecog'] = dict(
    archive_name='MNE-epilepsy-ecog-data.tar.gz',
    hash='md5:ffb139174afa0f71ec98adbbb1729dea',
    url='https://osf.io/z4epq/download?version=1',
    folder_name='MNE-epilepsy-ecog-data',
    config_key='MNE_DATASETS_EPILEPSY_ECOG_PATH',
)

# Fieldtrip CMC dataset
MNE_DATASETS['fieldtrip_cmc'] = dict(
    archive_name='SubjectCMC.zip',
    hash='md5:6f9fd6520f9a66e20994423808d2528c',
    url='https://osf.io/j9b6s/download?version=1',
    folder_name='MNE-fieldtrip_cmc-data',
    config_key='MNE_DATASETS_FIELDTRIP_CMC_PATH'
)

# brainstorm datasets:
MNE_DATASETS['bst_auditory'] = dict(
    archive_name='bst_auditory.tar.gz',
    hash='md5:fa371a889a5688258896bfa29dd1700b',
    url='https://osf.io/5t9n8/download?version=1',
    folder_name='MNE-brainstorm-data',
    config_key='MNE_DATASETS_BRAINSTORM_PATH',
)
MNE_DATASETS['bst_phantom_ctf'] = dict(
    archive_name='bst_phantom_ctf.tar.gz',
    hash='md5:80819cb7f5b92d1a5289db3fb6acb33c',
    url='https://osf.io/sxr8y/download?version=1',
    folder_name='MNE-brainstorm-data',
    config_key='MNE_DATASETS_BRAINSTORM_PATH',
)
MNE_DATASETS['bst_phantom_elekta'] = dict(
    archive_name='bst_phantom_elekta.tar.gz',
    hash='md5:1badccbe17998d18cc373526e86a7aaf',
    url='https://osf.io/dpcku/download?version=1',
    folder_name='MNE-brainstorm-data',
    config_key='MNE_DATASETS_BRAINSTORM_PATH',
)
MNE_DATASETS['bst_raw'] = dict(
    archive_name='bst_raw.tar.gz',
    hash='md5:fa2efaaec3f3d462b319bc24898f440c',
    url='https://osf.io/9675n/download?version=2',
    folder_name='MNE-brainstorm-data',
    config_key='MNE_DATASETS_BRAINSTORM_PATH',
)
MNE_DATASETS['bst_resting'] = dict(
    archive_name='bst_resting.tar.gz',
    hash='md5:70fc7bf9c3b97c4f2eab6260ee4a0430',
    url='https://osf.io/m7bd3/download?version=3',
    folder_name='MNE-brainstorm-data',
    config_key='MNE_DATASETS_BRAINSTORM_PATH',
)

# HF-SEF
MNE_DATASETS['hf_sef_raw'] = dict(
    archive_name='hf_sef_raw.tar.gz',
    hash='md5:33934351e558542bafa9b262ac071168',
    url='https://zenodo.org/record/889296/files/hf_sef_raw.tar.gz',
    folder_name='hf_sef',
    config_key='MNE_DATASETS_HF_SEF_PATH',
)
MNE_DATASETS['hf_sef_evoked'] = dict(
    archive_name='hf_sef_evoked.tar.gz',
    hash='md5:13d34cb5db584e00868677d8fb0aab2b',
    url=('https://zenodo.org/record/3523071/files/'
         'hf_sef_evoked.tar.gz'),
    folder_name='hf_sef',
    config_key='MNE_DATASETS_HF_SEF_PATH',
)

# "fake" dataset (for testing)
MNE_DATASETS['fake'] = dict(
    archive_name='foo.tgz',
    hash='md5:3194e9f7b46039bb050a74f3e1ae9908',
    url=('https://github.com/mne-tools/mne-testing-data/raw/master/'
         'datasets/foo.tgz'),
    folder_name='foo',
    config_key='MNE_DATASETS_FAKE_PATH'
)
