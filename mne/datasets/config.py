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
# update the checksum in `mne/data/dataset_checksums.txt` and change version
# here:                  ↓↓↓↓↓         ↓↓↓
RELEASES = dict(testing='0.112', misc='0.8')

# To update any other dataset besides `testing` or `misc`, upload the new
# version of the data archive itself (e.g., to https://osf.io or wherever) and
# then update the corresponding checksum in `mne/data/dataset_checksums.txt`.
TESTING_VERSIONED = f'mne-testing-data-{RELEASES["testing"]}'
MISC_VERSIONED = f'mne-misc-data-{RELEASES["misc"]}'

# remote locations of the various datasets
URLS = dict(
    bst_auditory='https://osf.io/5t9n8/download?version=1',
    bst_phantom_ctf='https://osf.io/sxr8y/download?version=1',
    bst_phantom_elekta='https://osf.io/dpcku/download?version=1',
    bst_raw='https://osf.io/9675n/download?version=2',
    bst_resting='https://osf.io/m7bd3/download?version=3',
    fake=('https://github.com/mne-tools/mne-testing-data/raw/master/'
          'datasets/foo.tgz'),
    misc=('https://codeload.github.com/mne-tools/mne-misc-data/tar.gz/'
          f'{RELEASES["misc"]}'),
    sample='https://osf.io/86qa2/download?version=5',
    somato='https://osf.io/tp4sg/download?version=7',
    spm='https://osf.io/je4s8/download?version=2',
    testing=('https://codeload.github.com/mne-tools/mne-testing-data/'
             f'tar.gz/{RELEASES["testing"]}'),
    multimodal='https://ndownloader.figshare.com/files/5999598',
    fnirs_motor='https://osf.io/dj3eh/download?version=1',
    opm='https://osf.io/p6ae7/download?version=2',
    visual_92_categories_1='https://osf.io/8ejrs/download?version=1',
    visual_92_categories_2='https://osf.io/t4yjp/download?version=1',
    mtrf='https://osf.io/h85s2/download?version=1',
    kiloword='https://osf.io/qkvf9/download?version=1',
    fieldtrip_cmc='https://osf.io/j9b6s/download?version=1',
    phantom_4dbti='https://osf.io/v2brw/download?version=2',
    refmeg_noise='https://osf.io/drt6v/download?version=1',
    hf_sef_raw='https://zenodo.org/record/889296/files/hf_sef_raw.tar.gz',
    hf_sef_evoked=('https://zenodo.org/record/3523071/files/'
                   'hf_sef_evoked.tar.gz'),
    ssvep='https://osf.io/z8h6k/download?version=5',
    erp_core='https://osf.io/rzgba/download?version=1',
    epilepsy_ecog='https://osf.io/z4epq/download?revision=1',
)

# the name of the zipped file for each dataset
ARCHIVE_NAMES = dict(
    bst_auditory='bst_auditory.tar.gz',
    bst_phantom_ctf='bst_phantom_ctf.tar.gz',
    bst_phantom_elekta='bst_phantom_elekta.tar.gz',
    bst_raw='bst_raw.tar.gz',
    bst_resting='bst_resting.tar.gz',
    fake='foo.tgz',
    fieldtrip_cmc='SubjectCMC.zip',
    kiloword='MNE-kiloword-data.tar.gz',
    misc=f'{MISC_VERSIONED}.tar.gz',
    mtrf='mTRF_1.5.zip',
    multimodal='MNE-multimodal-data.tar.gz',
    fnirs_motor='MNE-fNIRS-motor-data.tgz',
    opm='MNE-OPM-data.tar.gz',
    sample='MNE-sample-data-processed.tar.gz',
    somato='MNE-somato-data.tar.gz',
    spm='MNE-spm-face.tar.gz',
    testing=f'{TESTING_VERSIONED}.tar.gz',
    visual_92_categories_1='MNE-visual_92_categories-data-part1.tar.gz',
    visual_92_categories_2='MNE-visual_92_categories-data-part2.tar.gz',
    phantom_4dbti='MNE-phantom-4DBTi.zip',
    refmeg_noise='sample_reference_MEG_noise-raw.zip',
    hf_sef_raw='hf_sef_raw.tar.gz',
    hf_sef_evoked='hf_sef_evoked.tar.gz',
    ssvep='ssvep_example_data.zip',
    erp_core='MNE-ERP-CORE-data.tar.gz',
    epilepsy_ecog='MNE-epilepsy-ecog-data.tar.gz',
)
FOLDER_NAMES = dict(
    bst_auditory='MNE-brainstorm-data',
    bst_phantom_ctf='MNE-brainstorm-data',
    bst_phantom_elekta='MNE-brainstorm-data',
    bst_raw='MNE-brainstorm-data',
    bst_resting='MNE-brainstorm-data',
    fake='foo',
    fieldtrip_cmc='MNE-fieldtrip_cmc-data',
    kiloword='MNE-kiloword-data',
    misc='MNE-misc-data',
    mtrf='mTRF_1.5',
    multimodal='MNE-multimodal-data',
    fnirs_motor='MNE-fNIRS-motor-data',
    opm='MNE-OPM-data',
    sample='MNE-sample-data',
    somato='MNE-somato-data',
    spm='MNE-spm-face',
    testing='MNE-testing-data',
    visual_92_categories='MNE-visual_92_categories-data',
    phantom_4dbti='MNE-phantom-4DBTi',
    refmeg_noise='MNE-refmeg-noise-data',
    hf_sef_raw='HF_SEF',
    hf_sef_evoked='HF_SEF',
    ssvep='ssvep-example-data',
    erp_core='MNE-ERP-CORE-data',
    epilepsy_ecog='MNE-epilepsy-ecog-data',
)

# the environment configuration keys for each dataset name
CONFIG_KEYS = dict(
    fake='MNE_DATASETS_FAKE_PATH',
    misc='MNE_DATASETS_MISC_PATH',
    sample='MNE_DATASETS_SAMPLE_PATH',
    spm='MNE_DATASETS_SPM_FACE_PATH',
    somato='MNE_DATASETS_SOMATO_PATH',
    bst_auditory='MNE_DATASETS_BRAINSTORM_PATH',
    bst_phantom_ctf='MNE_DATASETS_BRAINSTORM_PATH',
    bst_phantom_elekta='MNE_DATASETS_BRAINSTORM_PATH',
    bst_raw='MNE_DATASETS_BRAINSTORM_PATH',
    bst_resting='MNE_DATASETS_BRAINSTORM_PATH',
    testing='MNE_DATASETS_TESTING_PATH',
    multimodal='MNE_DATASETS_MULTIMODAL_PATH',
    fnirs_motor='MNE_DATASETS_FNIRS_MOTOR_PATH',
    opm='MNE_DATASETS_OPM_PATH',
    visual_92_categories='MNE_DATASETS_VISUAL_92_CATEGORIES_PATH',
    kiloword='MNE_DATASETS_KILOWORD_PATH',
    mtrf='MNE_DATASETS_MTRF_PATH',
    fieldtrip_cmc='MNE_DATASETS_FIELDTRIP_CMC_PATH',
    phantom_4dbti='MNE_DATASETS_PHANTOM_4DBTI_PATH',
    refmeg_noise='MNE_DATASETS_REFMEG_NOISE_PATH',
    hf_sef_raw='MNE_DATASETS_HF_SEF_PATH',
    hf_sef_evoked='MNE_DATASETS_HF_SEF_PATH',
    ssvep='MNE_DATASETS_SSVEP_PATH',
    erp_core='MNE_DATASETS_ERP_CORE_PATH',
    epilepsy_ecog='MNE_DATASETS_EPILEPSY_ECOG_PATH',
)
assert set(ARCHIVE_NAMES) == set(URLS)
