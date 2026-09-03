#!/bin/bash -e

set -o pipefail

# Fetches the datasets circleci_triage.sh listed in datasets.txt, which is only
# the ones this build's examples actually use; everything else was either
# restored from a cache or is not needed at all. Most datasets just want their
# data_path, so only the ones that need something else are spelled out here.

export MNE_TQDM=off

if grep -qxF all datasets.txt; then
    python -c "import mne; mne.datasets._download_all_example_data()";
    exit 0
fi

while read -r NAME; do
    echo "Getting $NAME ...";
    case $NAME in
        fsaverage)
            python -c "import mne; print(mne.datasets.fetch_fsaverage())";;
        hcp_mmp_parcellation)
            python -c "import mne; print(mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=mne.datasets.sample.data_path() / 'subjects', accept=True))";;
        eegbci)
            python -c "import mne; print([mne.datasets.eegbci.load_data(subject, runs, update_path=True) for subject, runs in [(1, [3, 6, 10, 14]), (2, [3]), (3, [3]), (4, [3])]])";;
        sleep_physionet)
            python -c "import mne; print(mne.datasets.sleep_physionet.age.fetch_data([0, 1], recording=[1]))";;
        bst_*)
            python -c "import mne; print(mne.datasets.brainstorm.$NAME.data_path(update_path=True, accept=True))";;
        limo)
            python -c "import mne; print(mne.datasets.limo.data_path(subject=1, update_path=True))";;
        erp_core)
            python -c "import mne; print(mne.datasets.erp_core.data_path(update_path=True))";
            python -c "import mne; print([mne.datasets.erp_core.fetch_file(f'sub-001/eeg/sub-001_task-N170_{suffix}') for suffix in ['eeg.fdt', 'eeg.set', 'events.tsv']])";;
        ds004388)
            python -c "
import glob, mne, openneuro
target_dir = mne.datasets.default_path() / 'ds004388'
if not glob.glob(str(target_dir / 'sub-001/eeg/*median_run-03_eeg*.set')):
    target_dir.mkdir(exist_ok=True)
    openneuro.download(dataset='ds004388', target_dir=target_dir, include='sub-001/eeg/*median_run-03_eeg*')
";;
        *)
            python -c "import mne; print(mne.datasets.$NAME.data_path(update_path=True))";;
    esac
done < datasets.txt
