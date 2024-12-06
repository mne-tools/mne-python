#!/bin/bash -e

set -o pipefail
export MNE_TQDM=off
echo "export OPENBLAS_NUM_THREADS=4" >> $BASH_ENV
echo "export MNE_DOC_BUILD_N_JOBS=1" >> $BASH_ENV

if [ "$CIRCLE_BRANCH" == "main" ] || [[ $(cat gitlog.txt) == *"[circle full]"* ]] || [[ "$CIRCLE_BRANCH" == "maint/"* ]]; then
    echo "Doing a full build";
    echo html-memory > build.txt;
    echo "export OPENBLAS_NUM_THREADS=1" >> $BASH_ENV
    echo "export MNE_DOC_BUILD_N_JOBS=4" >> $BASH_ENV
    python -c "import mne; mne.datasets._download_all_example_data()";
else
    echo "Doing a partial build";
    FNAMES=$(git diff --name-only $(git merge-base $CIRCLE_BRANCH upstream/main) $CIRCLE_BRANCH);
    if [[ $(cat gitlog.txt) == *"[circle front]"* ]]; then
        FNAMES="tutorials/inverse/30_mne_dspm_loreta.py tutorials/machine-learning/30_strf.py tutorials/machine-learning/50_decoding.py tutorials/stats-source-space/20_cluster_1samp_spatiotemporal.py tutorials/evoked/20_visualize_evoked.py "${FNAMES};
        python -c "import mne; print(mne.datasets.testing.data_path(update_path=True))";
    fi;
    echo FNAMES="$FNAMES";
    for FNAME in $FNAMES; do
        if [[ $(echo "$FNAME" | grep -P '^(tutorials|examples)(/.*)?/((?!sgskip).)*\.py$') ]] ; then
            echo "Checking example $FNAME ...";
            PATTERN=$(basename $FNAME)"\\|"$PATTERN;
            if [[ $(cat $FNAME | grep -x ".*datasets.*sample.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.sample.data_path(update_path=True))";
            fi;
            if [[ $(cat $FNAME | grep -x ".*datasets.*fetch_fsaverage.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.fetch_fsaverage(verbose=True))";
            fi;
            if [[ $(cat $FNAME | grep -x ".*datasets.*spm_face.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.spm_face.data_path(update_path=True))";
            fi;
            if [[ $(cat $FNAME | grep -x ".*datasets.*somato.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.somato.data_path(update_path=True))";
            fi;
            if [[ $(cat $FNAME | grep -x ".*datasets.*eegbci.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.eegbci.load_data(1, [3, 6, 10, 14], update_path=True))";
                python -c "import mne; print(mne.datasets.eegbci.load_data(2, [3], update_path=True))";
                python -c "import mne; print(mne.datasets.eegbci.load_data(3, [3], update_path=True))";
                python -c "import mne; print(mne.datasets.eegbci.load_data(4, [3], update_path=True))";
            fi;
            if [[ $(cat $FNAME | grep -x ".*datasets.*sleep_physionet.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.sleep_physionet.age.fetch_data([0, 1], recording=[1]))";
            fi;
            if [[ $(cat $FNAME | grep -x ".*datasets.*hf_sef.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.hf_sef.data_path(update_path=True))";
            fi;
            if [[ $(cat $FNAME | grep -x ".*brainstorm.*bst_auditory.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.brainstorm.bst_auditory.data_path(update_path=True, accept=True))";
            fi;
            if [[ $(cat $FNAME | grep -x ".*brainstorm.*bst_resting.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.brainstorm.bst_resting.data_path(update_path=True, accept=True))";
            fi;
            if [[ $(cat $FNAME | grep -x ".*brainstorm.*bst_raw.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.brainstorm.bst_raw.data_path(update_path=True, accept=True))";
            fi;
            if [[ $(cat $FNAME | grep -x ".*brainstorm.*bst_phantom_ctf.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.brainstorm.bst_phantom_ctf.data_path(update_path=True, accept=True))";
            fi;
            if [[ $(cat $FNAME | grep -x ".*brainstorm.*bst_phantom_elekta.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.brainstorm.bst_phantom_elekta.data_path(update_path=True, accept=True))";
            fi;
            if [[ $(cat $FNAME | grep -x ".*datasets.*phantom_kernel.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.phantom_kernel.data_path(update_path=True))";
            fi;
            if [[ $(cat $FNAME | grep -x ".*datasets.*hcp_mmp_parcellation.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.sample.data_path(update_path=True))";
                python -c "import mne; print(mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=mne.datasets.sample.data_path() / 'subjects', accept=True))";
            fi;
            if [[ $(cat $FNAME | grep -x ".*datasets.*misc.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.misc.data_path(update_path=True))";
            fi;
            if [[ $(cat $FNAME | grep -x ".*datasets.*testing.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.testing.data_path(update_path=True))";
            fi;
            if [[ $(cat $FNAME | grep -x ".*datasets.*kiloword.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.kiloword.data_path(update_path=True))";
            fi;
            if [[ $(cat $FNAME | grep -x ".*datasets.*mtrf.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.mtrf.data_path(update_path=True))";
            fi;
            if [[ $(cat $FNAME | grep -x ".*datasets.*fieldtrip_cmc.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.fieldtrip_cmc.data_path(update_path=True))";
            fi;
            if [[ $(cat $FNAME | grep -x ".*datasets.*multimodal.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.multimodal.data_path(update_path=True))";
            fi;
            if [[ $(cat $FNAME | grep -x ".*datasets.*fnirs_motor.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.fnirs_motor.data_path(update_path=True))";
            fi;
            if [[ $(cat $FNAME | grep -x ".*datasets[^_]*opm.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.opm.data_path(update_path=True))";
            fi;
            if [[ $(cat $FNAME | grep -x ".*datasets.*phantom_4dbti.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.phantom_4dbti.data_path(update_path=True))";
            fi;
            if [[ $(cat $FNAME | grep -x ".*datasets.*limo.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.limo.data_path(subject=1, update_path=True))";
            fi;
            if [[ $(cat $FNAME | grep -x ".*datasets.*refmeg_noise.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.refmeg_noise.data_path(update_path=True))";
            fi;
            if [[ $(cat $FNAME | grep -x ".*datasets.*ssvep.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.ssvep.data_path(update_path=True))";
            fi;
            if [[ $(cat $FNAME | grep -x ".*datasets.*epilepsy_ecog.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.epilepsy_ecog.data_path(update_path=True))";
            fi;
            if [[ $(cat $FNAME | grep -x ".*datasets.*erp_core.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.erp_core.data_path(update_path=True))";
            fi;
            if [[ $(cat $FNAME | grep -x ".*datasets.*eyelink.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.eyelink.data_path(update_path=True))";
            fi;
            if [[ $(cat $FNAME | grep -x ".*datasets.*ucl_opm_auditory.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.ucl_opm_auditory.data_path(update_path=True))";
            fi;
            if [[ $(cat $FNAME | grep -x ".*datasets.*phantom_kit.*" | wc -l) -gt 0 ]]; then
                python -c "import mne; print(mne.datasets.phantom_kit.data_path(update_path=True))";
            fi;
        fi;
    done;
    echo PATTERN="$PATTERN";
    echo html-pattern-memory > build.txt;
    if [[ $PATTERN ]]; then
        PATTERN="\(${PATTERN::-2}\)";
    else
        PATTERN="run_no_examples_or_tutorials"
    fi;
fi;
echo "$PATTERN" > pattern.txt;
