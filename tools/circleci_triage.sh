#!/bin/bash -e

set -o pipefail

# Works out what this build needs, before anything is restored or downloaded:
# build.txt and pattern.txt for the doc build, datasets.txt for
# circleci_download.sh, and cache_keys/<cache>.txt for the cache keys in
# .circleci/config.yml. A cache marked noop hashes to a key nothing has ever
# saved, so its restore misses and costs nothing -- which is what most PRs want,
# since they touch no example and so need no dataset at all.

# dataset | cache holding it | an example needs the dataset if this matches one
# of its lines in full
DATASETS="
sample|sample|.*datasets.*sample.*
hcp_mmp_parcellation|sample|.*datasets.*hcp_mmp_parcellation.*
fsaverage|fsaverage|.*datasets.*fetch_fsaverage.*
spm_face|spm-face|.*datasets.*spm_face.*
somato|somato|.*datasets.*somato.*
eegbci|tiny|.*datasets.*eegbci.*
misc|tiny|.*datasets.*misc.*
kiloword|tiny|.*datasets.*kiloword.*
mtrf|tiny|.*datasets.*mtrf.*
phantom_4dbti|tiny|.*datasets.*phantom_4dbti.*
sleep_physionet|tiny|.*datasets.*sleep_physionet.*
fnirs_motor|tiny|.*datasets.*fnirs_motor.*
refmeg_noise|tiny|.*datasets.*refmeg_noise.*
hf_sef|hf-sef|.*datasets.*hf_sef.*
bst_auditory|bst-auditory|.*brainstorm.*bst_auditory.*
bst_resting|bst-resting|.*brainstorm.*bst_resting.*
bst_raw|bst-raw|.*brainstorm.*bst_raw.*
bst_phantom_ctf|bst-phantom-ctf|.*brainstorm.*bst_phantom_ctf.*
bst_phantom_elekta|bst-phantom-elekta|.*brainstorm.*bst_phantom_elekta.*
phantom_kernel|bst-phantom-kernel|.*datasets.*phantom_kernel.*
testing|testing|.*datasets.*testing.*
fieldtrip_cmc|fieldtrip|.*datasets.*fieldtrip_cmc.*
multimodal|multimodal|.*datasets.*multimodal.*
opm|opm|.*datasets[^_]*opm.*
limo|limo|.*datasets.*limo.*
ucl_opm_auditory|ucl-opm-auditory|.*datasets.*ucl_opm_auditory.*
phantom_kit|phantom-kit|.*datasets.*phantom_kit.*
visual_92_categories|visual|.*datasets.*visual_92_categories.*
ds004388|ds004388|.*ds004388.*
ssvep|ssvep|.*datasets.*ssvep.*
epilepsy_ecog|epilepsy-ecog|.*datasets.*epilepsy_ecog.*
erp_core|erp-core|.*datasets.*erp_core.*
eyelink|eyelink|.*datasets.*eyelink.*
"

want() {
    grep -qxF "$1" datasets.txt || echo "$1" >> datasets.txt
}

: > datasets.txt
echo "export OPENBLAS_NUM_THREADS=4" >> $BASH_ENV
echo "export MNE_DOC_BUILD_N_JOBS=1" >> $BASH_ENV

if { [[ "$CIRCLE_BRANCH" == "main" ]] || [[ $(cat gitlog.txt) == *"[circle full]"* ]] || [[ "$CIRCLE_BRANCH" == "maint/"* ]] ; } && [[ "$CIRCLE_PROJECT_USERNAME" == "mne-tools" ]]; then
    echo "Doing a full build";
    echo html-memory > build.txt;
    echo "export OPENBLAS_NUM_THREADS=1" >> $BASH_ENV
    echo "export MNE_DOC_BUILD_N_JOBS=4" >> $BASH_ENV
    # the full build downloads more than DATASETS covers (infant_template, the
    # phantom and parcellation fetchers, ...), so it is all or nothing
    want all
else
    echo "Doing a partial build";
    FNAMES=$(git diff --name-only $(git merge-base $CIRCLE_BRANCH upstream/main) $CIRCLE_BRANCH);
    if [[ $(cat gitlog.txt) == *"[circle front]"* ]]; then
        FNAMES="tutorials/inverse/30_mne_dspm_loreta.py tutorials/machine-learning/30_strf.py tutorials/machine-learning/50_decoding.py tutorials/stats-source-space/20_cluster_1samp_spatiotemporal.py tutorials/evoked/20_visualize_evoked.py "${FNAMES};
        want testing
    fi;
    echo FNAMES="$FNAMES";
    for FNAME in $FNAMES; do
        if [[ $(echo "$FNAME" | grep -P '^(tutorials|examples)(/.*)?/((?!sgskip).)*\.py$') ]] ; then
            echo "Checking example $FNAME ...";
            PATTERN=$(basename $FNAME)"\\|"$PATTERN;
            while IFS='|' read -r NAME CACHE MATCH; do
                if [[ -n "$NAME" ]] && grep -qx "$MATCH" $FNAME; then
                    want $NAME
                fi
            done <<< "$DATASETS"
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

mkdir -p cache_keys
for CACHE in $(echo "$DATASETS" | cut -d '|' -f 2 | sort -u); do
    echo noop > cache_keys/$CACHE.txt
done
while IFS='|' read -r NAME CACHE MATCH; do
    if [[ -n "$NAME" ]] && grep -qxF -e "$NAME" -e all datasets.txt; then
        echo real > cache_keys/$CACHE.txt
    fi
done <<< "$DATASETS"

echo "Datasets wanted: $(tr '\n' ' ' < datasets.txt)"
grep -H . cache_keys/*.txt
