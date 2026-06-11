#!/usr/bin/env bash

# Solution to handle error regarding variable SUBJECTS_DIR:
mkdir -p ~/dummy_freesurfer
export SUBJECTS_DIR=~/dummy_freesurfer

# Define relative paths:
RAW_DIR="../../dataset/raw"
DERIV_DIR="../../dataset/derivatives"
FS_LICENSE="../../freesurfer/license.txt"
WORK_BASE="/tmp/fmriprep_works"
LOG_FILE="fmriprep_status.txt"

# Define the number of subjects to process at once:
BATCH_SIZE=1

# Create log file if it does not exist (do NOT empty it):
touch "$LOG_FILE"

# Get subject list (removing "sub-" prefix):
subjects=$(ls "$RAW_DIR" | grep '^sub-' | sed 's/^sub-//')

# Function to check SUCCESS only:
is_success() {
    grep -q "^${1} SUCCESS" "$LOG_FILE"
}

# Build list of subjects to process (retry the FAILED subjects automatically):
to_process=()
for subj in $subjects; do
    if is_success "$subj"; then
        echo "fMRIPrep already done for $subj, skipping."
    else
        echo "Subject $subj will be processed (new or previous FAILURE)."
        to_process+=("$subj")
    fi
done

# Total subjects to process:
total=${#to_process[@]}

# Loop over all subjects:
for ((i=0; i<total; i+=BATCH_SIZE)); do
    batch=("${to_process[@]:i:BATCH_SIZE}")

    echo "Processing subjects: ${batch[*]}"

    ###### fMRIPrep: T1 and BOLD Preprocessing ######

    # Start timer for this batch:
    SECONDS=0

    # Define and create the subject-specific working directory:
    WORK_DIR="${WORK_BASE}/batch_${i}"
    mkdir -p "$WORK_DIR"

    # Run fmriprep for the current subject:
    if fmriprep \
        "$RAW_DIR" \
        "$DERIV_DIR" \
        participant \
        --derivatives "$DERIV_DIR" \
        --participant-label "${batch[@]}" \
        --skip-bids-validation \
        --n-cpus 8 \
        --omp-nthreads 7 \
        --mem-mb 16000 \
        --output-spaces MNI152NLin2009cAsym:res-2 \
        --bold2anat-init t2w \
        --random-seed 34 \
        --fs-license-file "$FS_LICENSE" \
        --fs-no-reconall \
        --work-dir "$WORK_DIR" \
        --notrack
    then
        status="SUCCESS"

        # Delete subject-specific work folder immediately after successful preprocessing:
        echo "Deleting temporary work folder for the current batch..."
        rm -rf "$WORK_DIR"
    else
        status="FAILURE"
    fi

    # Calculate elapsed time:
    elapsed=$SECONDS

    # Join subject names with commas
    batch_names=$(IFS=,; echo "${batch[*]}")

    # Log status with elapsed time:
    echo "$batch_names $status (Elapsed time: ${elapsed}s)" >> "$LOG_FILE"

    echo "Finished fMRIPrep for subjects [$batch_names] in ${elapsed}s"

done

echo "All subjects preprocessed with fmriprep. Log saved to $LOG_FILE."