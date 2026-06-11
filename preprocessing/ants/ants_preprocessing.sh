#!/usr/bin/env bash

# Define relative paths:
RAW_DIR="../../dataset/raw"
DERIV_DIR="../../dataset/derivatives"
TEMPLATE_DIR="../../templateflow/tpl-MNI152NLin2009cAsym"
ANTS_DIR="$PWD"
LOG_FILE="ants_status.txt"
FMRIPREP_LOG="../fmriprep/fmriprep_status.txt"

# Create log file if it does not exist (do NOT empty it):
touch "$LOG_FILE"

# Get subject list (removing "sub-" prefix):
subjects=$(ls "$RAW_DIR" | grep '^sub-' | sed 's/^sub-//')

# Loop over all subjects:
for subj in $subjects; do

    # Check fmriprep_status.txt: skip if not SUCCESS
    if ! grep -q "^$subj SUCCESS" "$FMRIPREP_LOG"; then
        echo "Skipping $subj: fMRIPrep did not succeed."
        continue
    fi

    echo "Processing subject: $subj"

    ###### T2 preprocessing ######
    
    # Skip T2 preprocessing if already done:
    if grep -q "^${subj} SUCCESS" "$LOG_FILE"; then
        echo "Skipping T2 preprocessing for $subj (already SUCCESS)."
        continue
    fi

    # Start timer for T2:
    SECONDS=0

    # Find T2 file: prefer *T2w*:
    T2_FILE=$(find "$RAW_DIR/sub-$subj"/ses-*/anat -maxdepth 1 -type f \( -name "*T2w*.nii*" ! -name "*TSE*" \) | head -n1)
    if [ -z "$T2_FILE" ]; then
        T2_FILE=$(find "$RAW_DIR/sub-$subj"/ses-*/anat -maxdepth 1 -type f -name "*T2w*.nii*" | head -n1)
    fi
    
    # Get the directory containing the T2 file:
    T2_DIR=$(dirname "$T2_FILE")

    # Go up one level to get the session folder (parent of 'anat')
    SESSION_DIR=$(dirname "$T2_DIR")
    
    # Get just the session name
    SESSION=$(basename "$SESSION_DIR")
    
    # Map raw path to derivatives path:
    OUT_DIR="${T2_DIR/$RAW_DIR/$DERIV_DIR/}/ants"

    # Create derivatives anat folder if it doesn't exist
    mkdir -p "$OUT_DIR"
        
    # Set output prefix for ANTs:
    OUT_PREFIX="$OUT_DIR/sub-${subj}_${SESSION}_desc-preproc_T2w_"
    
    singularity exec --bind "$ANTS_DIR":"$ANTS_DIR" \
      "$ANTS_DIR/ants.sif" \
      "$ANTS_DIR/antsBrainExtraction.sh" \
      -d 3 \
      -a "$T2_FILE" \
      -e "$TEMPLATE_DIR/tpl-MNI152NLin2009cAsym_res-02_T2w.nii.gz" \
      -m "$TEMPLATE_DIR/tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz" \
      -o "$OUT_PREFIX"

    # Define paths for warping:
    MOVING_IMAGE="${OUT_PREFIX}BrainExtractionBrain.nii.gz"
    TRANSFORM_FILE="$(dirname "$OUT_DIR")/sub-${subj}_${SESSION}_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5"
    OUTPUT_WARPED="$(dirname "$OUT_DIR")/sub-${subj}_${SESSION}_space-MNI152NLin2009cAsym_res-2_desc-preproc_T2w.nii.gz"

    singularity exec --bind "$ANTS_DIR":"$ANTS_DIR" \
      "$ANTS_DIR/ants.sif" \
      antsApplyTransforms \
        -d 3 \
        -i "$MOVING_IMAGE" \
        -r "$TEMPLATE_DIR/tpl-MNI152NLin2009cAsym_res-02_T2w.nii.gz" \
        -o "$OUTPUT_WARPED" \
        -t "$TRANSFORM_FILE" \
        -n Linear
      
    # Calculate elapsed time for T2:
    elapsed=$SECONDS
    echo "$subj SUCCESS (Elapsed: ${elapsed}s)" >> "$LOG_FILE"
    echo "Finished T2 preprocessing for $subj in ${elapsed}s"

done

echo "All subjects processed. Log saved to $LOG_FILE."