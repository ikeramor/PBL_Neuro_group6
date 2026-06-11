#!/bin/bash

echo "-----------------------------------------------------------------------------"
echo "T1T2 Processing Script (CI)"
echo "-----------------------------------------------------------------------------"
echo ""

# Save the current LC_NUMERIC setting
original_lc_numeric=$LC_NUMERIC
export LC_NUMERIC=en_US.UTF-8

# Path to derivatives directory (inside this folder all the preprocessed subjects are located)
base_dir="/home/jovyan/Desktop/PBL_Neuro/dataset/derivatives"

# Output directory for the results
output_central="/home/jovyan/Desktop/PBL_Neuro/myelin_maps"
mkdir -p "$output_central"

echo "-----------------------------------------------------------------------------"
echo "Starting Batch Processing..."
echo "Results will be saved in: $output_central"
echo "-----------------------------------------------------------------------------"

# --- LOOP THROUGH SUBJECTS ---
# looks for all 'anat' folders in sub-*/ses-*/ structure
for anat_dir in "${base_dir}"/sub-*/ses-*/anat; do
    
    # Check if directory exists, if not, subject is skipped (continue skips the following lines and starts next iteration (subject))
    [ -d "$anat_dir" ] || continue
    
    # Extract Subject ID from the path
    sub_id=$(echo "$anat_dir" | grep -o "sub-[^/]*") #grep saves the subject id in sub_id (coge todos los caracteres hasta la siguiente /)
    id="${sub_id}"
    
    echo ""
    echo ">>> Processing: $id"
    
    # just with the ending part of the file names (which is the same for all subjects) # /dev/null removes the error if no file is found 
    t1_file=$(ls "${anat_dir}"/*_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w.nii.gz 2>/dev/null)
    t2_file=$(ls "${anat_dir}"/*_space-MNI152NLin2009cAsym_res-2_desc-preproc_T2w.nii.gz 2>/dev/null)
    pve_gm=$(ls "${anat_dir}"/*_space-MNI152NLin2009cAsym_res-2_label-GM_*probseg.nii.gz 2>/dev/null | head -n 1) #algunos terminan en MNI152NLin2009cAsym_res-2_label-GM_probseg.nii.gz y otros MNI152NLin2009cAsym_res-2_label-GM_desc-preproc_probseg.nii.gz
    mask_brain=$(ls "${anat_dir}"/*_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz 2>/dev/null) #al pricnipio el fallo aqui porque se hacia referencia a las mascaras original y la del espacio mni152 a la vez
    
    # Check if all 4 files were found (tag -z checks if any of the variables are empty)
    if [[ -z "$t1_file" || -z "$t2_file" || -z "$pve_gm" || -z "$mask_brain" ]]; then
        echo "    [SKIP] Missing files for $id. Skipping..."
        continue # goes to the next subject
    fi
    
    # Local log for this subject (stored in the central output folder)
    log_file="${output_central}/${id}_log.txt"
    echo "T1T2 Log for ${id}" > "$log_file"
    echo "Processing started at: $(date +"%F %T")" >> "$log_file"
    
    # --- PROCESSING STEPS ---
    
    # Step 1: Gray matter mask
    fslmaths "$pve_gm" -thr 0.9 -bin "$output_central/${id}_GM_mask"
    
    # Step 2: Applying brain mask to T1w and T2w
    fslmaths "$t1_file" -mul "$mask_brain" "$output_central/${id}_T1w_mask.nii.gz"
    fslmaths "$t2_file" -mul "$mask_brain" "$output_central/${id}_T2w_mask.nii.gz"
    
    # Step 3: Scaling factor
    t1_cortex_median=$(fslstats "$output_central/${id}_T1w_mask.nii.gz" -k "$output_central/${id}_GM_mask" -p 50)
    t2_cortex_median=$(fslstats "$output_central/${id}_T2w_mask.nii.gz" -k "$output_central/${id}_GM_mask" -p 50)
    
    scaling_factor=$(bc <<< "scale=10 ; $t1_cortex_median / $t2_cortex_median")
    
    echo "    Scaling factor: $scaling_factor"
    echo "T1w median: $t1_cortex_median" >> "$log_file"
    echo "T2w median: $t2_cortex_median" >> "$log_file"
    echo "Scaling factor: $scaling_factor" >> "$log_file"
    
    # Step 4: Misaki formula
    fslmaths "$output_central/${id}_T2w_mask.nii.gz" -mul $scaling_factor "$output_central/${id}_T2w_scaled.nii.gz"
    fslmaths "$output_central/${id}_T1w_mask.nii.gz" -sub "$output_central/${id}_T2w_scaled.nii.gz" "$output_central/${id}_T1_minus_T2s.nii.gz"
    fslmaths "$output_central/${id}_T1w_mask.nii.gz" -add "$output_central/${id}_T2w_scaled.nii.gz" "$output_central/${id}_T1_plus_T2s.nii.gz"
    fslmaths "$output_central/${id}_T1_minus_T2s.nii.gz" -div "$output_central/${id}_T1_plus_T2s.nii.gz" "$output_central/${id}_sT1T2.nii.gz"
    
    # Check if result exists
    if [[ -e "$output_central/${id}_sT1T2.nii.gz" ]]; then
        echo "    [SUCCESS] Created ${id}_sT1T2.nii.gz"
        echo "Successfully created T1w/T2w image." >> "$log_file"
        
        # This removes the files from the itnermediate steps,as we only want to keep the myelin map
        rm "$output_central/${id}_GM_mask.nii.gz" "$output_central/${id}_T1w_mask.nii.gz" "$output_central/${id}_T2w_mask.nii.gz" "$output_central/${id}_T2w_scaled.nii.gz" "$output_central/${id}_T1_minus_T2s.nii.gz" "$output_central/${id}_T1_plus_T2s.nii.gz"
    else
        echo "    [ERROR] Failed for $id"
        echo "ERROR - Image creation failed." >> "$log_file"
    fi

done

echo ""
echo "-----------------------------------------------------------------------------"
echo "Bulk Processing Finished."
echo "-----------------------------------------------------------------------------"

# Restore the original LC_NUMERIC setting
export LC_NUMERIC=$original_lc_numeric