# Import necessary libraries:
from nilearn.datasets import fetch_atlas_schaefer_2018
from nilearn.maskers import NiftiLabelsMasker

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import matplotlib.patches as patches

import numpy as np
import pandas as pd
import os
from collections import defaultdict
from pathlib import Path
from tqdm.auto import tqdm


# Load Schaefer 2018 atlas (300 ROIs, 7 networks, 2 mm resolution):
atlas = fetch_atlas_schaefer_2018(n_rois=300, yeo_networks=7, resolution_mm=2)

# Get the corresponsing labels and remove the background:
labels = atlas["labels"][1:]

# Define a dictionary with network abbreviations as keys and network full names as values:
network_map = {
    "Vis": "Visual Network (VIN)",
    "SomMot": "Somatomotor Network (SMN)",
    "DorsAttn": "Dorsal Attention Network (DAN)",
    "SalVentAttn": "Salience / Ventral Attention Network (SAN/VAN)",
    "Limbic": "Limbic Network (LIN)",
    "Cont": "Control / Frontoparietal Network (FPN/CON)",
    "Default": "Default Mode Network (DMN)"
}

# Initialize an empty dictionary:
network_dict = defaultdict(lambda: {"name": None, "indices": [], "LH": [], "RH": []})

for i, label in enumerate(labels):

    # Get the different parts of the node label:
    parts = label.split("_")

    # Get the hemisphere of the node label:
    hemi = parts[1]

    # Get the abbreviation name of the network:
    network = parts[2]

    # Save both the network name and the index of the label: 
    network_dict[network]["name"] = network_map[network]
    network_dict[network]["indices"].append(i)

    # Save hemisphere information:
    if hemi == "LH":
        network_dict[network]["LH"].append(i)
    else:
        network_dict[network]["RH"].append(i)

# Turn the dictionary into a normal one:
network_dict = dict(network_dict)

# Print summary:
for net, info in network_dict.items():
    print("\n========================")
    print(net)
    print(info["name"])
    print("Total ROIs:", len(info["indices"]))
    print("LH ROIs:", len(info["LH"]))
    print("RH ROIs:", len(info["RH"]))

    # Get a list of network name abbreviations:
networks = list(network_map.keys())

# Define network color mapping:
network_colors = {
    "Vis": "#3700D4",
    "SomMot": "#2462B3",
    "DorsAttn": "#3A6D1C",
    "SalVentAttn": "#6F43B2",
    "Limbic": "#BEF264",
    "Cont": "#DD1952",
    "Default": "#F8A520"
}

# Initialize an empty sizes vector:
sizes = []

# LH blocks first:
for net in networks:
    n = len(network_dict[net]["LH"])
    sizes.append(n)

# RH blocks next:
for net in networks:
    n = len(network_dict[net]["RH"])
    sizes.append(n)

# Compute cumulative boundaries:
boundaries = np.cumsum(sizes)

# Get the functional atlas image:
atlas_img = atlas["maps"]

# Define myelin map directory:
myelin_maps_dir = "/home/jovyan/Desktop/PBL_Neuro/postprocessing/myelin_maps"

# Create a masker that uses the Schaefer atlas labels to define ROIs:
masker = NiftiLabelsMasker(labels_img=atlas_img)

# Get a sorted list of all _sT1T2.nii.gz files in the directory (filtering out other NIfTI files)
myelin_files = sorted(list(Path(myelin_maps_dir).glob("*_sT1T2.nii.gz")))
print(f"\nFound {len(myelin_files)} myelin maps. Processing...")

# Initialize a dictionary to store all vectors
all_subjects_vectors = {}

for file_path in tqdm(myelin_files, desc="Extracting myelin vectors"):
    # Extract subject ID from filename (e.g., "sub-OAS30001_sT1T2.nii.gz" -> "sub-OAS30001")
    filename = file_path.name
    sub_id = filename.split('_')[0]
    
    # Fit the masker to extract the mean myelin value for each ROI
    myelin_values_2d = masker.fit_transform(str(file_path))
    # Nos aseguramos de que sea un vector 1D (300,)
    myelin_vector = np.squeeze(myelin_values_2d)
    
    # Save the vector in the dictionary
    all_subjects_vectors[sub_id] = myelin_vector

# Create a final DataFrame explicitly passing the values as a list to avoid shape errors
final_df = pd.DataFrame(
    data=list(all_subjects_vectors.values()), 
    index=list(all_subjects_vectors.keys()), 
    columns=labels
)
final_df.index.name = 'Subject_ID'

print("\nExtracción completada. Primeros sujetos y regiones:")
print(final_df.iloc[:5, :5]) # Print first 5 subjects and first 5 ROIs

# Save the DataFrame to a CSV file (region level)
output_csv = "myelin_features_all_subjects.csv"
final_df.to_csv(output_csv)
print(f"\nResultados por regiones guardados exitosamente en: {output_csv}")

# --- Extract network-level myelin features (mean across each of the 7 networks) ---
myelin_networks_df = pd.DataFrame(index=final_df.index)

for net in networks:
    # Capture columns belonging to this network for BOTH Left and Right hemispheres
    net_cols = [col for col in final_df.columns if f"_LH_{net}_" in col or f"_RH_{net}_" in col]
    if net_cols:
        # Calculate the average across all regional columns for every subject
        myelin_networks_df[f"myelin_{net}"] = final_df[net_cols].mean(axis=1)

# Save the network averages DataFrame to a CSV file
output_networks_csv = "myelin_features_networks.csv"
myelin_networks_df.to_csv(output_networks_csv)
print(f"\nResultados por redes guardados exitosamente en: {output_networks_csv}")