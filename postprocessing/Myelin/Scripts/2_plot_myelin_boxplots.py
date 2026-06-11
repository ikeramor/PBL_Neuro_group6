import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from pathlib import Path

# Define the 7 networks
networks = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
network_map = {
    "Vis": "Visual Network (VIN)",
    "SomMot": "Somatomotor Network (SMN)",
    "DorsAttn": "Dorsal Attention Network (DAN)",
    "SalVentAttn": "Salience / Ventral Attention Network (SAN/VAN)",
    "Limbic": "Limbic Network (LIN)",
    "Cont": "Control / Frontoparietal Network (FPN/CON)",
    "Default": "Default Mode Network (DMN)"
}

# Define network color mapping (matching FC boxplots)
light_network_colors = {
    "Vis": "#9A8CFF",
    "SomMot": "#8BB9E0",
    "DorsAttn": "#8DAA5B",
    "SalVentAttn": "#B39DDB",
    "Limbic": "#DDF7A1",
    "Cont": "#F26A8D",
    "Default": "#FFD27A"
}

dark_network_colors = {
    "Vis": "#3700D4",
    "SomMot": "#2462B3",
    "DorsAttn": "#3A6D1C",
    "SalVentAttn": "#6F43B2",
    "Limbic": "#BEF264",
    "Cont": "#DD1952",
    "Default": "#F8A520"
}

# 1. Path to load data
# Default path in the Jupyter environment
myelin_file = Path("/home/jovyan/Desktop/PBL_Neuro/Scripts/myelin_features_networks_filtered.csv")

# Dynamic fallback paths
if not myelin_file.exists():
    myelin_file = Path("myelin_features_networks_filtered.csv")
    if not myelin_file.exists():
        myelin_file = Path("../myelin_features_networks_filtered.csv")
        if not myelin_file.exists():
            myelin_file = Path("Scripts/Myelin_Vector/myelin_features_networks_filtered.csv")
            if not myelin_file.exists():
                myelin_file = Path("Scripts/myelin_features_networks_filtered.csv")

if not myelin_file.exists():
    print(f"Error: Could not find 'myelin_features_networks_filtered.csv' at search path: {myelin_file.resolve()}")
    exit(1)

print(f"Loading filtered myelin network data from: {myelin_file}")
df_networks_filtered = pd.read_csv(myelin_file)

# Create plotting-friendly group labels (CN = Healthy Control, AD = Alzheimer's)
df_networks_filtered["group_plot"] = df_networks_filtered["group"].replace({"NORMCOG": "CN", "DEMENTED": "AD"})

# Mean myelin per group:
myelin_cols = [f"myelin_{net}" for net in networks]
cn_means = df_networks_filtered[df_networks_filtered["group"] == "NORMCOG"][myelin_cols].mean()
ad_means = df_networks_filtered[df_networks_filtered["group"] == "DEMENTED"][myelin_cols].mean()

# Customize outlier properties
flierprops = dict(marker='o', markerfacecolor='#F8C3C6', markeredgecolor='#D32F2F', markeredgewidth=2, markersize=8, linewidth=2)

# Set Seaborn styling
sns.set(style="whitegrid")

# Ensure output folder exists (saved relatively to where this script is run or in the parent directory's Figures_Myelin)
output_dir = Path("Figures_Myelin")
# If run inside Scripts/Myelin_Vector, save it to the project root directory's Figures_Myelin
if Path("Scripts").exists() and not Path("Myelin_Vector").exists():
    pass
elif Path("../../Figures_Myelin").exists() or Path("..").name == "Myelin_Vector":
    output_dir = Path("../Figures_Myelin")

output_dir.mkdir(parents=True, exist_ok=True)

# Iterate through each of the 7 networks to generate and save plots
for net in networks:
    myelin_col = f"myelin_{net}"
    
    fig, ax = plt.subplots(figsize=(6, 6))

    # Define light fill palette
    palette = {"CN": light_network_colors[net], "AD": light_network_colors[net]}

    # Create boxplot
    sns.boxplot(data=df_networks_filtered, x="group_plot", y=myelin_col, hue='group_plot', palette=palette, width=0.6, linewidth=1.8, flierprops=flierprops, ax=ax)

    # Create background distribution violin plot
    sns.violinplot(data=df_networks_filtered, x="group_plot", y=myelin_col, hue='group_plot', palette=palette, inner=None, cut=0, linewidth=1.2, alpha=0.25, ax=ax)

    # Overlay individual subject points
    sns.stripplot(data=df_networks_filtered, x="group_plot", y=myelin_col, color=dark_network_colors[net], jitter=True, size=5, alpha=0.8, marker='o', edgecolor='gray', linewidth=0.7, ax=ax)

    # Extract mean myelin values
    cn_val = cn_means[myelin_col]
    ad_val = ad_means[myelin_col]

    # Define custom legend labels displaying the numerical means
    legend_elements = [
        Line2D([], [], linestyle='None', markersize=0, label=f'CN Mean Myelin: {cn_val:.3f}'),
        Line2D([], [], linestyle='None', markersize=0, label=f'AD Mean Myelin: {ad_val:.3f}')
    ]

    # Add legend
    ax.legend(handles=legend_elements, loc="upper right", handlelength=0, handletextpad=0, prop={'weight': 'bold'})

    # Titles and Labels
    ax.set_title(network_map[net], fontsize=20, fontweight="bold", pad=15)
    ax.set_xlabel("")
    ax.set_ylabel("Mean Myelin Density", fontsize=14)

    # Clean layout and save
    plt.tight_layout()
    fig_path = output_dir / f"boxplot_Myelin_{net}.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Generated and saved: {fig_path}")
    
    # Close figures to conserve memory
    plt.close(fig)

print(f"\nSuccess: All myelin network boxplots have been generated in: {output_dir.resolve()}")
