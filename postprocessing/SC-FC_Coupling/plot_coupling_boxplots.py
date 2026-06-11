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

# Define network color mapping (matching previous plots)
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

def main():
    # 1. Paths to load data
    # Default remote server path
    coupling_file = Path("/home/jovyan/Desktop/PBL_Neuro/postprocessing/SC-FC_Coupling/SC_FC_coupling_vectors.csv")
    metadata_file = Path("/home/jovyan/Desktop/PBL_Neuro/eda/OASIS3_metadata_clean.csv")

    # Dynamic local fallback paths
    if not coupling_file.exists():
        coupling_file = Path("SC_FC_coupling_vectors.csv")
        if not coupling_file.exists():
            coupling_file = Path("../postprocessing/SC-FC_Coupling/SC_FC_coupling_vectors.csv")
            if not coupling_file.exists():
                coupling_file = Path("Desktop/PBL_Neuro/postprocessing/SC-FC_Coupling/SC_FC_coupling_vectors.csv")

    if not metadata_file.exists():
        metadata_file = Path("OASIS3_metadata_clean.csv")
        if not metadata_file.exists():
            metadata_file = Path("Scripts/Myelin_Vector/OASIS3_metadata_clean.csv")
            if not metadata_file.exists():
                metadata_file = Path("../Myelin_Vector/OASIS3_metadata_clean.csv")

    # Safety check for coupling file
    if not coupling_file.exists():
        print(f"Error: Could not find coupling CSV file at: {coupling_file.resolve()}")
        print("Please run this script inside the folder containing the CSV or provide a valid path.")
        return

    print(f"Loading coupling data from: {coupling_file}")
    df_coupling = pd.read_csv(coupling_file)
    print("Columns in coupling CSV:", list(df_coupling.columns))
    print(df_coupling.head(3))

    # Standardize subject column
    sub_col = None
    for col in df_coupling.columns:
        if col.lower() in ["subject", "subject_id", "subjectid", "sub"]:
            sub_col = col
            break
    if sub_col is None:
        sub_col = df_coupling.columns[0]
        print(f"Warning: Could not identify subject column. Using the first column: '{sub_col}'")
    
    df_coupling = df_coupling.rename(columns={sub_col: "subject"})
    # Normalize subject ID format (remove 'sub-' if present)
    df_coupling["subject_clean"] = df_coupling["subject"].astype(str).str.replace("sub-", "", regex=False)

    # Resolve Group labels
    group_col = None
    for col in df_coupling.columns:
        if col.lower() in ["group", "dx", "clinical_group", "diagnosis"]:
            group_col = col
            break
    
    if group_col is not None:
        print(f"Using group labels from column: '{group_col}'")
        df_coupling["group"] = df_coupling[group_col]
    else:
        print("Group column not found in coupling CSV. Merging with metadata...")
        if not metadata_file.exists():
            print(f"Error: Metadata file NOT found at: {metadata_file.resolve()}. Cannot determine subject groups.")
            return
        
        print(f"Loading metadata from: {metadata_file}")
        df_meta = pd.read_csv(metadata_file)
        df_meta = df_meta.rename(columns={"Subject_ID": "subject_clean"})
        df_meta["subject_clean"] = df_meta["subject_clean"].astype(str).str.replace("sub-", "", regex=False)
        df_meta['group'] = np.where(df_meta['DEMENTED'] == 1, 'DEMENTED', 'NORMCOG')
        
        # Merge
        df_coupling = pd.merge(df_coupling, df_meta[['subject_clean', 'group', 'Age']], on='subject_clean', how='inner')
        print(f"Merged with metadata. Shape: {df_coupling.shape}")

    # Standardize group plot labels
    df_coupling["group_plot"] = df_coupling["group"].replace({"NORMCOG": "CN", "DEMENTED": "AD", "Control": "CN", "AD": "AD"})

    # Map networks to columns in the CSV
    net_col_map = {}
    for net in networks:
        matched = [c for c in df_coupling.columns if net.lower() in c.lower()]
        if matched:
            net_col_map[net] = matched[0]
            print(f"Mapped network '{net}' to column: '{matched[0]}'")
        else:
            print(f"Error: Could not find column for network: '{net}'")
            return

    # Compute means per group
    cn_means = {}
    ad_means = {}
    for net in networks:
        col = net_col_map[net]
        cn_means[net] = df_coupling[df_coupling["group_plot"] == "CN"][col].mean()
        ad_means[net] = df_coupling[df_coupling["group_plot"] == "AD"][col].mean()

    # Outlier style
    flierprops = dict(marker='o', markerfacecolor='#F8C3C6', markeredgecolor='#D32F2F', markeredgewidth=2, markersize=8, linewidth=2)
    sns.set(style="whitegrid")

    # Set up output directory
    output_dir = Path("Figures_Coupling")
    if Path("Scripts").exists():
        output_dir = Path("Scripts/FC/Figures_Coupling")
    
    # Check if we are running in the target remote folder
    remote_fig_dir = Path("/home/jovyan/Desktop/PBL_Neuro/postprocessing/SC-FC_Coupling/Figures_Coupling")
    if remote_fig_dir.parent.exists():
        output_dir = remote_fig_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Figures will be saved to: {output_dir.resolve()}")

    # Plot each network
    for net in networks:
        col = net_col_map[net]
        fig, ax = plt.subplots(figsize=(6, 6))

        palette = {"CN": light_network_colors[net], "AD": light_network_colors[net]}

        # Boxplot
        sns.boxplot(data=df_coupling, x="group_plot", y=col, hue='group_plot', palette=palette, width=0.6, linewidth=1.8, flierprops=flierprops, ax=ax)

        # Violinplot
        sns.violinplot(data=df_coupling, x="group_plot", y=col, hue='group_plot', palette=palette, inner=None, cut=0, linewidth=1.2, alpha=0.25, ax=ax)

        # Stripplot
        sns.stripplot(data=df_coupling, x="group_plot", y=col, color=dark_network_colors[net], jitter=True, size=5, alpha=0.8, marker='o', edgecolor='gray', linewidth=0.7, ax=ax)

        # Legend showing means
        cn_val = cn_means[net]
        ad_val = ad_means[net]
        legend_elements = [
            Line2D([], [], linestyle='None', markersize=0, label=f'CN Mean Coupling: {cn_val:.3f}'),
            Line2D([], [], linestyle='None', markersize=0, label=f'AD Mean Coupling: {ad_val:.3f}')
        ]
        ax.legend(handles=legend_elements, loc="upper right", handlelength=0, handletextpad=0, prop={'weight': 'bold'})

        # Titles and Labels
        ax.set_title(f"Coupling: {network_map[net]}", fontsize=16, fontweight="bold", pad=15)
        ax.set_xlabel("")
        ax.set_ylabel("SC-FC Coupling Value", fontsize=14)

        plt.tight_layout()
        fig_path = output_dir / f"boxplot_Coupling_{net}.png"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Generated and saved: {fig_path.name}")

    print("\nSuccess: All SC-FC coupling network boxplots have been generated!")

if __name__ == "__main__":
    main()
