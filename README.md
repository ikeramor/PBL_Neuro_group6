# 🧠 PBL Neuro — Group 6: Alzheimer's Disease Classification

A project-based learning (PBL) study leveraging multimodal neuroimaging to classify Alzheimer's Disease (AD) using deep learning and machine learning approaches on the **OASIS-3** dataset.

---

## 📋 Overview

This repository contains the full analysis pipeline developed by **Group 6** for the Neuroimaging PBL course. The project investigates whether structural and functional brain imaging features — including T1w/T2w MRI, resting-state fMRI functional connectivity (FC), myelin maps, and structural–functional (SC–FC) coupling — can be used to distinguish cognitively normal individuals from those with Alzheimer's disease.

The pipeline is organized into four main stages:

```
EDA → Preprocessing → Postprocessing → Deep Learning / ML Modelling
```

---

## 📂 Repository Structure

```
PBL_Neuro_group6/
│
├── eda/                        # Exploratory Data Analysis
│   ├── EDA_raw.ipynb           # EDA on raw OASIS-3 metadata
│   ├── EDA_final.ipynb         # EDA on cleaned and finalized metadata
│   ├── OASIS3_demographics.csv
│   ├── OASIS3_metadata*.csv    # Raw, clean, and final metadata versions
│   ├── final_subjects.txt      # Final list of included subjects
│   ├── fig_final_correlation.png
│   └── scripts/                # Step-by-step data curation notebooks
│       ├── 1.explore_anat_func_files.ipynb
│       ├── 2.select_subjects_sessions.ipynb
│       ├── 3.delete_functional_files.ipynb
│       ├── 4.metadata_cleaning.ipynb
│       ├── 5.bad_T2.ipynb
│       └── 6.add_gender_data.ipynb
│
├── preprocessing/              # MRI Preprocessing Pipelines
│   ├── fmriprep/               # fMRIPrep for T1w + BOLD preprocessing
│   │   ├── fmriprep_preprocessing.sh
│   │   ├── fmriprep_status.txt
│   │   └── failures.txt
│   ├── ants/                   # ANTs for T2w brain extraction & MNI registration
│   │   ├── ants_preprocessing.sh
│   │   ├── antsBrainExtraction.sh
│   │   └── ants_status.txt
│   └── fd_gsr/                 # Framewise displacement & global signal regression
│       └── FD_GSR.ipynb
│
├── postprocessing/             # Feature Extraction & Analysis
│   ├── FC/                     # Functional Connectivity matrices
│   │   ├── FC_network.ipynb    # Network-level FC analysis
│   │   ├── FC_node.ipynb       # Node-level FC analysis
│   │   └── fc_metrics*.csv
│   ├── Myelin/                 # T1w/T2w myelin map extraction
│   │   ├── Scripts/
│   │   └── myelin_maps_logs/
│   ├── SC-FC_Coupling/         # Structural–Functional coupling analysis
│   │   ├── SC_FC_Coupling.ipynb
│   │   ├── ML.ipynb            # Classical ML on coupling features
│   │   ├── plot_coupling_boxplots.py
│   │   └── Myelin_FC_coupling_vectors*.csv
│   └── Timeseries/             # fMRI timeseries modelling (LSTM)
│       ├── 1_Binary_Activation.ipynb
│       ├── 2_Dataset_Windows.ipynb
│       ├── 3_LSTM_Baseline.ipynb
│       ├── 4_LSTM_NormCog_vs_Demented.ipynb
│       └── 5_LSTM_HiddenState_Classifier.ipynb
│
└── DL/                         # Deep Learning Models
    ├── MLP_FC/                 # MLP on FC network vectors
    │   └── fc_network_classification_optuna_stratified.ipynb
    ├── MLP_myelin/             # MLP on myelin network features
    │   └── myelin_networks_classification_optuna_stratified.ipynb
    ├── MLP_sc_fc_coupling/     # MLP on SC-FC coupling vectors
    │   └── sc_fc_classification_optuna_stratified.ipynb
    ├── 1D_CNN_SC-FC/           # 1D CNN on SC-FC coupling signals
    │   └── 1D_CNN_SC-FC.ipynb
    ├── 2D_CNN_FC/              # 2D CNN (BrainNetCNN-style) on FC matrices
    │   ├── 2D_CNN_FC.ipynb
    │   ├── brainnet_young_evaluation.ipynb
    │   └── brainnet_young_gradcam_fixed.ipynb  # Grad-CAM interpretability
    ├── 3D_CNN_T1w/             # 3D CNN on T1-weighted MRI volumes
    │   ├── alzheimer_3dcnn_optuna_fixedT1.ipynb
    │   ├── alzheimer_medicalnet_optunaT1.ipynb  # MedicalNet transfer learning
    │   ├── alzheimer_3dcnn_eval_gradcam.ipynb
    │   └── train_alzheimer_resnet50.py
    └── 3D_CNN_T2w/             # 3D CNN on T2-weighted MRI volumes
        ├── alzheimer_3dcnn_optuna_fixedT2.ipynb
        ├── alzheimer_medicalnet_optunaT2.ipynb
        └── alzheimer_3dcnn_eval_gradcam.ipynb
```

---

## 🗃️ Dataset

**OASIS-3** (Open Access Series of Imaging Studies — Longitudinal Multimodal Neuroimaging, Clinical, and Cognitive Dataset)

- Longitudinal dataset with cognitively normal (CN) and Alzheimer's disease (AD) subjects
- Modalities used: **T1w MRI**, **T2w MRI**, **resting-state fMRI**
- Subjects were selected and curated through the EDA pipeline (see `eda/`)

> ⚠️ Raw imaging data is **not included** in this repository due to size and data-sharing restrictions. Access the OASIS-3 dataset at [oasis-brains.org](https://www.oasis-brains.org/).

---

## 🔧 Pipeline

### 1. Exploratory Data Analysis (`eda/`)

- Exploration of raw OASIS-3 file structure and metadata
- Subject/session selection based on data quality and availability
- Metadata cleaning: removal of bad T2w scans, addition of demographic variables
- Final cohort defined in `final_subjects.txt`

### 2. Preprocessing (`preprocessing/`)

| Step | Tool | Description |
|------|------|-------------|
| Structural + functional MRI | **fMRIPrep** | T1w anatomical preprocessing + BOLD fMRI preprocessing, output to MNI152NLin2009cAsym space |
| T2w MRI | **ANTs** (via Singularity) | Brain extraction + nonlinear registration to MNI space using T1w-derived transforms |
| Signal cleaning | **FD/GSR** | Framewise displacement-based censoring + global signal regression on fMRI timeseries |

### 3. Postprocessing / Feature Extraction (`postprocessing/`)

| Feature | Description |
|---------|-------------|
| **Functional Connectivity (FC)** | Pearson correlation matrices at network- and node-level using an atlas parcellation |
| **Myelin maps** | T1w/T2w ratio as a proxy for cortical myelination, extracted per brain network |
| **SC–FC Coupling** | Region-wise coupling between myelin-based structural features and FC |
| **fMRI Timeseries** | Raw BOLD timeseries windowed for temporal sequence modelling |

### 4. Deep Learning & Machine Learning (`DL/`)

| Model | Input | Description |
|-------|-------|-------------|
| **MLP + Optuna** | FC / Myelin / SC-FC vectors | Hyperparameter-optimized MLPs with stratified cross-validation |
| **1D CNN** | SC-FC coupling signal | 1D convolutional network over coupling feature vectors |
| **2D CNN (BrainNetCNN)** | FC matrices | Convolutional model treating FC matrices as 2D images; Grad-CAM analysis |
| **3D CNN** | T1w / T2w volumes | Volumetric CNNs trained from scratch with Optuna search |
| **MedicalNet (ResNet)** | T1w / T2w volumes | Transfer learning from MedicalNet pretrained 3D ResNet |
| **LSTM** | fMRI timeseries | Recurrent models on windowed BOLD sequences for temporal classification |

All MLP/CNN models use **Optuna** for automated hyperparameter optimization with stratified k-fold cross-validation.

---

## 🛠️ Requirements

### Python
- Python ≥ 3.8
- `torch`, `torchvision`
- `optuna`
- `nilearn`, `nibabel`
- `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`

### Neuroimaging Tools
- [fMRIPrep](https://fmriprep.org/) (≥ 23.x)
- [ANTs](https://github.com/ANTsX/ANTs) (via Singularity container `ants.sif`)
- [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/) license (required by fMRIPrep)
- [Templateflow](https://www.templateflow.org/) — MNI152NLin2009cAsym template

---

## 🚀 Getting Started

1. **Obtain access to OASIS-3** at [oasis-brains.org](https://www.oasis-brains.org/) and download subjects listed in `eda/final_subjects.txt`.

2. **Organize data** in BIDS format under `dataset/raw/`.

3. **Run fMRIPrep preprocessing**:
   ```bash
   cd preprocessing/fmriprep
   bash fmriprep_preprocessing.sh
   ```

4. **Run ANTs T2w preprocessing** (requires `ants.sif` Singularity image):
   ```bash
   cd preprocessing/ants
   bash ants_preprocessing.sh
   ```

5. **Run postprocessing notebooks** in order within each `postprocessing/` subdirectory.

6. **Train models** using notebooks inside `DL/`.

---

## 👥 Authors

**PBL Neuro — Group 6**

---

## 📄 License

This project is for academic/research purposes. Data use is subject to the [OASIS-3 Data Use Agreement](https://www.oasis-brains.org/#access).
