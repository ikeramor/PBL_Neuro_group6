# ============================================================
# Alzheimer Detection with 3D CNN
# MRI T1W (.nii.gz) full-volume classification using PyTorch
# ============================================================
#
# Dataset assumptions:
# - Metadata CSV: OASIS3_metadata_clean.csv
# - Columns:
#       Subject_ID
#       Session_ID
#       DEMENTED
#       NORMCOG
#
# - MRI images:
#       *.nii.gz
# - Each filename contains the Subject_ID somewhere in the name
#
# - Images are already preprocessed (skull-stripped, registered).
#
# ============================================================
# INSTALL:
# pip install torch torchvision nibabel pandas scikit-learn tqdm matplotlib scipy
# ============================================================

import os
import glob
import random
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

from scipy.ndimage import zoom
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ============================================================
# CONFIG
# ============================================================

CSV_PATH = "OASIS3_metadata_clean.csv"
IMAGE_DIR = "."

BATCH_SIZE = 4          # Keep low — 3D volumes are memory-heavy
EPOCHS = 30
LEARNING_RATE = 1e-4

# Target volume size after resampling (D x H x W)
# Common choices: (64,64,64) or (96,96,96)
# Larger = better accuracy but more VRAM
TARGET_SHAPE = (64, 64, 64)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print(f"Using device: {DEVICE}")

# ============================================================
# LOAD METADATA
# ============================================================

df = pd.read_csv(CSV_PATH)

print("Original CSV size:", len(df))

# ============================================================
# CREATE BINARY LABEL
# ============================================================

# DEMENTED = 1  |  NORMCOG = 0
df["label"] = df["DEMENTED"].astype(int)

# Consistency check: exactly one of DEMENTED/NORMCOG must be 1
df = df[(df["DEMENTED"] + df["NORMCOG"]) == 1]

print("After consistency filtering:", len(df))

# ============================================================
# FIND AND MATCH MRI FILES
# ============================================================

all_images = glob.glob(os.path.join(IMAGE_DIR, "**/*.nii.gz"), recursive=True)

print(f"Found {len(all_images)} MRI files")

subject_to_path = {}

for img_path in all_images:
    filename = os.path.basename(img_path)
    for subject_id in df["Subject_ID"].unique():
        if subject_id in filename:
            subject_to_path[subject_id] = img_path
            break

df["image_path"] = df["Subject_ID"].map(subject_to_path)

missing = df["image_path"].isna().sum()
print(f"Subjects without MRI found: {missing}")

df = df.dropna(subset=["image_path"])
print("Final dataset size:", len(df))

# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.3,
    stratify=df["label"],
    random_state=SEED
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["label"],
    random_state=SEED
)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ============================================================
# PREPROCESSING UTILITY
# ============================================================

def load_and_preprocess_volume(path, target_shape=TARGET_SHAPE):
    """
    Load NIfTI volume, resample to target_shape, and normalize to [0, 1].
    Returns a float32 numpy array of shape target_shape.
    """
    nii = nib.load(path)
    volume = nii.get_fdata(dtype=np.float32)

    # Remove singleton dimensions if any (e.g. 4D -> 3D)
    volume = np.squeeze(volume)

    # Resample to target shape using zoom
    zoom_factors = [t / s for t, s in zip(target_shape, volume.shape)]
    volume = zoom(volume, zoom_factors, order=1)  # bilinear interpolation

    # Normalize to [0, 1]
    vmin, vmax = volume.min(), volume.max()
    volume = (volume - vmin) / (vmax - vmin + 1e-8)

    return volume

# ============================================================
# DATASET
# ============================================================

class OASIS3DDataset(Dataset):
    """
    Loads full 3D MRI volumes.
    Returns tensor of shape (1, D, H, W) — single channel.
    """

    def __init__(self, dataframe, target_shape=TARGET_SHAPE, augment=False):
        self.df = dataframe.reset_index(drop=True)
        self.target_shape = target_shape
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        volume = load_and_preprocess_volume(
            row["image_path"],
            target_shape=self.target_shape
        )

        # ====================================================
        # DATA AUGMENTATION (training only)
        # ====================================================

        if self.augment:
            # Random horizontal flip along depth axis
            if random.random() > 0.5:
                volume = np.flip(volume, axis=0).copy()

            # Random Gaussian noise
            if random.random() > 0.5:
                noise = np.random.normal(0, 0.01, volume.shape).astype(np.float32)
                volume = np.clip(volume + noise, 0, 1)

        # Add channel dimension: (1, D, H, W)
        volume = torch.tensor(volume, dtype=torch.float32).unsqueeze(0)

        label = torch.tensor(row["label"], dtype=torch.float32)

        return volume, label

# ============================================================
# DATALOADERS
# ============================================================

train_dataset = OASIS3DDataset(train_df, augment=True)
val_dataset   = OASIS3DDataset(val_df,   augment=False)
test_dataset  = OASIS3DDataset(test_df,  augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# ============================================================
# 3D CNN ARCHITECTURE
# ============================================================

class Conv3DBlock(nn.Module):
    """
    3D Convolutional block: Conv3d -> BatchNorm3d -> ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Alzheimer3DCNN(nn.Module):
    """
    A compact 3D CNN for binary Alzheimer classification.

    Architecture:
        Input:  (B, 1, 64, 64, 64)

        Block 1: Conv3D(1  -> 32)  + MaxPool3D  -> (B, 32, 32, 32, 32)
        Block 2: Conv3D(32 -> 64)  + MaxPool3D  -> (B, 64, 16, 16, 16)
        Block 3: Conv3D(64 -> 128) + MaxPool3D  -> (B, 128, 8, 8, 8)
        Block 4: Conv3D(128-> 256) + MaxPool3D  -> (B, 256, 4, 4, 4)

        Global Average Pooling               -> (B, 256)
        FC(256 -> 128) + ReLU + Dropout(0.5)
        FC(128 -> 1)   [logit]
    """

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            # Block 1
            Conv3DBlock(1, 32),
            Conv3DBlock(32, 32),
            nn.MaxPool3d(kernel_size=2),          # 64 -> 32

            # Block 2
            Conv3DBlock(32, 64),
            Conv3DBlock(64, 64),
            nn.MaxPool3d(kernel_size=2),          # 32 -> 16

            # Block 3
            Conv3DBlock(64, 128),
            Conv3DBlock(128, 128),
            nn.MaxPool3d(kernel_size=2),          # 16 -> 8

            # Block 4
            Conv3DBlock(128, 256),
            Conv3DBlock(256, 256),
            nn.MaxPool3d(kernel_size=2),          # 8 -> 4
        )

        # Global Average Pooling over spatial dims (D, H, W)
        self.gap = nn.AdaptiveAvgPool3d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 1)                     # Binary logit
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x


model = Alzheimer3DCNN().to(DEVICE)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {total_params:,}")

# ============================================================
# LOSS & OPTIMIZER
# ============================================================

criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

# Learning rate scheduler: reduce LR when validation AUC stagnates
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=5,
    verbose=True
)

# ============================================================
# TRAIN FUNCTION
# ============================================================

def train_one_epoch(model, loader):
    model.train()
    running_loss = 0.0

    for volumes, labels in tqdm(loader, desc="Training"):
        volumes = volumes.to(DEVICE)
        labels  = labels.to(DEVICE).unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(volumes)
        loss    = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)

# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(model, loader, desc="Evaluating"):
    model.eval()

    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for volumes, labels in tqdm(loader, desc=desc):
            volumes = volumes.to(DEVICE)

            outputs = model(volumes)
            probs   = torch.sigmoid(outputs)
            preds   = (probs > 0.5).int()

            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy().flatten())
            y_prob.extend(probs.cpu().numpy().flatten())

    return {
        "accuracy" : accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall"   : recall_score(y_true, y_pred, zero_division=0),
        "f1"       : f1_score(y_true, y_pred, zero_division=0),
        "auc"      : roc_auc_score(y_true, y_prob),
        "y_true"   : y_true,
        "y_pred"   : y_pred,
    }

# ============================================================
# TRAINING LOOP
# ============================================================

best_val_auc = 0.0
train_losses = []
val_aucs     = []

for epoch in range(1, EPOCHS + 1):

    print(f"\n{'='*50}")
    print(f"Epoch {epoch}/{EPOCHS}")

    train_loss = train_one_epoch(model, train_loader)
    val_metrics = evaluate(model, val_loader, desc="Validation")

    train_losses.append(train_loss)
    val_aucs.append(val_metrics["auc"])

    print(
        f"Train Loss : {train_loss:.4f}\n"
        f"Val AUC    : {val_metrics['auc']:.4f} | "
        f"ACC: {val_metrics['accuracy']:.4f} | "
        f"F1: {val_metrics['f1']:.4f} | "
        f"Recall: {val_metrics['recall']:.4f}"
    )

    # Step scheduler based on validation AUC
    scheduler.step(val_metrics["auc"])

    # Save best checkpoint
    if val_metrics["auc"] > best_val_auc:
        best_val_auc = val_metrics["auc"]
        torch.save(model.state_dict(), "best_alzheimer3dcnn.pth")
        print("  >> Best model saved!")

# ============================================================
# LOAD BEST MODEL & EVALUATE ON TEST SET
# ============================================================

model.load_state_dict(torch.load("best_alzheimer3dcnn.pth"))

test_metrics = evaluate(model, test_loader, desc="Test")

print("\n" + "="*50)
print("TEST RESULTS")
print("="*50)
print(
    f"Accuracy : {test_metrics['accuracy']:.4f}\n"
    f"Precision: {test_metrics['precision']:.4f}\n"
    f"Recall   : {test_metrics['recall']:.4f}\n"
    f"F1 Score : {test_metrics['f1']:.4f}\n"
    f"ROC AUC  : {test_metrics['auc']:.4f}"
)

# ============================================================
# CONFUSION MATRIX & CLASSIFICATION REPORT
# ============================================================

cm = confusion_matrix(test_metrics["y_true"], test_metrics["y_pred"])

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(
    test_metrics["y_true"],
    test_metrics["y_pred"],
    target_names=["Normal", "Demented"]
))

# ============================================================
# PLOTS
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Training loss curve
axes[0].plot(range(1, EPOCHS + 1), train_losses, marker="o", linewidth=2)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Training Loss")
axes[0].grid(True)

# Validation AUC curve
axes[1].plot(range(1, EPOCHS + 1), val_aucs, marker="o", color="orange", linewidth=2)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("ROC AUC")
axes[1].set_title("Validation AUC")
axes[1].grid(True)

plt.tight_layout()
plt.savefig("training_curves_3dcnn.png", dpi=150)
plt.show()

# ============================================================
# NOTES
# ============================================================
#
# 1. MEMORY:
#    3D volumes consume much more VRAM than 2D slices.
#    If you run out of memory:
#    - Reduce TARGET_SHAPE to (48, 48, 48) or (32, 32, 32)
#    - Reduce BATCH_SIZE to 2 or even 1
#    - Use gradient checkpointing (torch.utils.checkpoint)
#    - Use mixed precision: torch.cuda.amp.autocast()
#
# 2. MIXED PRECISION (recommended for GPU):
#    Add this inside train_one_epoch:
#
#    scaler = torch.cuda.amp.GradScaler()
#    with torch.cuda.amp.autocast():
#        outputs = model(volumes)
#        loss = criterion(outputs, labels)
#    scaler.scale(loss).backward()
#    scaler.step(optimizer)
#    scaler.update()
#
# 3. ARCHITECTURE ALTERNATIVES:
#    - ResNet3D (Med3D pretrained weights from Tencent)
#    - DenseNet3D
#    - MONAI (dedicated library for 3D medical imaging):
#          pip install monai
#          from monai.networks.nets import DenseNet121
#
# 4. WHY 3D CNN > 2D SLICE:
#    - Captures full spatial context across all three axes
#    - No information loss from slice selection
#    - Better detection of volumetric atrophy patterns
#    - Especially important for hippocampal and cortical thinning
#
# 5. METRICS PRIORITY (medical imaging):
#    Recall (Sensitivity) > AUC > F1 > Precision > Accuracy
#    A missed Alzheimer diagnosis is far worse than a false positive.
#
# ============================================================