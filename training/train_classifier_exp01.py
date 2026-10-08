import os
import csv
import time
import random
import shutil
import argparse
from collections import defaultdict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

DATA_ROOT = "D:/SmartPlate/datasets/IndianFood/extracted/Indian Food Images/Indian Food Images"
MODEL_DIR = "D:/SmartPlate/results/models"
LOG_DIR = "D:/SmartPlate/results/logs"
PLOT_DIR = "D:/SmartPlate/results/plots"

NUM_CLASSES = 80
SEED = 42

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# STRATIFIED SPLIT
# ============================================================

def create_stratified_split(
    data_root,
    train_per_class=40,
    val_per_class=10,
    seed=42
):
    """
    Create a reproducible stratified split from folder-per-class
    structure. Returns (train_samples, val_samples, class_names).
    Each sample is (image_path, class_index).
    """

    rng = random.Random(seed)

    class_names = sorted([
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d))
    ])

    train_samples = []
    val_samples = []

    for cls_idx, cls_name in enumerate(class_names):

        cls_dir = os.path.join(data_root, cls_name)

        images = sorted([
            f for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        rng.shuffle(images)

        train_imgs = images[:train_per_class]
        val_imgs = images[train_per_class:train_per_class + val_per_class]

        for img in train_imgs:
            train_samples.append((
                os.path.join(cls_dir, img),
                cls_idx
            ))

        for img in val_imgs:
            val_samples.append((
                os.path.join(cls_dir, img),
                cls_idx
            ))

    return train_samples, val_samples, class_names


# ============================================================
# DATASET
# ============================================================

class IndianFoodDataset(Dataset):

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path, label = self.samples[idx]

        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# METRICS
# ============================================================

def compute_topk_accuracy(outputs, targets, topk=(1, 5)):
    """Compute Top-1 and Top-5 accuracy."""

    maxk = max(topk)
    batch_size = targets.size(0)

    _, pred = outputs.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    results = []

    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        results.append(correct_k.item() / batch_size)

    return results


# ============================================================
# TRAINING
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=10)

    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("CLASSIFICATION EXP 1: ResNet-50 (Pretrained) — Indian Food")
    print("=" * 70)
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --------------------------------------------------------
    # STRATIFIED SPLIT
    # --------------------------------------------------------

    train_samples, val_samples, class_names = create_stratified_split(
        DATA_ROOT,
        train_per_class=40,
        val_per_class=10,
        seed=SEED
    )

    print(f"Classes: {len(class_names)}")
    print(f"Training samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")

    # --------------------------------------------------------
    # TRANSFORMS
    # --------------------------------------------------------

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(args.image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        ),
    ])

    # --------------------------------------------------------
    # DATASETS + LOADERS
    # --------------------------------------------------------

    train_dataset = IndianFoodDataset(
        train_samples,
        transform=train_transform
    )

    val_dataset = IndianFoodDataset(
        val_samples,
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # Replace final FC layer for 80 classes
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print(f"Model: ResNet-50 (pretrained)")
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    # --------------------------------------------------------
    # LOSS + OPTIMIZER
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    # --------------------------------------------------------
    # AMP
    # --------------------------------------------------------

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=torch.cuda.is_available()
    )

    # --------------------------------------------------------
    # LOG FILE
    # --------------------------------------------------------

    log_path = os.path.join(
        LOG_DIR,
        "classifier_exp01_resnet50.csv"
    )

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "val_loss",
            "top1_acc",
            "top5_acc",
            "precision",
            "recall",
            "f1",
            "lr",
            "epoch_time_sec"
        ])

    # --------------------------------------------------------
    # BEST MODEL TRACKING
    # --------------------------------------------------------

    best_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0

    best_model_path = os.path.join(
        MODEL_DIR,
        "classifier_exp01_resnet50_best.pth"
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "top1_acc": [],
        "top5_acc": [],
    }

    # ========================================================
    # EPOCH LOOP
    # ========================================================

    training_start = time.time()

    for epoch in range(1, args.epochs + 1):

        epoch_start = time.time()

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model.train()
        train_loss = 0.0

        for images, labels in train_loader:

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type="cuda",
                enabled=torch.cuda.is_available()
            ):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        top1_total = 0.0
        top5_total = 0.0
        val_batches = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast(
                    device_type="cuda",
                    enabled=torch.cuda.is_available()
                ):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()

                top1, top5 = compute_topk_accuracy(
                    outputs, labels, topk=(1, 5)
                )

                top1_total += top1
                top5_total += top5
                val_batches += 1

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        top1_acc = top1_total / val_batches
        top5_acc = top5_total / val_batches

        # Sklearn metrics
        all_preds_np = np.array(all_preds)
        all_targets_np = np.array(all_targets)

        precision = precision_score(
            all_targets_np,
            all_preds_np,
            average="macro",
            zero_division=0
        )

        recall = recall_score(
            all_targets_np,
            all_preds_np,
            average="macro",
            zero_division=0
        )

        f1 = f1_score(
            all_targets_np,
            all_preds_np,
            average="macro",
            zero_division=0
        )

        epoch_time = time.time() - epoch_start

        # ----------------------------------------------------
        # LR SCHEDULER
        # ----------------------------------------------------

        scheduler.step(top1_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        # ----------------------------------------------------
        # LOG
        # ----------------------------------------------------

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{train_loss:.6f}",
                f"{val_loss:.6f}",
                f"{top1_acc:.4f}",
                f"{top5_acc:.4f}",
                f"{precision:.4f}",
                f"{recall:.4f}",
                f"{f1:.4f}",
                f"{current_lr:.6f}",
                f"{epoch_time:.1f}"
            ])

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["top1_acc"].append(top1_acc)
        history["top5_acc"].append(top5_acc)

        # ----------------------------------------------------
        # PRINT
        # ----------------------------------------------------

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Top-1: {top1_acc:.4f} | "
            f"Top-5: {top5_acc:.4f} | "
            f"F1: {f1:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        if top1_acc > best_acc:

            best_acc = top1_acc
            best_epoch = epoch
            epochs_without_improvement = 0

            best_preds = all_preds_np.copy()
            best_targets = all_targets_np.copy()
            best_precision = precision
            best_recall = recall
            best_f1 = f1
            best_top5 = top5_acc

            torch.save(
                model.state_dict(),
                best_model_path
            )

            print(
                f"  >>> NEW BEST MODEL "
                f"(Top-1 = {best_acc:.4f})"
            )

        else:
            epochs_without_improvement += 1

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if epochs_without_improvement >= args.patience:

            print()
            print("=" * 70)
            print("EARLY STOPPING")
            print(f"Best Top-1 Accuracy: {best_acc:.4f}")
            print(f"Best Epoch: {best_epoch}")
            print("=" * 70)
            break

    total_time = time.time() - training_start

    # ========================================================
    # PLOTS
    # ========================================================

    epochs_range = range(1, len(history["train_loss"]) + 1)

    # Loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, history["train_loss"], label="Train Loss")
    plt.plot(epochs_range, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Classification Exp 1 — Loss Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOT_DIR, "classifier_exp01_loss.png"),
        dpi=150
    )
    plt.close()

    # Accuracy curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, history["top1_acc"], label="Top-1 Accuracy")
    plt.plot(epochs_range, history["top5_acc"], label="Top-5 Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Classification Exp 1 — Accuracy Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOT_DIR, "classifier_exp01_accuracy.png"),
        dpi=150
    )
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(best_targets, best_preds)

    plt.figure(figsize=(20, 18))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Classification Exp 1 — Confusion Matrix (Best Epoch)")
    plt.colorbar(shrink=0.8)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(
        os.path.join(PLOT_DIR, "classifier_exp01_confusion.png"),
        dpi=150
    )
    plt.close()

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 70)
    print("CLASSIFICATION EXP 1 — FINAL REPORT")
    print("=" * 70)
    print(f"Model:              ResNet-50 (pretrained ImageNet)")
    print(f"Dataset:            Indian Food Images (80 classes)")
    print(f"Train / Val:        3,200 / 800")
    print(f"Image size:         {args.image_size}x{args.image_size}")
    print(f"Best Epoch:         {best_epoch}")
    print(f"Epochs trained:     {len(history['train_loss'])}")
    print(f"Total time:         {total_time:.1f} sec ({total_time/60:.1f} min)")
    print()
    print(f"Top-1 Accuracy:     {best_acc:.4f}")
    print(f"Top-5 Accuracy:     {best_top5:.4f}")
    print(f"Precision (macro):  {best_precision:.4f}")
    print(f"Recall (macro):     {best_recall:.4f}")
    print(f"F1-score (macro):   {best_f1:.4f}")
    print()
    print(f"Best model:         {best_model_path}")
    print(f"Training log:       {log_path}")
    print(f"Loss plot:          {os.path.join(PLOT_DIR, 'classifier_exp01_loss.png')}")
    print(f"Accuracy plot:      {os.path.join(PLOT_DIR, 'classifier_exp01_accuracy.png')}")
    print(f"Confusion matrix:   {os.path.join(PLOT_DIR, 'classifier_exp01_confusion.png')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
