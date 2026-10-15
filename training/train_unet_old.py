import os
import csv
import time
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIG
# ============================================================

NUM_CLASSES = 104
BACKGROUND_CLASS = 0


# ============================================================
# 2. DATASET
# ============================================================

class FoodSegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=256):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size

        self.images = sorted(self.image_dir.glob("*.jpg"))

        if len(self.images) == 0:
            raise RuntimeError(f"No images found in {self.image_dir}")

        self.image_transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.mask_dir / (image_path.stem + ".png")

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        image = self.image_transform(image)

        # IMPORTANT:
        # Masks contain integer class IDs 0-103.
        # NEAREST interpolation preserves class IDs.
        mask = mask.resize(
            (self.image_size, self.image_size),
            Image.Resampling.NEAREST
        )

        mask = torch.from_numpy(
            np.array(mask, dtype=np.int64)
        )

        return image, mask


# ============================================================
# 3. DOUBLE CONV
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ============================================================
# 4. U-NET
# ============================================================

class UNet(nn.Module):
    def __init__(self, num_classes=104):
        super().__init__()

        self.enc1 = DoubleConv(3, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.output = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):

        e1 = self.enc1(x)

        e2 = self.enc2(self.pool(e1))

        e3 = self.enc3(self.pool(e2))

        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.output(d1)


# ============================================================
# 5. DICE SCORE
# ============================================================

def dice_score(pred, target, num_classes=104):

    pred = torch.argmax(pred, dim=1)

    scores = []

    for cls in range(1, num_classes):

        pred_cls = (pred == cls)
        target_cls = (target == cls)

        intersection = (pred_cls & target_cls).sum().float()

        denominator = pred_cls.sum().float() + target_cls.sum().float()

        if denominator == 0:
            continue

        dice = (2.0 * intersection + 1e-6) / (
            denominator + 1e-6
        )

        scores.append(dice)

    if len(scores) == 0:
        return torch.tensor(0.0, device=pred.device)

    return torch.stack(scores).mean()


# ============================================================
# 6. IoU SCORE
# ============================================================

def iou_score(pred, target, num_classes=104):

    pred = torch.argmax(pred, dim=1)

    scores = []

    for cls in range(1, num_classes):

        pred_cls = (pred == cls)
        target_cls = (target == cls)

        intersection = (pred_cls & target_cls).sum().float()

        union = (pred_cls | target_cls).sum().float()

        if union == 0:
            continue

        iou = (intersection + 1e-6) / (
            union + 1e-6
        )

        scores.append(iou)

    if len(scores) == 0:
        return torch.tensor(0.0, device=pred.device)

    return torch.stack(scores).mean()


# ============================================================
# 7. DICE LOSS
# ============================================================

def multiclass_dice_loss(logits, target, num_classes=104):

    probabilities = torch.softmax(logits, dim=1)

    target_one_hot = torch.nn.functional.one_hot(
        target,
        num_classes=num_classes
    )

    target_one_hot = target_one_hot.permute(
        0, 3, 1, 2
    ).float()

    intersection = (
        probabilities * target_one_hot
    ).sum(dim=(0, 2, 3))

    denominator = (
        probabilities.sum(dim=(0, 2, 3))
        +
        target_one_hot.sum(dim=(0, 2, 3))
    )

    dice = (
        2 * intersection + 1e-6
    ) / (
        denominator + 1e-6
    )

    # Ignore background for Dice loss
    dice = dice[1:]

    return 1 - dice.mean()


# ============================================================
# 8. COMBINED LOSS
# ============================================================

def combined_loss(logits, target):

    ce = nn.functional.cross_entropy(
        logits,
        target
    )

    dice = multiclass_dice_loss(
        logits,
        target,
        NUM_CLASSES
    )

    return ce + dice


# ============================================================
# 9. VALIDATION
# ============================================================

@torch.no_grad()
def validate(model, loader, device):

    model.eval()

    total_loss = 0
    total_dice = 0
    total_iou = 0

    batches = 0

    for images, masks in loader:

        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        loss = combined_loss(
            outputs,
            masks
        )

        dice = dice_score(
            outputs,
            masks
        )

        iou = iou_score(
            outputs,
            masks
        )

        total_loss += loss.item()
        total_dice += dice.item()
        total_iou += iou.item()

        batches += 1

    return (
        total_loss / batches,
        total_dice / batches,
        total_iou / batches
    )


# ============================================================
# 10. TRAINING
# ============================================================

def train(args):

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("\n====================================")
    print("SMART PLATE - U-NET TRAINING")
    print("====================================")

    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print("Epochs:", args.epochs)
    print("Image size:", args.image_size)
    print("Batch size:", args.batch_size)
    print("Learning rate:", args.lr)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = FoodSegDataset(
        "D:/SmartPlate/datasets/FoodSeg103/train/images",
        "D:/SmartPlate/datasets/FoodSeg103/train/masks",
        args.image_size
    )

    val_dataset = FoodSegDataset(
        "D:/SmartPlate/datasets/FoodSeg103/validation/images",
        "D:/SmartPlate/datasets/FoodSeg103/validation/masks",
        args.image_size
    )

    print(
        "Training samples:",
        len(train_dataset)
    )

    print(
        "Validation samples:",
        len(val_dataset)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = UNet(
        num_classes=NUM_CLASSES
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr
    )

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    model_dir = Path(
        "D:/SmartPlate/results/models"
    )

    log_dir = Path(
        "D:/SmartPlate/results/logs"
    )

    plot_dir = Path(
        "D:/SmartPlate/results/plots"
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    log_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    plot_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = (
        model_dir /
        f"unet_e{args.epochs}.pth"
    )

    csv_path = (
        log_dir /
        f"unet_e{args.epochs}.csv"
    )

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    with open(
        csv_path,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "epoch",
            "train_loss",
            "val_loss",
            "val_dice",
            "val_iou",
            "epoch_time_sec"
        ])

    history = {
        "train_loss": [],
        "val_loss": [],
        "dice": [],
        "iou": []
    }

    best_dice = -1

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(1, args.epochs + 1):

        start_time = time.time()

        model.train()

        running_loss = 0

        print(
            f"\nEpoch {epoch}/{args.epochs}"
        )

        for batch_idx, (images, masks) in enumerate(
            train_loader,
            start=1
        ):

            images = images.to(
                device,
                non_blocking=True
            )

            masks = masks.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad()

            outputs = model(images)

            loss = combined_loss(
                outputs,
                masks
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 50 == 0:

                print(
                    f"Batch {batch_idx}/{len(train_loader)} "
                    f"Loss: {loss.item():.4f}"
                )

        train_loss = (
            running_loss /
            len(train_loader)
        )

        val_loss, dice, iou = validate(
            model,
            val_loader,
            device
        )

        epoch_time = time.time() - start_time

        history["train_loss"].append(
            train_loss
        )

        history["val_loss"].append(
            val_loss
        )

        history["dice"].append(
            dice
        )

        history["iou"].append(
            iou
        )

        print(
            f"Train Loss: {train_loss:.4f}"
        )

        print(
            f"Val Loss: {val_loss:.4f}"
        )

        print(
            f"Dice: {dice:.4f}"
        )

        print(
            f"IoU: {iou:.4f}"
        )

        print(
            f"Time: {epoch_time:.1f} sec"
        )

        # Save CSV
        with open(
            csv_path,
            "a",
            newline=""
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                epoch,
                train_loss,
                val_loss,
                dice,
                iou,
                epoch_time
            ])

        # Save best model
        if dice > best_dice:

            best_dice = dice

            torch.save(
                model.state_dict(),
                model_path
            )

            print(
                "BEST MODEL SAVED!"
            )

    # --------------------------------------------------------
    # Plot results
    # --------------------------------------------------------

    epochs = range(
        1,
        args.epochs + 1
    )

    plt.figure()

    plt.plot(
        epochs,
        history["train_loss"],
        label="Train Loss"
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(
        f"U-Net Loss - {args.epochs} Epochs"
    )

    plt.legend()

    plt.savefig(
        plot_dir /
        f"loss_e{args.epochs}.png"
    )

    plt.close()

    plt.figure()

    plt.plot(
        epochs,
        history["dice"],
        label="Dice"
    )

    plt.plot(
        epochs,
        history["iou"],
        label="IoU"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title(
        f"U-Net Segmentation Metrics - {args.epochs} Epochs"
    )

    plt.legend()

    plt.savefig(
        plot_dir /
        f"metrics_e{args.epochs}.png"
    )

    plt.close()

    print("\n====================================")
    print("TRAINING COMPLETE")
    print("====================================")

    print(
        "Best Dice:",
        best_dice
    )

    print(
        "Model:",
        model_path
    )

    print(
        "Log:",
        csv_path
    )


# ============================================================
# 11. COMMAND LINE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=10
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=256
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001
    )

    args = parser.parse_args()

    train(args)