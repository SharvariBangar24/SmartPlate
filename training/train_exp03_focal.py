import os
import csv
import argparse
import random

import numpy as np
from PIL import Image, ImageEnhance

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "D:/SmartPlate/datasets/FoodSeg103"
MODEL_DIR = "D:/SmartPlate/results/models"
LOG_DIR = "D:/SmartPlate/results/logs"

NUM_CLASSES = 104
SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DATASET
# ============================================================

class FoodSegDataset(Dataset):

    def __init__(self, image_dir, mask_dir, image_size=256, augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.augment = augment

        self.images = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        filename = self.images[idx]

        image_path = os.path.join(self.image_dir, filename)

        mask_name = os.path.splitext(filename)[0] + ".png"
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # ----------------------------------------------------
        # DATA AUGMENTATION
        # ----------------------------------------------------

        if self.augment:

            # Horizontal flip
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

            # Small rotation
            if random.random() < 0.5:
                angle = random.uniform(-10, 10)

                image = image.rotate(
                    angle,
                    resample=Image.Resampling.BILINEAR,
                    fillcolor=(0, 0, 0)
                )

                mask = mask.rotate(
                    angle,
                    resample=Image.Resampling.NEAREST,
                    fillcolor=0
                )

            # Random crop + resize
            if random.random() < 0.3:

                w, h = image.size

                crop_scale = random.uniform(0.85, 1.0)

                new_w = int(w * crop_scale)
                new_h = int(h * crop_scale)

                if new_w < w and new_h < h:

                    left = random.randint(0, w - new_w)
                    top = random.randint(0, h - new_h)

                    image = image.crop(
                        (left, top, left + new_w, top + new_h)
                    )

                    mask = mask.crop(
                        (left, top, left + new_w, top + new_h)
                    )

            # Brightness
            if random.random() < 0.4:

                factor = random.uniform(0.8, 1.2)

                image = ImageEnhance.Brightness(
                    image
                ).enhance(factor)

            # Contrast
            if random.random() < 0.4:

                factor = random.uniform(0.8, 1.2)

                image = ImageEnhance.Contrast(
                    image
                ).enhance(factor)

        # ----------------------------------------------------
        # RESIZE
        # ----------------------------------------------------

        image = image.resize(
            (self.image_size, self.image_size),
            Image.Resampling.BILINEAR
        )

        mask = mask.resize(
            (self.image_size, self.image_size),
            Image.Resampling.NEAREST
        )

        # ----------------------------------------------------
        # IMAGE → TENSOR
        # ----------------------------------------------------

        image = np.array(image).astype(np.float32) / 255.0

        image = torch.from_numpy(
            image.transpose(2, 0, 1)
        ).float()

        # ----------------------------------------------------
        # MASK → TENSOR
        # ----------------------------------------------------

        mask = np.array(mask).astype(np.int64)

        mask = torch.from_numpy(mask).long()

        return image, mask


# ============================================================
# U-NET
# ============================================================

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):

    def __init__(self, num_classes=104):

        super().__init__()

        self.enc1 = DoubleConv(3, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(
            1024, 512, kernel_size=2, stride=2
        )
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(
            512, 256, kernel_size=2, stride=2
        )
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(
            256, 128, kernel_size=2, stride=2
        )
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(
            128, 64, kernel_size=2, stride=2
        )
        self.dec1 = DoubleConv(128, 64)

        self.final = nn.Conv2d(
            64,
            num_classes,
            kernel_size=1
        )

    def forward(self, x):

        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        e4 = self.enc4(
            self.pool(e3)
        )

        b = self.bottleneck(
            self.pool(e4)
        )

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

        return self.final(d1)


# ============================================================
# DICE LOSS
# ============================================================

def multiclass_dice_loss(
    logits,
    targets,
    num_classes=104,
    smooth=1.0
):

    probabilities = torch.softmax(logits, dim=1)

    targets_one_hot = torch.nn.functional.one_hot(
        targets,
        num_classes=num_classes
    )

    targets_one_hot = targets_one_hot.permute(
        0, 3, 1, 2
    ).float()

    probabilities = probabilities[:, 1:]
    targets_one_hot = targets_one_hot[:, 1:]

    intersection = (
        probabilities * targets_one_hot
    ).sum(dim=(2, 3))

    denominator = (
        probabilities.sum(dim=(2, 3))
        +
        targets_one_hot.sum(dim=(2, 3))
    )

    dice = (
        (2.0 * intersection + smooth)
        /
        (denominator + smooth)
    )

    return 1.0 - dice.mean()


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    logits,
    targets,
    num_classes=104
):

    predictions = torch.argmax(
        logits,
        dim=1
    )

    dice_scores = []
    iou_scores = []

    for cls in range(1, num_classes):

        pred_cls = predictions == cls
        target_cls = targets == cls

        intersection = (
            pred_cls & target_cls
        ).sum().float()

        pred_sum = pred_cls.sum().float()
        target_sum = target_cls.sum().float()

        union = (
            pred_cls | target_cls
        ).sum().float()

        if union == 0:
            continue

        dice = (
            2 * intersection + 1e-6
        ) / (
            pred_sum + target_sum + 1e-6
        )

        iou = (
            intersection + 1e-6
        ) / (
            union + 1e-6
        )

        dice_scores.append(dice.item())
        iou_scores.append(iou.item())

    if len(dice_scores) == 0:
        return 0.0, 0.0

    return (
        sum(dice_scores) / len(dice_scores),
        sum(iou_scores) / len(iou_scores)
    )


# ============================================================
# TRAINING
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=50
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

    parser.add_argument(
        "--patience",
        type=int,
        default=8
    )

    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("EXPERIMENT 3: U-NET + DATA AUGMENTATION + FOCAL LOSS")
    print("=" * 70)

    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = FoodSegDataset(
        os.path.join(
            DATA_DIR,
            "train",
            "images"
        ),
        os.path.join(
            DATA_DIR,
            "train",
            "masks"
        ),
        image_size=args.image_size,
        augment=True
    )

    val_dataset = FoodSegDataset(
        os.path.join(
            DATA_DIR,
            "validation",
            "images"
        ),
        os.path.join(
            DATA_DIR,
            "validation",
            "masks"
        ),
        image_size=args.image_size,
        augment=False
    )

    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))

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

    model = UNet(
        num_classes=NUM_CLASSES
    ).to(device)

    # --------------------------------------------------------
    # LOSS + OPTIMIZER
    # --------------------------------------------------------

    class FocalLoss(nn.Module):
        def __init__(self, gamma=2.0):
            super().__init__()
            self.gamma = gamma
            self.ce = nn.CrossEntropyLoss(reduction='none')

        def forward(self, logits, targets):
            ce_loss = self.ce(logits, targets)
            pt = torch.exp(-ce_loss)
            return ((1 - pt) ** self.gamma * ce_loss).mean()

    criterion_focal = FocalLoss(gamma=2.0)

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr
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
        "unet_exp03_focal.csv"
    )

    with open(
        log_path,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "epoch",
            "train_loss",
            "val_loss",
            "dice",
            "iou",
            "lr"
        ])

    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    best_dice = 0.0
    best_epoch = 0

    best_model_path = os.path.join(
        MODEL_DIR,
        "unet_exp03_focal_best.pth"
    )

    epochs_without_improvement = 0

    # ========================================================
    # EPOCH LOOP
    # ========================================================

    for epoch in range(1, args.epochs + 1):

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        model.train()

        train_loss = 0.0

        for images, masks in train_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            masks = masks.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.amp.autocast(
                device_type="cuda",
                enabled=torch.cuda.is_available()
            ):

                outputs = model(images)

                focal_loss = criterion_focal(
                    outputs,
                    masks
                )

                dice_loss = multiclass_dice_loss(
                    outputs,
                    masks,
                    NUM_CLASSES
                )

                loss = focal_loss + dice_loss

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
        dice_total = 0.0
        iou_total = 0.0
        metric_batches = 0

        with torch.no_grad():

            for images, masks in val_loader:

                images = images.to(
                    device,
                    non_blocking=True
                )

                masks = masks.to(
                    device,
                    non_blocking=True
                )

                with torch.amp.autocast(
                    device_type="cuda",
                    enabled=torch.cuda.is_available()
                ):

                    outputs = model(images)

                    focal_loss = criterion_focal(
                        outputs,
                        masks
                    )

                    dice_loss = multiclass_dice_loss(
                        outputs,
                        masks,
                        NUM_CLASSES
                    )

                    loss = focal_loss + dice_loss

                val_loss += loss.item()

                dice, iou = calculate_metrics(
                    outputs,
                    masks,
                    NUM_CLASSES
                )

                dice_total += dice
                iou_total += iou
                metric_batches += 1

        val_loss /= len(val_loader)

        val_dice = dice_total / metric_batches
        val_iou = iou_total / metric_batches

        # ----------------------------------------------------
        # LR SCHEDULER
        # ----------------------------------------------------

        scheduler.step(val_dice)

        current_lr = optimizer.param_groups[0]["lr"]

        # ----------------------------------------------------
        # LOG
        # ----------------------------------------------------

        with open(
            log_path,
            "a",
            newline=""
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                epoch,
                train_loss,
                val_loss,
                val_dice,
                val_iou,
                current_lr
            ])

        # ----------------------------------------------------
        # PRINT
        # ----------------------------------------------------

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Dice: {val_dice:.4f} | "
            f"IoU: {val_iou:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        if val_dice > best_dice:

            best_dice = val_dice
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                model.state_dict(),
                best_model_path
            )

            print(
                f"  >>> NEW BEST MODEL "
                f"(Dice = {best_dice:.4f})"
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
            print(
                f"Best Dice: {best_dice:.4f}"
            )
            print(
                f"Best Epoch: {best_epoch}"
            )
            print("=" * 70)

            break

    print()
    print("Training complete.")
    print("Best model:")
    print(best_model_path)

    print("Training log:")
    print(log_path)


if __name__ == "__main__":
    main()