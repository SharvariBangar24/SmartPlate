import os
import csv
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from train_unet import (
    DATA_DIR,
    MODEL_DIR,
    LOG_DIR,
    NUM_CLASSES,
    FoodSegDataset,
    UNet,
    multiclass_dice_loss,
    calculate_metrics
)

START_EPOCH = 50
INITIAL_BEST_DICE = 0.1461
INITIAL_BEST_IOU = 0.1121

RESUME_MODEL = os.path.join(
    MODEL_DIR,
    "unet_augmentation_best.pth"
)

EXTENDED_MODEL = os.path.join(
    MODEL_DIR,
    "unet_augmentation_extended_best.pth"
)

LOG_PATH = os.path.join(
    LOG_DIR,
    "unet_augmentation_extended.csv"
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.00025)
    parser.add_argument("--patience", type=int, default=12)

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\n" + "=" * 70)
    print("EXPERIMENT 2 CONTINUATION: U-NET + AUGMENTATION")
    print("=" * 70)

    print(f"Device: {device}")

    if not os.path.exists(RESUME_MODEL):
        raise FileNotFoundError(
            f"Best model not found:\n{RESUME_MODEL}"
        )

    # DATASETS
    train_dataset = FoodSegDataset(
    os.path.join(DATA_DIR, "train", "images"),
    os.path.join(DATA_DIR, "train", "masks"),
    image_size=args.image_size,
    augment=True
	)

    val_dataset = FoodSegDataset(
    os.path.join(DATA_DIR, "validation", "images"),
    os.path.join(DATA_DIR, "validation", "masks"),
    image_size=args.image_size,
    augment=False
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

    # MODEL
    model = UNet(
        num_classes=NUM_CLASSES
    ).to(device)

    # LOAD BEST EXPERIMENT 2 MODEL
    checkpoint = torch.load(
        RESUME_MODEL,
        map_location=device
    )

    model.load_state_dict(checkpoint)

    print("\nLoaded best Experiment 2 model:")
    print(f"Dice: {INITIAL_BEST_DICE:.4f}")
    print(f"IoU : {INITIAL_BEST_IOU:.4f}")
    print(f"Starting from epoch: {START_EPOCH}")

    # OPTIMIZER
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

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda")
    )

    best_dice = INITIAL_BEST_DICE
    best_epoch = START_EPOCH
    epochs_without_improvement = 0

    # Keep a copy of the starting best model
    torch.save(
        model.state_dict(),
        EXTENDED_MODEL
    )

    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "epoch",
            "train_loss",
            "val_loss",
            "dice",
            "iou",
            "lr"
        ])

        writer.writerow([
            START_EPOCH,
            "",
            "",
            INITIAL_BEST_DICE,
            INITIAL_BEST_IOU,
            args.lr
        ])

        # TRAINING
        for local_epoch in range(1, args.epochs + 1):

            epoch = START_EPOCH + local_epoch

            model.train()

            total_train_loss = 0.0

            for images, masks in train_loader:

                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(
                    device_type=device.type,
                    enabled=(device.type == "cuda")
                ):

                    outputs = model(images)

                    ce_loss = torch.nn.functional.cross_entropy(
                        outputs,
                        masks
                    )

                    dice_loss = multiclass_dice_loss(
                        outputs,
                        masks,
                        NUM_CLASSES
                    )

                    loss = ce_loss + dice_loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_train_loss += loss.item()

            train_loss = (
                total_train_loss / len(train_loader)
            )

            # VALIDATION
            model.eval()

            total_val_loss = 0.0
            total_dice = 0.0
            total_iou = 0.0

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
                        device_type=device.type,
                        enabled=(device.type == "cuda")
                    ):

                        outputs = model(images)

                        ce_loss = torch.nn.functional.cross_entropy(
                            outputs,
                            masks
                        )

                        dice_loss = multiclass_dice_loss(
                            outputs,
                            masks,
                            NUM_CLASSES
                        )

                        loss = ce_loss + dice_loss

                    total_val_loss += loss.item()

                    dice, iou = calculate_metrics(
                        outputs,
                        masks,
                        NUM_CLASSES
                    )

                    total_dice += dice
                    total_iou += iou

            val_loss = (
                total_val_loss / len(val_loader)
            )

            val_dice = (
                total_dice / len(val_loader)
            )

            val_iou = (
                total_iou / len(val_loader)
            )

            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch:03d}/100 | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Dice: {val_dice:.4f} | "
                f"IoU: {val_iou:.4f} | "
                f"LR: {current_lr:.6f}"
            )

            writer.writerow([
                epoch,
                train_loss,
                val_loss,
                val_dice,
                val_iou,
                current_lr
            ])

            f.flush()

            # LR scheduler
            scheduler.step(val_dice)

            # BEST MODEL
            if val_dice > best_dice:

                best_dice = val_dice
                best_epoch = epoch
                epochs_without_improvement = 0

                torch.save(
                    model.state_dict(),
                    EXTENDED_MODEL
                )

                print(
                    f"  >>> NEW BEST MODEL "
                    f"(Dice = {best_dice:.4f})"
                )

            else:

                epochs_without_improvement += 1

            # EARLY STOPPING
            if epochs_without_improvement >= args.patience:

                print(
                    f"\nEarly stopping at epoch {epoch}."
                )

                break

    print("\n" + "=" * 70)
    print("CONTINUATION COMPLETE")
    print("=" * 70)

    print(f"Best Dice : {best_dice:.4f}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best model: {EXTENDED_MODEL}")
    print(f"Log       : {LOG_PATH}")


if __name__ == "__main__":
    main()