# train.py

import argparse
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import glob
from model import get_model
from custom_dataset import get_dataset
import os


def train_model(
    model,
    train_data_dir,
    device,
    optimizer,
    criterion,
    total_epochs=50,
    save_path="models",
    model_weight_path=None,
):

    train_data = get_dataset(train_data_dir)

    # SAFE SETTINGS FOR CPU TRAINING
    train_loader = DataLoader(
        train_data,
        batch_size=16,
        num_workers=0,
        pin_memory=False,
        shuffle=True,
    )

    scaler = GradScaler()

    starting_epoch = 0
    train_losses = []

    # =============================
    # LOAD CHECKPOINT IF PROVIDED
    # =============================
    if model_weight_path and os.path.exists(model_weight_path):
        print(f"Loading checkpoint from {model_weight_path}")
        checkpoint = torch.load(model_weight_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        starting_epoch = checkpoint["epoch"] + 1
        train_losses = checkpoint.get("train_losses", [])

        model.train()

    else:
        # AUTO LOAD LATEST MODEL FROM SAVE PATH
        if os.path.exists(save_path):
            list_of_files = glob.glob(os.path.join(save_path, "model_epoch_*.pth"))

            if list_of_files:
                latest_file = max(list_of_files, key=os.path.getctime)
                print(f"Resuming from latest checkpoint: {latest_file}")

                checkpoint = torch.load(latest_file, map_location=device)

                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

                starting_epoch = checkpoint["epoch"] + 1
                train_losses = checkpoint.get("train_losses", [])

                model.train()

    # =============================
    # TRAINING LOOP
    # =============================
    for epoch in range(starting_epoch, total_epochs):

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch + 1}/{total_epochs} | Learning Rate: {current_lr:.6f}")

        model.train()
        total_loss = 0.0

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}", unit="batch") as progress_bar:

            for inputs, labels in train_loader:

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs.logits, labels)

                scaler.scale(loss).backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()

                progress_bar.set_postfix({"Loss": loss.item()})
                progress_bar.update()

                del inputs, labels, outputs

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        print(f"Epoch {epoch + 1} - Average Training Loss: {avg_train_loss:.4f}")

        # =============================
        # SAVE MODEL
        # =============================
        os.makedirs(save_path, exist_ok=True)

        save_file = os.path.join(save_path, f"model_epoch_{epoch}.pth")

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "avg_loss": avg_train_loss,
                "scaler_state_dict": scaler.state_dict(),
                "train_losses": train_losses,
            },
            save_file,
        )

        print(f"Model saved to: {save_file}")

        torch.cuda.empty_cache()


# =============================
# MAIN
# =============================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train AI Image Detector")

    parser.add_argument(
        "train_data_dir",
        type=str,
        help="Directory path for training data",
    )

    parser.add_argument(
        "--total_epochs",
        type=int,
        default=10,
        help="Total number of epochs",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="models",
        help="Folder to save checkpoints",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )

    parser.add_argument(
        "--model_weight_path",
        type=str,
        default=None,
        help="Path to specific checkpoint to resume",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    model = get_model(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-5,
        eps=1e-8,
    )

    criterion = torch.nn.CrossEntropyLoss()

    train_model(
        model,
        args.train_data_dir,
        device,
        optimizer,
        criterion,
        args.total_epochs,
        args.save_path,
        args.model_weight_path,
    )
