#############################################
#              CNN TRAINING LOOP            #
#############################################
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from ingestion import GuitarSetDataset
from tqdm.auto import tqdm


# -------------------------------
# Residual CNN
# -------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        residual = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class GuitarCNN(nn.Module):
    def __init__(self, n_strings=6, n_classes=21):
        super().__init__()
        self.features = nn.Sequential(
            ResidualBlock(1, 32),
            nn.MaxPool2d(2),
            ResidualBlock(32, 64),
            nn.MaxPool2d(2),
            ResidualBlock(64, 128),
            nn.MaxPool2d(2),
            ResidualBlock(128, 256),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, n_strings * n_classes),
        )

        self.n_strings = n_strings
        self.n_classes = n_classes

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x.view(-1, self.n_strings, self.n_classes)  # logits


# -------------------------------
# Loss function
# -------------------------------
def per_string_loss(output, target):
    # target: one-hot [B, strings, classes]
    loss = 0
    for s in range(target.shape[1]):
        loss += F.cross_entropy(output[:, s, :], target[:, s].argmax(dim=1))
    return loss / target.shape[1]


# -------------------------------
# Training utilities
# -------------------------------
def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0
    pbar = tqdm(dataloader, total=len(dataloader), desc="Batches", leave=False)
    for i, (xb, yb) in enumerate(pbar):
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = per_string_loss(out, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)

        # Optional: compute batch accuracy
        pred = out.argmax(dim=2)
        true = yb.argmax(dim=2)
        batch_acc = (pred == true).float().mean().item()

        pbar.set_postfix(loss=loss.item(), acc=f"{batch_acc*100:.2f}%")

    # return epoch loss
    return running_loss / len(dataloader.dataset)


def evaluate(model, dataloader, device):
    model.eval()
    running_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = per_string_loss(out, yb)
            running_loss += loss.item() * xb.size(0)

            pred = out.argmax(dim=2)
            true = yb.argmax(dim=2)
            correct += (pred == true).sum().item()
            total += torch.numel(true)
    return running_loss / len(dataloader.dataset), correct / total


def train_model(model, train_loader, val_loader, device, epochs=20, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=3, factor=0.5, min_lr=1e-5
    )

    best_acc = 0
    for epoch in tqdm(range(epochs), desc="Epochs"):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        scheduler.step(val_acc)

        print(
            f"Epoch {epoch+1}/{epochs} "
            f"- Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Acc: {val_acc:.4f}"
        )

        # save model checkpoints every epoch
        checkpoint_path = f"checkpoints/guitar_cnn_epoch{epoch+1}.pt"
        Path("checkpoints").mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        # track best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "checkpoints/guitar_cnn_best.pt")


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")

    cache_file = "data/preprocessed_guitardataset_mic.npz"
    audio_dir = "data/raw/audio_mono-mic"
    label_dir = "data/raw/annotation"
    file_list = [f.stem for f in list(Path(label_dir).glob("*.jams"))]

    # Train/val split indices
    dataset_full = GuitarSetDataset(
        audio_dir=None,
        label_dir=None,
        file_list=[],
        rmode="mic",
        cache_file=cache_file,
        augment=False,
    )
    n = len(dataset_full)
    idx = np.random.default_rng(42).permutation(n)
    split = int(0.9 * n)
    train_idx, val_idx = idx[:split], idx[split:]

    # Create separate dataset instances so we can toggle augment
    train_ds = GuitarSetDataset(
        audio_dir=None,
        label_dir=None,
        file_list=[],
        rmode="mic",
        cache_file=cache_file,
        augment=True,
    )
    val_ds = GuitarSetDataset(
        audio_dir=None,
        label_dir=None,
        file_list=[],
        rmode="mic",
        cache_file=cache_file,
        augment=False,
    )

    # Subset with same indices for consistency)
    train_ds = Subset(train_ds, train_idx)
    val_ds = Subset(val_ds, val_idx)

    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4)

    # Model + training
    model = GuitarCNN(
        n_strings=dataset_full.n_strings, n_classes=dataset_full.n_classes
    ).to(device)

    train_model(model, train_loader, val_loader, device, epochs=20)
