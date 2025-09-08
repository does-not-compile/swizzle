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
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=4)

    # Model + training
    model = GuitarCNN(
        n_strings=dataset_full.n_strings, n_classes=dataset_full.n_classes
    ).to(device)

    train_model(model, train_loader, val_loader, device, epochs=20)
