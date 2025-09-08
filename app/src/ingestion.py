#############################################
#          DATA INGESTION PIPELINE          #
#############################################
import os
from pathlib import Path
import numpy as np
import librosa
import jams
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm


class GuitarSetDataset(Dataset):
    def __init__(
        self,
        audio_dir: str,
        label_dir: str,
        file_list: list,
        rmode: str,
        cache_file: None,
        sr: int = 22050,
        hop_length: int = 512,
        n_bins: int = 192,
        bins_per_octave: int = 24,
        win_width: int = 9,
        frets: int = 19,
        tuning: list = [40, 45, 50, 55, 59, 64],
        augment: bool = False,
        normalize: bool = True,
        remove_noise: float = 0.95,
    ):
        """
        PyTorch Dataset for GuitarSet (Xi et al.).

        Args:
            audio_dir (str): path to audio files.
            label_dir (str): path to jams annotations.
            file_list (list): list of filenames (without extension).
            rmode (str): subset used of GuitarSet (hex_cln, or mic)
            sr (int): sample rate for librosa.load.
            hop_length (int): hop length for CQT.
            n_bins (int): number of CQT bins.
            bins_per_octave (int): bins per octave for CQT.
            win_width (int): sliding window width.
            frets (int): number of frets.
            tuning (list): MIDI numbers of open strings.
            augment (bool): whether to apply SpecAugment-style augmentation.
            normalize (bool): normalize spectrogram magnitudes.
            remove_noise (float): fraction of empty frames to drop.
        """
        self.audio_dir = audio_dir
        self.label_dir = label_dir
        self.file_list = file_list
        self.rmode = rmode
        self.sr = sr
        self.hop_length = hop_length
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.win_width = win_width
        self.half_width = win_width // 2
        self.frets = frets
        self.tuning = tuning
        self.n_strings = len(tuning)
        self.n_classes = frets + 2  # open and not played
        self.augment = augment
        self.normalize = normalize
        self.remove_noise_frac = remove_noise
        self.cache_file = cache_file
        # other kwargs like sr, n_bins, etc.

        if cache_file and os.path.exists(cache_file):
            # load preprocessed data
            data = np.load(cache_file, mmap_mode="r")
            self.windows = data["windows"]
            self.labels = data["labels"]
            print(f"Loaded preprocessed data from {cache_file}")
        else:
            # normal preprocessing
            self.windows = []
            self.labels = []
            self._preprocess_all_files()
            if cache_file:
                np.savez_compressed(
                    cache_file, windows=self.windows, labels=self.labels
                )
                print(f"Saved preprocessed data to {cache_file}")

    def _preprocess_all_files(self):
        for fname in tqdm(self.file_list, desc="Preprocessing GuitarSet files"):
            audio, labels = self._load_file(fname, self.rmode)
            cqt = self._preprocess_audio(audio)
            win_data = self._make_windows(cqt)
            win_labels = self._make_window_labels(labels, cqt.shape[0], audio)

            # optional: remove mostly-empty windows
            if self.remove_noise_frac:
                mask = self._remove_noise(win_labels)
                win_data = win_data[mask]
                win_labels = win_labels[mask]

            self.windows.append(win_data)
            self.labels.append(win_labels)

        self.windows = np.concatenate(self.windows, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)

    def _load_file(self, fname: str, rmode: str):
        audio_path = os.path.join(self.audio_dir, fname + "_" + rmode + ".wav")
        label_path = os.path.join(self.label_dir, fname + ".jams")
        audio, _ = librosa.load(audio_path, sr=self.sr, mono=True)
        labels = jams.load(label_path)
        return audio, labels

    def _preprocess_audio(self, audio):
        if self.normalize:
            audio = librosa.util.normalize(audio)
        cqt = np.abs(
            librosa.cqt(
                audio,
                sr=self.sr,
                hop_length=self.hop_length,
                n_bins=self.n_bins,
                bins_per_octave=self.bins_per_octave,
            )
        )
        cqt = np.log1p(cqt)  # log scaling
        return cqt.T  # shape: (time, freq)

    def _make_windows(self, cqt):
        n_frames, n_bins = cqt.shape
        windows = np.zeros((n_frames, n_bins, self.win_width))
        for t in range(n_frames):
            l = t - self.half_width
            r = t + self.half_width + 1
            frame = np.zeros((self.win_width, n_bins))
            if l < 0:
                pad = np.zeros((-l, n_bins))
                frame[: self.win_width + r, :] = np.vstack([pad, cqt[:r]])
            elif r > n_frames:
                pad = np.zeros((r - n_frames, n_bins))
                frame[: self.win_width - (r - n_frames), :] = cqt[l:n_frames]
                frame[self.win_width - (r - n_frames) :, :] = pad
            else:
                frame = cqt[l:r]
            windows[t] = frame.T
        return windows  # (time, freq, win_width)

    def _make_window_labels(self, labels, n_frames, audio):
        """
        Create label windows for each frame, mapping notes to string/fret positions.
        Handles time alignment in seconds and extracts string info from JAMS.

        Args:
            labels (jams.JAMS): annotations object
            n_frames (int): number of time frames in the CQT
            audio (np.ndarray): waveform, used to compute track duration
        """
        track_duration = librosa.get_duration(
            y=audio, sr=self.sr, hop_length=self.hop_length
        )
        notes = []

        # Iterate over all 'note_midi' annotations
        for string_idx, ann in enumerate(
            labels.annotations.search(namespace="note_midi")
        ):
            for obs in ann.data:
                notes.append([string_idx, int(obs.value), obs.time])

        notes = np.array(sorted(notes, key=lambda x: x[-1]))  # sort by time

        win_labels = np.zeros((n_frames, self.n_strings, self.n_classes))

        for t in range(n_frames):
            # compute window bounds in seconds
            lbound = max(0, (t - self.half_width) / n_frames * track_duration)
            rbound = min(
                track_duration, (t + self.half_width) / n_frames * track_duration
            )

            # select notes that are active in this window
            active = notes[(notes[:, 2] >= lbound) & (notes[:, 2] <= rbound)]
            for a in active:
                string = int(a[0])
                midi = int(a[1])

                fret = midi - self.tuning[string] + 1
                if 0 <= fret < self.n_classes:
                    win_labels[t, string, fret] = 1

            # if nothing played on string → mark as "mute"
            for s in range(self.n_strings):
                if not win_labels[t, s].any():
                    win_labels[t, s, 0] = 1

        return win_labels

    def _remove_noise(self, labels):
        noise_idx = []
        for i, frame in enumerate(labels):
            if np.all(frame[:, 0] == 1):  # all strings muted
                noise_idx.append(i)
        mask = np.ones(len(labels), dtype=bool)
        drop = int(len(noise_idx) * self.remove_noise_frac)
        mask[noise_idx[:drop]] = False
        return mask

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = self.windows[idx]  # (freq, time)
        y = self.labels[idx]  # (strings, classes)

        # convert to torch tensors
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # add channel dim
        y = torch.tensor(y, dtype=torch.float32)

        # augmentations
        if self.augment:
            if np.random.rand() < 0.5:
                # SpecAugment-style frequency masking
                f = np.random.randint(0, x.shape[1] // 8)
                f0 = np.random.randint(0, x.shape[1] - f)
                x[:, f0 : f0 + f, :] = 0.0

            if np.random.rand() < 0.5:
                # SpecAugment-style time masking
                t = np.random.randint(0, x.shape[2] // 8)
                t0 = np.random.randint(0, x.shape[2] - t)
                x[:, :, t0 : t0 + t] = 0.0

            if np.random.rand() < 0.3:
                # Gaussian noise injection
                noise = torch.randn_like(x) * 0.01
                x = x + noise

            if np.random.rand() < 0.3:
                # Random gain (volume scaling)
                gain = 0.8 + 0.4 * np.random.rand()
                x = x * gain

        return x, y

    def sanity_check(self, n_samples: int = 5):
        """
        Quick test to inspect a few windows and labels.

        Args:
            n_samples (int): number of frames to inspect
        """
        print(f"Dataset contains {len(self.windows)} windows.")
        print(
            f"Each window shape: {self.windows[0].shape}, each label shape: {self.labels[0].shape}"
        )

        for i in range(min(n_samples, len(self.windows))):
            x = self.windows[i]
            y = self.labels[i]
            print(f"\nSample {i}:")
            print(f"  Window min/max: {x.min():.4f} / {x.max():.4f}")
            print(
                f"  Label sum per string: {y.sum(axis=1)} (should be >=1 for each string)"
            )
            # Optionally, show which frets are active for first string
            active_frets = np.where(y[0] > 0)[0]
            print(f"  Active frets (string 0): {active_frets}")


if __name__ == "__main__":

    cache_file = "data/preprocessed_guitardataset_mic.npz"
    audio_dir = "data/raw/audio_mono-mic"
    label_dir = "data/raw/annotation"
    file_list = [f.stem for f in list(Path(label_dir).glob("*.jams"))]

    # point to your GuitarSet data root
    dataset = GuitarSetDataset(
        audio_dir,
        label_dir,
        file_list,
        cache_file=cache_file,
        rmode="mic",
        augment=True,
    )

    dataset.sanity_check()

    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Grab one batch
    xb, yb = next(iter(loader))

    print("Input batch shape:", xb.shape)  # expect: [B, 1, n_bins, frame_width]
    print("Label batch shape:", yb.shape)  # expect: [B, n_strings, n_classes]

    # quick sanity check
    print("Sample labels (first string, first 10 classes):")
    print(yb[0, 0, :10])
