from pathlib import Path
import torch
import librosa
import numpy as np
from cnn import GuitarCNN
import jams
import pretty_midi


def predict_notes_onset_driven_filtered(
    audio_path,
    model,
    tuning,
    sr=22050,
    hop_length=512,
    n_bins=192,
    bins_per_octave=24,
    win_width=9,
    device="cpu",
    min_onset_gap=0.05,  # merge onsets closer than 50ms
):
    # --- 1. Load audio ---
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    y = librosa.util.normalize(y)

    # --- 2. Onset detection ---
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    print(f"Detected {len(onsets)} raw onsets")

    # --- 3. Compute CQT ---
    cqt = np.abs(
        librosa.cqt(
            y,
            sr=sr,
            hop_length=hop_length,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
        )
    )
    cqt = np.log1p(cqt).T  # (time, freq)
    hop_time = hop_length / sr
    half_width = win_width // 2
    notes = []
    last_onset_time = -np.inf

    # --- 4. Extract windows for each onset ---
    for t_onset in onsets:
        # --- 4a. Merge nearby onsets ---
        if t_onset - last_onset_time < min_onset_gap:
            continue
        last_onset_time = t_onset

        frame_idx = int(round(t_onset / hop_time))
        n_frames, n_bins = cqt.shape

        # Build window
        l, r = frame_idx - half_width, frame_idx + half_width + 1
        if l < 0:
            frame = np.vstack([np.zeros((-l, n_bins)), cqt[:r]])
        elif r > n_frames:
            frame = np.vstack([cqt[l:], np.zeros((r - n_frames, n_bins))])
        else:
            frame = cqt[l:r]
        window = frame.T  # (freq, win_width)

        # Torch input
        x = (
            torch.tensor(window, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )

        # --- 4b. Predict ---
        with torch.no_grad():
            out = model(x)  # [1, strings, classes]
            probs = torch.softmax(out, dim=2)  # optional: get probabilities
            pred = out.argmax(dim=2).cpu().numpy()[0]
            pred_probs = probs.max(dim=2)[0].cpu().numpy()[0]

        # --- 4c. Skip if all strings are muted ---
        if np.all(pred == 0):
            continue

        # --- 4d. Decode ---
        for string, (fret_idx, p) in enumerate(zip(pred, pred_probs)):
            if fret_idx == 0:  # muted
                continue
            fret = fret_idx - 1
            midi = tuning[string] + fret
            notes.append(
                (t_onset, string, fret, midi, p)
            )  # optionally include confidence

    print(f"Filtered {len(notes)} predicted notes")
    return notes


def predict_notes_with_duration(
    audio_path,
    model,
    tuning,
    sr=22050,
    hop_length=512,
    n_bins=192,
    bins_per_octave=24,
    win_width=9,
    device="cpu",
    min_onset_gap=0.05,
):
    # --- Load audio ---
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    y = librosa.util.normalize(y)

    # --- Onset detection ---
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    hop_time = hop_length / sr
    print(f"Detected {len(onsets)} raw onsets")

    # --- Compute CQT for the entire audio ---
    cqt = np.abs(
        librosa.cqt(
            y,
            sr=sr,
            hop_length=hop_length,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
        )
    )
    cqt = np.log1p(cqt).T  # (frames, freq)
    n_frames, n_bins = cqt.shape
    half_width = win_width // 2

    notes = []
    last_onset_time = -np.inf

    # --- For each onset ---
    for t_onset in onsets:
        if t_onset - last_onset_time < min_onset_gap:
            continue
        last_onset_time = t_onset

        frame_idx = int(round(t_onset / hop_time))

        # Build window
        l, r = frame_idx - half_width, frame_idx + half_width + 1
        if l < 0:
            frame = np.vstack([np.zeros((-l, n_bins)), cqt[:r]])
        elif r > n_frames:
            frame = np.vstack([cqt[l:], np.zeros((r - n_frames, n_bins))])
        else:
            frame = cqt[l:r]
        window = frame.T  # (freq, win_width)
        x = (
            torch.tensor(window, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )

        # Predict at onset frame
        with torch.no_grad():
            out = model(x)  # [1, strings, classes]
            pred = out.argmax(dim=2).cpu().numpy()[0]

        # --- Track duration for each string ---
        for string, fret_idx in enumerate(pred):
            if fret_idx == 0:
                continue
            fret = fret_idx - 1
            midi = tuning[string] + fret

            # Start tracking from onset frame forward
            offset_frame = frame_idx
            while offset_frame + 1 < n_frames:
                # Build next window
                l, r = offset_frame + 1 - half_width, offset_frame + 1 + half_width + 1
                if l < 0:
                    f = np.vstack([np.zeros((-l, n_bins)), cqt[:r]])
                elif r > n_frames:
                    f = np.vstack([cqt[l:], np.zeros((r - n_frames, n_bins))])
                else:
                    f = cqt[l:r]
                next_window = f.T
                x_next = (
                    torch.tensor(next_window, dtype=torch.float32)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(device)
                )

                # Predict
                with torch.no_grad():
                    out_next = model(x_next)
                    pred_next = out_next.argmax(dim=2).cpu().numpy()[0]

                # Stop if string muted or fret changed
                if pred_next[string] != fret_idx:
                    break
                offset_frame += 1

            onset_time = frame_idx * hop_time
            offset_time = offset_frame * hop_time
            notes.append((onset_time, offset_time, string, fret, midi))

    print(f"Total predicted notes with duration: {len(notes)}")
    return notes


def export_to_midi(notes, tuning, out_file="predicted.mid"):
    pm = pretty_midi.PrettyMIDI()
    guitar = pretty_midi.Instrument(program=24)  # Acoustic Guitar (nylon)

    for onset, offset, string, fret, midi in notes:
        note = pretty_midi.Note(velocity=100, pitch=midi, start=onset, end=offset)
        guitar.notes.append(note)

    pm.instruments.append(guitar)
    pm.write(out_file)
    print(f"MIDI written to {out_file}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio_dir = Path("data/raw/audio_mono-mic")
    label_dir = Path("data/raw/annotation")
    file_list = [f.stem for f in list(label_dir.glob("*.jams"))]
    tuning = [40, 45, 50, 55, 59, 64]

    # filter file list for solos only
    file_list = [f for f in filter(lambda x: "solo" in x, file_list)]

    idx = np.random.choice(len(file_list), 1, replace=False).item()
    choice = file_list[idx]
    rec_mode = "mic"
    jams_file = label_dir / (choice + ".jams")
    audio_file = audio_dir / (choice + "_" + rec_mode + ".wav")

    print(choice, audio_file)
    jam = jams.load(str(jams_file))
    truth_notes = []
    for ann in jam.annotations:
        if ann.namespace in ["note_midi", "note_hz"]:
            truth_notes.extend(ann.data)

    print(f"Expecting {len(truth_notes)} predictions.")

    model = GuitarCNN().to(device)
    model.load_state_dict(torch.load("guitar_cnn_best.pt", map_location=device))
    model.eval()

    pred_notes = predict_notes_with_duration(audio_file, model, tuning)
    print(f"Predicted: {[n[3] for n in pred_notes]}")
    print(f"Truth: {[int(n.value) for n in truth_notes]}")
    export_to_midi(pred_notes, tuning, out_file="transcription.mid")
