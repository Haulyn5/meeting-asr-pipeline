import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from speechbrain.inference.speaker import EncoderClassifier
from tqdm import tqdm


DEFAULT_EMBEDDING_MODEL = os.environ.get(
    "MEETING_ASR_SPEAKER_MODEL_DIR",
    "models/spkrec-ecapa-voxceleb",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Energy VAD + speaker embedding diarization for meeting audio.")
    parser.add_argument("audio", help="Input wav file.")
    parser.add_argument("--output-dir", required=True, help="Directory for normalized audio and manifests.")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL, help="Local SpeechBrain embedding model dir.")
    parser.add_argument("--num-speakers", type=int, default=2, help="Expected number of speakers.")
    parser.add_argument("--device", default="cuda:0", help="Torch device for embedding extraction.")
    parser.add_argument("--target-sr", type=int, default=16000)
    parser.add_argument("--vad-frame-ms", type=float, default=30.0)
    parser.add_argument("--vad-hop-ms", type=float, default=10.0)
    parser.add_argument("--min-speech-sec", type=float, default=0.45)
    parser.add_argument("--merge-speech-gap-sec", type=float, default=0.35)
    parser.add_argument("--embed-window-sec", type=float, default=2.0)
    parser.add_argument("--embed-hop-sec", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cluster-method", choices=["kmeans", "agglomerative"], default="kmeans")
    parser.add_argument("--max-chunk-sec", type=float, default=30.0)
    parser.add_argument("--merge-same-speaker-gap-sec", type=float, default=0.8)
    return parser.parse_args()


def load_mono_resampled(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if sr != target_sr:
        gcd = math.gcd(sr, target_sr)
        mono = resample_poly(mono, target_sr // gcd, sr // gcd).astype(np.float32)
        sr = target_sr
    peak = float(np.max(np.abs(mono)) + 1e-9)
    if peak > 1.0:
        mono = mono / peak
    return mono.astype(np.float32), sr


def frame_rms(audio: np.ndarray, sr: int, frame_ms: float, hop_ms: float) -> tuple[np.ndarray, np.ndarray]:
    frame = int(sr * frame_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    starts = np.arange(0, max(1, len(audio) - frame + 1), hop)
    rms = np.empty(len(starts), dtype=np.float32)
    for i, start in enumerate(starts):
        chunk = audio[start : start + frame]
        rms[i] = float(np.sqrt(np.mean(chunk * chunk) + 1e-12))
    times = starts / sr
    return times, rms


def vad_segments(audio: np.ndarray, sr: int, args: argparse.Namespace) -> list[dict]:
    times, rms = frame_rms(audio, sr, args.vad_frame_ms, args.vad_hop_ms)
    db = 20 * np.log10(rms + 1e-8)
    noise = float(np.percentile(db, 20))
    speechish = float(np.percentile(db, 70))
    threshold = max(noise + 10.0, speechish - 6.0)
    active = db > threshold

    # Smooth isolated holes/spikes with a short moving majority filter.
    width = max(1, int(0.20 / (args.vad_hop_ms / 1000)))
    kernel = np.ones(width, dtype=np.float32)
    smoothed = np.convolve(active.astype(np.float32), kernel, mode="same") >= (0.35 * width)

    segments = []
    start = None
    hop = args.vad_hop_ms / 1000
    frame = args.vad_frame_ms / 1000
    for idx, is_active in enumerate(smoothed):
        if is_active and start is None:
            start = float(times[idx])
        elif not is_active and start is not None:
            end = float(times[idx] + frame)
            if end - start >= args.min_speech_sec:
                segments.append({"start": start, "end": min(end, len(audio) / sr)})
            start = None
    if start is not None:
        segments.append({"start": start, "end": len(audio) / sr})

    merged = []
    for seg in segments:
        if merged and seg["start"] - merged[-1]["end"] <= args.merge_speech_gap_sec:
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(seg)
    return merged


def embedding_windows(segments: list[dict], audio_len_sec: float, window_sec: float, hop_sec: float) -> list[dict]:
    windows = []
    for seg in segments:
        start, end = seg["start"], seg["end"]
        duration = end - start
        if duration <= 0:
            continue
        if duration <= window_sec:
            center = (start + end) / 2
            win_start = max(0.0, center - window_sec / 2)
            win_end = min(audio_len_sec, win_start + window_sec)
            win_start = max(0.0, win_end - window_sec)
            windows.append({"start": win_start, "end": win_end, "mid": center})
            continue
        pos = start
        while pos + window_sec <= end:
            windows.append({"start": pos, "end": pos + window_sec, "mid": pos + window_sec / 2})
            pos += hop_sec
        if windows and windows[-1]["end"] < end - 0.25:
            windows.append({"start": end - window_sec, "end": end, "mid": end - window_sec / 2})
    return windows


def extract_embeddings(
    audio: np.ndarray,
    sr: int,
    windows: list[dict],
    model_dir: Path,
    device: str,
    batch_size: int,
) -> np.ndarray:
    classifier = EncoderClassifier.from_hparams(
        source=str(model_dir),
        savedir=str(model_dir),
        overrides={"pretrained_path": str(model_dir)},
        run_opts={"device": device},
    )
    samples_per_window = int(round((windows[0]["end"] - windows[0]["start"]) * sr))
    embeddings = []
    for i in tqdm(range(0, len(windows), batch_size), desc="speaker embeddings"):
        batch_windows = windows[i : i + batch_size]
        batch = np.zeros((len(batch_windows), samples_per_window), dtype=np.float32)
        for row, win in enumerate(batch_windows):
            s = int(round(win["start"] * sr))
            e = min(len(audio), s + samples_per_window)
            chunk = audio[s:e]
            batch[row, : len(chunk)] = chunk
        wavs = torch.from_numpy(batch).to(device)
        with torch.no_grad():
            emb = classifier.encode_batch(wavs).squeeze(1).detach().cpu().numpy()
        embeddings.append(emb)
    matrix = np.vstack(embeddings)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    return matrix / norms


def cluster_embeddings(embeddings: np.ndarray, num_speakers: int, method: str) -> np.ndarray:
    if method == "kmeans":
        return KMeans(n_clusters=num_speakers, n_init=20, random_state=0).fit_predict(embeddings)
    try:
        clusterer = AgglomerativeClustering(n_clusters=num_speakers, metric="cosine", linkage="average")
    except TypeError:
        clusterer = AgglomerativeClustering(n_clusters=num_speakers, affinity="cosine", linkage="average")
    return clusterer.fit_predict(embeddings)


def assign_segments(segments: list[dict], windows: list[dict], labels: np.ndarray) -> list[dict]:
    diarized = []
    mids = np.array([w["mid"] for w in windows])
    for seg in segments:
        inside = np.where((mids >= seg["start"]) & (mids <= seg["end"]))[0]
        if len(inside) == 0:
            nearest = int(np.argmin(np.abs(mids - ((seg["start"] + seg["end"]) / 2))))
            inside = np.array([nearest])
        boundaries = [seg["start"]]
        if len(inside) > 1:
            local_mids = mids[inside]
            boundaries.extend(((local_mids[:-1] + local_mids[1:]) / 2).tolist())
        boundaries.append(seg["end"])
        for j, win_idx in enumerate(inside):
            start = float(boundaries[j])
            end = float(boundaries[j + 1])
            if end - start >= 0.20:
                diarized.append({"start": start, "end": end, "speaker": f"SPEAKER_{int(labels[win_idx]) + 1}"})
    return merge_adjacent(diarized, gap=0.25, max_duration=None)


def merge_adjacent(segments: list[dict], gap: float, max_duration: float | None) -> list[dict]:
    merged = []
    for seg in segments:
        if (
            merged
            and seg["speaker"] == merged[-1]["speaker"]
            and seg["start"] - merged[-1]["end"] <= gap
            and (max_duration is None or seg["end"] - merged[-1]["start"] <= max_duration)
        ):
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(dict(seg))
    return merged


def export_chunks(audio: np.ndarray, sr: int, diarized: list[dict], out_dir: Path, args: argparse.Namespace) -> list[dict]:
    chunks_dir = out_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunks = merge_adjacent(
        diarized,
        gap=args.merge_same_speaker_gap_sec,
        max_duration=args.max_chunk_sec,
    )
    manifest = []
    for idx, seg in enumerate(chunks):
        start = max(0.0, seg["start"] - 0.08)
        end = min(len(audio) / sr, seg["end"] + 0.08)
        s = int(round(start * sr))
        e = int(round(end * sr))
        path = chunks_dir / f"{idx:04d}_{seg['speaker']}_{start:.2f}_{end:.2f}.wav"
        sf.write(path, audio[s:e], sr, subtype="PCM_16")
        manifest.append(
            {
                "id": idx,
                "speaker": seg["speaker"],
                "start": round(start, 3),
                "end": round(end, 3),
                "audio": str(path),
            }
        )
    return manifest


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audio, sr = load_mono_resampled(Path(args.audio), args.target_sr)
    normalized_path = out_dir / "audio_16k_mono.wav"
    sf.write(normalized_path, audio, sr, subtype="PCM_16")

    speech = vad_segments(audio, sr, args)
    windows = embedding_windows(speech, len(audio) / sr, args.embed_window_sec, args.embed_hop_sec)
    if len(windows) < args.num_speakers:
        raise RuntimeError(f"Not enough speech windows for clustering: {len(windows)}")
    embeddings = extract_embeddings(audio, sr, windows, Path(args.embedding_model), args.device, args.batch_size)
    np.savez_compressed(
        out_dir / "speaker_embeddings.npz",
        embeddings=embeddings,
        starts=np.array([w["start"] for w in windows], dtype=np.float32),
        ends=np.array([w["end"] for w in windows], dtype=np.float32),
        mids=np.array([w["mid"] for w in windows], dtype=np.float32),
    )
    labels = cluster_embeddings(embeddings, args.num_speakers, args.cluster_method)
    diarized = assign_segments(speech, windows, labels)
    chunks = export_chunks(audio, sr, diarized, out_dir, args)

    write_json(out_dir / "speech_segments.json", speech)
    write_json(out_dir / "embedding_windows.json", [{**w, "speaker": f"SPEAKER_{int(l) + 1}"} for w, l in zip(windows, labels)])
    write_json(out_dir / "diarization_segments.json", diarized)
    write_json(out_dir / "asr_manifest.json", chunks)
    print(f"normalized_audio={normalized_path}")
    print(f"speech_segments={len(speech)}")
    print(f"embedding_windows={len(windows)}")
    print(f"diarization_segments={len(diarized)}")
    print(f"asr_chunks={len(chunks)}")


if __name__ == "__main__":
    main()
