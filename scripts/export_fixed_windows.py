import argparse
import json
from pathlib import Path

import soundfile as sf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export fixed overlapping audio windows for non-diarized ASR.")
    parser.add_argument("audio", help="Input audio, preferably normalized 16 kHz mono wav.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window-sec", type=float, default=300.0)
    parser.add_argument("--overlap-sec", type=float, default=30.0)
    parser.add_argument("--manifest-name", default="full_window_manifest.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.overlap_sec >= args.window_sec:
        raise ValueError("--overlap-sec must be smaller than --window-sec")

    out_dir = Path(args.output_dir)
    chunks_dir = out_dir / "full_windows"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    audio, sr = sf.read(args.audio, dtype="float32", always_2d=False)
    duration = len(audio) / sr
    step = args.window_sec - args.overlap_sec

    manifest = []
    start = 0.0
    idx = 0
    while start < duration:
        end = min(duration, start + args.window_sec)
        if end - start < 1.0:
            break
        start_i = int(round(start * sr))
        end_i = int(round(end * sr))
        path = chunks_dir / f"{idx:04d}_{start:.2f}_{end:.2f}.wav"
        sf.write(path, audio[start_i:end_i], sr, subtype="PCM_16")
        manifest.append(
            {
                "id": idx,
                "speaker": "FULL_AUDIO",
                "start": round(start, 3),
                "end": round(end, 3),
                "audio": str(path),
            }
        )
        if end >= duration:
            break
        start += step
        idx += 1

    manifest_path = out_dir / args.manifest_name
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest={manifest_path}")
    print(f"windows={len(manifest)}")
    print(f"duration_sec={duration:.3f}")


if __name__ == "__main__":
    main()
