import argparse
import json
import os
from pathlib import Path

import torch
from qwen_asr import Qwen3ASRModel


DEFAULT_MODEL_DIR = os.environ.get(
    "MEETING_ASR_QWEN_MODEL_DIR",
    "models/Qwen3-ASR-1.7B",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe a speaker diarization manifest with Qwen3-ASR.")
    parser.add_argument("--manifest", required=True, help="JSON manifest produced by the meeting diarization script.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


def fmt_time(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def write_outputs(records: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "final_transcript.jsonl"
    md_path = out_dir / "final_transcript.md"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    lines = ["# Meeting Transcript", ""]
    current_speaker = None
    for record in records:
        speaker = record["speaker"]
        if speaker != current_speaker:
            lines.extend([f"## {speaker}", ""])
            current_speaker = speaker
        lines.append(f"**[{fmt_time(record['start'])} - {fmt_time(record['end'])}]** {record['text']}")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"jsonl={jsonl_path}")
    print(f"markdown={md_path}")


def main() -> None:
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    model = Qwen3ASRModel.from_pretrained(
        args.model_dir,
        dtype=torch.bfloat16,
        device_map=args.device,
        max_inference_batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    records = []
    for i in range(0, len(manifest), args.batch_size):
        batch = manifest[i : i + args.batch_size]
        results = model.transcribe(
            audio=[item["audio"] for item in batch],
            language=[args.language for _ in batch],
        )
        for item, result in zip(batch, results):
            text = result.text.strip()
            if not text:
                continue
            records.append(
                {
                    "id": item["id"],
                    "speaker": item["speaker"],
                    "start": item["start"],
                    "end": item["end"],
                    "language": result.language,
                    "text": text,
                    "audio": item["audio"],
                }
            )
            print(f"{item['id']:04d} {item['speaker']} {fmt_time(item['start'])}-{fmt_time(item['end'])}: {text}")
        write_outputs(records, Path(args.output_dir))
    write_outputs(records, Path(args.output_dir))


if __name__ == "__main__":
    main()
