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
    parser = argparse.ArgumentParser(description="Transcribe audio with local Qwen3-ASR.")
    parser.add_argument("audio", help="Audio file path, URL, or another qwen-asr supported input.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Local model directory.")
    parser.add_argument("--device", default="cuda:0", help="Torch device, e.g. cuda:0 or cpu.")
    parser.add_argument("--language", default=None, help='Optional language hint, e.g. "Chinese" or "English".')
    parser.add_argument("--context", default="", help="Optional text context to help recognition.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Increase this for longer audio.")
    parser.add_argument("--batch-size", type=int, default=1, help="Max inference batch size.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference. This is slow.")
    parser.add_argument("--output-json", default=None, help="Optional path to write JSON result.")
    parser.add_argument("--output-md", default=None, help="Optional path to write Markdown result.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).expanduser()
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    use_cuda = torch.cuda.is_available() and not args.cpu
    device = args.device if use_cuda else "cpu"
    dtype = torch.bfloat16 if use_cuda else torch.float32

    model = Qwen3ASRModel.from_pretrained(
        str(model_dir),
        dtype=dtype,
        device_map=device,
        max_inference_batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    results = model.transcribe(
        audio=args.audio,
        language=args.language,
        context=args.context,
    )

    records = []
    for idx, result in enumerate(results):
        records.append(
            {
                "index": idx,
                "audio": args.audio,
                "language": result.language,
                "text": result.text.strip(),
            }
        )
        if len(results) > 1:
            print(f"=== Result {idx} ===")
        print(f"language: {result.language}")
        print(result.text)

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"json={output_json}")

    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# Full-Audio Meeting Transcript", ""]
        for record in records:
            lines.append(f"language: `{record['language']}`")
            lines.append("")
            lines.append(record["text"])
            lines.append("")
        output_md.write_text("\n".join(lines), encoding="utf-8")
        print(f"markdown={output_md}")


if __name__ == "__main__":
    main()
