"""Explore a text corpus: size, language mix, and length statistics.

Usage (each command on its own line):

    python scripts/explore_dataset.py --file data/sample_ar.txt
    python scripts/explore_dataset.py --hf --limit 2000

The --file mode needs nothing beyond the standard library. The --hf mode
streams a sample of a Hugging Face dataset (requires ``datasets``) instead
of downloading it in full.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ARABIC_RANGES = (("؀", "ۿ"), ("ݐ", "ݿ"), ("ࢠ", "ࣿ"))


def char_class(ch: str) -> str:
    if any(lo <= ch <= hi for lo, hi in ARABIC_RANGES):
        return "arabic"
    if "a" <= ch.lower() <= "z":
        return "latin"
    if ch.isdigit():
        return "digit"
    if ch.isspace():
        return "space"
    return "other"


def summarize(texts: list[str], label: str) -> None:
    total_chars = sum(len(t) for t in texts)
    counts = {"arabic": 0, "latin": 0, "digit": 0, "space": 0, "other": 0}
    for text in texts:
        for ch in text:
            counts[char_class(ch)] += 1
    word_counts = [len(t.split()) for t in texts]

    print(f"== {label} ==")
    print(f"records:        {len(texts)}")
    print(f"total chars:    {total_chars}")
    if total_chars:
        for name in ("arabic", "latin", "digit", "other"):
            share = 100 * counts[name] / total_chars
            print(f"{name + ' share:':16}{share:5.1f}%")
    if word_counts:
        print(f"words/record:   mean {statistics.mean(word_counts):.1f} | "
              f"median {statistics.median(word_counts):.0f} | "
              f"min {min(word_counts)} | max {max(word_counts)}")
        short = sum(1 for w in word_counts if w < 10)
        print(f"short records (<10 words): {short} "
              f"({100 * short / len(word_counts):.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", help="Path to a plain-text corpus file")
    parser.add_argument("--hf", action="store_true",
                        help="Stream a sample of the Jabarti dataset from HF")
    parser.add_argument("--limit", type=int, default=1000,
                        help="Max records to inspect in --hf mode")
    args = parser.parse_args()

    if args.hf:
        try:
            from datasets import load_dataset
        except ImportError:
            raise SystemExit("The 'datasets' package is required for --hf mode: "
                             "pip install datasets")
        ds = load_dataset("bakrianoo/jabarti-llm-dataset",
                          split="train", streaming=True)
        texts = []
        for i, row in enumerate(ds):
            if i >= args.limit:
                break
            texts.append(str(next(iter(row.values()))))
        summarize(texts, f"jabarti-llm-dataset (first {len(texts)} streamed)")
    elif args.file:
        raw = Path(args.file).read_text(encoding="utf-8")
        paragraphs = [p for p in raw.split("\n\n") if p.strip()]
        summarize(paragraphs, args.file)
    else:
        parser.error("pass --file or --hf")


if __name__ == "__main__":
    main()
