"""Clean + normalize a corpus, train the Kalam BPE tokenizer, and report.

Usage (each command on its own line):

    python scripts/train_tokenizer.py
    python scripts/train_tokenizer.py --vocab-size 16000 --corpus data/corpus.txt

The trained tokenizer is saved as JSON under artifacts/. The report
compares fertility (tokens per word) before and after training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kalam import BPETokenizer, clean_wikipedia_text, normalize_arabic

DEMO_SENTENCE = "اهلا بكم في مكتبة النيل، حيث يتعلم النموذج اللغة العربية من الصفر"


def load_corpus(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8")
    paragraphs = [p for p in raw.split("\n\n") if p.strip()]
    cleaned = []
    for paragraph in paragraphs:
        text = normalize_arabic(clean_wikipedia_text(paragraph, min_paragraph_chars=0))
        if text:
            cleaned.append(text)
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/sample_ar.txt")
    parser.add_argument("--vocab-size", type=int, default=800)
    parser.add_argument("--out", default="artifacts/tokenizer.json")
    args = parser.parse_args()

    texts = load_corpus(Path(args.corpus))
    total_words = sum(len(t.split()) for t in texts)
    print(f"corpus: {len(texts)} paragraphs, {total_words} words (after cleaning)")

    baseline = BPETokenizer()  # untrained: pure bytes
    print(f"fertility before training: {baseline.fertility(' '.join(texts)):.2f} tokens/word")

    tokenizer = BPETokenizer()
    tokenizer.train(texts, vocab_size=args.vocab_size)
    print(f"trained vocab size: {len(tokenizer)}")
    print(f"fertility after training:  {tokenizer.fertility(' '.join(texts)):.2f} tokens/word")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(out)
    print(f"saved tokenizer to {out}")

    demo = normalize_arabic(DEMO_SENTENCE)
    ids = tokenizer.encode(demo)
    print(f"demo sentence: {demo}")
    print(f"encoded as {len(ids)} tokens: {ids}")
    print(f"roundtrip ok: {tokenizer.decode(ids) == demo}")


if __name__ == "__main__":
    main()
