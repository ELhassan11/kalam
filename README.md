# Kalam — كلام

**A small Arabic-first language model, built from scratch with PyTorch.**

Kalam is a personal research project: build every stage of a language
model by hand — data, cleaning, Arabic normalization, tokenizer, model,
training, generation — instead of calling a ready-made pipeline, and
measure each stage honestly.

> Status: **Phase 1 (data + tokenizer)**. The model itself comes later;
> see [ROADMAP.md](ROADMAP.md).

## Why

Arabic is under-served by from-scratch learning projects, and its script
raises real design questions that English-first tutorials skip: optional
diacritics, multiple alef forms, alef maksura vs ya, ta marbuta, and
Eastern Arabic numerals. Kalam treats those as first-class engineering
decisions, documented in the code.

An original focus of this project is **evaluation**: a held-out set of
Egyptian-dialect prompts, and tokenizer-efficiency (fertility)
comparisons, so progress is measured — not vibes.

## What works today

| Piece | File | Notes |
|---|---|---|
| Arabic normalizer | `kalam/normalize.py` | Diacritics/tatweel removal, alef & ya unification, ta marbuta kept on purpose |
| Wikipedia cleaner | `kalam/clean.py` | Templates, reference sections, citation markers, stub paragraphs |
| BPE tokenizer | `kalam/bpe.py` | Byte-level BPE implemented from scratch — zero dependencies |
| Dataset explorer | `scripts/explore_dataset.py` | Size, language mix, length stats (local file or streamed HF sample) |
| Tokenizer training | `scripts/train_tokenizer.py` | Clean → normalize → train → fertility report → saved JSON tokenizer |

## Quick start

Each command on its own line:

```
python scripts/explore_dataset.py --file data/sample_ar.txt
```

```
python scripts/train_tokenizer.py
```

```
python -m unittest discover -s tests -v
```

The core package uses **only the Python standard library**. Exploring the
full Hugging Face dataset additionally needs `datasets`
(see `requirements.txt`).

## Training data

Experiments start from the publicly available
`bakrianoo/jabarti-llm-dataset` (bilingual Wikipedia + a curated
Egyptian-history collection, with a QA fine-tune subset), explored in
Phase 1 before any cleaning decisions were made.

## Design principles

1. **From scratch where it teaches something** — the tokenizer and
   (later) the model are hand-written, not wrappers.
2. **The same normalization at train and inference time** — the
   normalizer ships inside the tokenizer pipeline.
3. **Every stage is tested and measured** — unit tests for text rules,
   fertility numbers for the tokenizer, held-out articles for evaluation
   (no train/eval leakage).

## License

MIT — see [LICENSE](LICENSE).
