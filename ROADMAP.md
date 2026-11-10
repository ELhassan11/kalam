# Kalam Roadmap

## Phase 1 — Data & Tokenizer ✅ (in progress)
- [x] Dataset exploration: size, Arabic/English mix, length stats, quality issues
- [x] Wikipedia cleaning: templates, reference sections, citations, stub paragraphs
- [x] Arabic normalizer (same rules at train and inference time)
- [x] Byte-level BPE tokenizer from scratch, with fertility reporting
- [ ] Train the production tokenizer (16k vocab) on the full cleaned corpus
- [ ] Fertility benchmark vs a general-purpose tokenizer on Arabic text

## Phase 2 — Model
- [ ] Token + position embeddings
- [ ] Causal multi-head self-attention (with KV cache)
- [ ] Transformer blocks, final GPT-style model class
- [ ] Shape/unit tests at every stage

## Phase 3 — Pretraining
- [ ] Training loop: AdamW, LR schedule, gradient clipping, checkpoints
- [ ] Small run on a single consumer GPU; log train/val loss
- [ ] Text generation: greedy, temperature, top-k, top-p

## Phase 4 — Fine-tuning & Evaluation
- [ ] SFT on the Arabic/English QA subset
- [ ] Held-out evaluation (whole articles held out — no leakage)
- [ ] Egyptian-dialect prompt set: base vs fine-tuned comparison

## Phase 5 — Interface
- [ ] Simple chat demo (CLI first, then a small web UI)
- [ ] Model card: data, limits, honest sample outputs
