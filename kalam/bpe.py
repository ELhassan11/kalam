"""A byte-level BPE tokenizer, implemented from scratch.

Byte-level BPE starts from the 256 possible byte values, so any text —
Arabic, English, digits, punctuation — can be encoded with no unknown
tokens. Frequent byte pairs are merged step by step into new tokens.

Training learns an ordered list of merges. Encoding replays those merges
in the same order (lowest merge rank first). Decoding concatenates the
byte chunks and decodes them as UTF-8.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def _merge_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    """Byte-level BPE tokenizer (trainable, saveable, no dependencies)."""

    def __init__(self) -> None:
        # merges maps a pair of ids -> the id it was merged into.
        # Insertion order is the merge rank (earliest = most frequent).
        self.merges: dict[tuple[int, int], int] = {}
        # vocab maps an id -> its raw bytes.
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, texts: list[str], vocab_size: int) -> None:
        """Learn merges from ``texts`` until the vocab reaches ``vocab_size``."""
        if vocab_size <= 256:
            raise ValueError("vocab_size must be greater than the 256 base bytes")
        sequences = [list(text.encode("utf-8")) for text in texts if text]
        if not sequences:
            raise ValueError("cannot train on an empty corpus")

        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

        for new_id in range(256, vocab_size):
            counts: Counter[tuple[int, int]] = Counter()
            for seq in sequences:
                counts.update(zip(seq, seq[1:]))
            if not counts:
                break
            pair = counts.most_common(1)[0][0]
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            sequences = [_merge_pair(seq, pair, new_id) for seq in sequences]

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            # Find the adjacent pair with the lowest merge rank.
            best_pair: tuple[int, int] | None = None
            best_rank: int | None = None
            for pair in zip(ids, ids[1:]):
                rank = self.merges.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = pair
            if best_pair is None:
                break
            ids = _merge_pair(ids, best_pair, self.merges[best_pair])
        return ids

    def decode(self, ids: list[int]) -> str:
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def fertility(self, text: str) -> float:
        """Average tokens per word — lower means a more efficient tokenizer."""
        words = text.split()
        if not words:
            return 0.0
        return len(self.encode(text)) / len(words)

    # ------------------------------------------------------------------
    # Persistence (plain JSON, human-inspectable)
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        data = {
            "merges": [[a, b, new_id] for (a, b), new_id in self.merges.items()],
        }
        Path(path).write_text(
            json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer = cls()
        for a, b, new_id in data["merges"]:
            tokenizer.merges[(a, b)] = new_id
            tokenizer.vocab[new_id] = tokenizer.vocab[a] + tokenizer.vocab[b]
        return tokenizer

    def __len__(self) -> int:
        return len(self.vocab)
