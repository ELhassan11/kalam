"""Kalam — a small Arabic-first language model, built from scratch."""

from .bpe import BPETokenizer
from .clean import clean_wikipedia_text
from .normalize import normalize_arabic

__all__ = ["BPETokenizer", "clean_wikipedia_text", "normalize_arabic"]
__version__ = "0.1.0"
