"""Arabic text normalization for Kalam.

The same normalizer runs before training and at inference time, so the
model never sees a written form it was not trained on. Design decisions:

- Diacritics (harakat) and tatweel are removed: they are optional in
  everyday Arabic writing and would explode the vocabulary.
- All alef forms are unified to a bare alef, and alef maksura is unified
  to ya: the variants carry no difference in meaning.
- Ta marbuta is KEPT as-is. It carries grammatical information (feminine
  endings), and merging it with ha would lose that signal.
- Digits are kept as written by default; converting Eastern Arabic
  numerals (٠١٢٣) to Western ones is available as an explicit option.
"""

from __future__ import annotations

import re

# Fathatan, dammatan, kasratan, fatha, damma, kasra, shadda, sukun,
# plus the superscript (dagger) alef.
_DIACRITICS = re.compile(r"[ً-ْٰ]")

_TATWEEL = "ـ"

_ALEF_FORMS = {
    "أ": "ا",  # alef with hamza above
    "إ": "ا",  # alef with hamza below
    "آ": "ا",  # alef with madda
    "ٱ": "ا",  # alef wasla
}

_ALEF_MAKSURA = "ى"
_YA = "ي"

_EASTERN_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def normalize_arabic(text: str, *, convert_eastern_digits: bool = False) -> str:
    """Return a normalized copy of ``text`` (Arabic rules, Latin untouched)."""
    text = _DIACRITICS.sub("", text)
    text = text.replace(_TATWEEL, "")
    for form, bare in _ALEF_FORMS.items():
        text = text.replace(form, bare)
    text = text.replace(_ALEF_MAKSURA, _YA)
    if convert_eastern_digits:
        text = text.translate(_EASTERN_DIGITS)
    return text
