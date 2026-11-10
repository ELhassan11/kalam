import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kalam.normalize import normalize_arabic


class TestNormalizeArabic(unittest.TestCase):
    def test_removes_diacritics(self):
        self.assertEqual(normalize_arabic("أَهْلًا وَسَهْلًا"), "اهلا وسهلا")

    def test_removes_tatweel(self):
        self.assertEqual(normalize_arabic("كتــــاب"), "كتاب")

    def test_unifies_alef_forms(self):
        self.assertEqual(normalize_arabic("أ إ آ ٱ"), "ا ا ا ا")

    def test_unifies_alef_maksura_to_ya(self):
        self.assertEqual(normalize_arabic("على موسى"), "علي موسي")

    def test_keeps_ta_marbuta(self):
        # Ta marbuta carries grammar; it must NOT be merged away.
        self.assertEqual(normalize_arabic("مدرسة جميلة"), "مدرسة جميلة")

    def test_digits_kept_by_default(self):
        self.assertEqual(normalize_arabic("٣ كتب"), "٣ كتب")

    def test_digits_converted_when_asked(self):
        self.assertEqual(
            normalize_arabic("٣ كتب", convert_eastern_digits=True), "3 كتب"
        )

    def test_latin_text_untouched(self):
        self.assertEqual(normalize_arabic("Hello, world!"), "Hello, world!")


if __name__ == "__main__":
    unittest.main()
