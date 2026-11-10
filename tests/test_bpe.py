import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kalam.bpe import BPETokenizer

CORPUS = [
    "اللغة العربية لغة غنية بالمفردات والتراكيب",
    "يتعلم النموذج اللغة من النصوص خطوة بعد خطوة",
    "اللغة العربية تكتب من اليمين الي اليسار",
    "language models learn language from text step by step",
] * 5


class TestBPETokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = BPETokenizer()
        self.tokenizer.train(CORPUS, vocab_size=400)

    def test_vocab_grows(self):
        self.assertGreater(len(self.tokenizer), 256)
        self.assertLessEqual(len(self.tokenizer), 400)

    def test_roundtrip_arabic_and_english(self):
        for text in ("مرحبا بالعالم، هذا اختبار", "hello world, this is a test"):
            self.assertEqual(self.tokenizer.decode(self.tokenizer.encode(text)), text)

    def test_training_reduces_fertility(self):
        text = " ".join(CORPUS)
        untrained = BPETokenizer()
        self.assertLess(self.tokenizer.fertility(text), untrained.fertility(text))

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tokenizer.json"
            self.tokenizer.save(path)
            loaded = BPETokenizer.load(path)
        text = "اختبار الحفظ والاسترجاع"
        self.assertEqual(loaded.encode(text), self.tokenizer.encode(text))

    def test_rejects_tiny_vocab(self):
        with self.assertRaises(ValueError):
            BPETokenizer().train(CORPUS, vocab_size=100)


if __name__ == "__main__":
    unittest.main()
