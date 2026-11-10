import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kalam.clean import (
    clean_wikipedia_text,
    drop_short_paragraphs,
    remove_citations,
    remove_templates,
)


class TestClean(unittest.TestCase):
    def test_removes_numeric_citations(self):
        self.assertEqual(remove_citations("النيل نهر عظيم[1] جدا[٢، ٣]"),
                         "النيل نهر عظيم جدا")

    def test_removes_templates(self):
        self.assertEqual(remove_templates("نص {{قالب:معلومات}} مهم"), "نص  مهم")

    def test_removes_reference_section_but_keeps_next_section(self):
        doc = (
            "هذه فقرة افتتاحية طويلة بما يكفي لتبقى في النص بعد التنظيف الكامل.\n\n"
            "== المراجع ==\n"
            "كتاب أول وكتاب ثان وقائمة مصادر طويلة لا نريدها في بيانات التدريب.\n\n"
            "== الجغرافيا ==\n"
            "تتناول هذه الفقرة جغرافيا المنطقة بالتفصيل وتبقى في النص النهائي."
        )
        cleaned = clean_wikipedia_text(doc, min_paragraph_chars=0)
        self.assertNotIn("كتاب أول", cleaned)
        self.assertIn("جغرافيا المنطقة", cleaned)

    def test_drops_short_paragraphs(self):
        doc = "قصيرة\n\nهذه فقرة طويلة بما يكفي لاجتياز حد الطول المطلوب للبقاء هنا."
        self.assertNotIn("قصيرة", drop_short_paragraphs(doc, min_chars=40))


if __name__ == "__main__":
    unittest.main()
