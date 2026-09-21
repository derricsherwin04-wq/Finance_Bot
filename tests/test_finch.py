import tempfile
from pathlib import Path
import unittest

from finch.rag import ask
from finch.storage import Library, chunk_text


class FinchLibraryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "corporate_finance.md"
        self.source.write_text(
            "# Net Present Value\n\n"
            "Net present value (NPV) is the present value of future cash flows less the initial investment. "
            "Accept a project when NPV is positive at the required return.\n\n"
            "# Weighted Average Cost of Capital\n\n"
            "WACC weights the after-tax cost of debt and cost of equity by market value.",
            encoding="utf-8",
        )
        self.library = Library(self.root / "library.sqlite3")

    def tearDown(self):
        self.library.close()
        self.temporary.cleanup()

    def test_chunk_text_overlaps_long_content(self):
        pieces = chunk_text("word " * 400, size=300, overlap=50)
        self.assertGreater(len(pieces), 1)
        self.assertTrue(all(len(piece) <= 300 for piece in pieces))

    def test_ingest_search_and_feedback_export(self):
        self.assertGreater(self.library.ingest(self.source), 0)
        results = self.library.search("When should I accept an NPV project?")
        self.assertTrue(results)
        self.assertIn("NPV", results[0].content)

        answer = ask(self.library, "What is NPV?", sources_only=True)
        self.assertIn("no model generation", answer.response.answer)
        self.library.add_feedback(answer.response.id, "down", "Mention the required return explicitly.")
        exported = self.library.training_examples("down")
        self.assertEqual(len(exported), 1)
        self.assertIn("required return", exported[0]["messages"][-1]["content"])

    def test_unchanged_file_is_not_reindexed(self):
        self.assertGreater(self.library.ingest(self.source), 0)
        self.assertEqual(self.library.ingest(self.source), 0)
        self.assertEqual(self.library.counts()["documents"], 1)


if __name__ == "__main__":
    unittest.main()
