import unittest

from bench import _evaluate_general
from src.datasets import BenchmarkItem


def _pe_item(answer: str) -> BenchmarkItem:
    return BenchmarkItem(
        id="project_euler_test",
        dataset="project_euler",
        context="",
        question="test",
        answer=answer,
    )


class ProjectEulerScoringTests(unittest.TestCase):
    def test_scientific_notation_is_scored_correctly(self) -> None:
        item = _pe_item("6.3202e25093")
        extracted, correct = _evaluate_general(item, "6.3202e25093")
        self.assertEqual(extracted, "6.3202e25093")
        self.assertTrue(correct)

    def test_text_answer_is_scored_with_text_normalization(self) -> None:
        item = _pe_item("aprilfoolsjoke")
        _, correct_exact = _evaluate_general(item, "aprilfoolsjoke")
        _, correct_spaced = _evaluate_general(item, "APRIL FOOL")
        self.assertTrue(correct_exact)
        self.assertFalse(correct_spaced)

    def test_blank_expected_answer_is_unscored(self) -> None:
        item = _pe_item("")
        _, correct = _evaluate_general(item, "123")
        self.assertIsNone(correct)


if __name__ == "__main__":
    unittest.main()
