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
        extracted, correct, eval_error = _evaluate_general(item, "6.3202e25093")
        self.assertEqual(extracted, "6.3202e25093")
        self.assertTrue(correct)
        self.assertIsNone(eval_error)

    def test_text_answer_is_scored_with_text_normalization(self) -> None:
        item = _pe_item("aprilfoolsjoke")
        _, correct_exact, eval_error_exact = _evaluate_general(item, "aprilfoolsjoke")
        _, correct_spaced, eval_error_spaced = _evaluate_general(item, "APRIL FOOL")
        self.assertTrue(correct_exact)
        self.assertFalse(correct_spaced)
        self.assertIsNone(eval_error_exact)
        self.assertIsNone(eval_error_spaced)

    def test_blank_expected_answer_is_unscored(self) -> None:
        item = _pe_item("")
        _, correct, eval_error = _evaluate_general(item, "123")
        self.assertIsNone(correct)
        self.assertIsNone(eval_error)


if __name__ == "__main__":
    unittest.main()
