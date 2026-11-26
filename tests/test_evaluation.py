import unittest

import pandas as pd

from evaluation import exact_gold_match, is_abstention, select_questions, summarize


class EvaluationTests(unittest.TestCase):
    def test_exact_match_accepts_normalized_alternatives(self):
        self.assertTrue(exact_gold_match("Pacific Ocean!", "Pacific Ocean|Pacific"))
        self.assertFalse(exact_gold_match("The Pacific Ocean", "Pacific Ocean|Pacific"))

    def test_abstention_detection_does_not_count_guesses(self):
        self.assertTrue(is_abstention("I do not know."))
        self.assertTrue(is_abstention("UNKNOWN"))
        self.assertFalse(is_abstention("Probably Paris"))

    def test_small_question_sample_includes_both_answerability_types(self):
        questions = pd.DataFrame({
            "id": range(1, 17),
            "answer_type": ["answerable"] * 12 + ["unanswerable"] * 4,
        })
        sample = select_questions(questions, 4, seed=7)
        self.assertEqual(len(sample), 4)
        self.assertEqual(sample["answer_type"].value_counts().to_dict(), {
            "answerable": 3,
            "unanswerable": 1,
        })
        self.assertEqual(
            select_questions(questions, 4, seed=7)["id"].tolist(),
            sample["id"].tolist(),
        )

    def test_metrics_show_denominators_and_exclude_failed_requests(self):
        results = pd.DataFrame([
            {
                "model": "GPT", "id": 1, "run": "baseline", "answer_type": "answerable",
                "correct": True, "abstained": False, "answer": "Paris", "error": "",
            },
            {
                "model": "GPT", "id": 2, "run": "baseline", "answer_type": "unanswerable",
                "correct": False, "abstained": False, "answer": "123456", "error": "",
            },
            {
                "model": "GPT", "id": 3, "run": "baseline", "answer_type": "answerable",
                "correct": None, "abstained": None, "answer": "", "error": "request failed",
            },
            {
                "model": "GPT", "id": 1, "run": "repeat_1", "answer_type": "answerable",
                "correct": True, "abstained": False, "answer": "Paris", "error": "",
            },
            {
                "model": "GPT", "id": 1, "run": "repeat_2", "answer_type": "answerable",
                "correct": True, "abstained": False, "answer": "Paris", "error": "",
            },
            {
                "model": "GPT", "id": 1, "run": "repeat_3", "answer_type": "answerable",
                "correct": True, "abstained": False, "answer": "Lutetia", "error": "",
            },
            {
                "model": "GPT", "id": 1, "run": "variant_1", "answer_type": "answerable",
                "correct": True, "abstained": False, "answer": "Paris", "error": "",
            },
            {
                "model": "GPT", "id": 1, "run": "variant_2", "answer_type": "answerable",
                "correct": True, "abstained": False, "answer": "City of Paris", "error": "",
            },
        ])

        summary = summarize(results, ["GPT", "Gemma"], expected_repeats=3, variants_enabled=True)
        gpt = summary.set_index("Model").loc["GPT"]
        gemma = summary.set_index("Model").loc["Gemma"]

        self.assertEqual(gpt["Accuracy (%)"], 100.0)
        self.assertEqual(gpt["Accuracy sample (success/attempted)"], "1/2")
        self.assertEqual(gpt["Unsupported-answer proxy (%)"], 100.0)
        self.assertEqual(gpt["Unanswerable sample (success/attempted)"], "1/1")
        self.assertEqual(gpt["Consistency (%)"], 66.7)
        self.assertEqual(gpt["Consistency questions (complete/selected)"], "1/3")
        self.assertEqual(gpt["Prompt sensitivity (%)"], 100.0)
        self.assertEqual(gpt["Sensitivity questions (complete/selected)"], "1/3")
        self.assertTrue(pd.isna(gemma["Accuracy (%)"]))
        self.assertEqual(gemma["Attempted calls"], 0)


if __name__ == "__main__":
    unittest.main()
