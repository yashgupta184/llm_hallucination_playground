from collections import Counter

import pandas as pd


def normalize(value):
    """Apply light normalization; semantic equivalence still needs human review."""
    text = str(value or "").lower().strip()
    text = "".join(character for character in text if character.isalnum() or character.isspace())
    return " ".join(text.split())


def is_abstention(value):
    normalized = normalize(value)
    if normalized in {
        "unknown",
        "i dont know",
        "i do not know",
        "cannot determine",
        "i cannot determine",
        "not enough information",
        "insufficient information",
    }:
        return True
    words = normalized.split()
    return len(words) <= 12 and (
        normalized.startswith("i cannot determine")
        or normalized.startswith("i do not know")
    )


def exact_gold_match(answer, gold):
    """Match a short answer against pipe-separated accepted alternatives."""
    accepted = {normalize(candidate) for candidate in str(gold).split("|")}
    return normalize(answer) in accepted


def select_questions(questions, count, seed=42):
    """Choose a reproducible sample that includes both answerability classes."""
    if count < 1 or count > len(questions):
        raise ValueError("Question count must be between 1 and the dataset size.")
    if count == len(questions):
        return questions.reset_index(drop=True).copy()

    working = questions.reset_index(drop=True).copy()
    working["_sample_order"] = range(len(working))
    groups = list(working.groupby("answer_type", sort=True))
    allocations = [count * len(group) // len(working) for _, group in groups]

    if count >= len(groups):
        for index, allocation in enumerate(allocations):
            if allocation == 0:
                allocations[index] = 1

    while sum(allocations) > count:
        candidates = [i for i, allocation in enumerate(allocations) if allocation > 1]
        if not candidates:
            break
        largest = max(candidates, key=lambda i: allocations[i])
        allocations[largest] -= 1
    while sum(allocations) < count:
        largest_gap = max(
            range(len(groups)),
            key=lambda i: count * len(groups[i][1]) / len(working) - allocations[i],
        )
        allocations[largest_gap] += 1

    selected = []
    for index, ((_, group), allocation) in enumerate(zip(groups, allocations)):
        if allocation:
            selected.append(group.sample(n=allocation, random_state=seed + index))

    return (
        pd.concat(selected)
        .sort_values("_sample_order")
        .drop(columns="_sample_order")
        .reset_index(drop=True)
    )


def summarize(results, model_names, expected_repeats=3, variants_enabled=True):
    """Calculate metrics and their valid sample sizes for each configured model."""
    records = []
    for model in model_names:
        rows = results[results["model"] == model]
        successful = rows[rows["error"] == ""]
        baseline = rows[rows["run"] == "baseline"]
        valid_baseline = successful[successful["run"] == "baseline"]

        answerable_attempted = baseline[baseline["answer_type"] == "answerable"]
        answerable_valid = valid_baseline[valid_baseline["answer_type"] == "answerable"]
        unanswerable_attempted = baseline[baseline["answer_type"] == "unanswerable"]
        unanswerable_valid = valid_baseline[valid_baseline["answer_type"] == "unanswerable"]

        accuracy = (
            100 * answerable_valid["correct"].astype(bool).mean()
            if len(answerable_valid)
            else float("nan")
        )
        unsupported_proxy = (
            100 * (~unanswerable_valid["abstained"].astype(bool)).mean()
            if len(unanswerable_valid)
            else float("nan")
        )

        repeated_questions = []
        if expected_repeats:
            repeat_rows = successful[successful["run"].str.startswith("repeat_")]
            for question_id, group in repeat_rows.groupby("id"):
                answers = {
                    row["run"]: normalize(row["answer"])
                    for _, row in group.iterrows()
                }
                expected = {f"repeat_{index}" for index in range(1, expected_repeats + 1)}
                if expected.issubset(answers):
                    repeated_questions.append(
                        Counter(answers.values()).most_common(1)[0][1] / expected_repeats
                    )

        sensitivity_questions = []
        if variants_enabled:
            sensitivity_runs = successful[
                successful["run"].isin(["baseline", "variant_1", "variant_2"])
            ]
            for _, group in sensitivity_runs.groupby("id"):
                answers = {
                    row["run"]: normalize(row["answer"])
                    for _, row in group.iterrows()
                }
                required = {"baseline", "variant_1", "variant_2"}
                if required.issubset(answers):
                    sensitivity_questions.append(
                        any(answers[name] != answers["baseline"] for name in ("variant_1", "variant_2"))
                    )

        records.append({
            "Model": model,
            "Accuracy (%)": round(accuracy, 1) if pd.notna(accuracy) else float("nan"),
            "Accuracy sample (success/attempted)": f"{len(answerable_valid)}/{len(answerable_attempted)}",
            "Unsupported-answer proxy (%)": (
                round(unsupported_proxy, 1) if pd.notna(unsupported_proxy) else float("nan")
            ),
            "Unanswerable sample (success/attempted)": (
                f"{len(unanswerable_valid)}/{len(unanswerable_attempted)}"
            ),
            "Consistency (%)": (
                round(100 * sum(repeated_questions) / len(repeated_questions), 1)
                if repeated_questions
                else float("nan")
            ),
            "Consistency questions (complete/selected)": (
                f"{len(repeated_questions)}/{len(baseline)}"
                if expected_repeats
                else "Disabled"
            ),
            "Prompt sensitivity (%)": (
                round(100 * sum(sensitivity_questions) / len(sensitivity_questions), 1)
                if sensitivity_questions
                else float("nan")
            ),
            "Sensitivity questions (complete/selected)": (
                f"{len(sensitivity_questions)}/{len(baseline)}"
                if variants_enabled
                else "Disabled"
            ),
            "Successful calls": len(successful),
            "Attempted calls": len(rows),
            "Call success (%)": round(100 * len(successful) / len(rows), 1) if len(rows) else float("nan"),
        })

    return pd.DataFrame.from_records(records)
