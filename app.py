"""Streamlit app for comparing real model responses through OpenRouter."""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import dotenv_values
from openai import OpenAI

from evaluation import exact_gold_match, is_abstention, select_questions, summarize

APP_DIR = Path(__file__).resolve().parent
api_key = (dotenv_values(APP_DIR / ".env").get("OPENROUTER_API_KEY") or "").strip()

st.set_page_config(page_title="LLM Hallucination Comparison Lab", layout="wide")

MODEL_DEFAULTS = {
    "GPT": "openai/gpt-4o",
    "LLaMA": "meta-llama/llama-3.3-70b-instruct",
    "Gemma": "google/gemma-3-27b-it",
    "DeepSeek": "deepseek/deepseek-chat-v3-0324",
}

SYSTEM_PROMPT = (
    "Answer the question in a single short phrase. Do not explain. "
    "If the answer cannot be determined from available information, "
    "respond with exactly UNKNOWN. Do not invent facts or guess."
)
REQUIRED_COLUMNS = {
    "id",
    "category",
    "question",
    "variant_1",
    "variant_2",
    "gold_answer",
    "answer_type",
    "source_note",
}


def load_questions():
    dataset_path = APP_DIR / "questions.csv"
    questions = pd.read_csv(dataset_path).fillna("")
    missing_columns = REQUIRED_COLUMNS.difference(questions.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing_columns))}")
    if questions.empty or questions["id"].duplicated().any():
        raise ValueError("Dataset must contain questions with unique IDs.")
    if not set(questions["answer_type"]).issubset({"answerable", "unanswerable"}):
        raise ValueError("Each answer_type must be 'answerable' or 'unanswerable'.")
    return questions


def call_model(client, model_id, prompt, temperature, max_tokens):
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content
    if not content or not content.strip():
        raise ValueError("Model returned an empty response.")
    return content.strip()


def safe_error(error, api_key):
    message = str(error).replace(api_key, "[redacted]") if api_key else str(error)
    return message[:1000]


try:
    questions = load_questions()
except (OSError, ValueError, pd.errors.ParserError) as exc:
    st.error(f"Could not load questions.csv: {exc}")
    st.stop()

st.title("LLM Hallucination Comparison Lab")
st.caption("Compare GPT, LLaMA, Gemma and DeepSeek using real OpenRouter responses.")
st.warning(
    "A 100% accuracy result may be real on this small, mostly straightforward dataset. "
    "It is not proof of general reliability. Check the sample counts and per-question responses."
)

with st.sidebar:
    st.header("Experiment configuration")
    st.subheader("Models")
    st.caption("Use OpenRouter model IDs. Each configured model receives the same questions and settings.")
    models = {
        name: st.text_input(f"{name} model ID", value=model_id).strip()
        for name, model_id in MODEL_DEFAULTS.items()
    }

    st.subheader("Evaluation settings")
    question_count = st.slider(
        "Questions in this run",
        min_value=1,
        max_value=len(questions),
        value=min(4, len(questions)),
        help="Smaller samples are stratified to include answerable and unanswerable items when possible.",
    )
    sample_seed = st.number_input(
        "Sampling seed",
        min_value=0,
        max_value=2_147_483_647,
        value=42,
        step=1,
        help="The same seed and question count select the same sample.",
    )
    run_variants = st.checkbox(
        "Test both paraphrases per question",
        value=True,
        help="Adds variant_1 and variant_2 prompts to the baseline.",
    )
    run_consistency = st.checkbox(
        "Measure repeat consistency",
        value=True,
        help="Repeated calls use the baseline prompt and are separate API requests.",
    )
    repeat_count = (
        st.number_input(
            "Repeat calls per question",
            min_value=2,
            max_value=5,
            value=3,
            step=1,
            disabled=not run_consistency,
        )
        if run_consistency
        else 0
    )
    temperature = st.number_input(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.2,
        step=0.1,
        help="Applied equally to all models where supported by the selected provider/model.",
    )
    max_tokens = st.number_input(
        "Maximum output tokens",
        min_value=16,
        max_value=1024,
        value=100,
        step=16,
    )

    request_count = question_count * len(models) * (
        1 + (2 if run_variants else 0) + repeat_count
    )
    st.info(f"Estimated API requests: **{request_count}**. Pricing and rate limits vary by model.")
    st.caption("No dollar estimate is shown because model pricing varies and can change.")

st.subheader("1. Evaluation dataset")
with st.expander(f"Preview all {len(questions)} questions", expanded=False):
    st.dataframe(
        questions[["id", "category", "question", "gold_answer", "answer_type", "source_note"]],
        hide_index=True,
        use_container_width=True,
    )
    counts = questions["answer_type"].value_counts()
    st.caption(
        f"Dataset composition: {counts.get('answerable', 0)} answerable and "
        f"{counts.get('unanswerable', 0)} unanswerable questions."
    )

with st.expander("How the metrics are calculated", expanded=False):
    st.markdown(
        """
        - **Accuracy:** exact normalized match to one of the `|`-separated accepted answers on
          successful *baseline answerable* responses. Semantically correct alternate phrasing may
          be marked incorrect; inspect the response table.
        - **Unsupported-answer proxy:** percentage of successful baseline responses to
          unanswerable questions that did not abstain. This is not a general hallucination rate.
        - **Consistency:** average per-question share of repeat responses matching the modal
          normalized repeat answer. It measures repeat agreement, not truth.
        - **Prompt sensitivity:** percentage of questions where either paraphrase differs from the
          baseline. Only questions with all three successful responses are counted.
        - **Human-reviewed hallucination rate:** manually labeled baseline responses marked “Yes”
          divided by reviewed (“Yes” or “No”) responses. Review claims against reliable sources.

        Failed API requests are excluded from score denominators and shown separately. Every score
        includes its valid/attempted sample counts; unavailable metrics are not reported as zero.
        """
    )

if st.button("Run model comparison", type="primary"):
    missing_models = [name for name, model_id in models.items() if not model_id]
    if not api_key or api_key == "your-key-here":
        st.error("Set a valid OPENROUTER_API_KEY in the project's .env file.")
    elif missing_models:
        st.error(f"Enter a model ID for: {', '.join(missing_models)}.")
    else:
        subset = select_questions(questions, int(question_count), int(sample_seed))
        runs = ["baseline"]
        if run_variants:
            runs.extend(["variant_1", "variant_2"])
        if run_consistency:
            runs.extend([f"repeat_{index}" for index in range(1, int(repeat_count) + 1)])

        run_started = datetime.now().astimezone()
        run_id = run_started.strftime("%Y%m%dT%H%M%S%z")
        total_calls = len(subset) * len(models) * len(runs)
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            timeout=45,
            max_retries=1,
        )
        progress = st.progress(0, text=f"0/{total_calls} API requests")
        results = []
        completed = 0
        for model_name, model_id in models.items():
            for _, question in subset.iterrows():
                for run_name in runs:
                    prompt = (
                        str(question[run_name])
                        if run_name in {"variant_1", "variant_2"}
                        else str(question["question"])
                    )
                    try:
                        answer = call_model(
                            client,
                            model_id,
                            prompt,
                            float(temperature),
                            int(max_tokens),
                        )
                        error = ""
                    except Exception as exc:
                        answer = ""
                        error = safe_error(exc, api_key)

                    results.append({
                        "model": model_name,
                        "model_id": model_id,
                        "id": int(question["id"]),
                        "category": question["category"],
                        "run": run_name,
                        "question": prompt,
                        "gold_answer": question["gold_answer"],
                        "answer_type": question["answer_type"],
                        "answer": answer,
                        "correct": exact_gold_match(answer, question["gold_answer"]) if not error else None,
                        "abstained": is_abstention(answer) if not error else None,
                        "error": error,
                        "temperature": float(temperature),
                        "max_tokens": int(max_tokens),
                    })
                    completed += 1
                    progress.progress(completed / total_calls, text=f"{completed}/{total_calls} API requests")

        st.session_state["experiment_results"] = pd.DataFrame(results)
        st.session_state["experiment_metadata"] = {
            "run_id": run_id,
            "started_at": run_started.isoformat(timespec="seconds"),
            "models": models,
            "question_count": len(subset),
            "question_ids": [int(question_id) for question_id in subset["id"]],
            "sampling_seed": int(sample_seed),
            "paraphrases_enabled": bool(run_variants),
            "repeat_count": int(repeat_count),
            "temperature": float(temperature),
            "max_output_tokens": int(max_tokens),
            "system_prompt": SYSTEM_PROMPT,
            "attempted_api_requests": total_calls,
        }
        st.success("Run finished. Review the counts, metrics, and individual responses below.")

if "experiment_results" not in st.session_state:
    st.caption("Configure the experiment in the sidebar and run it to see results.")
else:
    results = st.session_state["experiment_results"]
    metadata = st.session_state["experiment_metadata"]
    model_names = list(metadata["models"])
    summary = summarize(
        results,
        model_names,
        expected_repeats=metadata["repeat_count"],
        variants_enabled=metadata["paraphrases_enabled"],
    )

    st.subheader("2. Comparison dashboard")
    st.caption(
        f"Results are from run {metadata['run_id']} — {metadata['question_count']} questions, "
        f"{len(model_names)} models, seed {metadata['sampling_seed']}. "
        "Changing sidebar settings does not change this completed run; click Run again to update it."
    )
    metric_columns = st.columns(3)
    metric_columns[0].metric("API requests", f"{int((results['error'] == '').sum())}/{len(results)} successful")
    metric_columns[1].metric("Questions", str(metadata["question_count"]))
    metric_columns[2].metric("Temperature / max tokens", f"{metadata['temperature']} / {metadata['max_output_tokens']}")

    display_summary = summary.copy()
    for metric in (
        "Accuracy (%)",
        "Unsupported-answer proxy (%)",
        "Consistency (%)",
        "Prompt sensitivity (%)",
    ):
        display_summary[metric] = display_summary[metric].map(
            lambda value: f"{value:.1f}" if pd.notna(value) else "Not measured"
        )
    st.dataframe(display_summary, hide_index=True, use_container_width=True)
    st.caption(
        "Accuracy and unsupported-answer sample columns show successful responses / attempted "
        "responses. Consistency and sensitivity count only complete question sets. "
        "“Not measured” means the run had no eligible responses for that metric."
    )

    chart_metrics = [
        "Accuracy (%)",
        "Unsupported-answer proxy (%)",
        "Consistency (%)",
        "Prompt sensitivity (%)",
    ]
    for metric in chart_metrics:
        chart_data = summary[["Model", metric]].dropna().set_index("Model")
        if not chart_data.empty:
            st.markdown(f"**{metric}**")
            st.bar_chart(chart_data)
        else:
            st.caption(f"{metric}: not measured for this run.")

    with st.expander("Run settings and provenance", expanded=False):
        st.json(metadata)
    st.download_button(
        "Download run metadata JSON",
        json.dumps(metadata, indent=2).encode("utf-8"),
        file_name="llm_run_metadata.json",
        mime="application/json",
    )
    st.download_button(
        "Download raw responses CSV",
        results.to_csv(index=False).encode("utf-8"),
        file_name="llm_responses.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download metrics CSV",
        summary.to_csv(index=False).encode("utf-8"),
        file_name="llm_metrics.csv",
        mime="text/csv",
    )

    st.subheader("3. Baseline responses")
    baseline = results[results["run"] == "baseline"].copy()
    st.dataframe(
        baseline[
            ["model", "model_id", "id", "category", "question", "gold_answer", "answer_type",
             "answer", "correct", "abstained", "error"]
        ],
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("4. Manual hallucination review")
    st.caption(
        "Check each baseline response against reliable sources. Mark “Yes” only when it asserts "
        "unsupported or fabricated claims; a wrong answer alone is not automatically a hallucination."
    )
    valid_baseline = baseline[baseline["error"] == ""].copy().reset_index(drop=True)
    if valid_baseline.empty:
        st.info("There are no successful baseline responses to review.")
    else:
        editor_key = f"manual_editor_{metadata['run_id']}"
        initial_labels = valid_baseline[
            ["model", "model_id", "id", "question", "gold_answer", "answer"]
        ].copy()
        initial_labels["Hallucination?"] = "Not reviewed"
        edited = st.data_editor(
            initial_labels,
            hide_index=True,
            disabled=["model", "model_id", "id", "question", "gold_answer", "answer"],
            column_config={
                "Hallucination?": st.column_config.SelectboxColumn(
                    "Hallucination?", options=["Not reviewed", "Yes", "No"], required=True
                )
            },
            use_container_width=True,
            key=editor_key,
        )
        labeled = edited[edited["Hallucination?"].isin(["Yes", "No"])]
        if not labeled.empty:
            audited = labeled.groupby("model")["Hallucination?"].agg(
                Reviewed="count",
                Hallucinated=lambda labels: (labels == "Yes").sum(),
            ).reset_index()
            audited["Reviewed hallucination rate (%)"] = (
                100 * audited["Hallucinated"] / audited["Reviewed"]
            ).round(1)
            st.dataframe(audited, hide_index=True, use_container_width=True)
        st.download_button(
            "Download reviewed responses CSV",
            edited.to_csv(index=False).encode("utf-8"),
            file_name="llm_reviewed.csv",
            mime="text/csv",
        )

    failed = results[results["error"] != ""]
    if not failed.empty:
        with st.expander(f"API errors ({len(failed)})", expanded=True):
            st.dataframe(
                failed[["model", "model_id", "id", "run", "error"]],
                hide_index=True,
                use_container_width=True,
            )
