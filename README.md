# LLM Hallucination Comparison Lab

A Streamlit research application for comparing real responses from GPT, LLaMA, Gemma and DeepSeek through OpenRouter. It runs the same controlled questions and generation settings against each configured model, calculates transparent automatic metrics, and supports manual review of baseline answers. Results are not hardcoded.

## What is included

- Four editable OpenRouter model IDs.
- A 16-item seed dataset (`questions.csv`): 12 answerable QA questions and 4 deliberately unanswerable private-information probes.
- Reproducible, stratified sampling. Samples of at least two questions include both answerability classes; the full dataset is selected when the count is 16.
- Optional runs of two paraphrases and 2–5 repeated baseline calls.
- Configurable temperature and maximum output-token limit.
- Per-question baseline responses, API error details, score denominators, run metadata, CSV downloads, and a manual hallucination review table.
- A small unit test suite for sampling, answer matching, abstention detection, and metrics.

## Requirements

- Python 3.10 or later.
- An OpenRouter account and API key, with access/balance for the selected models.
- Internet access for model calls.

## Install and run

From the project directory in Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Or skip activation and run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\streamlit.exe run app.py
```

For macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local Streamlit URL displayed in the terminal. The app reads `OPENROUTER_API_KEY` only from the project's `.env` file. Create or update `.env` in the project directory with:

```dotenv
OPENROUTER_API_KEY=your-real-key
```

The `.env` file is ignored by Git. Never commit it, paste a real key into source code, or share it in screenshots/chat. If a key has been exposed, revoke it in OpenRouter and create a replacement.

## Running a comparison

The API key is loaded from `.env` and is not entered in the UI. Configure these controls in the sidebar:

1. **Model IDs** — all four models run by default; edit IDs to compare different OpenRouter models.
2. **Questions in this run** — choose 1–16. Stratification balances answerable/unanswerable questions where the sample size permits; a sample of one cannot cover both.
3. **Sampling seed** — same dataset, count and seed select the same sample.
4. **Paraphrases** — optionally run `variant_1` and `variant_2` for every selected question.
5. **Repeat calls** — optionally run the baseline prompt 2–5 additional times to estimate repeat consistency.
6. **Temperature** and **maximum output tokens** — applied equally across models where their APIs support these settings.

The app estimates the number of API requests before the run:

```text
questions × configured models × (baseline + enabled paraphrases + enabled repeats)
```

For example, 4 questions × 4 models × (1 baseline + 2 paraphrases + 3 repeats) = 96 requests. Pricing and rate limits vary. The app does not estimate dollars because model prices can change.

Each completed run stores its model IDs, selected question IDs, seed, settings, system prompt, timestamp and request count. Changing sidebar settings does not change completed results; click **Run model comparison** again to start a new run. Download the run metadata JSON alongside the raw-response and metrics CSVs to retain the experiment configuration.

## Metrics and interpretation

- **Accuracy:** percentage of successful baseline responses to answerable questions that exactly match a normalized accepted answer. Alternatives in `gold_answer` are separated with `|`. This is not semantic grading; a correct longer answer can be marked wrong.
- **Unsupported-answer proxy:** percentage of successful baseline responses to unanswerable questions that did not match the app's abstention patterns. This is a narrow proxy, not a general hallucination rate.
- **Consistency:** for each question with every requested repeat successful, the share of repeat responses matching the modal normalized answer; the score averages those shares. Consistent answers can still be wrong.
- **Prompt sensitivity:** percentage of questions where either paraphrase differs from the baseline after normalization. A benign wording difference also counts as sensitivity. Only complete baseline + both paraphrase sets are scored.
- **Human-reviewed hallucination rate:** manually marked “Yes” responses divided by all baseline responses marked “Yes” or “No”. Fact-check claims against reliable sources; do not treat every factual error as a hallucination.
- **Call success:** successful API calls divided by attempted calls.

Score denominators and completion counts are displayed alongside each metric. API failures are excluded from score denominators and visible in the API errors section. A metric without eligible responses is “Not measured,” not zero. A 100% accuracy result can be genuine if the model answered all sampled questions correctly; the previous small, easy dataset is not challenging enough to establish broad model reliability. Review response-level outputs and sample counts before interpreting any score.

## Dataset and methodology limitations

`questions.csv` is a small illustrative seed dataset, not a validated benchmark. It has only 12 short-form answerable questions, so a model can score 100% on them. The four unanswerable items are synthetic probes; the proxy only checks whether answers abstain. Gold-answer matching uses light text normalization and will not detect semantic equivalence. API model aliases may change over time, and provider support for generation settings can differ.

For research or a final submission, expand the dataset with a documented, independently verified set across categories and difficulty levels; record sources and model/version IDs; use equal settings; fact-check claims with a human review protocol (ideally independent raters); report errors and uncertainty; and keep the unsupported-answer proxy separate from the human-reviewed hallucination rate. Do not call the automatic proxy a general hallucination rate.

## Tests

Run the unit tests from the project directory:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Check Python syntax:

```powershell
.\.venv\Scripts\python.exe -m compileall -q app.py evaluation.py tests
```

Unit tests do not make paid model API calls. To test the live integration manually, put a valid key in `.env`, start Streamlit, configure model IDs, then run one or two questions.
