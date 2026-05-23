# Exercise 3: Tri-Model AI Assistant

This exercise builds a local AI assistant that uses three specialized Hugging Face models orchestrated with LangGraph.

## Objective

Implement this flow:

`Large text -> summarization -> length refinement -> final summary -> question answering on the summary`

The app is fully local and interactive:

- accepts a large text input
- lets the user choose `short`, `medium`, or `long`
- generates a final summary
- starts a question-answer loop
- answers using the final summary as context

## Models

- Summarization model: `google/flan-t5-base`
- Refinement model: `google/flan-t5-base`
- Question answering model: `distilbert-base-cased-distilled-squad`

## Why Three Models

- The summarization model compresses the original text.
- The refinement model rewrites the summary to the selected length.
- The question answering model answers user questions using the final summary.

This keeps the application aligned with the assignment requirement for multi-model orchestration instead of relying on one general-purpose model.

## LangGraph Flow

The graph in [app.py](/C:/Users/Admin/Desktop/AI/exercise3/app.py:1) runs this sequence:

1. `summarize`
2. `refine`
3. `display_summary`
4. `prompt_question`
5. `answer_question`
6. loop back to `prompt_question` until the user types `exit`

## Files

- [app.py](/C:/Users/Admin/Desktop/AI/exercise3/app.py:1)
  Thin CLI entrypoint.

- [streamlit_app.py](/C:/Users/Admin/Desktop/AI/exercise3/streamlit_app.py:1)
  Thin Streamlit entrypoint.

- [tri_model_ai/assistant.py](/C:/Users/Admin/Desktop/AI/exercise3/tri_model_ai/assistant.py:1)
  Thin orchestration layer and LangGraph wiring.

- [tri_model_ai/config.py](/C:/Users/Admin/Desktop/AI/exercise3/tri_model_ai/config.py:1)
  Model names, prompt templates, and summary-length settings.

- [tri_model_ai/models.py](/C:/Users/Admin/Desktop/AI/exercise3/tri_model_ai/models.py:1)
  Lazy model loading for text generation and QA.

- [tri_model_ai/pipeline.py](/C:/Users/Admin/Desktop/AI/exercise3/tri_model_ai/pipeline.py:1)
  Summarization and refinement pipeline logic.

- [tri_model_ai/qa.py](/C:/Users/Admin/Desktop/AI/exercise3/tri_model_ai/qa.py:1)
  QA context selection and answer generation.

- [tri_model_ai/state.py](/C:/Users/Admin/Desktop/AI/exercise3/tri_model_ai/state.py:1)
  Shared LangGraph state definition.

- [tri_model_ai/parsing.py](/C:/Users/Admin/Desktop/AI/exercise3/tri_model_ai/parsing.py:1)
  Text normalization plus TXT, PDF, DOCX, CSV, and LOG parsing helpers.

- [tri_model_ai/text_splitter.py](/C:/Users/Admin/Desktop/AI/exercise3/tri_model_ai/text_splitter.py:1)
  Recursive splitter setup with a local fallback implementation.

- [tri_model_ai/utils.py](/C:/Users/Admin/Desktop/AI/exercise3/tri_model_ai/utils.py:1)
  Shared cleanup, deduplication, and keyword-overlap helpers.

- [tri_model_ai/cli.py](/C:/Users/Admin/Desktop/AI/exercise3/tri_model_ai/cli.py:1)
  CLI input flow and session execution.

- [tri_model_ai/streamlit_ui.py](/C:/Users/Admin/Desktop/AI/exercise3/tri_model_ai/streamlit_ui.py:1)
  Main Streamlit UI implementation.

- [requirements.txt](/C:/Users/Admin/Desktop/AI/exercise3/requirements.txt:1)
  Python dependencies.

- [data/sample_input.txt](/C:/Users/Admin/Desktop/AI/exercise3/data/sample_input.txt:1)
  Example large text for testing.

## Folder Structure

```text
exercise3/
  app.py
  streamlit_app.py
  requirements.txt
  README.md
  data/
    sample_input.txt
  tri_model_ai/
    __init__.py
    assistant.py
    config.py
    models.py
    parsing.py
    pipeline.py
    qa.py
    state.py
    text_splitter.py
    cli.py
    streamlit_ui.py
    utils.py
```

## Setup

From the repo root:

```bash
cd exercise3
python -m pip install -r requirements.txt
```

## Run

```bash
python app.py
```

## Run The UI

```bash
python -m streamlit run streamlit_app.py
```

## Example Run

1. Start the app.
2. Choose whether to load text from a file.
3. Provide a file path or paste a large text block.
4. Select `short`, `medium`, or `long`.
5. Read the final summary.
6. Ask questions about the summary.
7. Type `exit` to end the session.

## Streamlit UI Flow

1. Open the Streamlit app.
2. Paste large text, upload a supported file, or load the sample input.
3. Choose `short`, `medium`, or `long`.
4. Click `Generate Summaries`.
5. Review the initial summary and the refined final summary side by side.
6. Ask questions against the generated summary context.

## Notes

- The app chunks long input text before summarization using `RecursiveCharacterTextSplitter`, which preserves structure better than fixed-size chunking.
- The UI supports pasted text plus uploaded `.txt`, `.md`, `.pdf`, `.docx`, `.csv`, and `.log` files.
- The initial summary is prompt-driven: each chunk is summarized with an instruction to preserve key facts, names, dates, achievements, and major events, and the chunk summaries are stitched together to retain detail.
- The initial-summary stage uses soft prompt-driven control with a loose adaptive upper cap, so the model has room to preserve detail while still staying bounded on very large inputs.
- Summary length is controlled primarily through instruction-based refinement prompts, and the refinement generation limit is chosen adaptively from the size of the initial summary.
- The QA model answers from the generated summary context, which combines the initial summary and refined final summary.
- The first run will download the Hugging Face model files locally.
