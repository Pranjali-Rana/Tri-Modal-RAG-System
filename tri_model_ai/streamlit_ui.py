from __future__ import annotations

from pathlib import Path

import streamlit as st

from .assistant import LENGTH_SETTINGS, answer_from_context, generate_summary_bundle, refine_summary_bundle
from .parsing import extract_text_from_uploaded_file, normalize_source_text, strip_reference_artifacts


SAMPLE_INPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_input.txt"


def reset_outputs() -> None:
    st.session_state.pop("summary_bundle", None)
    st.session_state.pop("qa_history", None)


def render_app() -> None:
    st.set_page_config(
        page_title="Tri-Model AI Assistant",
        page_icon="AI",
        layout="wide",
    )

    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 15% 20%, rgba(245, 158, 11, 0.18), transparent 24%),
                radial-gradient(circle at 85% 18%, rgba(59, 130, 246, 0.16), transparent 26%),
                linear-gradient(135deg, #0b1020 0%, #121a2f 50%, #0f172a 100%);
            color: #e8edf7;
            font-family: Georgia, "Palatino Linotype", serif;
        }
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"],
        [data-testid="stMain"],
        [data-testid="stSidebar"] {
            color: #e8edf7;
        }
        h1, h2, h3, h4, h5, h6, p, label, span, div {
            color: #e8edf7;
        }
        .stMarkdown,
        .stMarkdown p,
        .stCaption,
        .stText,
        .stMetric,
        .stMetric label {
            color: #e8edf7 !important;
        }
        .hero {
            padding: 1.6rem 1.8rem;
            border-radius: 26px;
            background: rgba(15, 23, 42, 0.78);
            border: 1px solid rgba(148, 163, 184, 0.18);
            box-shadow: 0 18px 54px rgba(2, 6, 23, 0.28);
        }
        .hero h1 {
            margin: 0;
            font-size: 2.45rem;
            line-height: 1.05;
        }
        .hero p {
            margin: 0.8rem 0 0 0;
            color: #c4cee0;
            font-size: 1rem;
        }
        .panel-note {
            padding: 0.9rem 1rem;
            border-left: 4px solid #f59e0b;
            border-radius: 14px;
            background: rgba(120, 53, 15, 0.28);
            color: #fde7c7;
        }
        .metric-card {
            padding: 1rem 1.1rem;
            border-radius: 18px;
            background: rgba(15, 23, 42, 0.74);
            border: 1px solid rgba(148, 163, 184, 0.18);
            box-shadow: 0 12px 28px rgba(2, 6, 23, 0.22);
        }
        .summary-box {
            padding: 1rem 1.1rem;
            border-radius: 18px;
            background: rgba(15, 23, 42, 0.74);
            border: 1px solid rgba(148, 163, 184, 0.18);
            box-shadow: 0 12px 28px rgba(2, 6, 23, 0.22);
        }
        [data-testid="stTextArea"] textarea,
        [data-testid="stTextInput"] input {
            color: #e8edf7 !important;
            background: rgba(15, 23, 42, 0.92) !important;
        }
        [data-testid="stTextArea"] label,
        [data-testid="stTextInput"] label,
        [data-testid="stSelectbox"] label,
        [data-testid="stRadio"] label,
        [data-testid="stFileUploader"] label {
            color: #e8edf7 !important;
        }
        [data-testid="stFileUploader"] {
            color: #e8edf7 !important;
        }
        [data-testid="stFileUploader"] section {
            background: rgba(15, 23, 42, 0.82) !important;
            border: 1px dashed rgba(148, 163, 184, 0.28) !important;
            border-radius: 18px !important;
        }
        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] span,
        [data-testid="stFileUploader"] div {
            color: #e8edf7 !important;
        }
        [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
            background: rgba(15, 23, 42, 0.92) !important;
            color: #e8edf7 !important;
            border: 1px solid rgba(148, 163, 184, 0.24) !important;
        }
        [data-testid="stCodeBlock"],
        [data-testid="stCode"] {
            background: rgba(15, 23, 42, 0.9) !important;
            border: 1px solid rgba(148, 163, 184, 0.18) !important;
            border-radius: 16px !important;
        }
        [data-testid="stCodeBlock"] pre,
        [data-testid="stCode"] pre,
        [data-testid="stCodeBlock"] code,
        [data-testid="stCode"] code {
            color: #e8edf7 !important;
            background: transparent !important;
        }
        .stButton button {
            color: #0f172a !important;
            background: linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%) !important;
            border: 1px solid rgba(245, 158, 11, 0.22) !important;
        }
        .stAlert,
        [data-baseweb="notification"] {
            color: #e8edf7 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "source_text" not in st.session_state:
        st.session_state.source_text = ""
    if "source_label" not in st.session_state:
        st.session_state.source_label = "No source selected"
    if "summary_bundle" not in st.session_state:
        st.session_state.summary_bundle = None
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []

    st.markdown(
        """
        <div class="hero">
            <h1>Tri-Model AI Assistant</h1>
            <p>
                Explore the full local pipeline: prompt-driven intermediate summarization,
                adaptive refinement into short, medium, or long output, and question answering
                grounded in the generated summary context.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    controls_col, metrics_col = st.columns([1.2, 0.8], gap="large")

    with controls_col:
        st.markdown("## Source Text")
        input_mode = st.radio(
            "Input method",
            options=["Paste Text", "Upload File", "Sample Text"],
            horizontal=True,
        )

        if input_mode == "Paste Text":
            source_text = st.text_area(
                "Large input text",
                value=st.session_state.source_text,
                height=280,
                placeholder="Paste a long article, biography, report, or document here...",
            )
            st.session_state.source_text = normalize_source_text(source_text)
            st.session_state.source_label = "Pasted text"

        elif input_mode == "Upload File":
            uploaded_file = st.file_uploader(
                "Upload a document",
                type=["txt", "md", "pdf", "docx", "csv", "log"],
                help="Supported formats: TXT, MD, PDF, DOCX, CSV, LOG",
            )
            if uploaded_file is not None:
                try:
                    parsed_text = extract_text_from_uploaded_file(uploaded_file.name, uploaded_file.getvalue())
                    st.session_state.source_text = parsed_text
                    st.session_state.source_label = f"Uploaded file: {uploaded_file.name}"
                    st.success(f"Loaded and parsed `{uploaded_file.name}` successfully.")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Could not parse the uploaded file: {exc}")

            st.text_area(
                "Parsed preview",
                value=st.session_state.source_text,
                height=280,
                disabled=True,
            )

        else:
            if st.button("Load Sample Input", use_container_width=True):
                st.session_state.source_text = normalize_source_text(
                    SAMPLE_INPUT_PATH.read_text(encoding="utf-8")
                )
                st.session_state.source_label = "Built-in sample input"
                st.rerun()
            st.text_area(
                "Sample preview",
                value=st.session_state.source_text,
                height=280,
                disabled=True,
            )

        action_cols = st.columns([1, 1], gap="small")
        if action_cols[0].button("Clear Source", use_container_width=True):
            st.session_state.source_text = ""
            st.session_state.source_label = "No source selected"
            reset_outputs()
            st.rerun()
        if action_cols[1].button("Reset Session", use_container_width=True):
            st.session_state.source_text = ""
            st.session_state.source_label = "No source selected"
            reset_outputs()
            st.rerun()

    with metrics_col:
        st.markdown("## Controls")
        st.markdown(
            '<div class="panel-note">The initial summary is detail-preserving and prompt-driven. The final summary is then refined according to your chosen detail level.</div>',
            unsafe_allow_html=True,
        )
        summary_length = st.selectbox(
            "Summary style",
            options=["short", "medium", "long"],
            index=1,
            help="Short is concise, medium is balanced, and long preserves more context.",
        )
        st.markdown(
            f"""
            <div class="metric-card">
                <strong>Source</strong><br>
                {st.session_state.source_label}<br><br>
                <strong>Current mode</strong><br>
                {summary_length.title()}<br><br>
                <strong>Prompt goal</strong><br>
                {LENGTH_SETTINGS[summary_length]["instruction"].splitlines()[1]}
            </div>
            """,
            unsafe_allow_html=True,
        )

    text_metrics = st.columns(3)
    text_metrics[0].metric("Characters", len(st.session_state.source_text))
    text_metrics[1].metric("Words", len(st.session_state.source_text.split()))
    text_metrics[2].metric("Paragraphs", len([p for p in st.session_state.source_text.splitlines() if p.strip()]))

    generate_clicked = st.button("Generate Summaries", use_container_width=True, type="primary")

    if generate_clicked:
        if not st.session_state.source_text.strip():
            st.warning("Enter or load some source text first.")
        else:
            existing_bundle = st.session_state.summary_bundle
            source_text = st.session_state.source_text
            should_regenerate_initial = (
                not existing_bundle
                or existing_bundle.get("source_text") != source_text
                or not existing_bundle.get("initial_summary")
            )

            if should_regenerate_initial:
                reset_outputs()
                with st.spinner("Generating initial and final summaries with the local model pipeline..."):
                    st.session_state.summary_bundle = generate_summary_bundle(
                        source_text,
                        summary_length,
                    )
                    st.session_state.qa_history = []
            elif existing_bundle.get("summary_length") != summary_length:
                with st.spinner("Reusing the initial summary and refining it for the selected style..."):
                    st.session_state.summary_bundle = refine_summary_bundle(
                        existing_bundle["initial_summary"],
                        summary_length,
                        source_text=source_text,
                    )
                    st.session_state.qa_history = []

    summary_bundle = st.session_state.summary_bundle

    if summary_bundle:
        display_initial_summary = strip_reference_artifacts(summary_bundle["initial_summary"])
        display_final_summary = strip_reference_artifacts(summary_bundle["final_summary"])
        st.markdown("## Summary Results")
        initial_col, final_col = st.columns(2, gap="large")

        with initial_col:
            st.markdown("### Initial Summary")
            st.markdown(
                '<div class="summary-box">Detail-preserving intermediate summary used as the base for refinement and QA.</div>',
                unsafe_allow_html=True,
            )
            st.write(display_initial_summary)

        with final_col:
            st.markdown("### Final Summary")
            st.markdown(
                '<div class="summary-box">Refined output shaped to your selected summary style.</div>',
                unsafe_allow_html=True,
            )
            st.write(display_final_summary)

        st.markdown("## Question Answering")
        st.markdown(
            '<div class="panel-note">Questions are answered from the generated summary context, not from the raw source text.</div>',
            unsafe_allow_html=True,
        )

        with st.form("qa_form", clear_on_submit=True):
            question_col, ask_col = st.columns([5, 1], gap="small")
            question = question_col.text_input(
                "Ask a question about the generated summary",
                placeholder="What were the major achievements mentioned?",
                label_visibility="collapsed",
            )
            ask_clicked = ask_col.form_submit_button("Ask", use_container_width=True)

        if ask_clicked:
            if not question.strip():
                st.warning("Enter a question first.")
            else:
                with st.spinner("Searching the summary context for an answer..."):
                    answer = answer_from_context(question, summary_bundle["qa_context"])
                st.session_state.qa_history.append({"question": question, "answer": strip_reference_artifacts(answer)})

        if st.session_state.qa_history:
            st.markdown("### QA History")
            for index, item in enumerate(st.session_state.qa_history, start=1):
                st.markdown(f"**Q{index}.** {item['question']}")
                st.write(item["answer"])
