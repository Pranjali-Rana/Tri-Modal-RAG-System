from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .config import LENGTH_SETTINGS, SummaryLength
from .pipeline import get_pipeline
from .qa import get_qa_service
from .state import AssistantState


def get_assistant():
    print("\nLoading local Hugging Face models. This can take a while on the first run...")
    get_pipeline()
    return get_qa_service()


def summarize_node(state: AssistantState) -> AssistantState:
    initial_summary = get_pipeline().summarize_text(state["input_text"])
    return {"initial_summary": initial_summary}


def refine_node(state: AssistantState) -> AssistantState:
    final_summary = get_pipeline().refine_summary(
        state["initial_summary"],
        state["summary_length"],
    )
    qa_context = f"{state['initial_summary']}\n\n{final_summary}"
    return {"final_summary": final_summary, "qa_context": qa_context}


def display_summary_node(state: AssistantState) -> AssistantState:
    print("\nInitial summary")
    print("-" * 60)
    print(state["initial_summary"])
    print("-" * 60)

    print("\nFinal summary")
    print("-" * 60)
    print(state["final_summary"])
    print("-" * 60)
    return {}


def prompt_question_node(state: AssistantState) -> AssistantState:
    question = input("\nAsk a question about the summary (or type 'exit'): ").strip()
    return {"question": question}


def route_question(state: AssistantState) -> str:
    if state["question"].lower() in {"exit", "quit"}:
        return END
    return "answer_question"


def answer_question_node(state: AssistantState) -> AssistantState:
    answer = get_qa_service().answer_question(state["question"], state["qa_context"])
    history_line = f"Q: {state['question']}\nA: {answer}"
    print("\nAnswer")
    print("-" * 60)
    print(answer)
    print("-" * 60)
    return {"answer": answer, "qa_history": [history_line]}


def generate_summary_bundle(input_text: str, summary_length: SummaryLength) -> dict[str, str]:
    return get_pipeline().generate_summary_bundle(input_text, summary_length)


def refine_summary_bundle(
    initial_summary: str,
    summary_length: SummaryLength,
    *,
    source_text: str | None = None,
) -> dict[str, str]:
    return get_pipeline().refine_summary_bundle(
        initial_summary,
        summary_length,
        source_text=source_text,
    )


def answer_from_context(question: str, qa_context: str) -> str:
    return get_qa_service().answer_question(question, qa_context)


def build_graph():
    graph = StateGraph(AssistantState)
    graph.add_node("summarize", summarize_node)
    graph.add_node("refine", refine_node)
    graph.add_node("display_summary", display_summary_node)
    graph.add_node("prompt_question", prompt_question_node)
    graph.add_node("answer_question", answer_question_node)

    graph.add_edge(START, "summarize")
    graph.add_edge("summarize", "refine")
    graph.add_edge("refine", "display_summary")
    graph.add_edge("display_summary", "prompt_question")
    graph.add_conditional_edges("prompt_question", route_question, {"answer_question": "answer_question", END: END})
    graph.add_edge("answer_question", "prompt_question")

    return graph.compile()
