import os
from pathlib import Path
from typing import TypedDict

import chromadb
import uvicorn
from fastapi import FastAPI
from huggingface_hub import snapshot_download
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
DOCS_DIR = ROOT / "docs"
CHROMA_PATH = ROOT / "chroma_db"
MOCK_LLM = os.getenv("MOCK_LLM", "1")

PROMPT_TEMPLATE = """
You are Zepto policy assistant.

Role:
You answer customer policy questions only using Zepto's internal documents.

Context:
{retrieved_context}

Task:
Answer the user's question using only the retrieved context.

Format:
Return a concise response with clear references to policy facts, followed by a short summary.

Length:
Keep the answer brief but complete.

Negative constraint:
Do not answer using information not present in the provided context.

Few-shot example:
User: What is Zepto's refund window for damaged groceries?
Assistant: Grocery and perishable items may be reported for a return within 24 hours of delivery if damaged, spoiled, or incorrect.

User query:
{query}
""".strip()


class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class GraphState(TypedDict):
    query: str
    intent: str
    answer: str
    sources: list[str]
    confidence: float
    retrieved_chunks: list[str]


MODEL_DIR = ROOT / "hf_all_mini"
if not MODEL_DIR.exists():
    snapshot_download(repo_id="sentence-transformers/all-MiniLM-L6-v2", local_dir=str(MODEL_DIR), local_dir_use_symlinks=False)
embedding_model = SentenceTransformer(str(MODEL_DIR))
client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = client.get_or_create_collection(name="zepto_policy_collection")


def ingest_documents():
    files = sorted(DOCS_DIR.glob("doc_*.txt"))
    if not files:
        raise FileNotFoundError("No policy corpus files were found in support_assistant/docs.")

    documents = []
    ids = []
    metadatas = []
    for file_path in files:
        text = file_path.read_text(encoding="utf-8")
        doc_id = file_path.stem
        documents.append(text)
        ids.append(doc_id)
        metadatas.append({"source": file_path.name})

    existing = collection.get(include=[])
    if len(existing["ids"]) == 0:
        collection.add(documents=documents, ids=ids, metadatas=metadatas)


ingest_documents()


def classify_intent_fn(state: GraphState):
    lower_query = state["query"].lower()
    if MOCK_LLM == "0":
        if any(word in lower_query for word in ["delivery", "return", "refund", "membership", "tracking", "cancel", "gift card", "support hours"]):
            return {"intent": "policy_question"}
        return {"intent": "general_question"}

    keywords = ["delivery", "return", "refund", "membership", "tracking", "cancel", "gift card", "support hours"]
    if any(keyword in lower_query for keyword in keywords):
        return {"intent": "policy_question"}
    return {"intent": "general_question"}


def route_intent(state: GraphState):
    return state["intent"]


def retrieve_and_answer_fn(state: GraphState):
    result = collection.query(query_texts=[state["query"]], n_results=3, include=["documents", "metadatas", "distances"])
    docs = result["documents"][0]
    ids = result["ids"][0]
    top_chunk = docs[0]
    snippet = top_chunk[:200]
    state["retrieved_chunks"] = docs
    state["sources"] = ids

    if MOCK_LLM == "0":
        context = "\n\n".join(docs[:3])
        prompt = PROMPT_TEMPLATE.format(retrieved_context=context, query=state["query"])
        answer = f"Grounded answer (LLM extension): {prompt[:300]}"
    else:
        answer = f"Based on the retrieved context: {snippet}"

    state["answer"] = answer
    state["confidence"] = 1.0
    return state


def direct_answer_fn(state: GraphState):
    state["sources"] = []
    if MOCK_LLM == "0":
        state["answer"] = "I can answer general questions directly, but this module is limited to Zepto policy support."
    else:
        state["answer"] = "I can only answer questions about Zepto policies right now."
    state["confidence"] = 1.0
    return state


def build_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("classify_intent", classify_intent_fn)
    workflow.add_node("retrieve_and_answer", retrieve_and_answer_fn)
    workflow.add_node("direct_answer", direct_answer_fn)
    workflow.set_entry_point("classify_intent")
    workflow.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "policy_question": "retrieve_and_answer",
            "general_question": "direct_answer",
        },
    )
    workflow.add_edge("retrieve_and_answer", END)
    workflow.add_edge("direct_answer", END)
    return workflow.compile()


app_graph = build_graph()
app = FastAPI(title="Zepto Policy Assistant")


@app.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest):
    state = {
        "query": payload.query,
        "intent": "",
        "answer": "",
        "sources": [],
        "confidence": 0.0,
        "retrieved_chunks": [],
    }
    result = app_graph.invoke(state)
    response = AskResponse(
        answer=result.get("answer", ""),
        sources=result.get("sources", []),
        confidence=float(result.get("confidence", 1.0)),
    )
    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
