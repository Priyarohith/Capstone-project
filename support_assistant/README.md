# Support Assistant Module

This module builds a grounded Zepto policy assistant using a ChromaDB-backed retrieval layer, a LangGraph router, and a FastAPI API. The default mode is the graded offline mock path: MOCK_LLM is left unset or set to 1, so no external LLM is called.

## Architecture walkthrough

The full retrieval pipeline runs in the order ingestion → embedding → retrieval → generation.

- Ingestion: the document corpus is stored as plain text files in [support_assistant/docs](docs). The load step runs in [support_assistant/main.py](main.py) via the ingest_documents function.
- Embedding: each document is embedded locally using the SentenceTransformer model all-MiniLM-L6-v2. These vector embeddings are saved in the persistent Chroma collection named zepto_policy_collection.
- Retrieval: the classify_intent node decides whether a query is a policy_question or general_question. The policy route then calls the retrieve_and_answer node, which queries ChromaDB for the top three similar chunks.
- Generation: in mock mode, the final answer is produced by the retrieve_and_answer or direct_answer node using a canned response template. In the optional MOCK_LLM=0 path, the real LLM branch would use the structured prompt template and retrieved context instead.

The flow is graph-based rather than a single linear function: the conditional edge after classify_intent routes directly to the proper downstream node.

## Prompt template

The structured prompt template includes all required pieces: role, context, task, format, length, negative constraint, and a few-shot example. It is defined in [support_assistant/main.py](main.py) as PROMPT_TEMPLATE.

## Routing behavior

The requirement for the graded baseline is that classification happens through a keyword heuristic, not an LLM. For example:

- policy-style query: "What is Zepto's refund policy?" -> policy_question
- unrelated query: "What is the capital of France?" -> general_question

This path runs without any external LLM call.

## Example calls captured with the default mock mode

The project was run locally with MOCK_LLM left at the default. Two example API calls were executed:

1) Policy retrieval request:

```json
{"query": "What is Zepto's refund policy?"}
```

Response:

```json
{"answer":"Based on the retrieved context: Grocery and perishable items may be reported for a return within 24 hours of delivery if damaged, spoiled, or incorrect; non-perishable packaged items may be returned within 7 days of delivery in unopened, resalable condition.","sources":["doc_02"],"confidence":1.0}
```

2) General question request:

```json
{"query": "What is the capital of France?"}
```

Response:

```json
{"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}
```

## FastAPI application

The API is exposed via a POST /ask endpoint that accepts a JSON body of the form:

```json
{"query": "..."}
```

and returns a validated Pydantic response with answer, sources, and confidence.

## Docker run

The repository includes a local Dockerfile for the FastAPI service. It is designed to build and run in a container locally.

```bash
cd support_assistant
# build

docker build -t zepto-policy-assistant .

# run

docker run -p 7860:7860 zepto-policy-assistant
```

The app serves the POST /ask endpoint locally on port 7860.
