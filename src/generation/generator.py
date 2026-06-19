"""Day 6A — RAG generator.

Wires the winning retriever (3E: contextual chunks + hybrid + rerank) to an LLM
to produce grounded answers.  The pipeline is:

    query
      │
      ▼
    retriever.invoke(query)        ← top-k chunks (LangChain Documents)
      │
      ▼
    format_context(docs)           ← numbered passages for the prompt
      │
      ▼
    _RAG_PROMPT.format(...)        ← fill context + question into template
      │
      ▼
    llm.invoke(prompt).content     ← LLM generates the answer string
      │
      ▼
    RAGAnswer(question, answer, sources)

Design principle: keep the retriever and LLM fully swappable.
`ask()` accepts any BaseRetriever + any LangChain chat model — you can
benchmark 3A vs 3E answering quality by passing different retrievers.
"""

from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.retrievers import BaseRetriever


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_RAG_PROMPT = """\
You are an expert assistant on Aluminum-Air batteries. Answer the question using ONLY
the context passages below. Cite the passage number (e.g. [1], [2]) when you
use it. If the answer is not contained in the context, say exactly:
"I don't have enough information to answer this."

Context:
{context}

Question: {question}

Answer:"""


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class RAGAnswer:
    """Holds the generated answer together with its retrieval provenance."""
    question: str
    answer:   str
    sources:  list[Document] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper: format retrieved chunks into a numbered context block
# ---------------------------------------------------------------------------

# format_context: join a list of Documents into a single string for the prompt.
#
# Output format (one block per chunk):
#   [1] (filename.pdf)
#   <chunk text>
#
#   [2] (other_file.docx)
#   <chunk text>
#
# Input:  docs — list[Document] from retriever.invoke()
# Output: str  — the full context block, passages separated by "\n\n"


def format_context(docs: list[Document]) -> str:
    """Return a numbered, source-labelled context string for the RAG prompt."""
    parts = []
    for i, doc in enumerate(docs):
        # Extract just the filename from the full source path
        source = Path(doc.metadata.get("source", "unknown")).name   # (easy) filename only

        # Build the labelled passage block
        parts.append(f"[{i+1}] ({source})\n{doc.page_content}")      # (think) number, source, text field

    return "\n\n".join(parts)   # (easy) which str method joins a list?


# ---------------------------------------------------------------------------
# Core function: ask()
# ---------------------------------------------------------------------------

# ask: end-to-end RAG — retrieve, format, prompt, generate, return.
#
# Input:  query     — the user's natural-language question
#         retriever — any LangChain BaseRetriever (3E for best results)
#         llm       — any LangChain chat model (e.g. ChatOpenAI, ChatAnthropic)
#         k         — how many retrieved chunks to include in the prompt context
# Output: RAGAnswer with .answer (str) and .sources (list[Document])


def ask(
    query:     str,
    retriever: BaseRetriever,
    llm:       BaseLanguageModel,
    k:         int = 5,
) -> RAGAnswer:
    """Retrieve → format → generate → return RAGAnswer.

    Args:
        query     : the user's question
        retriever : any LangChain BaseRetriever
        llm       : any LangChain chat model
        k         : number of chunks to pass to the LLM as context
    """
    # Step 1: retrieve candidate chunks from the index
    # Hint: BaseRetriever exposes .invoke(query) → list[Document]
    docs = retriever.invoke(query)   # (think) method + argument

    # Step 2: cap to k — only feed the top-k chunks to the LLM
    # Background: passing too many chunks dilutes the signal and increases cost.
    # More context is not always better — LLMs lose focus in long prompts.
    context_docs = docs[:k]       # (easy) slice to k

    # Step 3: format the chunks into a numbered context string
    context = format_context(context_docs)     # (easy) call the helper above

    # Step 4: fill the prompt template with context and question
    prompt = _RAG_PROMPT.format(
        context=context,                # (easy)
        question=query,                 # (easy)
    )

    # Step 5: call the LLM and extract the answer string
    # Background: .invoke(prompt) returns an AIMessage; .content is the string.
    # Same pattern you used in contextual.py for generating chunk context.
    answer = llm.invoke(prompt).content   # (think) invoke + extract

    # Step 6: return a RAGAnswer bundling question, answer, and source docs
    return RAGAnswer(
        question=query,    # (easy)
        answer=answer,      # (easy)
        sources=context_docs,     # (easy) which variable holds the retrieved docs?
    )


# ---------------------------------------------------------------------------
# Convenience wrapper: build_rag_chain
# ---------------------------------------------------------------------------

def build_rag_chain(retriever: BaseRetriever, llm: BaseLanguageModel, k: int = 5):
    """Return a callable f(query) -> RAGAnswer with retriever and llm baked in.

    Usage:
        chain = build_rag_chain(retriever_3e, llm)
        result = chain("What is the peak power density of the Ag28Cu72 cathode?")
        print(result.answer)
    """
    def chain(query: str) -> RAGAnswer:
        return ask(query, retriever, llm, k=k)
    return chain


# ---------------------------------------------------------------------------
# Smoke test  (run: python3 -m src.generation.generator)
# Prerequisites: contextual_chunks.jsonl, indices/chroma_contextual
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path as _Path
    from langchain_openai import ChatOpenAI
    from langchain_huggingface import HuggingFaceEmbeddings
    from ..retrieval.index import load_chroma_index, build_bm25_retriever
    from ..retrieval.retrievers import make_hybrid_retriever
    from ..retrieval.rerank import make_reranking_retriever
    from ..chunking.stats import load_chunks_jsonl

    contextual_chunk_path = _Path("contextual_chunks.jsonl")
    contextual_index_path = _Path("indices/chroma_contextual")

    if not contextual_chunk_path.exists() or not contextual_index_path.exists():
        print("ERROR: run build_contextual_corpus.py first")
        raise SystemExit(1)

    # Build winning retriever (3E)
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )
    chunks      = load_chunks_jsonl(contextual_chunk_path)
    vectorstore = load_chroma_index(contextual_index_path, embeddings)
    bm25        = build_bm25_retriever(chunks, k=50)
    hybrid      = make_hybrid_retriever(vectorstore, bm25, dense_weight=0.5, k=50)
    retriever   = make_reranking_retriever(hybrid, top_n=5, fetch_k=50)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    probes = [
        "What is the peak power density achieved by the Ag28Cu72 parent alloy cathode?",
        "How does the generation of aluminate ions (Al(OH)4-) practically limit the Al-Air battery?",
        "What electrolyte concentration is used in the Al-Air battery tests?",
    ]

    for q in probes:
        result = ask(q, retriever, llm, k=5)
        print(f"\nQ: {q}")
        print(f"A: {result.answer}")
        print(f"   sources: {[_Path(d.metadata.get('source','?')).name for d in result.sources]}")

    # expected: grounded answers citing [1], [2] etc., with correct source filenames


# ---- HINTS (uncover only if stuck > 5 min) ----
# format_context blanks:
#   .name              ← Path attribute that returns just the filename
#   f"[{i+1}] ({source})\n{doc.page_content}"
#   "\n\n".join(parts)
#
# ask blanks:
#   retriever.invoke(query)
#   docs[:k]
#   format_context(context_docs)
#   _RAG_PROMPT.format(context=context, question=query)
#   llm.invoke(prompt).content
#   RAGAnswer(question=query, answer=answer, sources=context_docs)


# ---- Reflection questions ----
# Q13: The prompt says "use ONLY the context passages". What failure mode does
#      this guard against? What happens if you remove that instruction?
#      (Think: what does the LLM do when it knows the answer from training data
#      but the context doesn't contain it?)
#
# Q14: We pass k=5 chunks to the LLM by default. What are the trade-offs of
#      increasing k to 20? Consider: answer quality, token cost, and the
#      "lost in the middle" phenomenon in long-context LLMs.
