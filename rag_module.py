import os, re, time
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import openai

_HERE = os.path.dirname(os.path.abspath(__file__))

RAG_CONFIG = {
    "CHROMA_DB_PATH": os.path.join(_HERE, "chroma_db"),
    "COLLECTION_NAME" : "sbp_regulations",
    "EMBEDDING_MODEL" : "sentence-transformers/all-MiniLM-L6-v2",
    "OPENAI_MODEL"    : "gpt-4o-mini",
    "MAX_TOKENS"      : 1000,
    "TEMPERATURE"     : 0.1,
    "TOP_K_SEMANTIC"  : 10,
    "TOP_K_BM25"      : 10,
    "TOP_K_FINAL"     : 5,
    "FRAUD_THRESHOLD" : 0.5,
}

RISK_TIERS = {"CRITICAL": (0.85, 1.01), "HIGH": (0.65, 0.85), "MEDIUM": (0.50, 0.65)}

def get_risk_tier(prob):
    for tier, (lo, hi) in RISK_TIERS.items():
        if lo <= prob < hi:
            return tier
    return "LOW"

def decode_tx_type(features):
    for col, name in [("type_CASH_OUT","CASH_OUT"), ("type_DEBIT","DEBIT"),
                      ("type_PAYMENT","PAYMENT"),   ("type_TRANSFER","TRANSFER")]:
        if features.get(col, 0) >= 0.5:
            return name
    return "CASH_IN"

def _tokenize(text):
    return re.findall(r"\b[a-z0-9/\-]+\b", text.lower()) or ["<empty>"]

@dataclass
class StreamlitRAGResult:
    transaction_id   : str
    fraud_probability: float
    risk_tier        : str
    response_text    : str
    sources          : list
    citations        : list
    grounding_score  : float
    latency_seconds  : float
    no_evidence_flag : bool
    retrieved_chunks : list

_components = {}

def load_rag_components(openai_api_key=None):
    global _components
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
        # Rebuild client if key changed
        if _components:
            _components["client"] = openai.OpenAI(
                api_key=openai_api_key)
            return _components
    elif _components:
        return _components
    elif _components:
        return _components
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    embed_model   = SentenceTransformer(RAG_CONFIG["EMBEDDING_MODEL"])
    chroma_client = chromadb.PersistentClient(path=RAG_CONFIG["CHROMA_DB_PATH"])
    # Patch: delete stored embedding function metadata that causes _type error
    try:
        col = chroma_client.get_collection(name=RAG_CONFIG["COLLECTION_NAME"])
    except Exception:
        col = None
    # AFTER
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    collection = chroma_client.get_or_create_collection(
        name=RAG_CONFIG["COLLECTION_NAME"],
        embedding_function=None   # we handle embeddings ourselves — skip deserialization
    )
    client        = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
    all_data      = collection.get(include=["documents", "metadatas"])
    texts         = all_data["documents"]
    metas         = all_data["metadatas"]
    bm25          = BM25Okapi([_tokenize(t) for t in texts])
    try:
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception:
        reranker = None
    _components = {
        "embed_model": embed_model, "collection": collection,
        "client": client, "bm25": bm25, "texts": texts,
        "metas": metas, "reranker": reranker,
    }
    return _components

def rag_pipeline_for_streamlit(fraud_probability, features, transaction_id="TXN-STREAMLIT", openai_api_key=None, embed_model=None, reranker=None):
    comp        = load_rag_components(openai_api_key)
    # Override with pre-cached models if passed in
    if embed_model is not None:
        comp["embed_model"] = embed_model
    if reranker is not None:
        comp["reranker"] = reranker
    embed_model = comp["embed_model"]
    collection  = comp["collection"]
    client      = comp["client"]
    bm25        = comp["bm25"]
    texts       = comp["texts"]
    metas       = comp["metas"]
    reranker    = comp["reranker"]

    tx_type   = decode_tx_type(features)
    risk_tier = get_risk_tier(fraud_probability)
    amount    = features.get("amount", 0)
    old_bal   = features.get("oldbalanceOrg", 0)
    new_bal   = features.get("newbalanceOrig", 0)

    query = (
        f"A {tx_type} transaction of PKR {amount:,.0f} flagged "
        f"({fraud_probability:.1%} {risk_tier} risk). "
        f"Balance PKR {old_bal:,.0f} to PKR {new_bal:,.0f}. "
        "SBP AML CFT regulations STR reporting suspicious transaction CDD EDD."
    )

    t0  = time.time()
    qe  = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True).tolist()
    fraud_categories = ["AML/CFT/CPF", "Branchless Banking"]
    sem = collection.query(
        query_embeddings=qe,
        n_results=RAG_CONFIG["TOP_K_SEMANTIC"],
        where={"category": {"$in": fraud_categories}},
        include=["documents", "metadatas", "distances"],
    )
    scores   = bm25.get_scores(_tokenize(query))
    bm25_top = np.argsort(scores)[::-1][:RAG_CONFIG["TOP_K_BM25"]]

    cands = {}
    for rank, (doc, meta, dist) in enumerate(zip(
        sem["documents"][0], sem["metadatas"][0], sem["distances"][0]
    )):
        key = (meta["doc_name"], meta["page_number"])
        cands[key] = {"text": doc, "meta": meta, "score": 1.0 / (61 + rank)}

    for rank, idx in enumerate(bm25_top):
        if scores[idx] > 0:
            meta = metas[idx]
            key  = (meta["doc_name"], meta["page_number"])
            if key not in cands:
                cands[key] = {"text": texts[idx], "meta": meta, "score": 0.0}
            cands[key]["score"] += 1.0 / (61 + rank)

    sorted_c = sorted(cands.values(), key=lambda x: x["score"], reverse=True)

    if reranker and sorted_c:
        pairs    = [(query, c["text"]) for c in sorted_c]
        rs       = reranker.predict(pairs).tolist()
        sorted_c = [c for _, c in sorted(zip(rs, sorted_c), key=lambda x: x[0], reverse=True)]

    final          = sorted_c[:RAG_CONFIG["TOP_K_FINAL"]]
    context_blocks = []
    sources        = []
    chunks_display = []

    for i, c in enumerate(final, 1):
        m    = c["meta"]
        cite = m.get("citation", f"[{m.get('short_name','SBP')}, Page {m.get('page_number',0)}]")
        context_blocks.append(f"[Context {i}] {cite}\n{c['text']}")
        sources.append(
            f"- {m.get('short_name')} | {m.get('category')} | "
            f"Page {m.get('page_number')} | {m.get('section')}"
        )
        chunks_display.append({
            "citation": cite,
            "text"    : c["text"][:300] + "...",
            "page"    : m.get("page_number"),
        })

    context_str = "\n\n".join(context_blocks)
    system = (
        "You are an SBP regulatory compliance assistant. "
        "Ground every claim in the provided context. "
        "Cite sources as [Document, Section, Page X]. Never hallucinate. "
        "If context is insufficient, state: No grounded SBP regulatory evidence found."
    )
    user = (
        f"Transaction: {tx_type} | PKR {amount:,.0f} | {fraud_probability:.1%} | {risk_tier}\n\n"
        f"RETRIEVED SBP CONTEXT:\n{context_str}\n\n"
        "Provide: 1) Regulations triggered (cite each), "
        "2) Reporting obligations (STR/CTR), "
        "3) Compliance actions, 4) Risk justification, "
        "5) Regulatory basis summary. End with SOURCES."
    )

    try:
        completion    = client.chat.completions.create(
            model     = RAG_CONFIG["OPENAI_MODEL"],
            messages  = [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens  = RAG_CONFIG["MAX_TOKENS"],
            temperature = RAG_CONFIG["TEMPERATURE"],
        )
        response_text = completion.choices[0].message.content.strip()
    except Exception as e:
        response_text = f"ERROR: {e}"

    latency   = time.time() - t0
    pattern   = r"\[([^\[\]]+,\s*[^\[\]]+,\s*Page\s*\d+[^\[\]]*)\]"
    citations = list(dict.fromkeys(re.findall(pattern, response_text)))
    sents     = [s.strip() for s in re.split(r"[.!?]", response_text) if len(s.strip()) > 20]
    grounding = sum(1 for s in sents if "[" in s and "]" in s) / max(len(sents), 1)
    no_ev     = "no grounded" in response_text.lower()

    return StreamlitRAGResult(
        transaction_id    = transaction_id,
        fraud_probability = fraud_probability,
        risk_tier         = risk_tier,
        response_text     = response_text,
        sources           = list(dict.fromkeys(sources)),
        citations         = citations,
        grounding_score   = round(grounding, 3),
        latency_seconds   = round(latency, 2),
        no_evidence_flag  = no_ev,
        retrieved_chunks  = chunks_display,
    )
