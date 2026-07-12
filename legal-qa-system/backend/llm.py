import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL, MAX_ANSWER_TOKENS

SYSTEM_PROMPT = """You are a legal research assistant answering questions strictly from the \
provided source excerpts. Rules:
1. Only use facts contained in the excerpts below. Never use outside knowledge.
2. Every factual claim must end with an inline citation in the exact form [Doc: <doc_title>, p. <page>].
3. If the excerpts do not contain enough information to answer, say so explicitly instead of guessing.
4. Be concise: a short direct answer, then supporting detail, then citations woven in throughout.
5. Do not fabricate a citation that is not in the provided excerpts."""


def build_context_block(chunks):
    parts = []
    for c in chunks:
        page = c.get("page")
        page_str = f", p. {page}" if page else ""
        parts.append(f"[Doc: {c['doc_title']}{page_str}]\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def generate_answer(question: str, chunks: list) -> str:
    """Calls Claude via the Anthropic API to synthesize a cited answer from retrieved chunks.
    Falls back to a plain retrieval summary if no API key is configured."""
    if not chunks:
        return "I couldn't find any relevant passages in the indexed documents for this question."

    if not ANTHROPIC_API_KEY:
        return _retrieval_only_fallback(chunks)

    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    context = build_context_block(chunks)
    user_message = f"Question: {question}\n\nSource excerpts:\n\n{context}"

    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=MAX_ANSWER_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def _retrieval_only_fallback(chunks: list) -> str:
    """Used when ANTHROPIC_API_KEY isn't set — returns the raw top passages so the
    system is still usable (and testable) without an LLM call."""
    lines = ["No ANTHROPIC_API_KEY configured — showing the top retrieved passages directly:\n"]
    for c in chunks:
        page = f", p. {c.get('page')}" if c.get("page") else ""
        lines.append(f"[Doc: {c['doc_title']}{page}]\n{c['text'][:400]}...\n")
    return "\n".join(lines)
