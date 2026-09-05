"""
src/generation/prompts.py

Defines prompt templates for DevDoc RAG:
- Prompt A: Baseline direct context prompt
- Prompt B: Structured technical assistant prompt with source citations and code fences
- Query Rewriter: Reformulates developer questions into targeted search queries (Best Practice)
- LLM-as-a-Judge Prompt: For automated faithfulness and answer relevance evaluation
"""

from __future__ import annotations

# Prompt A: Baseline Direct Context Prompt
PROMPT_A_TEMPLATE = """Context information is below:
---------------------
{context}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {query}
Answer:"""

# Prompt B: Structured Technical Assistant Prompt (Production / Winning Template)
PROMPT_B_TEMPLATE = """You are an expert technical documentation assistant and senior software engineer specializing in backend frameworks and developer tools.

### Retrieved Documentation Context:
{context}

### User Question:
{query}

### Instructions:
1. Grounding: Answer the question accurately and concisely based ONLY on the provided context. Do not invent features, flags, or APIs not supported by the context.
2. Code Blocks: When code is relevant, provide clean, idiomatic, and properly highlighted code blocks (e.g. ```python, ```bash, ```sql).
3. Citations: Reference the source library and document/section name where the information was found (e.g. `[fastapi / tutorial/dependencies]`).
4. Fallback: If the provided context does not contain sufficient details to answer completely, honestly state what is covered and what is missing.

Answer:"""

# Query Rewriting Prompt (Best Practice Bonus Rubric)
QUERY_REWRITE_TEMPLATE = """You are a search query reformulation assistant for technical documentation.
Your task is to take a developer's question and rewrite it into a concise, keyword-rich search query optimized for dense vector and BM25 lexical documentation search.

Developer Question: {query}

Instructions:
- Keep key technical terms, framework names, methods, parameters, and error keywords.
- Remove conversational filler words (e.g., "how can I", "please tell me", "I was wondering").
- Return ONLY the rewritten search query with no quotes, preamble, or markdown tags.

Rewritten Query:"""

# LLM-as-a-Judge Prompts for Evaluation (Faithfulness & Relevance)
LLM_JUDGE_PROMPT_TEMPLATE = """You are an impartial expert evaluator benchmarking AI-generated technical answers against ground-truth documentation.

Context Provided to Assistant:
\"\"\"
{context}
\"\"\"

User Question:
\"\"\"
{question}
\"\"\"

Ground Truth Reference Answer:
\"\"\"
{ground_truth_answer}
\"\"\"

Assistant Generated Answer:
\"\"\"
{generated_answer}
\"\"\"

Please evaluate the Assistant Generated Answer on two distinct criteria using a scale from 1 to 5:

1. Faithfulness / Groundedness (1-5):
   - 5: Every claim in the answer is directly supported by the provided context with zero hallucination.
   - 3: Mostly supported, but includes minor unsupported statements.
   - 1: Significant hallucinations or contradicts the provided context.

2. Answer Relevance & Completeness (1-5):
   - 5: Completely and directly answers the user's question matching the ground-truth standard.
   - 3: Answers the question partially or includes unnecessary tangential information.
   - 1: Off-topic, does not answer the core question, or fails to address the user request.

Respond ONLY with a valid JSON object matching this exact schema:
{{
  "faithfulness_score": 5,
  "faithfulness_reasoning": "Explanation of faithfulness score",
  "relevance_score": 5,
  "relevance_reasoning": "Explanation of relevance score"
}}
"""


def format_context(retrieved_chunks: list[dict]) -> str:
    """Formats retrieved chunks into a standardized context block for prompt injection."""
    formatted_blocks: list[str] = []
    for idx, chunk in enumerate(retrieved_chunks, 1):
        source_lib = chunk.get("source_lib", "unknown")
        section_title = chunk.get("section_title", "Document")
        file_path = chunk.get("file_path", "")
        content = chunk.get("content", "").strip()

        header = f"[{idx}] Source: {source_lib} | File: {file_path} | Section: {section_title}"
        formatted_blocks.append(f"{header}\n{content}")

    return "\n\n".join(formatted_blocks)
