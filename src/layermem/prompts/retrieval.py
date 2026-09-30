"""The retrieval planner prompt (LayerMem.md §2.4).

One LLM call per query decides *which perspectives to consult* and extracts
the exact keys the state chain needs. It deliberately does **not** decide how
many memories to return or which layer to read — LayerMem_imple.md §三 argues
that granularity alignment falls out of the embedding distance, so a router
that guesses at query difficulty only adds a failure mode.
"""

RETRIEVAL_PLANNER_PROMPT = """
You route a question to the memory stores of a long-term memory system.

Available stores:
- factual:    events that happened, preferences, plans, biographical details
- relational: relationships between people, entities or concepts
- state:      a subject's current or past value for one attribute
              (residence, occupation, pet_name, employer, ...)

Return ONE JSON object:

{
  "tracks": ["factual"],
  "keywords": ["keyword one", "keyword two"],
  "subject": null,
  "state_attr": null,
  "time_ref": null
}

Rules:
- "tracks" is a subset of ["factual", "relational", "state"]. Include every
  store that could plausibly hold the answer. When unsure, include more than
  one; the retriever merges and re-ranks.
- "keywords": 3 to 6 short search terms taken from the question and its likely
  answer, used by the sparse (lexical) retriever. Include proper nouns.
- "subject": only when "state" is in tracks — the person or entity whose
  attribute is asked about, exactly as it appears in the question. Otherwise null.
- "state_attr": only when "state" is in tracks — the attribute being asked
  about, as a short lowercase English name (e.g. "residence", "occupation").
  Otherwise null.
- "time_ref": only when "state" is in tracks AND the question asks about the
  past rather than the present. Copy the time expression from the question
  verbatim (e.g. "last year", "in 2022"). For present-tense questions use null.
- Return JSON only, with no markdown fences and no commentary.

Question:
{query}
""".strip()

PLANNER_DATE_PLACEHOLDER = "{current_date}"


def render_planner_prompt(query: str) -> str:
    return RETRIEVAL_PLANNER_PROMPT.replace("{query}", query)
