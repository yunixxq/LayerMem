"""Prompts for hierarchical memory compression (LayerMem.md §2.2).

The three levels describe the same person at increasing abstraction and
decreasing temporal resolution:

* L2 — what repeatedly happens *within one topic*;
* L3 — what stays true across topics over a period (a stage portrait);
* L4 — the stable overall profile.
"""

L2_TOPIC_SUMMARY_PROMPT = """
You are compressing one topic of a long-term memory into a single summary.

The entries below all belong to the same semantic topic and are ordered by
event time. Write ONE paragraph in English that captures what this topic is
about and how it developed.

Requirements:
- Merge repeated or near-duplicate facts instead of listing them separately.
- Keep concrete specifics (names, places, numbers, dates) that a later
  question would need; drop only the redundancy.
- Describe recurring behaviour as a pattern, not as a list of occurrences.
- Do not invent anything that is not supported by the entries.
- Output plain prose only, no JSON, no headings, no bullet points.

Topic entries:
{entries}
""".strip()


L3_STAGE_PORTRAIT_PROMPT = """
You are building a stage portrait of a person from several topic summaries.

The summaries below are the topics currently active in this person's life.
Write ONE paragraph in English describing the stable characteristics that hold
across them — enduring preferences, commitments, relationships and
circumstances — rather than any single topic in isolation.

Requirements:
- Describe what is stable across time, not one-off events.
- Attribute characteristics to the right person when several people appear.
- Do not invent anything that is not supported by the summaries.
- Output plain prose only, no JSON, no headings, no bullet points.

Topic summaries:
{summaries}
""".strip()


L4_PROFILE_PROMPT = """
You are maintaining the long-term profile of a person.

Inputs:
1. Stage portraits distilled from their memory so far.
2. A snapshot of the attributes they are currently recorded as having.

Write ONE paragraph in English that is the durable profile of this person:
who they are, what they consistently care about, and their present situation.

Requirements:
- Prefer durable facts over temporary ones; drop anything already outdated.
- The current attributes take precedence over older portraits when they
  disagree.
- Do not invent anything that is not supported by the inputs.
- Output plain prose only, no JSON, no headings, no bullet points.

Stage portraits:
{portraits}

Current attributes:
{attributes}
""".strip()


L2_MERGE_PROMPT = """
You are de-duplicating the summaries of one topic.

Several summaries below cover the same topic and overlap. Merge them into ONE
paragraph in English that keeps every distinct fact exactly once.

Requirements:
- Remove redundancy and contradiction, keeping the more recent statement when
  two summaries disagree.
- Preserve concrete specifics.
- Do not invent anything.
- Output plain prose only.

Summaries:
{summaries}
""".strip()


L4_MERGE_PROMPT = """
You are de-duplicating a person's profile.

Several profile paragraphs are given. Merge them into ONE paragraph in English
that is the current, non-redundant profile of this person.

Requirements:
- Remove repetition; when two profiles disagree, keep the more recent claim.
- Preserve concrete specifics.
- Do not invent anything.
- Output plain prose only.

Profiles:
{profiles}
""".strip()
