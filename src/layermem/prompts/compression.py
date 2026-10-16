"""Online L1 -> L2 -> L3 -> L4 prompts."""

FACTUAL_L2_PROMPT = """
Compress the supplied L1 factual memories about one semantic topic into ONE concise standalone English paragraph.
Combine duplicate propositions but preserve distinct events, people, places, names, dates, quantities, outcomes, negation, uncertainty, and whether a claim is an event, plan, intention, preference, or opinion. Inputs are not chronological: use only explicit time information and never invent dates, order, currentness, habits, or personality traits. Repeated descriptions of one event do not prove repetition. Use only the supplied memories; output plain prose with no JSON, heading, bullets, or commentary.

L1 factual memories:
{entries}
""".strip()

RELATIONAL_L2_PROMPT = """
Compress the supplied L1 relational memories about one semantic topic into ONE concise standalone English paragraph.
Preserve who acted or spoke, to whom, what was expressed or done, and the concrete context. Keep direction, asymmetry, responses, boundaries, conflict, emotion, negation, uncertainty, and stated relationship labels. Merge duplicate accounts of one interaction, but keep distinct people, pairs, and interactions separate. Do not infer friendship, romance, kinship, trust, reciprocity, motives, or lasting closeness from politeness or one exchange. Inputs are not chronological; use only explicit time information. Use only the supplied memories; output plain prose with no JSON, heading, bullets, or commentary.

L1 relational memories:
{entries}
""".strip()

FACTUAL_L3_PROMPT = """
Synthesize the supplied L2 factual summaries into ONE concise standalone English L3 stage portrait for this semantic topic.
Connect supported activities, goals, preferences, commitments, progress, milestones, and changes while retaining defining names, dates, quantities, and outcomes. Describe a recurring pattern only when explicitly supported by distinct occurrences; overlapping summaries may describe one event. Keep people separate, plans distinct from achievements, opinions attributed, and uncertainty or negation intact. Inputs are not chronological, so do not infer order, permanence, or an exact date. Do not invent habits, personality, motives, or facts beyond the summaries. Output plain prose with no JSON, heading, bullets, or commentary.

L2 factual summaries:
{summaries}
""".strip()

RELATIONAL_L3_PROMPT = """
Synthesize the supplied L2 relational summaries into ONE concise standalone English relationship stage portrait.
Describe supported interaction patterns: who encourages, advises, praises, confides in, disagrees with, or responds to whom and around which concerns. Keep people and pairs separate; preserve direction, asymmetry, responses, conflict, boundaries, changes, uncertainty, and offers versus completed actions. Repeated summaries of one interaction do not prove a persistent bond. Do not invent relationship labels, reciprocity, motives, emotions, trust, reconciliation, or permanence. Inputs are not chronological; use only explicit time information. Output plain prose with no JSON, heading, bullets, or commentary.

L2 relational summaries:
{summaries}
""".strip()

FACTUAL_L4_PROMPT = """
Distill the supplied L3 factual portraits into ONE concise standalone English long-term factual profile for this semantic topic, not a complete biography.
Prioritize well-supported enduring interests, commitments, preferences, capabilities, and significant experiences. Consolidate overlap, but repeated wording about one event is not evidence of a lifelong trait. Keep people separate, plans distinct from achievements, opinions attributed, and uncertainty or negation intact. Preserve historical dates and changes; do not infer chronology, permanence, or currentness from input order. Use only the supplied portraits; output plain prose with no JSON, heading, bullets, or commentary.

L3 factual portraits:
{portraits}

""".strip()

RELATIONAL_L4_PROMPT = """
Distill the supplied L3 relationship portraits into ONE concise standalone English long-term relational profile for this semantic topic.
Center specific people and relationship dynamics: supported enduring connections, characteristic exchanges, support, shared concerns, tensions, and boundaries. Keep each pair and direction distinct, including asymmetric or mixed evidence. Repeated summaries of one interaction do not prove a persistent bond. Do not invent relationship labels, motives, emotional dependence, trust, reciprocity, reconciliation, or permanence. Preserve stated dates, changes, uncertainty, and negation; input order is not chronological. Include biographical facts only when needed to explain an interaction. Use only supplied portraits; output plain prose with no JSON, heading, bullets, or commentary.

L3 relational portraits:
{portraits}
""".strip()

L2_PROMPTS = {"factual": FACTUAL_L2_PROMPT, "relational": RELATIONAL_L2_PROMPT}
L3_PROMPTS = {"factual": FACTUAL_L3_PROMPT, "relational": RELATIONAL_L3_PROMPT}
L4_PROMPTS = {"factual": FACTUAL_L4_PROMPT, "relational": RELATIONAL_L4_PROMPT}
