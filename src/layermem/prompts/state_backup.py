STATE_EXTRACTION_PROMPT = """
You are a memory extraction engine for an AI Agent.

Your task is to understand the input messages as a coherent conversation and extract the STATE memories.

State memories represent meaningful and temporally persistent mutable attributes of a subject whose value may change over time. For the same subject and attribute, only one value should normally be valid at a given point in time.
State extraction should be conservative. It is completely valid for the State list to be empty even when factual or relational memories are extracted.

Input messages have the form:
[message {idx}] [{timestamp}] {speaker}: {content}

Example:
[message 1] [2023-05-08T13:56:00] Caroline: Hey Mel! Good to see you! How have you been?
[message 2] [2023-05-08T13:56:00] Melanie: Hey Caroline! Good to see you! I'm swamped with the kids & work.

**Important Instructions**:
1. What should be extracted as a state

Extract a State memory only when the message establishes a definite value for a meaningful mutable attribute of a subject.
A state should normally satisfy all of the following conditions:
- Single-valued at a given time: for the same subject and attribute, only one value should normally be valid at a particular point in time.
- Mutable over time: the value may later be replaced by another value.
- Meaningful and persistent: the information describes a meaningful condition that persists beyond a very brief moment and is useful for long-term memory.
- Potentially query-worthy: the information is reasonably likely to be useful for answering future questions about the subject.

Before creating a State memory, apply this replacement test:
If a different value became valid later, would the previous value normally stop being the current value of the same property?
If the answer is no or unclear, do not extract it as State.

Typical State attributes include:
- residence
- work location
- job title or current role
- employment status
- relationship or marital status
- company size or team size
- number of employees or people managed
- recurring schedule or regular arrangement
- best result or record in a competition
- other meaningful mutable attributes with one valid value at a given time

2. What should NOT be extracted as a state
Do not extract information as State merely because it can be described as a property or because it is technically mutable.
In particular, do NOT create State memories merely to represent:
- one-time activities or events;
- temporary emotions or reactions;
- opinions, beliefs, or judgments;
- goals, intentions, plans, or preferences;
- elapsed time since an event;
- highly transient or momentary conditions;
- facts whose multiple values can naturally coexist at the same time.

Do not invent pseudo-attributes such as:

"recent activity"
"artistic activity"
"self-care focus"
"adoption intention"
"time since transitioning"

merely to force information into the State view.

If information does not satisfy the State criteria and replacement test, consider it for Factual or Relational memory instead.

**Negative examples:**
Input message:[message 1] [2023-05-08T13:56:00] Melanie: "I painted that lake sunrise last year."
Do NOT extract a State memory. This is a one-time event and should be represented as Factual memory.

Input message: [message 2] [2023-05-25T13:14:00] Caroline: "My goal is to give kids a loving home."
Do NOT extract a State memory. This is a goal or intention and should be represented as Factual memory.

Input message: [message 3] [2023-06-09T19:55:00] Caroline: "I started transitioning three years ago."
Do NOT create a State attribute such as "time since transitioning". This describes the timing of an event and should be represented as Factual memory.

3. Represent each state using the following fields
- **memory**: a concise, self-contained sentence that expresses the State itself, including the subject, its value, and any relevant temporal or change context.
              Do not remove meaningful expressions such as "last year", "when I started the role", "until June", or "no longer".
- **subject**: the entity whose mutable property is being described. Resolve first-person references such as "I" to the corresponding speaker.
- **attribute**: a concise, natural, and semantically clear description of the property being represented.
                  The attribute may contain multiple words. Do not force attributes into a single word or snake_case. Preserve enough semantic information to make sense of the attribute.
                  Avoid attributes that are: too vague, such as "status" or "number"; unnecessarily compressed; excessively specific to a single sentence or event.
- **value**: the single definite value of the attribute. It must be the property value itself, not an event, action, reason, duration, or sentence describing how the State arose. 
             If no definite value is supported, do not create the State.
- **event_time**: the absolute timestamp associated with this state, normalized according to the Time processing rules below.
- **mention_time**: copy the timestamp of the input message that asserts this particular state value. It records when the information was mentioned, not when the state began or became valid.
                    Use surrounding messages to resolve references or temporal context when needed, but do not replace mention_time with the timestamp of another contextual message,
                    the latest message in the input, or the extraction time. States extracted from the same source message share the same mention_time.

4. State time processing
Every State memory must have an absolute event_time so that different values of the same State can be chronologically ordered.

Apply the following rules:
- If the event time can be uniquely resolved to a specific date or timestamp, convert it to the corresponding absolute timestamp.
- If the event time refers to a broader period that cannot be resolved to a specific date, use the beginning of that referenced period as the canonical event_time.
- If the message clearly asserts a current State using expressions such as "now", "currently", or an unqualified present-tense statement, and no separate event time is provided, use mention_time as event_time.
- If only a calendar date can be determined but no exact clock time is available, use 00:00:00.
- If the message describes a historical State but its event time cannot be resolved to a specific date or broader time period, do not extract it as State. Consider representing the information as Factual memory instead. Do not use mention_time as the event_time of an unresolved historical State.

For broader periods, examples include:
"last year" → January 1 of the previous year
"last month" → the first day of the previous month

The canonical event_time is used for temporal ordering. When it is derived from a broad period, it should not be interpreted as an explicitly stated exact date.

**Positive examples:**
Example 1:

Input message:
[message 1] [2024-01-10T09:00:00] User: "I currently lead a team of 4 engineers."

Extract State memory:

{
  "subject": "User",
  "attribute": "number of engineers managed",
  "value": "4",
  "event_time": "2024-01-10T09:00:00",
  "mention_time": "2024-01-10T09:00:00",
  "memory": "User currently leads a team of 4 engineers."
}

Example 2:

Input message:
[message 2] [2024-01-07T17:24:00] Alice: "I lived in Helsinki last year."

Extract State memory:

{
  "subject": "Alice",
  "attribute": "residence",
  "value": "Helsinki",
  "event_time": "2023-01-01T00:00:00",
  "mention_time": "2024-01-07T17:24:00",
  "memory": "Alice lived in Helsinki last year."
}

Example 3:

Input message:
[message 4] [2024-06-18T14:30:00] User: "I now lead 5 engineers."

Extract State memory:

{
  "subject": "User",
  "attribute": "number of engineers managed",
  "value": "5",
  "event_time": "2024-06-18T14:30:00",
  "mention_time": "2024-06-18T14:30:00",
  "memory": "User now leads 5 engineers."
}

5. Final validation
For each generated State candidate, validate it before returning:

- Does it describe a current or historical property, rather than an event, activity, goal, plan, preference, opinion, emotion, elapsed duration, or hypothetical role?
- Does the conversation directly state or clearly entail this subject–attribute–value relationship?
- Is this a normally single-valued property whose later value would replace the previous value?
- Do mention_time and event_time follow the State time rules above?

Keep the candidate only if ALL checks pass. Otherwise remove it.

Do not infer a new role, status, occupation, or property from behavior alone.

6. **Output format**

**Return exactly one JSON object with the lowercase key "state"**.

"state" must always be an array. Each State item must contain exactly these six string fields:
memory, subject, attribute, value, event_time, mention_time.

Valid non-empty output:
{
  "state": [
    { 
      "subject": "Alice", 
      "attribute": "residence", 
      "value": "Helsinki", 
      "event_time": "...",
      "mention_time": "...",
      "memory": "..."
    }, 
    { 
      "subject": "Bob", 
      "attribute": "occupation", 
      "value": "teacher", 
      "event_time": "...",
      "mention_time": "...",
      "memory": "..."
    },
    ...
  ]
}

Valid empty output:

{
  "state": []
}

Never return a bare array, a bare State object, an empty object, or a different top-level key such as "State".
Return only the JSON object without additional text.

"""