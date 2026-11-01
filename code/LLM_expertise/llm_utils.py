#!/usr/bin/env python3
"""
Shared LLM utilities for expertise scoring.
Used by llm_expert_extract_new.py and context_size_calibration.py.
"""

import re
import sys
from pathlib import Path
from typing import Optional, Tuple

from openai import OpenAI

# Ensure code/ is on sys.path so shared.topic_maps is importable regardless
# of which directory this module is imported from.
_CODE_DIR = str(Path(__file__).resolve().parent.parent)
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from shared.topic_maps import extract_topic  # noqa: E402 (path set above)


def count_tokens(text: str, encoding) -> int:
    """Count tokens in text using tiktoken."""
    try:
        return len(encoding.encode(text))
    except Exception as e:
        print(f"Warning: Token counting failed: {e}")
        return int(len(text.split()) * 1.3)  # Rough estimate


def score_expertise_with_llm(
    contribution: str,
    context: str,
    topic: str,
    api_key: str,
    model: str = "gpt-4o-mini",
) -> Tuple[Optional[int], str]:
    """
    Score expertise level of a contribution using an LLM.

    Args:
        contribution: The speech act to evaluate.
        context: Previous speech acts for context.
        topic: Topic of deliberation.
        api_key: OpenAI API key.
        model: Model to use.

    Returns:
        Tuple of (rating 1-5, justification string).
        Rating is None if the API call failed.
    """
    client = OpenAI(api_key=api_key)

    system_prompt = """You are a political innovations expert assisting a research study on deliberative communication.
Your task is to assess how much a given speech act demonstrates domain expertise on the topic under discussion.
Refer to previous context to assess how the current speech act signifies expertise in the current discussion topics.

Definition:
"Expertise" here refers to the expression of relevant and well-grounded knowledge about the topic, which may be demonstrated through technical understanding, analytical reasoning, or situated experiential knowledge.

Expertise can be expressed in different forms. In deliberative settings, participants may legitimately draw on professional knowledge, formal training, policy familiarity, or lived experience and practical engagement with the issue.
Statements grounded in personal or lived experience should not be penalised, provided they are used to illuminate, generalise, or reason about the issue rather than to express purely subjective preference or emotion.

Evaluate one speaker's statement in the context of an ongoing deliberation.
The input may include minor transcription errors (from speech-to-text).

When deciding, rely on well-established linguistic indicators of expertise, such as:

- Use of domain-specific or technical vocabulary, where appropriate (precision, correct terminology);

- Structured reasoning and inferential coherence, including explicit or implicit logical relations such as:

    - Cause–effect: because, since, therefore, thus, hence.

    - Entailment or consequence: it follows that, as a consequence, this implies that.

    - Conditional inference / modus ponens: if … then …, given that … we can conclude ….

    - Syllogistic or categorical reasoning: all X are Y; this is X; therefore ….

- Evidence or grounding markers, including references to data, institutions, policy mechanisms, or systematically described personal experience (e.g. repeated exposure, first-hand involvement, comparison across cases);

- Epistemic calibration (balanced use of hedges like "may", "might", "suggests", showing awareness of uncertainty);

- Syntactic and lexical complexity where it serves explanation or reasoning clarity;

- Analytical use of experience (personal or lived experience used to support general claims, causal reasoning, or problem understanding);

- Relevance to the current topic (staying on-topic with informed contributions).

Do not penalise statements solely because they rely on lived experience or personal involvement.
However, distinguish analytical experiential reasoning from statements that are purely anecdotal, emotive, or preference-based without broader reasoning.

Avoid judging based on fluency, confidence tone, or opinion alignment.
Focus purely on linguistic, reasoning and content cues that signal expertise or informed understanding.

You must return a rating on a 1–5 Likert scale:

1 — Strongly Disagree: clearly non-expert (uninformed, vague, purely emotive, unreasoned anecdote, or incorrect)
2 — Disagree: likely non-expert (limited reasoning, weak grounding, minimal analytical content)
3 — Undecided: ambiguous or generic statement (no clear expert or non-expert cues)
4 — Agree: likely expert (clear reasoning, grounded claims, or analytically used experience)
5 — Strongly Agree: clearly expert (precise terminology and/or well-articulated experiential insight, coherent reasoning, strong grounding)

Give a one-sentence justification referencing linguistic or reasoning cues.

Format your answer exactly as:
Rating: x/5. Justification: [One concise sentence]."""

    user_prompt = f"""Topic of deliberation: {topic}

Here is the transcript of the deliberation from before the statement which is to be evaluated. This serves as the context for the evaluation:

{context}

Here is the statement which you have to evaluate:

{contribution}

Please evaluate whether the speaker demonstrates expertise on this topic."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=150,
        )

        result = response.choices[0].message.content.strip()

        rating = None
        justification = ""

        # Try strict pattern first, then fallback
        rating_match = re.search(r'Rating:\s*(\d)/5', result, re.IGNORECASE)
        if not rating_match:
            rating_match = re.search(r'(\d)/5', result)

        if rating_match:
            rating = int(rating_match.group(1))

        just_match = re.search(r'Justification:\s*(.+)', result, re.IGNORECASE)
        if just_match:
            justification = just_match.group(1).strip()
        else:
            justification = result

        if rating is None or rating < 1 or rating > 5:
            print(f"  Warning: Invalid rating parsed from response: {result}")
            rating = 3  # Default to neutral

        return rating, justification

    except Exception as e:
        print(f"  API call failed: {str(e)}")
        return None, str(e)
