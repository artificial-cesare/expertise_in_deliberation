"""
Canonical topic mappings shared across all analysis scripts.

Two distinct uses live here:

1. ASSEMBLY_TOPICS / EXPERT_TOPICS  — short display labels used as DataFrame
   'topic' column values in correlation, regression, and summary scripts.

2. _LLM_TOPIC_PROMPTS / extract_topic()  — long-form descriptions sent to the
   LLM in expertise-scoring prompts (llm_utils.py and callees).
   These are intentionally different from the display labels.
"""

from pathlib import Path
from typing import Dict

# ---------------------------------------------------------------------------
# Short display labels (used in DataFrames, plots, LaTeX tables)
# ---------------------------------------------------------------------------

ASSEMBLY_TOPICS: Dict[str, str] = {
    "8th_amendment": "Constitutional Reform",
    "gendereq": "Gender Equality",
    "scotland": "Public Policy",
}

EXPERT_TOPICS: Dict[str, str] = {
    "constitution": "Constitutional Reform",
    "future": "Public Policy",
    "gender": "Gender Equality",
}

# ---------------------------------------------------------------------------
# Long-form LLM prompt descriptions (used only in expertise scoring)
# ---------------------------------------------------------------------------

_LLM_TOPIC_PROMPTS: Dict[str, str] = {
    "8th_amendment": "the 8th Amendment and reproductive rights in Ireland",
    "gendereq": "gender equality and related social policies",
    "gender": "gender equality and related social policies",
    "scotland": "proposals for Scottish governance and policy",
    "constitution": "constitutional reform and parliamentary governance",
    "future": "public policy and future governance",
}


def extract_topic(identifier: str) -> str:
    """
    Return the LLM-appropriate topic description for an assembly/expert folder.

    Accepts either a bare folder name ('gendereq', '8th_amendment', …) or a
    full CSV file path; Path(...).parts handles both identically.

    Covers all six folder types:
      Assemblies: 8th_amendment, gendereq, scotland
      Experts:    constitution, future, gender

    Args:
        identifier: Folder name string or full file path.

    Returns:
        Long-form topic description suitable for LLM prompts.

    Raises:
        ValueError: If the topic cannot be inferred from the identifier.
    """
    for part in reversed(Path(identifier).parts):
        p = part.lower()
        if "scotland" in p:
            return _LLM_TOPIC_PROMPTS["scotland"]
        if "gendereq" in p:
            return _LLM_TOPIC_PROMPTS["gendereq"]
        if "8th" in p or "amendment" in p:
            return _LLM_TOPIC_PROMPTS["8th_amendment"]
        if "constitution" in p:
            return _LLM_TOPIC_PROMPTS["constitution"]
        if "future" in p:
            return _LLM_TOPIC_PROMPTS["future"]
        if "gender" in p:
            return _LLM_TOPIC_PROMPTS["gender"]
    raise ValueError(f"Could not extract topic from: {identifier!r}")
