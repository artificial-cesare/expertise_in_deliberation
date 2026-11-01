"""Broader evidential-marker detector (v2), by type of evidence.

The paper's pattern only catches formal citation ("according to", "studies show"), which is rare in
spoken assembly talk. v2 adds the ways people actually back claims in these transcripts.
Returns counts per type; evidential_markers_v2 = 100 * total matches / alphabetic tokens (as in v1).
"""
import re

TYPES = {
    # formal attribution to a source (v1 patterns plus common variants)
    "source": r"\baccording to\b|\b(the )?(data|evidence|research|figures|statistics|numbers|report|survey|studies|study)\s+(show|shows|showed|indicate|indicates|suggest|suggests|found|finds|says|said|tells us)\b"
              r"|\b(as|per) (the )?(report|guidelines|recommendations|figures|statistics)\b|\b(a|the|one) (study|survey|report|poll)\b"
              r"|\b(research|studies|evidence|statistics|data) (on|about|from|suggest|show)\b|\bthere is (no |some |good |strong |little )?evidence\b",
    # evidence heard during the assembly (presenters, experts, briefings)
    "testimony": r"\bwe (heard|were told|learned|learnt|were shown)\b|\b(the |our )?(experts?|speakers?|presenters?|witness(es)?|doctors?|professors?|panel)\b[^.]{0,40}\b(said|told us|showed|explained|mentioned|pointed out)\b"
                 r"|\bas (dr|professor|the chair|the speaker)\b|\bin (the|their) (presentation|briefing|evidence)\b",
    # quantitative evidence
    "quantity": r"\b\d+(\.\d+)?\s?(%|per ?cent|percent)\b|\b(one|two|three|four|five|half|a third|a quarter) (in|out of) (every )?(two|three|four|five|ten|\d+)\b"
                r"|\b\d{1,3}(,\d{3})+\b|\b(million|billion|thousand)\b|£\s?\d|€\s?\d|\$\s?\d",
    # first-hand experience offered as grounds
    "experience": r"\bin my (experience|job|work|area|community|family)\b|\bi know (from|of)\b|\bi('ve| have) (seen|worked|experienced|witnessed|dealt with)\b"
                  r"|\bwhen i (worked|was working)\b|\bas a (nurse|teacher|doctor|carer|parent|mother|father|farmer|worker|manager|social worker)\b|\bfrom (my|our) own experience\b",
    # concrete case or example offered as support
    "example": r"\bfor (example|instance)\b|\bsuch as\b|\bin (countries|places) like\b|\b(look|looking) at (what|how)\b|\bthe case of\b",
}
TYPES = {k: re.compile(v, re.I) for k, v in TYPES.items()}


def evidential_counts(text: str) -> dict:
    t = str(text)
    return {k: len(p.findall(t)) for k, p in TYPES.items()}


def evidential_v2(text: str, n_alpha_tokens: int) -> float:
    return 100.0 * sum(evidential_counts(text).values()) / max(n_alpha_tokens, 1)
