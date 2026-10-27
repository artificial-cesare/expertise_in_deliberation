#!/usr/bin/env python3
"""
Context Size Calibration for Expertise Scoring
Tests sensitivity of LLM expertise scores to varying amounts of context.
Implements stratified sampling, multiple token budgets, and statistical analysis.
"""

import os
import csv
import sys
import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import time

import numpy as np
import pandas as pd
import tiktoken
from openai import OpenAI
from scipy.stats import spearmanr
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Import the existing scoring function
sys.path.append(os.path.dirname(__file__))

# Constants
CONTEXT_BUDGETS = [0, 500, 1000, 1500, 2000]
SUBSET_SIZE = 200
RANDOM_SEED = 42
MODEL = "gpt-4o-mini"
ASSEMBLIES = ["8th_amendment", "gendereq", "scotland"]
CONTENT_WORD_THRESHOLD = 5

# Set random seeds for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


@dataclass
class Utterance:
    """Represents a single utterance from the assembly transcripts."""
    utterance_id: str
    assembly: str
    file_name: str
    row_index: int
    speaker: str
    text: str
    token_count: int
    content_word_count: int
    is_substantial: bool
    date: str = ""


def count_tokens(text: str, encoding) -> int:
    """Count tokens in text using tiktoken."""
    try:
        return len(encoding.encode(text))
    except Exception as e:
        print(f"Warning: Token counting failed: {e}")
        return len(text.split()) * 1.3  # Rough estimate


def count_content_words(text: str) -> int:
    """
    Count non-function words (content words) in text.
    This should match the logic in your existing code.
    """
    # Common English function words
    function_words = {
        'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'until',
        'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
        'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
        'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
        'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
        'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
        'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
        's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'i', 'you',
        'he', 'she', 'it', 'we', 'they', 'them', 'their', 'what', 'which', 'who',
        'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
        'had', 'do', 'does', 'did', 'would', 'could', 'should', 'may', 'might',
        'must', 'shall'
    }

    words = text.lower().split()
    content_words = [w for w in words if w not in function_words and len(w) > 0]
    return len(content_words)


def load_assembly_data(base_dir: str, assembly: str, encoding) -> List[Utterance]:
    """
    Load all CSV files for a given assembly and return list of Utterance objects.

    Args:
        base_dir: Base directory containing processed files
        assembly: Assembly name (8th_amendment, gendereq, scotland)
        encoding: Tiktoken encoding for token counting

    Returns:
        List of Utterance objects
    """
    assembly_path = os.path.join(base_dir, assembly)
    csv_files = sorted([f for f in os.listdir(assembly_path) if f.endswith('.csv')])

    utterances = []

    for csv_file in csv_files:
        csv_path = os.path.join(assembly_path, csv_file)

        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                for row_idx, row in enumerate(reader):
                    text = row.get('text', '').strip()
                    if not text:
                        continue

                    # Count tokens and content words
                    token_count = count_tokens(text, encoding)

                    # Use existing content word count if available, otherwise compute
                    if 'num_non_function_words' in row and row['num_non_function_words']:
                        try:
                            content_word_count = int(row['num_non_function_words'])
                        except (ValueError, TypeError):
                            content_word_count = count_content_words(text)
                    else:
                        content_word_count = count_content_words(text)

                    is_substantial = content_word_count >= CONTENT_WORD_THRESHOLD

                    utterance_id = f"{assembly}_{csv_file}_{row_idx}"

                    utterances.append(Utterance(
                        utterance_id=utterance_id,
                        assembly=assembly,
                        file_name=csv_file,
                        row_index=row_idx,
                        speaker=row.get('speaker', 'Unknown'),
                        text=text,
                        token_count=token_count,
                        content_word_count=content_word_count,
                        is_substantial=is_substantial,
                        date=row.get('date', '')
                    ))

        except Exception as e:
            print(f"Warning: Error loading {csv_path}: {e}")
            continue

    return utterances


def stratified_sample(
    all_utterances: List[Utterance],
    sample_size: int,
    random_seed: int
) -> List[Utterance]:
    """
    Perform stratified sampling across assemblies and temporal positions.

    Strategy:
    - Proportional allocation by assembly
    - Within each assembly, sample equally from early/mid/late thirds
    - Oversample long utterances (>300 tokens) slightly

    Args:
        all_utterances: List of all substantial utterances
        sample_size: Target sample size
        random_seed: Random seed for reproducibility

    Returns:
        List of sampled utterances
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Group by assembly
    by_assembly = defaultdict(list)
    for u in all_utterances:
        if u.is_substantial:
            by_assembly[u.assembly].append(u)

    total_substantial = sum(len(utts) for utts in by_assembly.values())

    sampled = []

    for assembly, utterances in by_assembly.items():
        # Proportional allocation
        n_from_assembly = round(sample_size * (len(utterances) / total_substantial))

        # Sort by row_index (temporal order) within each file
        utterances_sorted = sorted(utterances, key=lambda u: (u.file_name, u.row_index))

        # Split into thirds (early, mid, late)
        n_per_third = n_from_assembly // 3
        remainder = n_from_assembly % 3

        third_size = len(utterances_sorted) // 3
        thirds = [
            utterances_sorted[:third_size],
            utterances_sorted[third_size:2*third_size],
            utterances_sorted[2*third_size:]
        ]

        # Sample from each third
        for i, third in enumerate(thirds):
            # Add 1 to first thirds if there's a remainder
            n_to_sample = n_per_third + (1 if i < remainder else 0)

            if n_to_sample > len(third):
                n_to_sample = len(third)

            # Slightly oversample long utterances (>300 tokens)
            long_utterances = [u for u in third if u.token_count > 300]
            short_utterances = [u for u in third if u.token_count <= 300]

            # Take 30% from long if available, rest from all
            n_long = min(int(n_to_sample * 0.3), len(long_utterances))
            n_short = n_to_sample - n_long

            sampled_third = []
            if n_long > 0:
                sampled_third.extend(random.sample(long_utterances, n_long))
            if n_short > 0:
                # Sample from remaining (could be short or long that weren't picked)
                remaining = [u for u in third if u not in sampled_third]
                if len(remaining) >= n_short:
                    sampled_third.extend(random.sample(remaining, n_short))
                else:
                    sampled_third.extend(remaining)

            sampled.extend(sampled_third)

    print(f"\nStratified Sampling Summary:")
    print(f"Total substantial utterances: {total_substantial}")
    print(f"Sampled: {len(sampled)}")

    for assembly in by_assembly.keys():
        assembly_sampled = [u for u in sampled if u.assembly == assembly]
        print(f"  {assembly}: {len(assembly_sampled)} sampled")

    return sampled


def build_context(
    target_utterance: Utterance,
    all_utterances_in_file: List[Utterance],
    token_budget: int,
    encoding
) -> Tuple[str, int, int]:
    """
    Build context window by walking backwards from target, respecting token budget.

    Rules:
    - Never truncate an utterance mid-text
    - Always include at least one preceding utterance (even if it exceeds budget)
    - If budget is 0, return empty string
    - Skip the target utterance itself

    Args:
        target_utterance: The utterance to score
        all_utterances_in_file: All utterances from the same file, sorted by row_index
        token_budget: Maximum tokens for context
        encoding: Tiktoken encoding

    Returns:
        Tuple of (context_string, actual_tokens_used, n_utterances_in_context)
    """
    if token_budget == 0:
        return "", 0, 0

    # Get all utterances before the target in the same file
    preceding = [
        u for u in all_utterances_in_file
        if u.file_name == target_utterance.file_name
        and u.row_index < target_utterance.row_index
    ]

    if not preceding:
        return "", 0, 0

    # Sort by row_index (chronological)
    preceding = sorted(preceding, key=lambda u: u.row_index, reverse=True)

    context_utterances = []
    tokens_used = 0

    for i, utt in enumerate(preceding):
        utt_tokens = count_tokens(utt.text, encoding)

        # First utterance: always include
        if i == 0:
            context_utterances.append(utt)
            tokens_used += utt_tokens
            continue

        # Subsequent: check budget
        if tokens_used + utt_tokens > token_budget:
            break

        context_utterances.append(utt)
        tokens_used += utt_tokens

    # Reverse to restore chronological order
    context_utterances.reverse()

    # Format as expected by LLM
    context_lines = [
        f"[{u.row_index}] {u.speaker}: {u.text}"
        for u in context_utterances
    ]

    context_string = "\n".join(context_lines)

    return context_string, tokens_used, len(context_utterances)


def build_context_20_utterances(
    target_utterance: Utterance,
    all_utterances_in_file: List[Utterance],
    encoding
) -> Tuple[str, int, int]:
    """
    Build context using the existing 20-utterance approach (baseline).

    Args:
        target_utterance: The utterance to score
        all_utterances_in_file: All utterances from the same file
        encoding: Tiktoken encoding

    Returns:
        Tuple of (context_string, actual_tokens_used, n_utterances=20)
    """
    # Get preceding utterances from same file
    preceding = [
        u for u in all_utterances_in_file
        if u.file_name == target_utterance.file_name
        and u.row_index < target_utterance.row_index
    ]

    if not preceding:
        return "", 0, 0

    # Sort by row_index
    preceding = sorted(preceding, key=lambda u: u.row_index)

    # Take last 20 (or fewer if not available)
    context_utterances = preceding[-20:]

    # Format
    context_lines = [
        f"[{u.row_index}] {u.speaker}: {u.text}"
        for u in context_utterances
    ]

    context_string = "\n".join(context_lines)
    tokens_used = count_tokens(context_string, encoding)

    return context_string, tokens_used, len(context_utterances)


def score_expertise_with_llm(
    contribution: str,
    context: str,
    topic: str,
    api_key: str,
    model: str = "gpt-4o-mini"
) -> Tuple[Optional[int], str]:
    """
    Score expertise level using LLM.
    This uses the EXISTING prompt from your codebase.

    Args:
        contribution: The speech act to evaluate
        context: Previous speech acts for context
        topic: Topic of deliberation
        api_key: OpenAI API key
        model: Model to use

    Returns:
        Tuple of (rating 1-5, justification string)
    """
    import re

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
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=150
        )

        result = response.choices[0].message.content.strip()

        # Parse rating
        rating = None
        justification = result

        rating_match = re.search(r'Rating:\s*(\d)', result)
        if rating_match:
            rating = int(rating_match.group(1))

        # Extract justification
        just_match = re.search(r'Justification:\s*(.+)', result, re.IGNORECASE)
        if just_match:
            justification = just_match.group(1).strip()

        # Validate rating
        if rating is None or rating < 1 or rating > 5:
            print(f"  Warning: Invalid rating parsed: {result}")
            rating = 3  # Default to neutral

        return rating, justification

    except Exception as e:
        print(f"  API call failed: {e}")
        return None, str(e)


def extract_topic_from_assembly(assembly: str) -> str:
    """Extract topic description from assembly name."""
    if 'scotland' in assembly.lower():
        return "proposals for Scottish governance and policy"
    elif 'gender' in assembly.lower() or 'gendereq' in assembly.lower():
        return "gender equality and related social policies"
    elif '8th' in assembly.lower() or 'amendment' in assembly.lower():
        return "the 8th Amendment and reproductive rights in Ireland"
    return "policy and governance matters"


def run_calibration_experiment(
    base_dir: str,
    api_key: str,
    output_dir: str,
    model: str = MODEL,
    rate_limit_delay: float = 0.5
) -> pd.DataFrame:
    """
    Run the full context size calibration experiment.

    Args:
        base_dir: Base directory containing processed CSV files
        api_key: OpenAI API key
        output_dir: Directory to save results
        model: LLM model to use
        rate_limit_delay: Delay between API calls

    Returns:
        DataFrame with all results
    """
    print("="*80)
    print("CONTEXT SIZE CALIBRATION EXPERIMENT")
    print("="*80)
    print(f"Model: {model}")
    print(f"Context budgets: {CONTEXT_BUDGETS}")
    print(f"Sample size: {SUBSET_SIZE}")
    print(f"Random seed: {RANDOM_SEED}")
    print("="*80)

    # Initialize tokenizer
    encoding = tiktoken.encoding_for_model(model)

    # 1. Load all data
    print("\n[1/5] Loading assembly data...")
    all_utterances = []
    for assembly in ASSEMBLIES:
        print(f"  Loading {assembly}...")
        utts = load_assembly_data(base_dir, assembly, encoding)
        all_utterances.extend(utts)
        print(f"    Loaded {len(utts)} utterances")

    print(f"\nTotal utterances loaded: {len(all_utterances)}")
    substantial = [u for u in all_utterances if u.is_substantial]
    print(f"Substantial utterances (>={CONTENT_WORD_THRESHOLD} content words): {len(substantial)}")

    # 2. Stratified sampling
    print("\n[2/5] Performing stratified sampling...")
    sampled_utterances = stratified_sample(all_utterances, SUBSET_SIZE, RANDOM_SEED)

    # 3. Build lookup for fast context retrieval
    print("\n[3/5] Organizing utterances by file...")
    utterances_by_file = defaultdict(list)
    for u in all_utterances:
        utterances_by_file[f"{u.assembly}/{u.file_name}"].append(u)

    # Sort each file's utterances by row_index
    for key in utterances_by_file:
        utterances_by_file[key] = sorted(
            utterances_by_file[key],
            key=lambda u: u.row_index
        )

    # 4. Run LLM scoring across all conditions
    print("\n[4/5] Running LLM scoring...")
    print(f"Total API calls: {len(sampled_utterances) * (len(CONTEXT_BUDGETS) + 1)}")

    results = []

    # Randomize order to avoid API caching effects
    scoring_order = list(range(len(sampled_utterances)))
    random.shuffle(scoring_order)

    for i, idx in enumerate(scoring_order):
        utt = sampled_utterances[idx]

        print(f"\nScoring utterance {i+1}/{len(sampled_utterances)}: {utt.utterance_id}")
        print(f"  Assembly: {utt.assembly}, Length: {utt.token_count} tokens")

        file_key = f"{utt.assembly}/{utt.file_name}"
        all_in_file = utterances_by_file[file_key]
        topic = extract_topic_from_assembly(utt.assembly)

        # Randomize condition order for this utterance
        conditions = CONTEXT_BUDGETS + ["20_utterances"]
        random.shuffle(conditions)

        for condition in conditions:
            if condition == "20_utterances":
                context, actual_tokens, n_utts = build_context_20_utterances(
                    utt, all_in_file, encoding
                )
                budget_label = "20_utterances"
            else:
                context, actual_tokens, n_utts = build_context(
                    utt, all_in_file, condition, encoding
                )
                budget_label = condition

            print(f"  Condition: {budget_label} (actual: {actual_tokens} tokens, {n_utts} utts)...", end=' ')

            score, justification = score_expertise_with_llm(
                utt.text,
                context,
                topic,
                api_key,
                model
            )

            if score is not None:
                print(f"Score: {score}/5")
            else:
                print(f"FAILED: {justification}")

            results.append({
                "utterance_id": utt.utterance_id,
                "assembly": utt.assembly,
                "file_name": utt.file_name,
                "row_index": utt.row_index,
                "speaker": utt.speaker,
                "target_text": utt.text,
                "target_token_count": utt.token_count,
                "target_content_words": utt.content_word_count,
                "budget_condition": budget_label,
                "actual_tokens_used": actual_tokens,
                "n_utterances_in_context": n_utts,
                "score": score,
                "justification": justification
            })

            # Rate limiting
            time.sleep(rate_limit_delay)

    # 5. Save results
    print("\n[5/5] Saving results...")
    os.makedirs(output_dir, exist_ok=True)

    results_df = pd.DataFrame(results)
    results_path = os.path.join(output_dir, "context_calibration_results.csv")
    results_df.to_csv(results_path, index=False)

    print(f"\n✓ Results saved to: {results_path}")
    print(f"  Total rows: {len(results_df)}")

    return results_df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Context Size Calibration for Expertise Scoring"
    )
    parser.add_argument(
        '--base-dir',
        default='./data/assemblies/processed_files',
        help='Base directory containing processed CSV files'
    )
    parser.add_argument(
        '--output-dir',
        default='./results/context_calibration',
        help='Output directory for results and plots'
    )
    parser.add_argument(
        '--model',
        default=MODEL,
        help=f'OpenAI model to use (default: {MODEL})'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.5,
        help='Delay between API calls in seconds (default: 0.5)'
    )

    args = parser.parse_args()

    # Get API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    # Run experiment
    results_df = run_calibration_experiment(
        base_dir=args.base_dir,
        api_key=api_key,
        output_dir=args.output_dir,
        model=args.model,
        rate_limit_delay=args.delay
    )

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print("\nNext step: Run analysis with:")
    print(f"  python analyze_calibration.py --results-file {args.output_dir}/context_calibration_results.csv")


if __name__ == "__main__":
    main()
