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
from concurrent.futures import ThreadPoolExecutor
import threading
import time

import numpy as np
import pandas as pd
import tiktoken
from scipy.stats import spearmanr
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'handcrafted_features'))

from filter_function import count_content_words
from llm_utils import count_tokens, extract_topic, score_expertise_with_llm

# Constants
CONTEXT_BUDGETS = [0, 100, 500, 1000]
SUBSET_SIZE = 200
RANDOM_SEED = 42
MODEL = "gpt-4o-mini"
# The ageing-population voting session belongs to the same Irish assembly as the Eighth Amendment session.
# It was scored from the 8th_amendment folder, so its published scores used the 8th_amendment topic string;
# the calibration reproduces that (TOPIC_FOLDER) and pools it with that assembly for stratification.
ASSEMBLIES = ["8th_amendment", "ageing_population", "gendereq", "scotland"]
ASSEMBLY_OF = {"ageing_population": "8th_amendment"}
TOPIC_FOLDER = {"ageing_population": "8th_amendment"}
CONTENT_WORD_THRESHOLD = 5  # default; --thresholds overrides. Always ">=" (substantial = at least N content words)

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


def load_assembly_data(base_dir: str, assembly: str, encoding, threshold: int = CONTENT_WORD_THRESHOLD) -> List[Utterance]:
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
                            content_word_count = int(float(row['num_non_function_words']))  # stored as '779.0'
                        except (ValueError, TypeError):
                            content_word_count = count_content_words(text)
                    else:
                        content_word_count = count_content_words(text)

                    is_substantial = content_word_count >= threshold

                    utterance_id = f"{assembly}_{csv_file}_{row_idx}"

                    utterances.append(Utterance(
                        utterance_id=utterance_id,
                        assembly=ASSEMBLY_OF.get(assembly, assembly),
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
        f"[{u.row_index + 1}] {u.speaker}: {u.text}"
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

    # The 20 preceding CSV rows, empty rows skipped (as in llm_expert_extract_new.get_context_window)
    context_utterances = [u for u in preceding if u.row_index >= target_utterance.row_index - 20]

    # Format
    context_lines = [
        f"[{u.row_index + 1}] {u.speaker}: {u.text}"
        for u in context_utterances
    ]

    context_string = "\n".join(context_lines)
    tokens_used = count_tokens(context_string, encoding)

    return context_string, tokens_used, len(context_utterances)


def run_calibration_experiment(
    base_dir: str,
    api_key: str,
    output_dir: str,
    model: str = MODEL,
    rate_limit_delay: float = 0.0,
    thresholds: Tuple[int, ...] = (CONTENT_WORD_THRESHOLD,),
    prompt: str = "paper",
    workers: int = 8,
) -> Dict[int, pd.DataFrame]:
    """
    Run the context size calibration once per content-word threshold (substantial = >= threshold).

    Each threshold gets its own stratified sample (same seed). An utterance sampled under several
    thresholds is scored once per context condition and the score is reused, so the runs differ only
    in which utterances they contain. Results go to <output_dir>/ge<threshold>/context_calibration_results.csv.
    """
    print("=" * 80)
    print("CONTEXT SIZE CALIBRATION EXPERIMENT")
    print("=" * 80)
    print(f"Model: {model} | prompt: {prompt} | thresholds (>=): {list(thresholds)}")
    print(f"Context conditions: {CONTEXT_BUDGETS + ['20_utterances']} | sample size: {SUBSET_SIZE} | seed: {RANDOM_SEED}")

    encoding = tiktoken.encoding_for_model(model)

    samples = {}
    utterances_by_file = None
    for thr in thresholds:
        all_utterances = []
        for assembly in ASSEMBLIES:
            all_utterances.extend(load_assembly_data(base_dir, assembly, encoding, threshold=thr))
        n_sub = sum(u.is_substantial for u in all_utterances)
        print(f"\n>= {thr} content words: {n_sub} substantial of {len(all_utterances)} non-empty utterances")
        samples[thr] = stratified_sample(all_utterances, SUBSET_SIZE, RANDOM_SEED)
        if utterances_by_file is None:  # context is built from all non-empty rows, whatever the threshold
            utterances_by_file = defaultdict(list)
            for u in all_utterances:
                utterances_by_file[f"{u.assembly}/{u.file_name}"].append(u)
            for key in utterances_by_file:
                utterances_by_file[key].sort(key=lambda u: u.row_index)

    unique = {u.utterance_id: u for smp in samples.values() for u in smp}
    conditions = CONTEXT_BUDGETS + ["20_utterances"]
    jobs = [(uid, c) for uid in unique for c in conditions]
    random.shuffle(jobs)  # interleave conditions and utterances
    print(f"\nUnique utterances: {len(unique)} | API calls: {len(jobs)} | workers: {workers}")

    cache: Dict[Tuple[str, str], dict] = {}
    lock = threading.Lock()
    done = [0]

    def score_one(job):
        uid, condition = job
        utt = unique[uid]
        all_in_file = utterances_by_file[f"{utt.assembly}/{utt.file_name}"]
        if condition == "20_utterances":
            context, actual_tokens, n_utts = build_context_20_utterances(utt, all_in_file, encoding)
        else:
            context, actual_tokens, n_utts = build_context(utt, all_in_file, condition, encoding)
        topic = extract_topic(utt.assembly)
        score, justification = None, ""
        for attempt in range(4):
            score, justification = score_expertise_with_llm(utt.text, context, topic, api_key, model, prompt=prompt)
            if score is not None:
                break
            time.sleep(2 ** attempt * 5)
        if rate_limit_delay:
            time.sleep(rate_limit_delay)
        rec = {"budget_condition": condition, "actual_tokens_used": actual_tokens,
               "n_utterances_in_context": n_utts, "score": score, "justification": justification}
        with lock:
            cache[(uid, str(condition))] = rec
            done[0] += 1
            if done[0] % 100 == 0:
                print(f"  {done[0]}/{len(jobs)} scored")
        return rec

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(score_one, jobs))

    out = {}
    for thr, smp in samples.items():
        rows = []
        for utt in smp:
            for c in conditions:
                rows.append({
                    "utterance_id": utt.utterance_id, "assembly": utt.assembly, "file_name": utt.file_name,
                    "row_index": utt.row_index, "speaker": utt.speaker, "target_text": utt.text,
                    "target_token_count": utt.token_count, "target_content_words": utt.content_word_count,
                    **cache[(utt.utterance_id, str(c))]})
        df = pd.DataFrame(rows)
        d = os.path.join(output_dir, f"ge{thr}")
        os.makedirs(d, exist_ok=True)
        df.to_csv(os.path.join(d, "context_calibration_results.csv"), index=False)
        print(f"\n>= {thr}: {df.utterance_id.nunique()} utterances, {df.score.isna().sum()} failed calls -> {d}")
        out[thr] = df
    return out


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Context Size Calibration for Expertise Scoring")
    parser.add_argument('--base-dir', default='./data/assemblies/processed_files',
                        help='Base directory containing processed CSV files')
    parser.add_argument('--output-dir', default='./results/context_calibration',
                        help='Output directory; one ge<threshold>/ subfolder per threshold')
    parser.add_argument('--model', default=MODEL, help=f'OpenAI model to use (default: {MODEL})')
    parser.add_argument('--delay', type=float, default=0.0, help='Extra delay per call and worker (s)')
    parser.add_argument('--thresholds', type=int, nargs='+', default=[CONTENT_WORD_THRESHOLD],
                        help='Content-word thresholds; substantial = >= threshold (e.g. 5 10 20)')
    parser.add_argument('--prompt', default='paper', choices=['paper', 'lived_experience'],
                        help='System prompt: "paper" = Appendix A, the prompt behind the published scores')
    parser.add_argument('--workers', type=int, default=8, help='Parallel API calls')
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    run_calibration_experiment(base_dir=args.base_dir, api_key=api_key, output_dir=args.output_dir,
                               model=args.model, rate_limit_delay=args.delay, thresholds=tuple(args.thresholds),
                               prompt=args.prompt, workers=args.workers)
    print("\nNext: python code/LLM_expertise/analyze_calibration.py --results-file <output-dir>/ge<N>/context_calibration_results.csv --output-dir <output-dir>/ge<N>")


if __name__ == "__main__":
    main()
