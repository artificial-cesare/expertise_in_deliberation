#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sample speech acts with context from Citizens' Assemblies and expert transcripts.
Stratified by: group, topic, and expertise_llm_score.
"""

import sys
import json
import csv
import random
import argparse
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.topic_maps import ASSEMBLY_TOPICS as ASSEMBLIES_TOPICS, EXPERT_TOPICS as EXPERTS_TOPICS

# ============================================================================
# Configuration
# ============================================================================
SEED = 42
SAMPLE_SIZE = 20
CONTEXT_WINDOW = 20

ASSEMBLIES_DIR = Path("data/assemblies/processed_files")
EXPERTS_DIR = Path("data/assemblies/processed_files/experts")


OUTPUT_FILE = Path("data/validation/sampled_speeches.json")

# ============================================================================
# Helper Functions
# ============================================================================
def get_text(row: Dict) -> str:
    """Extract text from row."""
    return row.get('text', row.get('comment', '')).strip()

def get_speaker(row: Dict) -> str:
    """Extract speaker from row."""
    return row.get('speaker', row.get('name', row.get('Speaker', 'Unknown')))

def is_valid_score(score_str: str) -> bool:
    """Check if expertise score is valid integer 1-5."""
    try:
        score = int(score_str.strip())
        return 1 <= score <= 5
    except (ValueError, AttributeError):
        return False

# ============================================================================
# Main Logic
# ============================================================================
def load_speeches(base_dir: Path, topic_map: Dict[str, str], group: str) -> List[Dict]:
    """Load all valid speeches with context."""
    all_speeches = []
    
    for folder_name, topic in topic_map.items():
        folder_path = base_dir / folder_name
        if not folder_path.exists():
            continue
        
        for csv_file in folder_path.glob("*.csv"):
            # Load all rows from file
            with open(csv_file, 'r', encoding='utf-8') as f:
                rows = list(csv.DictReader(f))
            
            # Process each row
            for idx, row in enumerate(rows):
                # Validate expertise score
                score_str = row.get('expertise_llm_score', '')
                if not is_valid_score(score_str):
                    continue
                
                text = get_text(row)
                if not text:
                    continue
                
                # Build context from previous speeches
                context = []
                for i in range(max(0, idx - CONTEXT_WINDOW), idx):
                    ctx_text = get_text(rows[i])
                    if ctx_text:
                        context.append(f"{get_speaker(rows[i])}: {ctx_text}")
                
                all_speeches.append({
                    'speech_id': f"{group}/{folder_name}/{csv_file.name}/row_{idx}",
                    'text': text,
                    'speaker': get_speaker(row),
                    'context': context,
                    'group': group,
                    'topic': topic,
                    'expertise_score': int(score_str),
                    'expertise_justification': row.get('expertise_justification', '')
                })
    
    return all_speeches

def stratified_sample(speeches: List[Dict], n: int, seed: int) -> List[Dict]:
    """Stratified sampling by (group, topic, score)."""
    random.seed(seed)
    
    # Group by strata
    strata = defaultdict(list)
    for speech in speeches:
        key = (speech['group'], speech['topic'], speech['expertise_score'])
        strata[key].append(speech)
    
    # Proportional allocation
    total = len(speeches)
    samples_per_stratum = {
        key: max(1, round(len(items) / total * n))
        for key, items in strata.items()
    }
    
    # FIX: This loop can get stuck if all strata have count=1
    # Changed to also adjust upward if under budget
    total_allocated = sum(samples_per_stratum.values())
    
    # Adjust if over budget - only reduce strata with >1
    while total_allocated > n:
        reducible = {k: v for k, v in samples_per_stratum.items() if v > 1}
        if not reducible:
            break  # Can't reduce further
        largest = max(reducible, key=reducible.get)
        samples_per_stratum[largest] -= 1
        total_allocated -= 1
    
    # Sample from each stratum
    sampled = []
    for key, count in samples_per_stratum.items():
        items = strata[key]
        sampled.extend(random.sample(items, min(count, len(items))))
    
    random.shuffle(sampled)
    return sampled[:n]

# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Sample speeches with context for annotation")
    parser.add_argument("--data-dir", type=Path, default=ASSEMBLIES_DIR,
                        help="Processed files directory (experts read from its experts/ subdir)")
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE,
                        help="Output JSON file path")
    args = parser.parse_args()

    print(f"Sampling {SAMPLE_SIZE} speeches (seed={SEED}, context={CONTEXT_WINDOW})\n")

    # Load all speeches
    assemblies = load_speeches(args.data_dir, ASSEMBLIES_TOPICS, "assemblies")
    experts = load_speeches(args.data_dir / "experts", EXPERTS_TOPICS, "experts")
    all_speeches = assemblies + experts
    
    print(f"Total valid speeches: {len(all_speeches)}")
    print(f"  - Assemblies: {len(assemblies)}")
    print(f"  - Experts: {len(experts)}\n")
    
    # Sample
    sampled = stratified_sample(all_speeches, SAMPLE_SIZE, SEED)
    
    # Report distribution
    print(f"Sampled {len(sampled)} speeches:")
    for key in ['group', 'topic', 'expertise_score']:
        counts = defaultdict(int)
        for s in sampled:
            counts[s[key]] += 1
        print(f"\n  By {key}:")
        for val, count in sorted(counts.items()):
            print(f"    {str(val):25} {count:3}")
    
    # Save
    output = {
        'metadata': {
            'sample_size': len(sampled),
            'context_window': CONTEXT_WINDOW,
            'random_seed': SEED
        },
        'speeches': sampled
    }
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved to {args.output}")

if __name__ == "__main__":
    main()