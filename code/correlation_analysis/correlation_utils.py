"""
Shared utilities for correlation and temporal analysis.

Imported by corr.py and detailed_temporal_analysis.py.  All functions are
standalone (no class dependency) so they can be called directly or used in
other analysis scripts.

Topic assignment in load_and_combine_data is driven by the topic_folders
parameter; callers should pass ASSEMBLY_TOPICS / EXPERT_TOPICS from
shared.topic_maps so the labels stay canonical.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Default exclusion patterns for get_dynamic_features.
# Merged from both corr.py and detailed_temporal_analysis.py.
# ---------------------------------------------------------------------------
DEFAULT_EXCLUDED_COLUMNS: List[str] = [
    # Free-text content
    'text', 'speech', 'comment', 'statement', 'original_text',
    # Speaker / identity metadata
    'speaker', 'name', 'speaker_id', 'speech_id', 'member_id',
    # Temporal / provenance metadata
    'date', 'year', 'source_file', 'topic',
    # Annotation text
    'justification',
    # Pandas artefacts
    'Unnamed',
    # Derived / redundant columns
    'hedge_ratio',
    # Embedding cluster labels (not distances)
    'cluster_',
    # Generic identifiers
    'id', 'index', 'row',
]


def filter_syntactically_relevant(
    df: pd.DataFrame,
    threshold: int = 5,
) -> pd.DataFrame:
    """
    Keep only rows with strictly more than *threshold* non-function words.

    Args:
        df: DataFrame containing a 'num_non_function_words' column.
        threshold: Minimum content-word count (exclusive lower bound).

    Returns:
        Filtered copy of df.  If the column is absent, df is returned
        unchanged with a warning.
    """
    if 'num_non_function_words' not in df.columns:
        print("   ⚠️  'num_non_function_words' not found, no filtering applied")
        return df

    original = len(df)
    df = df.copy()
    df['num_non_function_words'] = pd.to_numeric(
        df['num_non_function_words'], errors='coerce'
    )
    filtered = df[df['num_non_function_words'] > threshold].copy()
    removed = original - len(filtered)
    print(f"   ✓ Filtered (threshold={threshold}): {len(filtered)} rows kept ({removed} removed)")
    return filtered


def get_dynamic_features(
    df: pd.DataFrame,
    model_type: str,
    excluded_columns: Optional[List[str]] = None,
) -> List[str]:
    """
    Automatically select relevant numeric feature columns from a DataFrame.

    Selection rules:
    - Only numeric dtype columns are considered.
    - Columns whose name contains any pattern in *excluded_columns*
      (case-insensitive partial match) are dropped.
    - Columns containing 'base' are kept only when model_type == 'base'.
    - Columns containing 'finetuned' are kept only when model_type == 'finetuned'.

    Return order: linguistic features, overall scores, distances,
    similarities, expertise scores — so expertise columns always appear
    last for interpretability.

    Args:
        df: Source DataFrame.
        model_type: 'base' or 'finetuned'.
        excluded_columns: Patterns to skip.  Defaults to DEFAULT_EXCLUDED_COLUMNS.

    Returns:
        Ordered, deduplicated list of feature column names.
    """
    if excluded_columns is None:
        excluded_columns = DEFAULT_EXCLUDED_COLUMNS

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Defensive: drop columns that are also present in object/string dtypes.
    text_cols = set(df.select_dtypes(include=['object', 'string', 'category']).columns)
    numeric_cols = [c for c in numeric_cols if c not in text_cols]

    features = []
    for col in numeric_cols:
        col_lower = col.lower()
        if col == 'id':
            continue
        if any(ex.lower() in col_lower for ex in excluded_columns):
            continue

        is_base_col = 'base' in col_lower
        is_fine_col = 'finetuned' in col_lower
        if is_base_col and model_type != 'base':
            continue
        if is_fine_col and model_type != 'finetuned':
            continue

        features.append(col)

    # Deduplicate, preserving order
    seen: set = set()
    unique_features = [x for x in features if not (x in seen or seen.add(x))]

    # Priority ordering: linguistic rest → overall → distance → similarity → expertise
    expertise_cols: List[str] = []
    distance_cols: List[str] = []
    similarity_cols: List[str] = []
    overall_cols: List[str] = []
    rest_cols: List[str] = []

    for f in unique_features:
        f_lower = f.lower()
        if 'expertise' in f_lower:
            expertise_cols.append(f)
        elif 'distance' in f_lower:
            distance_cols.append(f)
        elif 'similarity' in f_lower:
            similarity_cols.append(f)
        elif 'overall' in f_lower:
            overall_cols.append(f)
        else:
            rest_cols.append(f)

    return rest_cols + overall_cols + distance_cols + similarity_cols + expertise_cols


def format_feature_name(name: str) -> str:
    """
    Convert an internal column name to a human-readable display label.

    Examples:
        'distance_base'        → 'Distance To Centroid'
        'similarity_finetuned' → 'Similarity To Centroid'
        'jargon_density_norm'  → 'Jargon Density'
        'expertise_llm_score'  → 'Expertise Llm Score'
    """
    name = name.replace('_norm', '').replace('_base', '').replace('_finetuned', '')
    if name.startswith('distance'):
        name = name.replace('distance', 'Distance to Centroid', 1)
    elif name.startswith('similarity'):
        name = name.replace('similarity', 'Similarity to Centroid', 1)
    return name.replace('_', ' ').title()


def load_and_combine_data(
    base_path: Path,
    topic_folders: Dict[str, str],
    threshold: int = 5,
) -> Dict[str, pd.DataFrame]:
    """
    Load CSV files from each topic subfolder, tag with topic label, and filter.

    The 'topic' column in each loaded DataFrame is set to the value from
    *topic_folders*, so callers should pass ASSEMBLY_TOPICS or EXPERT_TOPICS
    from shared.topic_maps to ensure canonical labels.

    Falls back to a '<folder>_csv' directory name if the plain folder does not
    exist (e.g. 'scotland_csv').

    Args:
        base_path: Root directory that contains the topic subfolders.
        topic_folders: Mapping of {folder_name: topic_display_label}.
        threshold: Passed to filter_syntactically_relevant.

    Returns:
        Dict of {topic_display_label: combined_filtered_DataFrame}.
        Topics with no usable data are omitted.
    """
    data_by_topic: Dict[str, pd.DataFrame] = {}

    for folder, topic_name in topic_folders.items():
        folder_path = base_path / folder
        if not folder_path.exists():
            alt = base_path / f"{folder}_csv"
            if alt.exists():
                folder_path = alt
            else:
                print(f"   ⚠️  Folder not found: {base_path / folder}")
                continue

        csv_files = list(folder_path.glob('*.csv'))
        if not csv_files:
            print(f"   ⚠️  No CSV files in {folder_path}")
            continue

        dfs = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                df['source_file'] = csv_file.name
                df['topic'] = topic_name
                dfs.append(df)
            except Exception as e:
                print(f"   ❌ Error reading {csv_file.name}: {e}")

        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            filtered = filter_syntactically_relevant(combined, threshold=threshold)
            if len(filtered) > 0:
                data_by_topic[topic_name] = filtered
                print(f"   ✓ {topic_name}: {len(csv_files)} files, {len(filtered)} rows")

    return data_by_topic
