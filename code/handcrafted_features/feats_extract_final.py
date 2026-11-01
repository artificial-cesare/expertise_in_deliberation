#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final feature extraction script.
Creates new_feats/ directory with processed CSVs.

Features extracted:
- jargon_density
- lexical_sophistication  
- syntactic_complexity
- hedge_density
- booster_density
- epistemic_balance
- reasoning_markers
- evidential_markers
- specificity_numbers 
- specificity_word_length
- subjectivity_score
- mrc_familiarity
- mrc_concreteness
- mrc_imageability
- length_tokens (control)
- num_non_function_words (control)
"""

import re
import sys
import json
import shutil
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared.topic_maps import ASSEMBLY_TOPICS as ASSEMBLY_TOPIC_MAP, EXPERT_TOPICS as EXPERT_TOPIC_MAP
from mrc_lookup import MRCLookup, compute_mrc_features
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher, Matcher
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================
try:
    NLP = spacy.load("en_core_web_sm", disable=["ner"])
except OSError:
    raise RuntimeError("spaCy model 'en_core_web_sm' not found.")

RELEVANCE = 5  # Minimum content words for substantive speech

# Feature columns to extract
FEATURE_NAMES = [
    "jargon_density",
    "lexical_sophistication",
    "syntactic_complexity",
    "hedge_density",
    "booster_density",
    "epistemic_balance",
    "reasoning_markers",
    "evidential_markers",
    "subjectivity_score",
    "mrc_familiarity",
    "mrc_concreteness",
    "mrc_imageability",
    "length_tokens",
    "num_non_function_words",
]

# Columns to preserve (never drop)
PROTECTED_COLUMNS = {
    "cluster_base", "distance_base", "similarity_base",
    "cluster_finetuned", "distance_finetuned", "similarity_finetuned",
    "overall_score", "expertise_llm_score"
}

# Lexical sets
HEDGE_SET = {
    "apparently", "appear", "appears", "approximately", "arguably", "assume",
    "broadly", "could", "fairly", "frequently", "generally", "likely", "may",
    "maybe", "might", "mostly", "often", "perhaps", "plausible", "possibly",
    "presumably", "probable", "probably", "roughly", "seems", "seem", "seemed",
    "seemingly", "sometimes", "suggest", "suggests", "tend", "tends",
    "typically", "usually"
}

BOOSTER_SET = {
    "absolutely", "always", "certainly", "clearly", "definitely", "evidently",
    "incontrovertibly", "indeed", "must", "never", "obviously", "undeniably",
    "unequivocally"
}

REASONING_TOKENS = {
    "because", "since", "therefore", "thus", "hence", "consequently",
    "accordingly", "so", "wherefore", "follows", "entails", "implies",
    "means", "indicates", "demonstrates", "shows", "proves", "suggests",
    "reveals", "conclude", "concludes", "concluded", "conclusion",
    "infer", "infers", "inferred", "inference", "ergo", "whereby",
    "whence", "whereupon"
}

# Evidential patterns (expanded)
EVIDENTIAL_PATTERNS = [
    r"\baccording to\b",
    r"\b(as )?(cited|reported|stated|noted|mentioned) (by|in)\b",
    r"\b(the |a )?(research|study|studies|survey|surveys|analysis|report|reports|data|evidence|findings|results) (show|shows|indicate|indicates|suggest|suggests|demonstrate|demonstrates|reveal|reveals|found|find|finds|confirm|confirms)\b",
    r"\b(research|studies|surveys|evidence|data) (has |have )?(shown|indicated|suggested|demonstrated|revealed|found|confirmed)\b",
    r"\b(statistics|figures|numbers|polls?|percentages?) (show|indicate|suggest|reveal)\b",
    r"\b\d+\s*(%|percent|per cent)\b",
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten|\d+) (in|out of) (ten|twenty|\d+)\b",
    r"\b(experts?|researchers?|scientists?|economists?|analysts?|specialists?) (say|says|argue|argues|believe|believes|suggest|suggests|recommend|recommends|conclude|concludes|found|find)\b",
    r"\b(the )?(government|commission|committee|council|department|ministry|agency|organization|organisation|WHO|UN|EU|OECD|IMF) (says?|stated?|reported?|found|recommend|concluded)\b",
    r"\b(official|government|published) (data|statistics|figures|report|reports)\b",
    r"\b(the |it is a )?(fact|facts) (is|are|that|show)\b",
    r"\bit (is|has been) (proven|shown|demonstrated|established|documented)\b",
    r"\bevidence (shows|suggests|indicates|supports|confirms)\b",
]
EVIDENTIAL_REGEX = re.compile("|".join(EVIDENTIAL_PATTERNS), flags=re.IGNORECASE)

# Syntactic labels
SUBORD_LABELS = {"mark", "advcl", "csubj", "csubjpass", "relcl"}
CNOM_LABELS = {"amod", "compound", "nmod", "appos"}

# MPQA POS mapping
MPQA_POS_MAP = {"adj": "ADJ", "noun": "NOUN", "verb": "VERB", "adverb": "ADV", "anypos": "ANY"}

# Topic mappings

# ============================================================================
# Word frequency
# ============================================================================
try:
    from wordfreq import zipf_frequency
    def token_zipf(tok: str) -> float:
        return zipf_frequency(tok, "en")
except ImportError:
    print("⚠️  wordfreq not installed. Using length-based fallback.")
    def token_zipf(tok: str) -> float:
        if len(tok) <= 2: return 6.0
        elif len(tok) <= 4: return 5.0
        elif len(tok) <= 7: return 4.0
        else: return 3.0

# ============================================================================
# Subjectivity Lexicon
# ============================================================================
def load_mpqa_lexicon(tff_path: Path) -> Tuple[Dict, Dict]:
    """Load MPQA subjectivity lexicon."""
    lemma_lexicon, surface_lexicon = {}, {}
    
    if not tff_path.exists():
        print(f"⚠️  Subjectivity lexicon not found: {tff_path}")
        return {}, {}
    
    with open(tff_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if not line.strip():
                continue
            entry = {}
            for part in line.split():
                if '=' in part:
                    k, v = part.split('=', 1)
                    entry[k] = v
            
            subj_type = entry.get('type')
            word = entry.get('word1')
            pos_raw = entry.get('pos1')
            stemmed = entry.get('stemmed1')
            
            if not (word and pos_raw):
                continue
            
            weight = 1.0 if subj_type == 'strongsubj' else 0.5
            pos_tag = MPQA_POS_MAP.get(pos_raw, 'ANY')
            
            if stemmed == 'y':
                lemma_lexicon[(word, pos_tag)] = weight
                lemma_lexicon[(word, 'ANY')] = weight
            else:
                surface_lexicon[(word, pos_tag)] = weight
                surface_lexicon[(word, 'ANY')] = weight
    
    print(f"✅ MPQA: {len(lemma_lexicon)} lemma, {len(surface_lexicon)} surface entries")
    return lemma_lexicon, surface_lexicon

# ============================================================================
# Matcher Builders
# ============================================================================
def build_jargon_matcher(terms: List[str]) -> Tuple[Optional[PhraseMatcher], Set[str], Set[Tuple[str, ...]]]:
    """Build jargon matcher with unigram set and ngram tuples for efficient matching."""
    if not terms:
        return None, set(), set()
    
    cleaned = list({t.strip().lower() for t in terms if len(t.strip()) >= 2})
    if not cleaned:
        return None, set(), set()
    
    # Separate unigrams and ngrams
    unigrams = [t for t in cleaned if ' ' not in t]
    ngrams = [t for t in cleaned if ' ' in t]
    
    # Build unigram lemma set
    unigram_lemma_set = set()
    for term in unigrams:
        doc = NLP(term)
        if doc:
            unigram_lemma_set.add(doc[0].lemma_.lower())
            unigram_lemma_set.add(term)
    
    # Build ngram lemma tuples
    ngram_lemma_tuples = set()
    for ngram in ngrams:
        doc = NLP(ngram)
        lemma_tuple = tuple(t.lemma_.lower() for t in doc if t.is_alpha and not t.is_stop)
        if len(lemma_tuple) >= 2:
            ngram_lemma_tuples.add(lemma_tuple)
        original_tuple = tuple(t.text.lower() for t in doc if t.is_alpha and not t.is_stop)
        if len(original_tuple) >= 2:
            ngram_lemma_tuples.add(original_tuple)
    
    # Build phrase matcher
    matcher = PhraseMatcher(NLP.vocab, attr="LOWER")
    try:
        matcher.add("DOMAIN_TERMS", [NLP.make_doc(t) for t in cleaned])
    except Exception as e:
        print(f"⚠️  Matcher error: {e}")
        return None, unigram_lemma_set, ngram_lemma_tuples
    
    return matcher, unigram_lemma_set, ngram_lemma_tuples


def build_reasoning_matcher() -> Matcher:
    """Build comprehensive reasoning marker matcher."""
    matcher = Matcher(NLP.vocab)
    
    # Single tokens
    matcher.add("REASON_TOKEN", [[{"LOWER": {"IN": list(REASONING_TOKENS)}}]])
    
    # Causal patterns
    causal_patterns = [
        [{"LOWER": "as"}, {"LOWER": "a"}, {"LOWER": w}] for w in ["result", "consequence"]
    ] + [
        [{"LOWER": w1}, {"LOWER": w2}] for w1, w2 in [
            ("due", "to"), ("owing", "to"), ("results", "in"),
            ("leads", "to"), ("causes", "to"), ("results", "from")
        ]
    ] + [
        [{"LOWER": w1}, {"LOWER": w2}, {"LOWER": w3}] for w1, w2, w3 in [
            ("for", "this", "reason"), ("on", "account", "of"),
            ("in", "light", "of"), ("in", "view", "of"), ("by", "virtue", "of")
        ]
    ]
    
    # Entailment patterns
    entailment_patterns = [
        [{"LOWER": "it"}, {"LOWER": "follows"}, {"LOWER": "that"}],
        [{"LOWER": "it"}, {"LOWER": "follows"}],
        [{"LOWER": {"IN": ["this", "that", "which"]}},
         {"LEMMA": {"IN": ["imply", "mean", "show", "demonstrate", "prove", "suggest", "indicate", "reveal", "entail"]}},
         {"LOWER": "that"}],
        [{"LOWER": "as"}, {"LOWER": "a"}, {"LOWER": {"IN": ["consequence", "result", "corollary"]}}],
    ]
    
    # Conditional patterns
    conditional_patterns = [
        [{"LOWER": "given"}, {"LOWER": "that"}],
        [{"LEMMA": {"IN": ["assume", "suppose", "grant"]}}, {"LOWER": "that"}],
        [{"LOWER": {"IN": ["provided", "providing"]}}, {"LOWER": "that"}],
        [{"LOWER": "on"}, {"LOWER": "condition"}, {"LOWER": "that"}],
    ]
    
    # Explanatory patterns
    explanatory_patterns = [
        [{"LOWER": "the"}, {"LOWER": "reason"}, {"LOWER": {"IN": ["is", "why", "being"]}}],
        [{"LOWER": "that"}, {"LOWER": "is"}, {"LOWER": "why"}],
        [{"LOWER": "this"}, {"LOWER": "is"}, {"LOWER": "why"}],
        [{"LOWER": "in"}, {"LOWER": "order"}, {"LOWER": "to"}],
        [{"LOWER": "so"}, {"LOWER": "as"}, {"LOWER": "to"}],
        [{"LOWER": "so"}, {"LOWER": "that"}],
    ]
    
    # Contrastive patterns
    contrastive_patterns = [
        [{"LOWER": {"IN": ["however", "nevertheless", "nonetheless", "notwithstanding"]}}],
        [{"LOWER": "on"}, {"LOWER": "the"}, {"LOWER": "other"}, {"LOWER": "hand"}],
        [{"LOWER": {"IN": ["in", "by"]}}, {"LOWER": "contrast"}],
        [{"LOWER": {"IN": ["although", "though", "while", "whereas", "whilst"]}}],
        [{"LOWER": "despite"}],
        [{"LOWER": "in"}, {"LOWER": "spite"}, {"LOWER": "of"}],
    ]
    
    matcher.add("CAUSAL", causal_patterns)
    matcher.add("ENTAILMENT", entailment_patterns)
    matcher.add("CONDITIONAL", conditional_patterns)
    matcher.add("EXPLANATORY", explanatory_patterns)
    matcher.add("CONTRASTIVE", contrastive_patterns)
    
    return matcher

# ============================================================================
# Feature Extraction
# ============================================================================
def compute_jargon_density(
    doc,
    non_funct_tokens: List,
    jargon_matcher: Optional[PhraseMatcher],
    jargon_lemma_set: Set[str],
    jargon_ngram_tuples: Set[Tuple[str, ...]],
) -> float:
    """Compute jargon density with comprehensive matching."""
    if not non_funct_tokens:
        return 0.0
    
    matched_indices = set()
    
    # Phrase matcher matches
    if jargon_matcher:
        for _, start, end in jargon_matcher(doc):
            for idx in range(start, end):
                if not doc[idx].is_stop and doc[idx].is_alpha:
                    matched_indices.add(idx)
    
    # Unigram lemma matches
    if jargon_lemma_set:
        for i, token in enumerate(doc):
            if token.is_alpha and not token.is_stop:
                if token.lemma_.lower() in jargon_lemma_set or token.text.lower() in jargon_lemma_set:
                    matched_indices.add(i)
    
    # Ngram matches
    if jargon_ngram_tuples:
        content_tokens = [(i, doc[i].lemma_.lower()) for i in range(len(doc)) if doc[i].is_alpha and not doc[i].is_stop]
        max_ngram_len = max((len(t) for t in jargon_ngram_tuples), default=2)
        for ngram_len in range(2, max_ngram_len + 1):
            if len(content_tokens) < ngram_len:
                continue
            for start_pos in range(len(content_tokens) - ngram_len + 1):
                window_indices = [content_tokens[start_pos + j][0] for j in range(ngram_len)]
                window_lemmas = tuple(content_tokens[start_pos + j][1] for j in range(ngram_len))
                if window_lemmas in jargon_ngram_tuples:
                    matched_indices.update(window_indices)
    
    n_content = len(non_funct_tokens)
    jargon_count = len(matched_indices)
    
    # Log-normalized density
    return np.log1p(jargon_count) / np.log1p(n_content) if n_content > 0 else 0.0


def compute_subjectivity(doc, lemma_lexicon: Dict, surface_lexicon: Dict, n_tokens: int) -> float:
    """Compute subjectivity score from MPQA lexicon."""
    if not lemma_lexicon and not surface_lexicon:
        return 0.0
    
    subjectivity_accumulated = 0.0
    
    for t in doc:
        if not t.is_alpha:
            continue
        
        weight = 0.0
        found = False
        
        # Surface match first
        for key in [(t.text.lower(), t.pos_), (t.text.lower(), 'ANY')]:
            if key in surface_lexicon:
                weight = surface_lexicon[key]
                found = True
                break
        
        # Lemma match as fallback
        if not found:
            for key in [(t.lemma_.lower(), t.pos_), (t.lemma_.lower(), 'ANY')]:
                if key in lemma_lexicon:
                    weight = lemma_lexicon[key]
                    found = True
                    break
        
        subjectivity_accumulated += weight
    
    # Log-normalized
    return np.log1p(subjectivity_accumulated) / np.log1p(max(n_tokens, 1))


def extract_features(
    text: str,
    jargon_matcher: Optional[PhraseMatcher],
    jargon_lemma_set: Set[str],
    jargon_ngram_tuples: Set[Tuple[str, ...]],
    reasoning_matcher: Matcher,
    lemma_lexicon: Dict,
    surface_lexicon: Dict,
    mrc_lookup: MRCLookup,
) -> Dict[str, float]:
    """Extract all features from text."""
    
    if not isinstance(text, str) or not text.strip():
        text = ""
    
    doc = NLP(text)
    tokens = [t for t in doc if t.is_alpha]
    tokens_lower = [t.text.lower() for t in tokens]
    n_tokens = len(tokens_lower)
    non_funct_tokens = [t for t in tokens if not t.is_stop]
    n_non_funct = len(non_funct_tokens)
    norm_tok = max(n_tokens, 1)
    
    # Non-substantive text - return NaN for linguistic features
    if n_non_funct < RELEVANCE:
        return {
            "jargon_density": np.nan,
            "lexical_sophistication": np.nan,
            "syntactic_complexity": np.nan,
            "hedge_density": np.nan,
            "booster_density": np.nan,
            "epistemic_balance": np.nan,
            "reasoning_markers": np.nan,
            "evidential_markers": np.nan,
            "specificity_numbers": np.nan,
            "specificity_word_length": np.nan,
            "subjectivity_score": np.nan,
            "mrc_familiarity": np.nan,
            "mrc_concreteness": np.nan,
            "mrc_imageability": np.nan,
            "length_tokens": float(n_tokens),
            "num_non_function_words": float(n_non_funct),
        }
    
    # 1. Jargon density
    jargon_density = compute_jargon_density(
        doc, non_funct_tokens, jargon_matcher, jargon_lemma_set, jargon_ngram_tuples
    )
    
    # 2. Lexical sophistication (inverse of frequency)
    sophisticated_threshold = 4.0
    n_sophisticated = sum(1 for t in tokens_lower if token_zipf(t) < sophisticated_threshold)
    lexical_sophistication = np.log1p(n_sophisticated) / np.log1p(n_tokens) if n_tokens > 0 else 0.0
    
    # 3. Syntactic complexity
    subord = sum(1 for t in doc if t.dep_ in SUBORD_LABELS)
    cnom = sum(1 for t in doc if t.dep_ in CNOM_LABELS)
    raw_complexity = (2 * subord) + cnom
    syntactic_complexity = np.log1p(raw_complexity) / np.log1p(norm_tok)
    
    # 4-6. Epistemic markers
    hedge_count = sum(tok in HEDGE_SET for tok in tokens_lower)
    booster_count = sum(tok in BOOSTER_SET for tok in tokens_lower)
    hedge_density = np.log1p(hedge_count) / np.log1p(norm_tok)
    booster_density = np.log1p(booster_count) / np.log1p(norm_tok)
    
    denom = hedge_count + booster_count
    epistemic_balance = 1.0 - (abs(hedge_count - booster_count) / denom) if denom > 0 else np.nan
    
    # 7. Reasoning markers
    reasoning_matches = reasoning_matcher(doc)
    reasoning_markers = np.sqrt(len(reasoning_matches)) / np.sqrt(norm_tok) if norm_tok > 0 else 0.0
    
    # 8. Evidential markers
    evidential_hits = len(EVIDENTIAL_REGEX.findall(text))
    evidential_markers = np.log1p(evidential_hits) / np.log1p(norm_tok) if norm_tok > 0 else 0.0
    
    # 9. Specificity
    mean_word_len = float(np.mean([len(t) for t in tokens_lower])) if n_tokens > 0 else 0.0
    num_numbers = sum(1 for t in doc if t.like_num)
    # Average word length mapped to [0, 1]: 3-char words (function words) → 0,
    # 8-char words (complex content words) → 1. Clip handles extremes.
    # Longer words signal more specific/technical vocabulary.
    # Parameter justification: Preoţiuc-Pietro et al. (2012), arXiv:1208.6109 —
    # average word length tracks lexical complexity across registers.
    spec_word_len = float(np.clip((mean_word_len - 3) / 5, 0, 1))
    spec_nums = num_numbers

    # 10. Subjectivity
    subjectivity_score = compute_subjectivity(doc, lemma_lexicon, surface_lexicon, n_tokens)
    
    # 10-12. MRC features
    mrc_features = compute_mrc_features(doc, mrc_lookup)
    
    return {
        "jargon_density": float(jargon_density),
        "lexical_sophistication": float(lexical_sophistication),
        "syntactic_complexity": float(syntactic_complexity),
        "hedge_density": float(hedge_density),
        "booster_density": float(booster_density),
        "epistemic_balance": float(epistemic_balance) if not np.isnan(epistemic_balance) else np.nan,
        "reasoning_markers": float(reasoning_markers),
        "evidential_markers": float(evidential_markers),
        "specificity_numbers": float(spec_nums),
        "specificity_word_length": float(spec_word_len),
        "subjectivity_score": float(subjectivity_score),
        **mrc_features,
        "length_tokens": float(n_tokens),
        "num_non_function_words": float(n_non_funct),
    }

# ============================================================================
# File Processing
# ============================================================================
def get_columns_to_drop(df: pd.DataFrame, text_col: str) -> List[str]:
    """Identify columns to drop (old features not in FEATURE_NAMES or PROTECTED)."""
    # Common old feature patterns to drop
    old_feature_patterns = [
        "specificity", "subjectivity_count", "_norm",
        "mean_word_len", "numbers_norm", "content_prop"
    ]
    
    cols_to_drop = []
    for col in df.columns:
        # Skip protected columns
        if col in PROTECTED_COLUMNS:
            continue
        # Skip new feature columns
        if col in FEATURE_NAMES:
            continue
        # Skip essential columns
        if col in [text_col, "text", "comment", "speaker", "turn_id", "session", 
                   "timestamp", "date", "id", "index", "Unnamed: 0"]:
            continue
        # Check if it's an old feature column
        for pattern in old_feature_patterns:
            if pattern in col.lower():
                cols_to_drop.append(col)
                break
    
    return cols_to_drop


def process_csv(
    src_path: Path,
    dst_path: Path,
    text_col: str,
    jargon_matcher: Optional[PhraseMatcher],
    jargon_lemma_set: Set[str],
    jargon_ngram_tuples: Set[Tuple[str, ...]],
    reasoning_matcher: Matcher,
    lemma_lexicon: Dict,
    surface_lexicon: Dict,
    mrc_lookup: MRCLookup,
) -> None:
    """Process single CSV: extract features, drop old columns, save to new location."""
    print(f"\n   📄 {src_path.name}")
    
    try:
        df = pd.read_csv(src_path)
        print(f"      Loaded: {len(df)} rows, {len(df.columns)} columns")
        
        # Find text column
        actual_col = next((c for c in [text_col, "text", "comment"] if c in df.columns), None)
        if not actual_col:
            raise ValueError("No text column found")
        
        # Identify and drop old feature columns
        cols_to_drop = get_columns_to_drop(df, actual_col)
        if cols_to_drop:
            print(f"      Dropping {len(cols_to_drop)} old columns: {cols_to_drop[:5]}{'...' if len(cols_to_drop) > 5 else ''}")
            df = df.drop(columns=cols_to_drop, errors='ignore')
        
        # Extract features
        print(f"      Extracting features...")
        for idx, row in tqdm(df.iterrows(), total=len(df), leave=False, desc="      Processing"):
            text = str(row[actual_col]) if pd.notna(row[actual_col]) else ""
            try:
                features = extract_features(
                    text, jargon_matcher, jargon_lemma_set, jargon_ngram_tuples,
                    reasoning_matcher, lemma_lexicon, surface_lexicon, mrc_lookup
                )
                for k, v in features.items():
                    df.at[idx, k] = v
            except Exception as e:
                print(f"\n      ⚠️  Row {idx} error: {e}")
                for k in FEATURE_NAMES:
                    df.at[idx, k] = np.nan
        
        # Ensure destination directory exists
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save
        df.to_csv(dst_path, index=False)
        
        substantive = (df['num_non_function_words'] >= RELEVANCE).sum()
        print(f"      ✅ Saved to {dst_path.name} ({substantive}/{len(df)} substantive, {len(df.columns)} columns)")
        
    except Exception as e:
        print(f"      ❌ Error: {e}")
        import traceback
        traceback.print_exc()


def process_folder(
    src_folder: Path,
    dst_folder: Path,
    topic_key: str,
    domain_terms_dict: Dict[str, List[str]],
    text_col: str,
    lemma_lexicon: Dict,
    surface_lexicon: Dict,
    mrc_lookup: MRCLookup,
) -> None:
    """Process all CSVs in a folder."""
    if not src_folder.exists():
        print(f"\n⚠️  Source folder not found: {src_folder}")
        return
    
    domain_terms = domain_terms_dict.get(topic_key.lower(), [])
    jargon_matcher, jargon_lemma_set, jargon_ngram_tuples = build_jargon_matcher(domain_terms)
    reasoning_matcher = build_reasoning_matcher()
    
    print(f"\n📂 {src_folder.name} → {topic_key} ({len(domain_terms)} terms)")
    
    csv_files = sorted(src_folder.glob("*.csv"))
    if not csv_files:
        print(f"   No CSV files found")
        return
    
    for csv_file in csv_files:
        dst_file = dst_folder / csv_file.name
        process_csv(
            csv_file, dst_file, text_col,
            jargon_matcher, jargon_lemma_set, jargon_ngram_tuples,
            reasoning_matcher, lemma_lexicon, surface_lexicon, mrc_lookup
        )


def process_all_files(
    processed_dir: Path,
    output_dir: Path,
    domain_terms_json: Path,
    tff_path: Path,
    mrc_db_path: Path,
    text_col: str,
) -> None:
    """Process all files, creating new_feats/ structure."""
    
    print(f"\n{'='*80}")
    print("FEATURE EXTRACTION - Creating new_feats/ directory")
    print(f"{'='*80}")
    
    # Load resources
    print("\n⏳ Loading resources...")
    
    # Domain terms
    try:
        with open(domain_terms_json, "r") as f:
            # Lowercase keys: the JSON uses "gender equality" while topic_maps
            # yields "Gender Equality"; an exact match returns 0 terms
            domain_terms_dict = {k.lower(): v for k, v in json.load(f).items()}
        print(f"✅ Domain terms: {sum(len(v) for v in domain_terms_dict.values())} terms across {len(domain_terms_dict)} topics")
    except Exception as e:
        # A missing lexicon would make jargon_density silently ~0 for every row
        sys.exit(f"❌ Domain terms error ({domain_terms_json}): {e}")
    
    # Subjectivity lexicon
    lemma_lexicon, surface_lexicon = load_mpqa_lexicon(tff_path)
    
    # MRC database
    mrc_lookup = MRCLookup(mrc_db_path)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Output directory: {output_dir}")
    
    # Process experts
    expert_src = processed_dir / "experts"
    expert_dst = output_dir / "experts"
    if expert_src.exists():
        print("\n" + "="*40)
        print("📚 EXPERTS")
        print("="*40)
        for topic_folder in sorted(expert_src.iterdir()):
            if topic_folder.is_dir() and topic_folder.name in EXPERT_TOPIC_MAP:
                process_folder(
                    topic_folder,
                    expert_dst / topic_folder.name,
                    EXPERT_TOPIC_MAP[topic_folder.name],
                    domain_terms_dict, text_col,
                    lemma_lexicon, surface_lexicon, mrc_lookup
                )
    
    # Process assemblies
    print("\n" + "="*40)
    print("👥 ASSEMBLIES")
    print("="*40)
    for folder_name, topic_key in ASSEMBLY_TOPIC_MAP.items():
        src_folder = processed_dir / folder_name
        dst_folder = output_dir / folder_name
        process_folder(
            src_folder, dst_folder, topic_key,
            domain_terms_dict, text_col,
            lemma_lexicon, surface_lexicon, mrc_lookup
        )
    
    print(f"\n{'='*80}")
    print("✅ COMPLETE")
    print(f"{'='*80}")

# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Extract features and save to new_feats/ directory"
    )
    parser.add_argument(
        "--processed_dir", type=Path,
        default=Path("data/assemblies/processed_files"),
        help="Source processed files directory"
    )
    parser.add_argument(
        "--output_dir", type=Path,
        default=Path("data/assemblies/new_feats"),
        help="Output directory for processed files"
    )
    parser.add_argument(
        "--domain_terms", type=Path,
        default=Path(__file__).resolve().parent / "domain_terms_clean_exp.json",
        help="Path to domain terms JSON (output of dom_terms_final_exp.py)"
    )
    parser.add_argument(
        "--subjectivity_lexicon", type=Path,
        default=Path("data/subjectivity_clues_hltemnlp05/subjclueslen1-HLTEMNLP05.tff"),
        help="Path to MPQA .tff file"
    )
    parser.add_argument(
        "--mrc_db", type=Path,
        default=Path("data/mrc-psycholinguistics-master/mrc2.db"),
        help="Path to MRC database"
    )
    parser.add_argument(
        "--text_col", type=str, default="text",
        help="Name of text column"
    )
    
    args = parser.parse_args()
    
    process_all_files(
        processed_dir=args.processed_dir,
        output_dir=args.output_dir,
        domain_terms_json=args.domain_terms,
        tff_path=args.subjectivity_lexicon,
        mrc_db_path=args.mrc_db,
        text_col=args.text_col,
    )


if __name__ == "__main__":
    main()