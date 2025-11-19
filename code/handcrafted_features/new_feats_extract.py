#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract expert-language features from transcripts.
Processes CSVs and adds 9 linguistic feature columns.
"""

import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher, Matcher
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# ============================================================================
# Configuration
# ============================================================================
try:
    NLP = spacy.load("en_core_web_sm", disable=["ner"])
except OSError:
    raise RuntimeError(
        "spaCy model 'en_core_web_sm' not found.\n"
        "Install with: python -m spacy download en_core_web_sm"
    )

# Constants
RELEVANCE = 5  # Minimum content words for substantive speech
FEATURE_NAMES = [
    "jargon_density", "lexical_sophistication", "syntactic_complexity",
    "hedge_density", "booster_density", "epistemic_balance",
    "reasoning_markers", "evidential_markers", "specificity_length_norm",
    "specificity_mean_word_len_norm", "specificity_numbers_norm",
    "specificity_content_prop", 
    "length_tokens", "num_non_function_words"
]

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

# ============================================================================
# Reasoning marker tokens (expanded for structured reasoning)
# ============================================================================
REASONING_TOKENS = {
    # Causal
    "because", "since", "therefore", "thus", "hence", "consequently", 
    "accordingly", "so", "wherefore",
    
    # Entailment/Inference
    "follows", "entails", "implies", "means", "indicates", "demonstrates",
    "shows", "proves", "suggests", "reveals",
    
    # Conclusion markers
    "conclude", "concludes", "concluded", "conclusion",
    "infer", "infers", "inferred", "inference",
    
    # Logical operators
    "ergo", "whereby", "whence", "whereupon"
}

# Regex patterns
EVIDENTIAL_REGEX = re.compile(
    r"\baccording to\b|"
    r"\bthe (data|evidence|research) (show|shows|indicate|indicates|suggest|suggests)\b|"
    r"\bstudies? (show|shows|find|finds|indicate|indicates|suggest|suggests)\b|"
    r"\bresearch (shows|finds|indicates|suggests)\b|"
    r"\bmeta[- ]?analysis (shows|finds|indicates|suggests)\b|"
    r"\b(as|per) (the )?(report|guidelines|recommendations)\b",
    flags=re.IGNORECASE
)

# spaCy dependency labels
SUBORD_LABELS = {"mark", "advcl", "csubj", "csubjpass", "relcl"}
CNOM_LABELS = {"amod", "compound", "nmod", "appos"}

# Topic mappings
EXPERT_TOPIC_MAP = {
    "constitution": "constitutional reform",
    "future": "public policy",
    "gender": "gender equality"
}

ASSEMBLY_TOPIC_MAP = {
    "8th_amendment": "constitutional reform",
    "gendereq": "gender equality",
    "scotland": "public policy"
}

# ============================================================================
# Word frequency
# ============================================================================
try:
    from wordfreq import zipf_frequency
    token_zipf = lambda tok: zipf_frequency(tok, "en")
except ImportError:
    print("⚠️  wordfreq not installed. Using length-based fallback.")
    def token_zipf(tok: str) -> float:
        """Fallback: length-based heuristic."""
        return {True: 6.0, False: {True: 5.0, False: {True: 4.0, False: 3.0}[len(tok) > 7]}[len(tok) > 4]}[len(tok) <= 2]

# ============================================================================
# Matcher builders
# ============================================================================
def build_phrase_matcher(terms: List[str]) -> Optional[PhraseMatcher]:
    """Build spaCy PhraseMatcher for domain terms."""
    if not terms:
        return None
    
    cleaned = list({t.strip().lower() for t in terms if len(t.strip()) >= 2})
    if not cleaned:
        return None
    
    matcher = PhraseMatcher(NLP.vocab, attr="LOWER")
    try:
        matcher.add("DOMAIN_TERMS", [NLP.make_doc(t) for t in cleaned])
        return matcher
    except Exception as e:
        print(f"⚠️  Matcher error: {e}")
        return None

def build_reasoning_matcher() -> Matcher:
    """Build spaCy Matcher for reasoning markers (expanded for structured reasoning)."""
    matcher = Matcher(NLP.vocab)
    
    # Single tokens
    matcher.add("REASON_TOKEN", [[{"LOWER": {"IN": list(REASONING_TOKENS)}}]])
    
    # ========================================================================
    # CAUSAL REASONING
    # ========================================================================
    causal_patterns = [
        # Basic causal
        [{"LOWER": "as"}, {"LOWER": "a"}, {"LOWER": w}] 
        for w in ["result", "consequence"]
    ] + [
        [{"LOWER": w1}, {"LOWER": w2}] 
        for w1, w2 in [
            ("due", "to"), ("owing", "to"), ("results", "in"), 
            ("leads", "to"), ("causes", "to"), ("results", "from")
        ]
    ] + [
        [{"LOWER": w1}, {"LOWER": w2}, {"LOWER": w3}]
        for w1, w2, w3 in [
            ("for", "this", "reason"), ("on", "account", "of"),
            ("in", "light", "of"), ("in", "view", "of"),
            ("by", "virtue", "of")
        ]
    ] + [
        [{"LEMMA": v}, {"LOWER": "in"}] for v in ["result", "culminate", "end"]
    ] + [
        [{"LEMMA": v}, {"LOWER": "to"}] for v in ["lead", "contribute", "give rise"]
    ]
    
    # ========================================================================
    # ENTAILMENT & CONSEQUENCE
    # ========================================================================
    entailment_patterns = [
        # "it follows that"
        [{"LOWER": "it"}, {"LOWER": "follows"}, {"LOWER": "that"}],
        [{"LOWER": "it"}, {"LOWER": "follows"}],
        
        # "this implies/means/shows that"
        [{"LOWER": {"IN": ["this", "that", "which"]}}, 
         {"LEMMA": {"IN": ["imply", "mean", "show", "demonstrate", "prove", "suggest", "indicate", "reveal", "entail"]}}, 
         {"LOWER": "that"}],
        
        # "as a consequence/result"
        [{"LOWER": "as"}, {"LOWER": "a"}, 
         {"LOWER": {"IN": ["consequence", "result", "corollary"]}}],
        
        # "we can conclude/infer that"
        [{"LOWER": {"IN": ["we", "one"]}}, 
         {"LOWER": {"IN": ["can", "may", "must"]}},
         {"LEMMA": {"IN": ["conclude", "infer", "deduce", "derive"]}},
         {"LOWER": "that"}],
        
        # "the conclusion is"
        [{"LOWER": "the"}, 
         {"LOWER": {"IN": ["conclusion", "inference", "implication"]}},
         {"LEMMA": "be"}],
    ]
    
    # ========================================================================
    # CONDITIONAL REASONING (if-then, modus ponens)
    # ========================================================================
    conditional_patterns = [
        # "if X then Y" (capture the structure)
        [{"LOWER": "if"}, {"OP": "*"}, {"LOWER": "then"}],
        
        # "given that X, Y" / "given X, we can conclude Y"
        [{"LOWER": "given"}, {"LOWER": "that"}],
        [{"LOWER": "given"}, {"OP": "*"}, 
         {"LOWER": {"IN": ["we", "one"]}}, 
         {"LOWER": {"IN": ["can", "may", "must"]}}],
        
        # "assuming/supposing that"
        [{"LEMMA": {"IN": ["assume", "suppose", "grant"]}}, {"LOWER": "that"}],
        
        # "provided that" / "on condition that"
        [{"LOWER": {"IN": ["provided", "providing"]}}, {"LOWER": "that"}],
        [{"LOWER": "on"}, {"LOWER": "condition"}, {"LOWER": "that"}],
        
        # "in case/event that"
        [{"LOWER": "in"}, 
         {"LOWER": {"IN": ["case", "event"]}}, 
         {"LOWER": "that"}],
    ]
    
    # ========================================================================
    # SYLLOGISTIC REASONING
    # ========================================================================
    syllogistic_patterns = [
        # "all X are Y"
        [{"LOWER": "all"}, {"POS": {"IN": ["NOUN", "PROPN"]}}, 
         {"LEMMA": "be"}],
        
        # "every X is Y"
        [{"LOWER": "every"}, {"POS": {"IN": ["NOUN", "PROPN"]}}, 
         {"LEMMA": "be"}],
        
        # "no X is/are Y"
        [{"LOWER": "no"}, {"POS": {"IN": ["NOUN", "PROPN"]}}, 
         {"LEMMA": "be"}],
        
        # "some X are Y"
        [{"LOWER": "some"}, {"POS": {"IN": ["NOUN", "PROPN"]}}, 
         {"LEMMA": "be"}],
        
        # "X is a Y" (categorical membership)
        [{"POS": {"IN": ["NOUN", "PROPN"]}}, 
         {"LEMMA": "be"}, 
         {"LOWER": "a"}, 
         {"POS": {"IN": ["NOUN", "PROPN"]}}],
    ]
    
    # ========================================================================
    # EXPLANATORY REASONING
    # ========================================================================
    explanatory_patterns = [
        # "the reason is/why"
        [{"LOWER": "the"}, {"LOWER": "reason"}, 
         {"LOWER": {"IN": ["is", "why", "being"]}}],
        
        # "that is why"
        [{"LOWER": "that"}, {"LOWER": "is"}, {"LOWER": "why"}],
        [{"LOWER": "this"}, {"LOWER": "is"}, {"LOWER": "why"}],
        
        # "in order to" / "so as to"
        [{"LOWER": "in"}, {"LOWER": "order"}, {"LOWER": "to"}],
        [{"LOWER": "so"}, {"LOWER": "as"}, {"LOWER": "to"}],
        
        # "for the purpose of"
        [{"LOWER": "for"}, {"LOWER": "the"}, {"LOWER": "purpose"}, {"LOWER": "of"}],
        
        # "so that"
        [{"LOWER": "so"}, {"LOWER": "that"}],
    ]
    
    # ========================================================================
    # CONTRASTIVE/ADVERSATIVE REASONING (but, however, although)
    # ========================================================================
    contrastive_patterns = [
        # "however" / "nevertheless" / "nonetheless"
        [{"LOWER": {"IN": ["however", "nevertheless", "nonetheless", "notwithstanding"]}}],
        
        # "on the other hand"
        [{"LOWER": "on"}, {"LOWER": "the"}, {"LOWER": "other"}, {"LOWER": "hand"}],
        
        # "in contrast" / "by contrast"
        [{"LOWER": {"IN": ["in", "by"]}}, {"LOWER": "contrast"}],
        
        # "although/though/while"
        [{"LOWER": {"IN": ["although", "though", "while", "whereas", "whilst"]}}],
        
        # "despite/in spite of"
        [{"LOWER": "despite"}],
        [{"LOWER": "in"}, {"LOWER": "spite"}, {"LOWER": "of"}],
    ]
    
    # ========================================================================
    # Add all patterns to matcher
    # ========================================================================
    matcher.add("CAUSAL", causal_patterns)
    matcher.add("ENTAILMENT", entailment_patterns)
    matcher.add("CONDITIONAL", conditional_patterns)
    matcher.add("SYLLOGISTIC", syllogistic_patterns)
    matcher.add("EXPLANATORY", explanatory_patterns)
    matcher.add("CONTRASTIVE", contrastive_patterns)
    
    return matcher

# ============================================================================
# Feature extraction
# ============================================================================
def extract_features(
    text: str,
    jargon_matcher: Optional[PhraseMatcher],
    reasoning_matcher: Matcher,
    norm_params: Dict[str, float]
) -> Dict[str, float]:
    """
    Extract linguistic features.
    
    Args:
        text: Input text
        jargon_matcher: Domain term matcher
        reasoning_matcher: Reasoning marker matcher
        norm_params: Dict with keys: max_tokens, max_word_len, max_numbers, 
                     zipf_common, zipf_rare
    
    Returns:
        Dict of feature values (NaN for non-substantive text)
    """
    # Parse
    doc = NLP(str(text) if isinstance(text, str) else "")
    tokens = [t.text.lower() for t in doc if t.is_alpha]
    n_tokens = len(tokens)
    content_tokens = [t for t in doc if t.is_alpha and not t.is_stop]
    n_content = len(content_tokens)
    
    # Check substantiveness
    if n_content < RELEVANCE:
        # Return NaN for most features, but keep length_tokens and num_non_function_words
        features = {f: np.nan for f in FEATURE_NAMES}
        features["length_tokens"] = n_tokens
        features["num_non_function_words"] = n_content
        return features
    
    norm_tok = max(n_tokens, 1)
    
    # 1. Jargon density
    jargon_density = 0.0
    if jargon_matcher:
        matched_indices = set()
        for _, start, end in jargon_matcher(doc):
            matched_indices.update(range(start, end))
        jargon_density = len(matched_indices) / norm_tok
    
    # 2. Lexical sophistication (rarer words => higher sophistication)
    avg_zipf = np.mean([token_zipf(t) for t in tokens]) if tokens else 0.0
    zipf_range = norm_params["zipf_common"] - norm_params["zipf_rare"]
    lexical_sophistication = np.clip(
        (norm_params["zipf_common"] - avg_zipf) / max(zipf_range, 1.0),
        0.0, 1.0
    )
    
    # 3. Syntactic complexity
    subord = sum(1 for t in doc if t.dep_ in SUBORD_LABELS)
    cnom = sum(1 for t in doc if t.dep_ in CNOM_LABELS)
    syntactic_complexity = (2 * subord + cnom) / norm_tok
    
    # 4-6. Epistemic markers
    hedge_count = sum(t in HEDGE_SET for t in tokens)
    booster_count = sum(t in BOOSTER_SET for t in tokens)
    total_epistemic = hedge_count + booster_count
    
    hedge_density = 100.0 * hedge_count / norm_tok
    booster_density = 100.0 * booster_count / norm_tok
    epistemic_balance = (
        1.0 - abs(hedge_count - booster_count) / total_epistemic
        if total_epistemic > 0 else np.nan
    )
    
    # 7. Reasoning markers
    reasoning_markers = 100.0 * len(reasoning_matcher(doc)) / norm_tok
    
    # 8. Evidential markers
    evidential_markers = 100.0 * len(EVIDENTIAL_REGEX.findall(text)) / norm_tok
    
    # 9. Specificity components
    mean_word_len = np.mean([len(t) for t in tokens]) if tokens else 0.0
    num_numbers = sum(1 for t in doc if t.like_num)
    
    # Individual normalized components
    specificity_length_norm = (
        min(n_tokens, norm_params["max_tokens"]) /
        max(norm_params["max_tokens"], 1)
    )
    
    specificity_mean_word_len_norm = (
        min(mean_word_len, norm_params["max_word_len"]) /
        max(norm_params["max_word_len"], 1.0)
    )
    
    specificity_numbers_norm = (
        min(num_numbers, norm_params["max_numbers"]) /
        max(norm_params["max_numbers"], 1)
    )
    
    # Proportion of content (non-stopword) tokens
    specificity_content_prop = n_content / norm_tok if norm_tok > 0 else 0.0
    
    return {
        "jargon_density": jargon_density,
        "lexical_sophistication": lexical_sophistication,
        "syntactic_complexity": syntactic_complexity,
        "hedge_density": hedge_density,
        "booster_density": booster_density,
        "epistemic_balance": epistemic_balance,
        "reasoning_markers": reasoning_markers,
        "evidential_markers": evidential_markers,
        # individual specificity components:
        "specificity_length_norm": specificity_length_norm,
        "specificity_mean_word_len_norm": specificity_mean_word_len_norm,
        "specificity_numbers_norm": specificity_numbers_norm,
        "specificity_content_prop": specificity_content_prop,
        # control features:
        "length_tokens": n_tokens,
        "num_non_function_words": n_content,
    }

# ============================================================================
# Corpus statistics
# ============================================================================
def compute_corpus_statistics(texts: List[str]) -> Dict[str, float]:
    """
    Compute normalization parameters from corpus.
    Uses 95th/5th percentiles for robustness.
    """
    valid_texts = [str(t) for t in texts if t and str(t).strip()]
    
    if not valid_texts:
        return {
            "max_tokens": 40, "max_word_len": 8.0, "max_numbers": 3,
            "zipf_common": 6.5, "zipf_rare": 3.0
        }
    
    # Compute statistics
    lengths = [len(text.split()) for text in valid_texts]
    mean_word_lens = [
        np.mean([len(w) for w in text.split()]) 
        for text in valid_texts if text.split()
    ]
    num_counts = [len(re.findall(r'\b\d+\b', text)) for text in valid_texts]
    
    # Zipf scores
    zipf_scores = []
    for text in valid_texts:
        doc = NLP(text.lower())
        for t in doc:
            if t.is_alpha and not t.is_stop:
                z = token_zipf(t.text)
                if z > 0:
                    zipf_scores.append(z)
    
    # Compute percentiles
    params = {
        "max_tokens": max(1, int(np.percentile(lengths, 95))) if lengths else 40,
        "max_word_len": max(1.0, float(np.percentile(mean_word_lens, 95))) if mean_word_lens else 8.0,
        "max_numbers": max(1, int(np.percentile(num_counts, 95))) if num_counts else 3,
        "zipf_common": float(np.percentile(zipf_scores, 95)) if zipf_scores else 6.5,
        "zipf_rare": float(np.percentile(zipf_scores, 5)) if zipf_scores else 3.0
    }
    
    # Validate Zipf range
    if params["zipf_common"] <= params["zipf_rare"]:
        params["zipf_common"], params["zipf_rare"] = 6.5, 3.0
    
    return params

# ============================================================================
# DataFrame processing
# ============================================================================
def process_dataframe(
    df: pd.DataFrame,
    text_col: str,
    jargon_matcher: Optional[PhraseMatcher],
    reasoning_matcher: Matcher,
    normalize: bool = True
) -> pd.DataFrame:
    """Add features to dataframe and optionally normalize."""
    
    # Detect text column
    text_col = next((c for c in [text_col, "text", "comment"] if c in df.columns), None)
    if not text_col:
        raise ValueError("No text column found")
    
    # Check which features already exist
    existing_features = [f for f in FEATURE_NAMES if f in df.columns]
    new_features = [f for f in FEATURE_NAMES if f not in df.columns]
    
    if existing_features:
        print(f"      Rescoring existing features: {', '.join(existing_features)}")
    if new_features:
        print(f"      Adding new features: {', '.join(new_features)}")
    
    # Compute normalization params
    print(f"      Computing corpus statistics...")
    texts = df[text_col].dropna().astype(str).tolist()
    norm_params = compute_corpus_statistics(texts)
    
    print(f"      Norm bounds: tokens={norm_params['max_tokens']}, "
          f"word_len={norm_params['max_word_len']:.2f}, "
          f"zipf=[{norm_params['zipf_rare']:.2f}, {norm_params['zipf_common']:.2f}]")
    
    # Extract features
    print(f"      Extracting features from {len(df)} rows...")
    features_list = []
    for text in tqdm(df[text_col], desc="      Processing", leave=False):
        try:
            features_list.append(
                extract_features(text, jargon_matcher, reasoning_matcher, norm_params)
            )
        except Exception as e:
            print(f"\n      ⚠️  Error: {e}")
            features_list.append({f: np.nan for f in FEATURE_NAMES})
    
    # Update or add feature columns
    for feature in FEATURE_NAMES:
        df[feature] = [f[feature] for f in features_list]
    
    # Remove old normalized columns if they exist
    old_norm_cols = [f"{f}_norm" for f in FEATURE_NAMES if f"{f}_norm" in df.columns]
    if old_norm_cols:
        print(f"      Removing old normalized columns: {len(old_norm_cols)} columns")
        df = df.drop(columns=old_norm_cols)
    
    # Normalize if requested
    if normalize:
        print(f"      Normalizing features...")
        mask = df[FEATURE_NAMES].notna().all(axis=1)
        if mask.sum() > 0:
            scaler = StandardScaler()
            df.loc[mask, [f"{f}_norm" for f in FEATURE_NAMES]] = scaler.fit_transform(
                df.loc[mask, FEATURE_NAMES]
            )
    
    return df

# ============================================================================
# File processing
# ============================================================================
def process_csv(
    csv_file: Path,
    text_col: str,
    domain_terms: List[str],
    normalize: bool
) -> None:
    """Process single CSV file."""
    print(f"\n   📄 {csv_file.name}")
    
    try:
        df = pd.read_csv(csv_file)
        print(f"      Loaded: {len(df)} rows")
        
        # Build matchers
        jargon_matcher = build_phrase_matcher(domain_terms)
        reasoning_matcher = build_reasoning_matcher()
        
        # Process
        df = process_dataframe(df, text_col, jargon_matcher, reasoning_matcher, normalize)
        
        # Save
        df.to_csv(csv_file, index=False)
        
        # Report
        complete = df[FEATURE_NAMES].notna().all(axis=1).sum()
        print(f"      ✅ Complete ({complete}/{len(df)} substantive)")
        
    except Exception as e:
        print(f"      ❌ Error: {e}")

def process_folder(
    folder: Path,
    topic_key: str,
    domain_terms_dict: Dict[str, List[str]],
    text_col: str,
    normalize: bool
) -> None:
    """Process all CSVs in folder."""
    if not folder.exists():
        return
    
    domain_terms = domain_terms_dict.get(topic_key, [])
    print(f"\n📂 {folder.name} → {topic_key} ({len(domain_terms)} terms)")
    
    for csv_file in sorted(folder.glob("*.csv")):
        process_csv(csv_file, text_col, domain_terms, normalize)

# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Extract expert features")
    parser.add_argument("--processed_dir", type=Path, 
                       default=Path("data/assemblies/processed_files"))
    parser.add_argument("--domain_terms", type=Path,
                       default=Path("code/handcrafted_features/domain_terms_clean_exp.json"))
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--no-normalize", action="store_true")
    args = parser.parse_args()
    
    # Load domain terms
    with open(args.domain_terms) as f:
        domain_terms_dict = json.load(f)
    
    print(f"\n{'='*80}\nEXPERT FEATURE EXTRACTION\n{'='*80}")
    
    # Process experts
    expert_dir = args.processed_dir / "experts"
    if expert_dir.exists():
        print("\n📚 EXPERTS")
        for folder in sorted(expert_dir.iterdir()):
            if folder.is_dir() and folder.name in EXPERT_TOPIC_MAP:
                process_folder(
                    folder, EXPERT_TOPIC_MAP[folder.name],
                    domain_terms_dict, args.text_col, not args.no_normalize
                )
    
    # Process assemblies
    print("\n\n👥 ASSEMBLIES")
    for folder_name, topic_key in ASSEMBLY_TOPIC_MAP.items():
        process_folder(
            args.processed_dir / folder_name, topic_key,
            domain_terms_dict, args.text_col, not args.no_normalize
        )
    
    print(f"\n{'='*80}\n✅ COMPLETE\n{'='*80}")

if __name__ == "__main__":
    main()