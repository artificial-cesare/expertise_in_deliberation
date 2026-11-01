#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust domain jargon extraction with OCR error detection.
"""

import json
import re
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Set
from functools import lru_cache

import spacy
from bs4 import BeautifulSoup
#from autocorrect import Speller

#spell = Speller(lang='en')

try:
    from wordfreq import zipf_frequency, word_frequency
    WORDFREQ_AVAILABLE = True
except ImportError:
    print("❌ wordfreq required: pip install wordfreq")
    exit(1)

try:
    from spellchecker import SpellChecker
    SPELL_CHECKER = SpellChecker()
    SPELLCHECK_AVAILABLE = True
except ImportError:
    SPELLCHECK_AVAILABLE = False
    print("⚠️  Spell checking disabled (install: pip install pyspellchecker)")

# ============================================================================
# Configuration
# ============================================================================
NLP = spacy.load("en_core_web_sm", disable=["ner"])

CONTENT_POS = {"NOUN", "VERB", "ADJ", "PROPN"}
MIN_DOC_FREQ = 2
ENRICHMENT_THRESHOLD = 1.5
ZIPF_MIN = 1.5
ZIPF_MAX = 5.5 #tried harder filter
MIN_TERM_LENGTH = 2
EXTRACT_NGRAMS = True
NGRAM_RANGE = (2, 3)
MAX_UNIGRAMS = 100
MAX_MULTI_WORD = 100
ENABLE_SPELL_CHECK = True
MIN_EDIT_DISTANCE = 2

# Whitelist (condensed for brevity)
DOMAIN_WHITELIST = {
    # UK Political Entities
    'westminster', 'holyrood', 'stormont', 'senedd',
    'labour', 'conservative', 'snp', 'plaid', 'dup', 'ukip',
    
    # Government Terms
    'mep', 'mp', 'mps', 'pm', 'brexit', 'devolution', 'devolved',
    'whitehall', 'sewel',
    
    # Organizations
    'nhs', 'bbc', 'uk', 'eu', 'un', 'nato', 'ehrc', 'ippr', 'oecd',
    'ofcom', 'ico', 'ofsted', 'hmrc',
    
    # Constitutional Terms
    'magna carta', 'commons', 'lords',
    
    # People (common in your corpus)
    'badenoch', 'kemi', 'dinenage', 'penrose', 'gillan', 'heywood', 'colburn',
    
    # Domain-specific terms
    'justiciable', 'enshrine', 'datum', 'gva', 'jobcentre',
    'scrutinise', 'galvanise', 'catalyse', 'privatise',
}

# ============================================================================
# OCR Error Patterns (Centralized)
# ============================================================================
OCR_ERROR_PATTERNS = {
    # Missing first letter
    r'^nder$': 'under',
    r'^elcome$': 'welcome',
    r'^ecause$': 'because',
    r'^ourt$': 'court',
    r'^ommission$': 'commission',
    r'^ection$': 'section',
    r'^eople$': 'people',
    r'^iven$': 'given',
    r'^ork$': 'work',
    r'^ith$': 'with',
    r'^hen$': 'then',
    
    # Truncated words
    r'^practi$': 'practice',
    r'^qualit$': 'quality',
    r'^peech$': 'speech',
    r'^eech$': 'speech',
}

UNCOMMON_STARTS = {
    'ec', 'ou', 'iv', 'or', 'us', 'ig', 'oo', 'ee',
    'ust', 'ith', 'ork', 'ight', 'hen', 'ome', 'urt', 'nder'
}

UNCOMMON_ENDINGS = {
    'cti', 'pti', 'ech', 'eech', 'alit', 'olit', 'nder'
}

# ============================================================================
# Core Validation (Single Source of Truth)
# ============================================================================
@lru_cache(maxsize=10000)
def is_valid_english_word(word: str) -> bool:
    """Check if word is valid English (cached)."""
    if not WORDFREQ_AVAILABLE:
        return len(word) >= 2 and word.isalpha()
    
    freq = word_frequency(word, 'en')
    if freq <= 0:
        return False
    
    zipf = zipf_frequency(word, 'en')
    return zipf >= ZIPF_MIN

@lru_cache(maxsize=10000)
def is_ocr_error(word: str) -> bool:
    """Detect OCR errors using pattern matching (cached)."""
    word = word.lower()
    
    # Whitelist bypass
    if word in DOMAIN_WHITELIST:
        return False
    
    # Explicit pattern matching
    for pattern in OCR_ERROR_PATTERNS:
        if re.match(pattern, word):
            return True
    
    # Generic patterns
    if len(word) >= 4:
        # Uncommon starts
        if any(word.startswith(start) for start in UNCOMMON_STARTS):
            return True
        
        # Uncommon endings
        if any(word.endswith(end) for end in UNCOMMON_ENDINGS):
            return True
        
        # Missing vowels
        vowel_count = sum(1 for c in word if c in 'aeiouy')
        if vowel_count == 0 or (len(word) >= 5 and vowel_count < len(word) / 4):
            return True
        
        # Truncation check
        common_suffixes = ['e', 'es', 'ed', 'ing', 'ly', 'er', 'ion', 'ce', 'ty', 'h']
        if any(is_valid_english_word(word + suffix) for suffix in common_suffixes):
            return True
    
    return False

def is_valid_term(term: str, enrichment_ratio: float = 0.0) -> bool:
    """Validate term (rejects OCR errors, checks English validity)."""
    term_lower = term.lower()
    
    # Whitelist
    if term_lower in DOMAIN_WHITELIST:
        return True
    
    words = term_lower.split()
    
    # Single-letter spacing error
    if re.match(r'^[a-z] [a-z]+$', term_lower):
        return False
    
    # Check each word
    for word in words:
        if len(word) <= 2:
            continue
        
        if word in DOMAIN_WHITELIST:
            continue
        
        # ✅ Reject OCR errors
        if is_ocr_error(word):
            return False
    
    # Single word
    if len(words) == 1:
        if is_valid_english_word(term_lower):
            return True
        
        # High enrichment acronyms
        if term.isupper() and len(term) <= 6 and enrichment_ratio > 50.0:
            return True
        
        return False
    
    # Multi-word: at least 50% valid
    valid_count = sum(
        1 for w in words
        if is_valid_english_word(w) or w in DOMAIN_WHITELIST
    )
    return valid_count / len(words) >= 0.5

# ============================================================================
# Spell Checking (Simplified)
# ============================================================================
def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate edit distance."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

@lru_cache(maxsize=10000)
def spell_check_word(word: str) -> str:
    """Correct single word spelling (cached)."""
    if not ENABLE_SPELL_CHECK or not SPELLCHECK_AVAILABLE:
        return word
    
    word_lower = word.lower()
    
    # Don't correct valid words
    if word_lower in DOMAIN_WHITELIST:
        return word
    
    # Try spell correction
    if word in SPELL_CHECKER.unknown([word]):
        correction = SPELL_CHECKER.correction(word)
        if correction and correction != word:
            dist = levenshtein_distance(word, correction)
            if dist <= MIN_EDIT_DISTANCE:
                return correction
    
    return word

def spell_check_term(term: str) -> str:
    """Correct term spelling."""
    words = term.split()
    corrected = [spell_check_word(w) for w in words]
    return ' '.join(corrected)

# ============================================================================
# Text Extraction
# ============================================================================
def extract_text_from_html(html_file: Path) -> str:
    """Extract clean text from HTML."""
    try:
        with open(html_file, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
        
        for script in soup(["script", "style"]):
            script.decompose()
        
        text = soup.get_text(separator=' ', strip=True)
        return re.sub(r'\s+', ' ', text)
    
    except Exception as e:
        print(f"   ⚠️  Error reading {html_file.name}: {e}")
        return ""

def extract_ngrams(tokens: List[str], n: int) -> List[str]:
    """Extract n-grams."""
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tokens[i:i+n]
        if all(len(w) >= 2 for w in ngram):
            ngrams.append(' '.join(ngram))
    return ngrams

def extract_terms(text: str) -> Tuple[List[str], List[str]]:
    """Extract unigrams and multi-word terms."""
    doc = NLP(text)
    
    unigrams = []
    lemmatized = []
    
    for token in doc:
        if (token.pos_ in CONTENT_POS and token.is_alpha and
            len(token.text) >= MIN_TERM_LENGTH and not token.is_stop):
            lemma = token.lemma_.lower()
            unigrams.append(lemma)
            lemmatized.append(lemma)
    
    multi_word = []
    
    # Noun phrases
    for chunk in doc.noun_chunks:
        phrase = chunk.text.lower().strip()
        words = phrase.split()
        
        if (2 <= len(words) <= 4 and len(phrase) >= 6 and
            words[0] not in {'the', 'this', 'that', 'a', 'an'}):
            multi_word.append(phrase)
    
    # N-grams
    if EXTRACT_NGRAMS and len(lemmatized) >= NGRAM_RANGE[0]:
        for n in range(NGRAM_RANGE[0], NGRAM_RANGE[1] + 1):
            multi_word.extend(extract_ngrams(lemmatized, n))
    
    return unigrams, multi_word

# ============================================================================
# Frequency & Enrichment
# ============================================================================
def get_english_frequency(term: str) -> float:
    """Get English frequency (per million)."""
    words = term.split()
    if len(words) == 1:
        return word_frequency(term, 'en') * 1_000_000
    else:
        freqs = [word_frequency(w, 'en') * 1_000_000 for w in words]
        return min(freqs) if freqs else 0.01

def compute_enrichment_score(
    term: str,
    corpus_freq: float,
    doc_count: int,
    total_docs: int
) -> float:
    """Compute enrichment score."""
    eng_freq = max(get_english_frequency(term), 0.01)
    freq_ratio = corpus_freq / eng_freq
    
    # Document spread bonus
    spread = doc_count / total_docs
    spread_score = max(0.1, 1.0 - abs(spread - 0.5) * 2)
    
    # Zipf penalty
    zipf = zipf_frequency(term.split()[0], 'en')
    if zipf < ZIPF_MIN:
        zipf_penalty = 0.5
    elif zipf > ZIPF_MAX:
        zipf_penalty = 0.7
    else:
        zipf_penalty = 1.0
    
    return freq_ratio * spread_score * zipf_penalty

# ============================================================================
# Main Pipeline
# ============================================================================
def extract_domain_jargon(topic_dir: Path) -> List[Dict]:
    """Extract domain jargon from HTML files."""
    print(f"\n📂 Processing: {topic_dir.name}")
    
    html_files = sorted(topic_dir.glob("*.html"))
    if not html_files:
        print(f"   ⚠️  No HTML files found")
        return []
    
    print(f"   Found {len(html_files)} HTML files")
    
    # Extract terms from all documents
    all_unigrams = []
    all_multi_word = []
    doc_term_sets = []
    
    for html_file in html_files:
        text = extract_text_from_html(html_file)
        if not text:
            continue
        
        unigrams, multi_word = extract_terms(text)
        all_unigrams.extend(unigrams)
        all_multi_word.extend(multi_word)
        doc_term_sets.append(set(unigrams + multi_word))
    
    total_docs = len(doc_term_sets)
    if total_docs == 0:
        print(f"   ⚠️  No text extracted")
        return []
    
    print(f"   Extracted from {total_docs} documents")
    
    # Compute frequencies
    all_terms = all_unigrams + all_multi_word
    total_tokens = len(all_terms)
    term_counts = Counter(all_terms)
    doc_freq = Counter()
    for doc_terms in doc_term_sets:
        for term in doc_terms:
            doc_freq[term] += 1
    
    print(f"   Unique terms: {len(term_counts)}")
    print(f"   Filtering OCR errors, spell-checking, and validating...")
    
    # ========================================================================
    # ✅ FIX: Pre-filter OCR errors BEFORE processing
    # ========================================================================
    ocr_filtered = 0
    spell_corrected = 0
    invalid_count = 0
    term_data = []
    
    for term, count in term_counts.items():
        # 1. Skip low document frequency
        if doc_freq[term] < MIN_DOC_FREQ:
            continue
        
        # 2. ✅ Reject OCR errors FIRST (before spell check)
        if is_ocr_error(term):
            ocr_filtered += 1
            continue
        
        # 3. Compute frequencies
        corpus_freq = (count / total_tokens) * 1_000_000
        eng_freq = max(get_english_frequency(term), 0.01)
        freq_ratio = corpus_freq / eng_freq
        
        if freq_ratio < ENRICHMENT_THRESHOLD:
            continue
        
        # 4. Spell check (only non-OCR errors)
        original_term = term
        term = spell_check_term(term)
        if term != original_term:
            spell_corrected += 1
            # Recalculate with corrected term
            if term in term_counts:
                count = term_counts[term]
                corpus_freq = (count / total_tokens) * 1_000_000
                freq_ratio = corpus_freq / eng_freq
        
        # 5. Validate corrected term
        if not is_valid_term(term, freq_ratio):
            invalid_count += 1
            continue
        
        # 6. Compute enrichment
        enrichment = compute_enrichment_score(
            term, corpus_freq, doc_freq[original_term], total_docs
        )
        
        word_count = len(term.split())
        term_type = {1: 'unigram', 2: 'bigram', 3: 'trigram'}.get(word_count, 'phrase')
        
        term_data.append({
            'term': term,
            'corpus_freq': corpus_freq,
            'english_freq': eng_freq,
            'freq_ratio': freq_ratio,
            'doc_count': doc_freq[original_term],
            'doc_coverage': doc_freq[original_term] / total_docs,
            'enrichment_score': enrichment,
            'term_type': term_type,
            'is_phrase': ' ' in term
        })
    
    term_data.sort(key=lambda x: x['enrichment_score'], reverse=True)
    
    print(f"   ✅ Found {len(term_data)} domain-enriched terms")
    print(f"   🚫 Filtered: {ocr_filtered} OCR errors, {invalid_count} invalid terms")
    if SPELLCHECK_AVAILABLE:
        print(f"   ✏️  Spell-corrected {spell_corrected} terms")
    
    return term_data

def format_output(topic_name: str, term_data: List[Dict]) -> List[Dict]:
    """Format and print results with dynamic allocation."""
    if not term_data:
        print(f"\n   No jargon found for {topic_name}")
        return []
    
    # Separate unigrams and multi-word
    unigrams = [t for t in term_data if not t['is_phrase']]
    multi_word = [t for t in term_data if t['is_phrase']]
    
    # ========================================================================
    # ✅ DYNAMIC ALLOCATION: Fill shortage with unigrams
    # ========================================================================
    n_multi = min(len(multi_word), MAX_MULTI_WORD)
    multi_shortage = MAX_MULTI_WORD - n_multi
    
    # If multi-word is short, allocate extra slots to unigrams
    n_uni = min(len(unigrams), MAX_UNIGRAMS + multi_shortage)
    
    # ========================================================================
    
    selected = unigrams[:n_uni] + multi_word[:n_multi]
    selected.sort(key=lambda x: x['enrichment_score'], reverse=True)
    
    type_counts = Counter([t['term_type'] for t in selected])
    
    print(f"\n   📊 Selected: {len(selected)} terms")
    print(f"      - Unigrams: {n_uni}/{len(unigrams)}")
    print(f"      - Multi-word: {n_multi}/{len(multi_word)}")
    
    # ✅ Show allocation adjustment
    if multi_shortage > 0:
        print(f"      ⚙️  Allocated {multi_shortage} extra unigram slots (multi-word shortage)")
    
    print(f"\n   {'Rank':<5} {'Term':<35} {'Type':<8} {'Ratio':<10} {'Score':<8}")
    print(f"   {'-'*70}")
    
    for i, item in enumerate(selected[:20], 1):
        print(f"   {i:<5} {item['term'][:33]:<35} "
              f"{item['term_type']:<8} "
              f"{item['freq_ratio']:.1f}x{'':<6} "
              f"{item['enrichment_score']:.2f}")
    
    if len(selected) > 20:
        print(f"   ... and {len(selected) - 20} more")
    
    return selected

def build_domain_vocabulary(vocab_dir: Path, output_file: Path) -> None:
    """Build vocabulary JSON."""
    print("=" * 80)
    print("🔍 DOMAIN JARGON EXTRACTION")
    print("=" * 80)
    print(f"Config: enrichment={ENRICHMENT_THRESHOLD}x, "
          f"min_docs={MIN_DOC_FREQ}, zipf=[{ZIPF_MIN}, {ZIPF_MAX}]")
    
    topics = {
        "constitution": "constitutional reform",
        "future": "public policy",
        "gender": "gender equality"
    }
    
    domain_terms = {}
    
    for folder, topic_key in topics.items():
        topic_dir = vocab_dir / folder
        if not topic_dir.exists():
            print(f"\n⚠️  Topic directory not found: {topic_dir}")
            continue
        
        term_data = extract_domain_jargon(topic_dir)
        selected = format_output(topic_key, term_data)
        domain_terms[topic_key] = [item['term'] for item in selected]
    
    print("\n" + "=" * 80)
    print(f"💾 Saving to: {output_file}")
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(domain_terms, f, indent=2, ensure_ascii=False)
    
    print("\n📊 SUMMARY:")
    for topic, terms in domain_terms.items():
        uni = len([t for t in terms if ' ' not in t])
        multi = len([t for t in terms if ' ' in t])
        print(f"   {topic}: {len(terms)} terms ({uni} uni, {multi} multi)")
    
    print("\n✅ COMPLETE!\n" + "=" * 80)

# ============================================================================
# CLI
# ============================================================================
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract domain jargon")
    parser.add_argument("--vocab_dir", type=Path, default=Path("data/vocab"))
    parser.add_argument("--output", type=Path,
                       default=Path("code/handcrafted_features/domain_terms_clean_exp.json"))
    parser.add_argument("--enrichment", type=float, default=1.5)
    parser.add_argument("--no-spell-check", action="store_true")
    
    args = parser.parse_args()
    
    global ENRICHMENT_THRESHOLD, ENABLE_SPELL_CHECK
    ENRICHMENT_THRESHOLD = args.enrichment
    ENABLE_SPELL_CHECK = not args.no_spell_check
    
    build_domain_vocabulary(args.vocab_dir, args.output)

if __name__ == "__main__":
    main()