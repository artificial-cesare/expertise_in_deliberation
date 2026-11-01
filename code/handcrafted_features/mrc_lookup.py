#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MRC Psycholinguistic Database lookup for Familiarity, Concreteness, and Imageability.
"""

import sqlite3
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np


class MRCLookup:
    """
    Efficient lookup for MRC psycholinguistic features.
    
    Features (scale 100-700, where 0 = missing):
    - FAM (Familiarity): How familiar the word is (700 = very familiar)
    - CONC (Concreteness): How concrete vs abstract (700 = very concrete)
    - IMAG (Imageability): How easily it evokes a mental image (700 = very imageable)
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Load MRC data into memory for fast lookups.
        
        Args:
            db_path: Path to mrc2.db file
        """
        self.word_data: Dict[str, Tuple[int, int, int]] = {}
        self.loaded = False
        
        # Store corpus-level statistics for imputation
        self.corpus_mean_fam = 488.0  # Approximate corpus mean
        self.corpus_mean_conc = 438.0
        self.corpus_mean_imag = 450.0
        
        if db_path is None:
            # Default path relative to thesis root
            db_path = Path(__file__).parent.parent.parent / "data" / "mrc-psycholinguistics-master" / "mrc2.db"
        
        if db_path.exists():
            self._load_database(db_path)
        else:
            print(f"⚠️  MRC database not found at {db_path}")
            print("   Run: cd data/mrc-psycholinguistics-master && python extract.py")
    
    def _load_database(self, db_path: Path) -> None:
        """Load relevant columns into memory dictionary."""
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Select only words that have at least one non-zero rating
            cursor.execute("""
                SELECT word, fam, conc, imag 
                FROM word 
                WHERE fam > 0 OR conc > 0 OR imag > 0
            """)
            
            all_fam, all_conc, all_imag = [], [], []
            
            for row in cursor.fetchall():
                word = row[0].lower().strip()
                fam, conc, imag = row[1], row[2], row[3]
                self.word_data[word] = (fam, conc, imag)
                
                # Collect for corpus statistics
                if fam > 0:
                    all_fam.append(fam)
                if conc > 0:
                    all_conc.append(conc)
                if imag > 0:
                    all_imag.append(imag)
            
            conn.close()
            self.loaded = True
            
            # Compute actual corpus means for better imputation
            if all_fam:
                self.corpus_mean_fam = float(np.mean(all_fam))
            if all_conc:
                self.corpus_mean_conc = float(np.mean(all_conc))
            if all_imag:
                self.corpus_mean_imag = float(np.mean(all_imag))
            
            # Stats
            n_with_fam = len(all_fam)
            n_with_conc = len(all_conc)
            n_with_imag = len(all_imag)
            
            print(f"✅ Loaded MRC database: {len(self.word_data)} words")
            print(f"   FAM: {n_with_fam} (mean={self.corpus_mean_fam:.0f})")
            print(f"   CONC: {n_with_conc} (mean={self.corpus_mean_conc:.0f})")
            print(f"   IMAG: {n_with_imag} (mean={self.corpus_mean_imag:.0f})")
            
        except Exception as e:
            print(f"❌ Error loading MRC database: {e}")
            self.loaded = False
    
    def lookup(self, word: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Look up a word's psycholinguistic ratings.
        
        Args:
            word: The word to look up (case-insensitive)
            
        Returns:
            (familiarity, concreteness, imageability) - None if not found or 0
        """
        if not self.loaded:
            return None, None, None
        
        word_lower = word.lower().strip()
        
        if word_lower in self.word_data:
            fam, conc, imag = self.word_data[word_lower]
            # Convert 0 to None (0 means "not rated" in MRC)
            return (
                fam if fam > 0 else None,
                conc if conc > 0 else None,
                imag if imag > 0 else None
            )
        
        return None, None, None
    
    def lookup_lemma(self, word: str, lemma: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Try word first, then fall back to lemma.
        """
        result = self.lookup(word)
        if result == (None, None, None):
            result = self.lookup(lemma)
        return result


def compute_mrc_features(doc, mrc: MRCLookup) -> Dict[str, float]:
    """
    Compute MRC-based features for a spaCy Doc.
    
    Returns dict with:
    - mrc_familiarity: Weighted familiarity score (0-1)
    - mrc_concreteness: Weighted concreteness score (0-1)
    - mrc_imageability: Weighted imageability score (0-1)
    
    Strategy to avoid sparsity and handle sentence length:
    1. Use lemma fallback for better coverage
    2. Normalize from 100-700 scale to 0-1
    3. Only consider content words (not stopwords)
    4. Weight by coverage: blend found-word average with corpus mean
       based on how many words were actually found
    5. Apply confidence weighting based on number of found words
    """
    # Default neutral values (normalized corpus means)
    default_fam = (mrc.corpus_mean_fam - 100) / 600 if mrc.loaded else 0.5
    default_conc = (mrc.corpus_mean_conc - 100) / 600 if mrc.loaded else 0.5
    default_imag = (mrc.corpus_mean_imag - 100) / 600 if mrc.loaded else 0.5
    
    if not mrc.loaded:
        return {
            "mrc_familiarity": default_fam,
            "mrc_concreteness": default_conc,
            "mrc_imageability": default_imag,
        }
    
    # Collect ratings for content words only
    fam_scores = []
    conc_scores = []
    imag_scores = []
    n_content_words = 0
    
    for token in doc:
        # Skip non-content words
        if not token.is_alpha or token.is_stop:
            continue
        
        n_content_words += 1
        
        # Try word, then lemma
        fam, conc, imag = mrc.lookup_lemma(token.text, token.lemma_)
        
        # Collect non-None scores
        if fam is not None:
            fam_scores.append(fam)
        if conc is not None:
            conc_scores.append(conc)
        if imag is not None:
            imag_scores.append(imag)
    
    def compute_weighted_score(scores: list, corpus_mean: float, n_content: int) -> float:
        if n_content == 0 or len(scores) == 0:
            return np.nan  # Return NaN instead of corpus mean - let imputation handle it
        
        n_found = len(scores)
        coverage = n_found / n_content
        
        # Only blend if coverage is very low (< 20%)
        # Otherwise, trust the sample
        if coverage < 0.2 and n_found < 3:
            # Low confidence - blend with corpus mean
            confidence = coverage * (n_found / 3)  # Max confidence at 3+ words
            blended = confidence * np.mean(scores) + (1 - confidence) * corpus_mean
        else:
            # High confidence - use sample mean directly
            blended = np.mean(scores)
        
        normalized = (blended - 100) / 600
        return float(np.clip(normalized, 0, 1))
    
    familiarity = compute_weighted_score(fam_scores, mrc.corpus_mean_fam, n_content_words)
    concreteness = compute_weighted_score(conc_scores, mrc.corpus_mean_conc, n_content_words)
    imageability = compute_weighted_score(imag_scores, mrc.corpus_mean_imag, n_content_words)
    
    return {
        "mrc_familiarity": familiarity,
        "mrc_concreteness": concreteness,
        "mrc_imageability": imageability,
    }


# Quick test
if __name__ == "__main__":
    import spacy
    nlp = spacy.load("en_core_web_sm")
    
    mrc = MRCLookup()
    
    # Test individual lookups
    test_words = ["democracy", "apple", "freedom", "table", "justice", "dog"]
    print("\n📊 Individual word lookups:")
    print(f"{'Word':<15} {'FAM':>6} {'CONC':>6} {'IMAG':>6}")
    print("-" * 40)
    for word in test_words:
        fam, conc, imag = mrc.lookup(word)
        print(f"{word:<15} {str(fam):>6} {str(conc):>6} {str(imag):>6}")
    
    # Test on sample texts - including short and long sentences
    texts = [
        # Very short
        "Good.",
        "I agree.",
        "Yes, definitely.",
        # Short
        "The apple is red.",
        "Democracy matters.",
        # Medium
        "The constitutional reform requires careful deliberation by elected representatives.",
        "I saw a big red apple on the wooden table in the kitchen.",
        "Democracy and freedom are fundamental values in our society.",
        # Long
        "The comprehensive analysis of constitutional amendments demonstrates the complex interplay between legislative intent and judicial interpretation in modern democratic governance systems.",
        # Abstract vs concrete
        "Justice, equality, and freedom are abstract concepts that shape our moral framework.",
        "The dog ran across the green grass and jumped over the wooden fence into the garden.",
    ]
    
    print("\n📊 Document-level features (with length-aware weighting):")
    print(f"{'Tokens':<8} {'Content':<8} {'FAM':>8} {'CONC':>8} {'IMAG':>8}  Text")
    print("-" * 100)
    
    for text in texts:
        doc = nlp(text)
        features = compute_mrc_features(doc, mrc)
        n_tokens = len([t for t in doc if t.is_alpha])
        n_content = len([t for t in doc if t.is_alpha and not t.is_stop])
        
        print(f"{n_tokens:<8} {n_content:<8} {features['mrc_familiarity']:>8.3f} "
              f"{features['mrc_concreteness']:>8.3f} {features['mrc_imageability']:>8.3f}  "
              f"{text[:50]}{'...' if len(text) > 50 else ''}")
    
    # Show the effect of sentence length on confidence
    print("\n📊 Confidence weighting demonstration:")
    print("(Same word 'apple' repeated N times)")
    for n in [1, 2, 3, 5, 10, 20]:
        text = " ".join(["apple"] * n)
        doc = nlp(text)
        features = compute_mrc_features(doc, mrc)
        print(f"  N={n:>2}: FAM={features['mrc_familiarity']:.3f}, "
              f"CONC={features['mrc_concreteness']:.3f}, "
              f"IMAG={features['mrc_imageability']:.3f}")