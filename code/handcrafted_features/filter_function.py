#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filter function to identify substantive speech acts based on content word count.
"""

def is_substantive_speech_act(num_non_function_words: int, threshold: int = 5) -> bool:
    """
    Determine if a speech act is substantive based on non-function word count.
    The logic is that simple opinions are not relevant to expert behavior, they represent rather stance than argument,
    so we filter out utterances with very few content words.
    
    Args:
        num_non_function_words: Count of content words (non-stopwords)
        threshold: Minimum number of content words required (default: 5)
        
    Returns:
        True if speech act meets threshold, False otherwise
    
    Suggested thresholds:
        - 3: Very permissive (includes brief responses like "I think so too")
        - 5: Moderate (default, filters very short utterances)
        - 10: Strict (only substantive contributions)
        - 15: Very strict (only extended arguments)
    """
    return num_non_function_words >= threshold


def filter_dataframe(df, threshold: int = 5, column: str = "num_non_function_words"):
    """
    Filter a dataframe to keep only substantive speech acts.
    
    Args:
        df: Pandas DataFrame with feature columns
        threshold: Minimum content words required
        column: Name of the column containing non-function word counts
        
    Returns:
        Filtered DataFrame
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    mask = df[column] >= threshold
    filtered_df = df[mask].copy()
    
    print(f"Original rows: {len(df)}")
    print(f"Filtered rows (>= {threshold} content words): {len(filtered_df)}")
    print(f"Removed: {len(df) - len(filtered_df)} ({100*(len(df)-len(filtered_df))/len(df):.1f}%)")
    
    return filtered_df


# Example usage:
if __name__ == "__main__":
    import pandas as pd
    
    # Test the function
    test_data = {
        'text': ['Yes', 'I agree with that', 'This is a complex issue that requires careful consideration'],
        'num_non_function_words': [1, 2, 8]
    }
    df = pd.DataFrame(test_data)
    
    print("Testing different thresholds:\n")
    for thresh in [3, 5, 10]:
        print(f"\nThreshold = {thresh}:")
        filtered = filter_dataframe(df, threshold=thresh)
        print(filtered[['text', 'num_non_function_words']])